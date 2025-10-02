"""
Cliente e utilitários para integração com a API OpenAI.
"""
import os
import json
import time
import logging
import hashlib
import pickle
from typing import Optional, List, Dict, Any, Sequence
from openai import OpenAI

from .config import (
    SUMMARY_SYS_INSTR, TOPIC_SUMMARY_SYS_INSTR, 
    SUMMARY_BATCH_SIZE, SUMMARY_MAX_CHARS_PER_ITEM,
    TOPIC_SUMMARY_BATCH_SIZE, TOPIC_SUMMARY_MAX_ITEMS, 
    TOPIC_SUMMARY_MAX_CHARS, MAX_API_RETRIES
)
from .cost_tracker import registrar_uso
from .text_classifier import truncate, gerar_resumo_fallback, gerar_resumo_topico_fallback


def build_client() -> Optional[OpenAI]:
    """
    Constrói e retorna um cliente OpenAI se a chave da API estiver disponível.
    
    Returns:
        OpenAI client se a chave estiver configurada, None caso contrário
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OPENAI_API_KEY ausente; seguirei apenas com heurísticas locais.")
        return None
    return OpenAI(api_key=api_key)


def resumir_em_lote(
    client: Optional[OpenAI],
    textos: List[str],
    model: str,
    batch_size: int = SUMMARY_BATCH_SIZE,
    max_chars: int = SUMMARY_MAX_CHARS_PER_ITEM,
    max_retries: int = MAX_API_RETRIES,
    cache_file: str = "cache_summaries.pkl",
) -> List[str]:
    """
    Gera resumos para uma lista de textos usando a API OpenAI em lotes.
    
    Args:
        client: Cliente OpenAI (None para usar apenas heurísticas locais)
        textos: Lista de textos para resumir
        model: Nome do modelo OpenAI a usar
        batch_size: Número de textos por lote
        max_chars: Máximo de caracteres por texto
        max_retries: Número máximo de tentativas em caso de erro
        cache_file: Arquivo para cache local dos resumos
        
    Returns:
        Lista de resumos (mesmo tamanho da lista de entrada)
    """
    # Carrega cache existente
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    resultados = ["" for _ in textos]

    # Verifica cache
    for i, texto in enumerate(textos):
        h = hashlib.sha256(texto.encode("utf-8")).hexdigest()
        if h in cache:
            resultados[i] = cache[h]

    # Se não há cliente OpenAI, usa apenas heurísticas locais
    if client is None:
        for i, t in enumerate(textos):
            if not resultados[i]:
                resumo = gerar_resumo_fallback(t)
                resultados[i] = resumo
                cache[hashlib.sha256(t.encode("utf-8")).hexdigest()] = resumo
        
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
        return resultados

    # Processa textos em lotes
    for start in range(0, len(textos), batch_size):
        bloco_idx = list(range(start, min(start + batch_size, len(textos))))
        bloco = [truncate(textos[i], max_chars) for i in bloco_idx]
        
        user_lines = [f"{i-start}. {bloco[i-start]}" for i in bloco_idx if not resultados[i]]
        
        if not user_lines:
            continue
            
        user_prompt = (
            "Resuma os itens numerados abaixo. Produza JSON com [{\"i\": <indice_global>, \"resumo\": \"...\"}].\n\n"
            "ITENS:\n" + "\n".join(user_lines) + "\n\n"
            "Observação: 'i' deve ser o ÍNDICE GLOBAL original que estou te passando agora: "
            f"{bloco_idx}"
        )

        last_exc: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SUMMARY_SYS_INSTR},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    registrar_uso(model, {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    })
                
                content = (resp.choices[0].message.content or "").strip()
                
                if content.startswith("```"):
                    content = content.strip("`")
                    if content.startswith("json"):
                        content = content[4:]
                
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("JSON não é uma lista")
                
                for item in data:
                    gi = int(item.get("i", -1))
                    resumo = str(item.get("resumo", "")).strip()
                    
                    if 0 <= gi < len(resultados):
                        if not resumo:
                            resumo = gerar_resumo_fallback(textos[gi])
                        resultados[gi] = resumo
                        cache[hashlib.sha256(textos[gi].encode("utf-8")).hexdigest()] = resumo
                break
            except Exception as exc:
                last_exc = exc
                logging.warning(
                    "Falha ao resumir bloco %s-%s (tentativa %s/%s): %s",
                    start, start + len(bloco) - 1, attempt, max_retries, exc,
                )
                time.sleep(attempt)
        else:
            logging.error(
                "Falha definitiva ao resumir bloco %s-%s: %s",
                start, start + len(bloco) - 1, last_exc,
            )
            for gi in bloco_idx:
                if not resultados[gi]:
                    resumo = gerar_resumo_fallback(textos[gi])
                    resultados[gi] = resumo
                    cache[hashlib.sha256(textos[gi].encode("utf-8")).hexdigest()] = resumo

    # Garantia final
    for idx, resumo in enumerate(resultados):
        if not str(resumo).strip():
            resumo = gerar_resumo_fallback(textos[idx])
            resultados[idx] = resumo
            cache[hashlib.sha256(textos[idx].encode("utf-8")).hexdigest()] = resumo

    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)
    
    return resultados


def gerar_resumos_por_topico(
    client: Optional[OpenAI],
    topico_para_textos: Dict[str, List[str]],
    model: str,
    topico_para_rotulos: Optional[Dict[str, Sequence[str]]] = None,
    batch_size: int = TOPIC_SUMMARY_BATCH_SIZE,
    max_items: int = TOPIC_SUMMARY_MAX_ITEMS,
    max_chars: int = TOPIC_SUMMARY_MAX_CHARS,
    max_retries: int = MAX_API_RETRIES,
) -> Dict[str, str]:
    """
    Gera resumos por tópico usando a API OpenAI.
    
    Args:
        client: Cliente OpenAI
        topico_para_textos: Mapeamento de tópicos para listas de textos
        model: Nome do modelo OpenAI
        topico_para_rotulos: Mapeamento de tópicos para rótulos (opcional)
        batch_size: Número de tópicos por lote
        max_items: Máximo de itens por tópico
        max_chars: Máximo de caracteres por item
        max_retries: Número máximo de tentativas
        
    Returns:
        Dicionário mapeando tópicos para seus resumos
    """
    if not topico_para_textos:
        return {}

    resultados: Dict[str, str] = {}
    itens = list(topico_para_textos.items())

    for inicio in range(0, len(itens), batch_size):
        bloco = itens[inicio:inicio + batch_size]
        chunk_topics = [topico for topico, _ in bloco]

        secoes_usuario: List[str] = []
        for topico, textos in bloco:
            exemplos = textos[:max_items] or [""]
            trechos = [truncate(texto, max_chars) for texto in exemplos]
            linhas = "\n".join(f"- {linha}" if linha else "- (conteúdo vazio)" for linha in trechos)
            secoes_usuario.append(f"Tópico: {topico}\nOcorrências:\n{linhas}")

        user_prompt = (
            "Analise os tópicos abaixo. Para cada tópico, identifique os principais problemas relatados, "
            "padrões de causa e impactos observados. Escreva um resumo curto, objetivo e organizado. "
            "Use o nome EXATO do tópico.\n\n" + "\n\n".join(secoes_usuario) +
            "\n\nSaída: forneça estritamente JSON como uma lista de objetos {\"topico\": <nome_exato>, \"resumo\": <texto>}"
        )

        if client is None:
            for t, textos in bloco:
                rotulos = topico_para_rotulos.get(t) if topico_para_rotulos else None
                resultados[t] = gerar_resumo_topico_fallback(t, textos, rotulos)
            continue

        ultima_excecao = None
        for tentativa in range(1, max_retries + 1):
            try:
                resposta = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": TOPIC_SUMMARY_SYS_INSTR},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                
                usage = getattr(resposta, "usage", None)
                if usage is not None:
                    registrar_uso(model, {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    })
                
                content = (resposta.choices[0].message.content or "").strip()
                
                if content.startswith("```"):
                    content = content.strip("`")
                    if content.startswith("json"):
                        content = content[4:]
                
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("Resposta não é uma lista JSON")
                
                for item in data:
                    topico_nome = item.get("topico", "").strip()
                    resumo_texto = item.get("resumo", "").strip()
                    
                    if topico_nome in chunk_topics:
                        resultados[topico_nome] = resumo_texto or f"Resumo vazio para '{topico_nome}'"
                
                break
            except Exception as exc:
                ultima_excecao = exc
                logging.warning(f"Falha ao gerar resumos por tópico (tentativa {tentativa}/{max_retries}): {exc}")
                time.sleep(tentativa)
        else:
            logging.error(f"Falha definitiva ao gerar resumos por tópico: {ultima_excecao}")
            for t, textos in bloco:
                if t not in resultados:
                    rotulos = topico_para_rotulos.get(t) if topico_para_rotulos else None
                    resultados[t] = gerar_resumo_topico_fallback(t, textos, rotulos)

    return resultados