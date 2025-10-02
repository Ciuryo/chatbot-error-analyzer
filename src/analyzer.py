import json          # Para manipulação de dados JSON
import logging       # Para sistema de logs
import os           # Para variáveis de ambiente e sistema
import sys          # Para operações do sistema
import time         # Para delays e timestamps
import re           # Para expressões regulares
import textwrap     # Para formatação de texto
from pathlib import Path                    # Para manipulação de caminhos de arquivos
from typing import List, Optional, Sequence, Dict, Any  # Para type hints
from datetime import datetime              # Para manipulação de datas
from collections import Counter            # Para contagem de elementos

# Imports para análise de dados e machine learning
import numpy as np                         # Para operações numéricas
import pandas as pd                        # Para manipulação de DataFrames
from dotenv import load_dotenv            # Para carregar variáveis de ambiente
from openai import OpenAI                 # Cliente da API OpenAI
import hashlib                            # Para gerar hashes (cache)
import pickle                             # Para serialização de objetos

# Imports para machine learning
from sklearn.linear_model import LogisticRegression      # Modelo de classificação
from sklearn.metrics import classification_report, f1_score  # Métricas de avaliação
from sklearn.model_selection import train_test_split    # Divisão de dados
from sklearn.pipeline import make_pipeline              # Pipeline de ML
from sklearn.feature_extraction.text import TfidfVectorizer  # Vetorização TF-IDF

# Imports para geração de PDF e visualizações
import matplotlib.pyplot as plt           # Para gráficos
import seaborn as sns                     # Para visualizações estatísticas
from matplotlib.backends.backend_pdf import PdfPages  # Para geração de PDF

# Dependência opcional para detecção de encoding
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    logging.info("chardet não disponível. Detecção automática de encoding limitada.")

# =============================
# CONFIGURAÇÕES PADRÃO DO SISTEMA
# =============================
# Estas configurações podem ser sobrescritas por variáveis de ambiente

# Arquivos CSV padrão para análise (dados históricos do chatbot)
DEFAULT_CONVERSATION_FILE = "21_filtrado_21.csv"

# Configurações de colunas dos dados
DEFAULT_MESSAGE_COLUMN = "Observação"  # Coluna que contém o texto das interações
DEFAULT_LABEL_COLUMN = "rotulo"        # Coluna de classificação (criada se não existir)

# Arquivos de saída
DEFAULT_OUTPUT_FILE = "relatorio_erro_input.csv"           # Relatório CSV
DEFAULT_PDF_OUTPUT = "Analise_Erros_Chatbot_Diagnostico.pdf"  # Relatório PDF

# Modelos da OpenAI utilizados
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"              # Modelo para geração de resumos
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"  # Modelo para embeddings

# Configurações de API
MAX_API_RETRIES = 3  # Número máximo de tentativas em caso de falha na API

# =============================
# SISTEMA DE CONTROLE DE CUSTOS DA API
# =============================

# Dicionário global para rastrear uso de tokens e custos
USAGE_STATS = {
    "tokens_in": 0,      # Tokens de entrada (prompts)
    "tokens_out": 0,     # Tokens de saída (respostas)
    "tokens_total": 0,   # Total de tokens usados
    "emb_tokens": 0,     # Tokens usados para embeddings
    "usd_total": 0.0     # Custo total estimado em USD
}

# Tabela de preços da OpenAI por 1000 tokens (valores aproximados em USD)
MODEL_PRICES = {
    "gpt-4.1-mini": {"in": 0.00015, "out": 0.0006},        # GPT-4 mini: entrada e saída
    "text-embedding-3-large": {"in": 0.00013},             # Embeddings: apenas entrada
}

def registrar_uso(model: str, usage: Dict[str, Any]):
    """
    Registra o uso de tokens e calcula custos para modelos de chat.
    
    Args:
        model: Nome do modelo utilizado
        usage: Dicionário com informações de uso retornado pela API OpenAI
    """
    global USAGE_STATS
    if not usage:
        return
    
    # Extrai informações de uso da resposta da API
    in_toks = usage.get("prompt_tokens", 0)      # Tokens do prompt
    out_toks = usage.get("completion_tokens", 0)  # Tokens da resposta
    total_toks = usage.get("total_tokens", in_toks + out_toks)
    
    # Atualiza estatísticas globais
    USAGE_STATS["tokens_in"] += in_toks
    USAGE_STATS["tokens_out"] += out_toks
    USAGE_STATS["tokens_total"] += total_toks
    
    # Calcula custo se o modelo estiver na tabela de preços
    if model in MODEL_PRICES:
        p = MODEL_PRICES[model]
        custo = (in_toks/1000.0)*p.get("in",0) + (out_toks/1000.0)*p.get("out",0)
        USAGE_STATS["usd_total"] += custo

def registrar_emb_usage(model: str, tokens: int):
    """
    Registra o uso de tokens para embeddings e calcula custos.
    
    Args:
        model: Nome do modelo de embedding utilizado
        tokens: Número de tokens processados
    """
    global USAGE_STATS
    USAGE_STATS["emb_tokens"] += tokens
    
    # Calcula custo para embeddings
    if model in MODEL_PRICES:
        p = MODEL_PRICES[model]
        custo = (tokens/1000.0)*p.get("in",0)
        USAGE_STATS["usd_total"] += custo
# =============================
# CONFIGURAÇÕES DE PROCESSAMENTO
# =============================

# PROCESSAMENTO MULTI-ARQUIVO
# Permite processar múltiplos CSVs de duas formas:
# 1. INPUT_FILES="arquivo1.csv,arquivo2.csv,arquivo3.csv" (lista específica)
# 2. INPUT_DIR="/caminho/para/pasta" (todos os CSVs da pasta)

# CONFIGURAÇÕES DE RESUMO EM LOTE
SUMMARY_BATCH_SIZE = int(os.getenv("SUMMARY_BATCH_SIZE", "100"))           # Itens processados por lote
SUMMARY_MAX_CHARS_PER_ITEM = int(os.getenv("SUMMARY_MAX_CHARS_PER_ITEM", "800"))  # Limite de caracteres por item

# Instrução do sistema para geração de resumos individuais
SUMMARY_SYS_INSTR = (
    "Você é um assistente especializado em análise de dados. "
    "Para CADA item fornecido, gere um resumo curto (2–3 frases) destacando "
    "os PRINCIPAIS erros de input do usuário (se houver). "
    "Saia ESTRITAMENTE em JSON como uma lista, onde cada elemento é {\"i\": <indice>, \"resumo\": <texto>}."
)

# CONFIGURAÇÕES DE RESUMO POR TÓPICO
DEFAULT_TOPIC_SUMMARY_FILE = "resumo_topicos.csv"                         # Arquivo de saída dos resumos por tópico
TOPIC_SUMMARY_BATCH_SIZE = int(os.getenv("TOPIC_SUMMARY_BATCH_SIZE", "8"))        # Tópicos processados por lote
TOPIC_SUMMARY_MAX_ITEMS = int(os.getenv("TOPIC_SUMMARY_MAX_ITEMS", "30"))         # Máximo de itens por tópico
TOPIC_SUMMARY_MAX_CHARS = int(os.getenv("TOPIC_SUMMARY_MAX_CHARS", "500"))        # Limite de caracteres por tópico

# Instrução do sistema para análise de tópicos
TOPIC_SUMMARY_SYS_INSTR = (
    "Você é um analista de atendimento especializado em encontrar padrões de erro. "
    "Para CADA tópico fornecido, identifique os principais problemas, causas recorrentes "
    "e impactos percebidos. "
    "Retorne ESTRITAMENTE em JSON como uma lista de objetos com o formato "
    "{\"topico\": <nome_exato>, \"resumo\": <texto>}. "
    "Use frases curtas ou bullet points para destacar os pontos críticos de forma objetiva."
)

# OPÇÕES DE MACHINE LEARNING (configuráveis via variáveis de ambiente)
USE_OVERSAMPLING = os.getenv("USE_OVERSAMPLING", "true").lower() == "true"        # Balanceamento de classes
USE_TFIDF_BASELINE = os.getenv("USE_TFIDF_BASELINE", "true").lower() == "true"    # Usar baseline TF-IDF
TFIDF_SOURCE = os.getenv("TFIDF_SOURCE", "resumo")                               # Fonte para TF-IDF: "resumo" ou nome da coluna
MIN_SAMPLES_PER_CLASS = int(os.getenv("MIN_SAMPLES_PER_CLASS", "10"))            # Mínimo de amostras por classe

# =============================
# CLIENTE DA API OPENAI
# =============================

def build_client() -> Optional[OpenAI]:
    """
    Constrói e retorna um cliente OpenAI se a chave da API estiver disponível.
    
    Returns:
        OpenAI client se a chave estiver configurada, None caso contrário
        
    Note:
        Se não houver chave da API, o sistema funcionará apenas com heurísticas locais
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OPENAI_API_KEY ausente; seguirei apenas com heurísticas locais.")
        return None
    return OpenAI(api_key=api_key)

# =============================
# LEITURA ROBUSTA DE ARQUIVOS CSV
# =============================

def read_csv_com_fallback(path: Path, sep: str = ";") -> pd.DataFrame:
    """
    Lê um arquivo CSV tentando diferentes encodings e separadores automaticamente.
    
    Esta função é robusta e tenta múltiplas combinações de encoding e separadores
    para garantir que o arquivo seja lido corretamente, mesmo com diferentes formatos.
    
    Args:
        path: Caminho para o arquivo CSV
        sep: Separador preferencial (padrão: ";")
        
    Returns:
        DataFrame do pandas com os dados do CSV
        
    Raises:
        RuntimeError: Se não conseguir ler o arquivo com nenhuma combinação
    """
    # Lista de encodings para tentar (do mais comum ao menos comum)
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "utf-16", "iso8859-15"]
    
    # Lista de separadores para tentar
    seps_to_try = [sep, ",", "\t", "|"]
    
    last_err: Optional[Exception] = None
    
    # Tenta todas as combinações de encoding e separador
    for enc in encodings_to_try:
        for s in seps_to_try:
            try:
                df = pd.read_csv(path, sep=s, encoding=enc, on_bad_lines="skip")
                logging.info(f"Lido {path} com encoding={enc}, sep={repr(s)} ({len(df)} linhas)")
                return df
            except Exception as e:
                last_err = e
                continue
    
    # Fallback: tenta detectar encoding automaticamente usando chardet
    if HAS_CHARDET:
        try:
            with open(path, "rb") as f:
                rawdata = f.read(50000)  # Lê primeiros 50KB para detecção
            enc = chardet.detect(rawdata).get("encoding", "utf-8")
            df = pd.read_csv(path, sep=sep, encoding=enc, on_bad_lines="skip")
            logging.info(f"Lido {path} via chardet: encoding={enc}")
            return df
        except Exception as e:
            pass
    
    raise RuntimeError(f"Falha ao ler CSV {path}. Último erro: {last_err}. Instale 'chardet' para melhor detecção de encoding.")

# =============================
# FUNÇÕES UTILITÁRIAS
# =============================

def truncate(s: str, max_chars: int) -> str:
    """
    Trunca uma string para um número máximo de caracteres.
    
    Args:
        s: String a ser truncada
        max_chars: Número máximo de caracteres permitidos
        
    Returns:
        String truncada se necessário, ou string original se dentro do limite
        
    Note:
        Converte automaticamente valores não-string para string
    """
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    return s if len(s) <= max_chars else s[:max_chars]

# =============================
# Resumo em lote (JSON estável)
# =============================

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
    
    Esta função processa textos em lotes para otimizar o uso da API e implementa
    um sistema de cache local para evitar reprocessamento de textos já resumidos.
    
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
        
    Note:
        - Usa cache baseado em hash SHA256 dos textos
        - Fallback para heurísticas locais se API não disponível
        - Implementa retry automático em caso de falhas da API
    """
    # Carrega cache existente de resumos (baseado em hash dos textos)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    # Inicializa lista de resultados
    resultados = ["" for _ in textos]

    # Verifica se algum texto já foi processado (existe no cache)
    for i, texto in enumerate(textos):
        h = hashlib.sha256(texto.encode("utf-8")).hexdigest()
        if h in cache:
            resultados[i] = cache[h]

    # Se não há cliente OpenAI, usa apenas heurísticas locais
    if client is None:
        for i, t in enumerate(textos):
            if not resultados[i]:  # Só processa se não estiver no cache
                resumo = gerar_resumo_fallback(t)
                resultados[i] = resumo
                cache[hashlib.sha256(t.encode("utf-8")).hexdigest()] = resumo
        # Salva cache atualizado
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
        return resultados

    # Processa textos em lotes para otimizar chamadas da API
    for start in range(0, len(textos), batch_size):
        # Define índices do bloco atual
        bloco_idx = list(range(start, min(start + batch_size, len(textos))))
        
        # Trunca textos do bloco para evitar excesso de tokens
        bloco = [truncate(textos[i], max_chars) for i in bloco_idx]
        
        # Cria linhas apenas para textos que ainda não foram processados
        user_lines = [f"{i-start}. {bloco[i-start]}" for i in bloco_idx if not resultados[i]]
        
        # Pula bloco se todos os textos já foram processados
        if not user_lines:
            continue
            
        # Monta prompt para a API com instruções específicas
        user_prompt = (
            "Resuma os itens numerados abaixo. Produza JSON com [{\"i\": <indice_global>, \"resumo\": \"...\"}].\n\n"
            "ITENS:\n" + "\n".join(user_lines) + "\n\n"
            "Observação: 'i' deve ser o ÍNDICE GLOBAL original que estou te passando agora: "
            f"{bloco_idx}"
        )

        last_exc: Optional[Exception] = None

        # Implementa retry com backoff exponencial
        for attempt in range(1, max_retries + 1):
            try:
                # Chama API OpenAI para gerar resumos
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SUMMARY_SYS_INSTR},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                
                # Registra uso de tokens para controle de custos
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    registrar_uso(model, {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    })
                
                # Processa resposta da API
                content = (resp.choices[0].message.content or "").strip()
                
                # Remove markdown code blocks se presentes
                if content.startswith("```"):
                    content = content.strip("`")
                    if content.startswith("json"):
                        content = content[4:]
                
                # Parse do JSON retornado
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("JSON não é uma lista")
                
                # Processa cada item do resultado
                for item in data:
                    gi = int(item.get("i", -1))  # Índice global
                    resumo = str(item.get("resumo", "")).strip()
                    
                    # Valida índice e atualiza resultado
                    if 0 <= gi < len(resultados):
                        if not resumo:  # Fallback se resumo vazio
                            resumo = gerar_resumo_fallback(textos[gi])
                        resultados[gi] = resumo
                        # Atualiza cache
                        cache[hashlib.sha256(textos[gi].encode("utf-8")).hexdigest()] = resumo
                break  # Sucesso - sai do loop de retry
            except Exception as exc:
                last_exc = exc
                logging.warning(
                    "Falha ao resumir bloco %s-%s (tentativa %s/%s): %s",
                    start, start + len(bloco) - 1, attempt, max_retries, exc,
                )
                time.sleep(attempt)  # Backoff simples
        else:
            # Se todas as tentativas falharam, usa fallback local
            logging.error(
                "Falha definitiva ao resumir bloco %s-%s: %s",
                start, start + len(bloco) - 1, last_exc,
            )
            # Gera resumos usando heurísticas locais para itens não processados
            for gi in bloco_idx:
                if not resultados[gi]:
                    resumo = gerar_resumo_fallback(textos[gi])
                    resultados[gi] = resumo
                    cache[hashlib.sha256(textos[gi].encode("utf-8")).hexdigest()] = resumo

    # Garantia final: verifica se algum resumo ficou vazio
    for idx, resumo in enumerate(resultados):
        if not str(resumo).strip():
            resumo = gerar_resumo_fallback(textos[idx])
            resultados[idx] = resumo
            cache[hashlib.sha256(textos[idx].encode("utf-8")).hexdigest()] = resumo

    # Salva cache atualizado
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)
    
    return resultados

# =============================
# SISTEMA DE CLASSIFICAÇÃO HEURÍSTICA
# =============================

# Dicionário de padrões regex para classificação automática de interações
# Cada chave representa um tipo de erro/situação, e os valores são listas de regex patterns
KEYS = {
    # Problemas relacionados a CPF
    "Solicita CPF": [r"\bcpf\b", r"solicit(a|ou).*cpf", r"inform(e|ar).*cpf"],
    
    # Perguntas genéricas ou desvios do fluxo
    "Pergunta Algo Mais": [r"algo mais", r"mais alguma coisa", r"posso ajudar em.*mais"],
    
    # Problemas com propostas iniciais
    "Proposta Inicial": [r"\bproposta inicial\b", r"iniciar proposta", r"simular proposta"],
    "Proposta Inicial Creli": [r"\bproposta inicial creli\b", r"creli.*proposta"],
    
    # Questões sobre parcelamento
    "Lista Opções Parcelamento": [r"opç(ao|ões) de parcelamento\b", r"parcelas?\b"],
    "Lista Mais Opções Parcelamento": [r"mais opç(ao|ões) parcelamento"],
    "Lista Opções Parcelamento Réplica": [r"parcelamento.*r[eé]plica", r"r[eé]plica.*parcel"],
    
    # Questões específicas do produto Creli
    "Mensagem Solicita Valor Entrada Creli": [r"valor de entrada.*creli", r"entrada creli"],
    "Lista Mais Opções Creli": [r"mais opç(ao|ões).*creli"],
    "Lista Opções Parcelamento Creli": [r"opç(ao|ões) de parcelamento.*creli", r"creli.*parcel"],
    "Lista Mais Opções Parcelamento Creli ": [r"mais opç(ao|ões) parcelamento.*creli"],
    
    # Problemas de navegação e interface
    "Lista Data Desejada": [r"data desejada", r"agendar data", r"escolher data"],
    "Lista Mais Opções": [r"mais opç(ao|ões)\b", r"ver mais"],
    
    # Erros do sistema
    "Menu CPF Não Localizado": [r"cpf n[aã]o localizado", r"n[aã]o encontrei.*cpf"],
    "Decisão Erro Persistente": [r"erro persistente", r"tentativa.*falha.*(de novo|novamente)"],
    
    # Problemas de atendimento
    "Botão Negociou Recentemente": [r"negociou recentemente", r"bot[aã]o.*negociou"],
    "Pergunta Inatividade Atendente": [r"inatividade.*atendente", r"sem resposta.*atendente"],
}

# Resumos padrão para cada tipo de classificação (usado quando API não disponível)
FALLBACK_SUMMARIES = {
    # Problemas de CPF
    "Solicita CPF": "Usuário não forneceu um CPF válido quando solicitado.",
    
    # Desvios do fluxo
    "Pergunta Algo Mais": "Usuário desviou do fluxo padrão e pediu outra ajuda.",
    
    # Problemas com propostas
    "Proposta Inicial": "Usuário não informou os dados necessários para iniciar a proposta.",
    "Proposta Inicial Creli": "Usuário não completou as informações para a proposta Creli.",
    
    # Questões de parcelamento
    "Lista Opções Parcelamento": "Usuário teve dúvida sobre as opções de parcelamento disponíveis.",
    "Lista Mais Opções Parcelamento": "Usuário pediu alternativas extras de parcelamento.",
    "Lista Opções Parcelamento Réplica": "Usuário repetiu a solicitação de opções de parcelamento.",
    
    # Questões específicas do Creli
    "Mensagem Solicita Valor Entrada Creli": "Usuário questionou o valor de entrada do produto Creli.",
    "Lista Mais Opções Creli": "Usuário pediu mais opções específicas do Creli.",
    "Lista Opções Parcelamento Creli": "Usuário consultou as parcelas do produto Creli.",
    "Lista Mais Opções Parcelamento Creli ": "Usuário busca mais alternativas de parcelamento do Creli.",
    
    # Problemas de navegação
    "Lista Data Desejada": "Usuário solicitou negociar em outra data.",
    "Lista Mais Opções": "Usuário pediu para ver opções adicionais do fluxo.",
    
    # Erros do sistema
    "Menu CPF Não Localizado": "CPF informado não foi localizado no sistema.",
    "Decisão Erro Persistente": "Usuário relata erro recorrente sem solução.",
    
    # Problemas de atendimento
    "Botão Negociou Recentemente": "Usuário já negociou recentemente e não encontra o botão correspondente.",
    "Pergunta Inatividade Atendente": "Usuário pede atendimento humano após falta de resposta.",
    
    # Casos especiais
    "Campo vazio ou nulo": "Interação vazia; nenhuma mensagem do usuário foi registrada.",
}


def atribuir_rotulo(texto: str) -> str:
    """
    Classifica um texto usando regras heurísticas baseadas em regex.
    
    Esta função aplica padrões de expressão regular para identificar
    automaticamente o tipo de problema ou situação descrita no texto.
    
    Args:
        texto: Texto da interação a ser classificado
        
    Returns:
        String com o rótulo/categoria identificada
        
    Note:
        - Retorna "Campo vazio ou nulo" para textos vazios
        - Usa fallback "Pergunta Algo Mais" se nenhum padrão for encontrado
        - Inclui detecção especial para pedidos de atendimento humano
    """
    # Verifica se texto é válido
    if not isinstance(texto, str) or not texto.strip():
        return "Campo vazio ou nulo"
    
    # Converte para minúsculas para comparação case-insensitive
    t = texto.lower()
    
    # Testa cada padrão regex definido em KEYS
    for label, pats in KEYS.items():
        for pat in pats:
            if re.search(pat, t):
                return label
    
    # Detecção especial para pedidos de atendimento humano
    if any(x in t for x in ["atendente", "humano", "suporte"]):
        return "Pergunta Inatividade Atendente"
    
    # Fallback padrão
    return "Pergunta Algo Mais"

# =============================
# SISTEMA DE EXTRAÇÃO DE TÓPICOS
# =============================

# Regex para extrair tópicos do formato "pergunta: <conteúdo>"
TOPICO_REGEX = re.compile(r"pergunta:\s*([^|]+)", re.IGNORECASE)

def extrair_topico(texto: str) -> str:
    """
    Extrai o tópico principal de um texto baseado no padrão "pergunta: <conteúdo>".
    
    Esta função procura por um padrão específico no texto que indica o tópico
    da pergunta ou problema relatado pelo usuário.
    
    Args:
        texto: Texto da interação
        
    Returns:
        String com o tópico extraído ou "Outro" se não encontrado
        
    Note:
        - Procura pelo padrão "pergunta: <conteúdo>"
        - Remove espaços extras e caracteres de pontuação
        - Retorna "Outro" como fallback
    """
    if not isinstance(texto, str):
        return "Outro"
    
    # Busca pelo padrão regex
    match = TOPICO_REGEX.search(texto)
    if match:
        bruto = match.group(1)  # Captura o conteúdo após "pergunta:"
        if bruto:
            # Limpa espaços extras e caracteres indesejados
            topico = re.sub(r"\s+", " ", bruto).strip(" -:")
            if topico:
                return topico
    
    return "Outro"


def agrupar_por_topico(questions: List[str]) -> Dict[str, List[str]]:
    """
    Agrupa uma lista de perguntas por tópico extraído.
    
    Args:
        questions: Lista de textos/perguntas para agrupar
        
    Returns:
        Dicionário onde as chaves são tópicos e valores são listas de perguntas
        
    Note:
        - Ignora entradas que não são strings
        - Usa a função extrair_topico() para classificar cada pergunta
    """
    topicos: Dict[str, List[str]] = {}
    
    for pergunta in questions:
        if not isinstance(pergunta, str):
            continue
        
        # Extrai tópico e adiciona à lista correspondente
        topico = extrair_topico(pergunta)
        topicos.setdefault(topico, []).append(pergunta)
    
    return topicos


def gerar_resumo_fallback(texto: str) -> str:
    """
    Gera um resumo usando heurísticas locais quando a API não está disponível.
    
    Esta função serve como fallback quando não é possível usar a API OpenAI,
    combinando classificação automática com trechos do texto original.
    
    Args:
        texto: Texto original da interação
        
    Returns:
        String com resumo gerado localmente
        
    Note:
        - Usa classificação heurística para identificar o tipo de problema
        - Inclui trecho do texto original para contexto
        - Fornece mensagens padrão para cada tipo de classificação
    """
    texto = texto or ""
    
    # Classifica o texto usando regras heurísticas
    rotulo = atribuir_rotulo(texto)
    
    # Busca resumo padrão para o rótulo identificado
    base = FALLBACK_SUMMARIES.get(rotulo)
    
    # Cria trecho do texto original para contexto
    trecho = truncate(texto, 140)
    
    # Combina resumo padrão com trecho original
    if base:
        return f"{base} Trecho original: {trecho}" if trecho else base
    
    # Fallback final se não houver resumo padrão
    return f"Resumo automático indisponível. Revisar interação: {trecho}" if trecho else "Resumo automático indisponível. Texto original ausente."


def gerar_resumo_topico_fallback(
    topico: str,
    textos: Sequence[str],
    rotulos: Optional[Sequence[str]] = None,
) -> str:
    textos_validos = [truncate(t, 140) for t in textos if isinstance(t, str) and t.strip()]
    descricao_rotulos = ""
    if rotulos:
        contagem = Counter(r for r in rotulos if isinstance(r, str) and r.strip())
        if contagem:
            principais = ", ".join(
                f"{rotulo} ({quantidade})" for rotulo, quantidade in contagem.most_common(3)
            )
            descricao_rotulos = f" Principais rótulos: {principais}."

    if textos_validos:
        exemplos = "; ".join(textos_validos[:3])
        return (
            f"Resumo automático indisponível.{descricao_rotulos} "
            f"Exemplos recorrentes: {exemplos}"
        )

    if descricao_rotulos:
        return f"Resumo automático indisponível.{descricao_rotulos}"

    return f"Resumo automático indisponível para '{topico}'. Revisar interações."


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
            # Sem API → tudo via fallback
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
                conteudo = (resposta.choices[0].message.content or "").strip()
                if conteudo.startswith("```"):
                    conteudo = conteudo.strip("`")
                    if conteudo.startswith("json"):
                        conteudo = conteudo[4:]
                dados = json.loads(conteudo)
                if not isinstance(dados, list):
                    raise ValueError("JSON não é uma lista")

                lookup_exato = {topico: topico for topico in chunk_topics}
                lookup_cf = {topico.casefold(): topico for topico in chunk_topics}

                for item in dados:
                    nome = str(item.get("topico", "")).strip()
                    resumo = str(item.get("resumo", "")).strip()
                    if not nome or not resumo:
                        continue
                    chave = lookup_exato.get(nome) or lookup_cf.get(nome.casefold())
                    if chave:
                        resultados[chave] = resumo
                break
            except Exception as exc:
                ultima_excecao = exc
                logging.warning(
                    "Falha ao resumir tópicos %s-%s (tentativa %s/%s): %s",
                    inicio, inicio + len(bloco) - 1, tentativa, max_retries, exc,
                )
                time.sleep(tentativa)
        else:
            logging.error(
                "Falha definitiva ao resumir tópicos %s-%s: %s",
                inicio, inicio + len(bloco) - 1, ultima_excecao,
            )

    for topico, textos in topico_para_textos.items():
        if topico not in resultados:
            rotulos = topico_para_rotulos.get(topico) if topico_para_rotulos else None
            resultados[topico] = gerar_resumo_topico_fallback(topico, textos, rotulos)

    return resultados

# =============================
# Embeddings em lote (robusto)
# =============================

def generate_embeddings_for_texts(client: Optional[OpenAI], texts: List[str], model: str, batch_size: int = 128) -> np.ndarray:
    """
    Gera embeddings vetoriais para uma lista de textos usando a API OpenAI.
    
    Os embeddings são representações numéricas dos textos que capturam
    seu significado semântico, úteis para machine learning e análise de similaridade.
    
    Args:
        client: Cliente OpenAI (None para retornar vetores nulos)
        texts: Lista de textos para converter em embeddings
        model: Nome do modelo de embedding da OpenAI
        batch_size: Número de textos processados por lote
        
    Returns:
        Array numpy com os embeddings (shape: [n_texts, embedding_dim])
        
    Raises:
        ValueError: Se a lista de textos estiver vazia
        RuntimeError: Se falhar ao gerar embeddings após todas as tentativas
        
    Note:
        - Processa textos em lotes para otimizar uso da API
        - Implementa retry automático com backoff
        - Registra uso de tokens para controle de custos
        - Retorna vetores nulos se cliente não disponível
    """
    if not texts:
        raise ValueError("Lista de textos vazia")
    
    # Se não há cliente OpenAI, retorna vetores nulos (pula treino ML)
    if client is None:
        return np.zeros((len(texts), 1), dtype=np.float32)

    vetores: List[List[float]] = []
    
    # Garante que todos os textos são válidos (substitui vazios)
    validos = [t if (t and str(t).strip()) else "(resumo ausente)" for t in texts]

    # Processa textos em lotes para otimizar chamadas da API
    for i in range(0, len(validos), batch_size):
        bloco = validos[i:i + batch_size]
        last_exc: Optional[Exception] = None
        
        # Implementa retry com backoff exponencial
        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                # Chama API para gerar embeddings
                resp = client.embeddings.create(model=model, input=bloco)
                
                # Registra uso de tokens para controle de custos
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    registrar_emb_usage(model, usage.prompt_tokens)
                else:
                    # Estimativa de tokens se não disponível
                    registrar_emb_usage(model, len("".join(bloco).split()))
                
                # Extrai vetores da resposta
                vetores.extend([d.embedding for d in resp.data])
                break  # Sucesso - sai do loop de retry
                
            except Exception as exc:
                last_exc = exc
                logging.warning(
                    "Embeddings falharam para bloco %s-%s (tentativa %s/%s): %s",
                    i, i + len(bloco) - 1, attempt, MAX_API_RETRIES, exc,
                )
                time.sleep(attempt)  # Backoff simples
        else:
            # Se todas as tentativas falharam
            raise RuntimeError(
                f"Falha definitiva ao gerar embeddings para bloco {i}-{i + len(bloco) - 1}: {last_exc}"
            )

    return np.array(vetores, dtype=np.float32)

# =============================
# Classificador com melhorias
# =============================

def train_classifier_balanced(embeddings: np.ndarray, labels: Sequence[str]) -> str:
    """
    Treina um classificador de regressão logística usando embeddings e avalia sua performance.
    
    Esta função implementa um pipeline completo de machine learning:
    - Divisão treino/teste estratificada
    - Balanceamento de classes (opcional)
    - Treinamento com regularização
    - Avaliação com métricas detalhadas
    
    Args:
        embeddings: Array numpy com embeddings dos textos
        labels: Sequência com rótulos correspondentes
        
    Returns:
        String com relatório detalhado de classificação
        
    Raises:
        ValueError: Se arrays estiverem vazios ou com tamanhos incompatíveis
        
    Note:
        - Usa stratified split para manter proporção de classes
        - Aplica oversampling se configurado (USE_OVERSAMPLING=true)
        - Implementa class_weight="balanced" para lidar com desbalanceamento
        - Pula treinamento se embeddings forem mock (sem API)
    """
    # Validações básicas
    if embeddings.size == 0:
        raise ValueError("Embeddings array is empty")
    if len(labels) != len(embeddings):
        raise ValueError("Label count mismatch")
    
    # Se embeddings são mock (sem API), pula o treinamento
    if embeddings.shape[1] == 1 and np.allclose(embeddings, 0):
        return "(Classificador com embeddings pulado: embeddings indisponíveis)"

    # Analisa distribuição de classes
    label_counts = Counter(labels)
    logging.info(f"Distribuição de classes: {dict(label_counts)}")

    # Divisão estratificada treino/teste (mantém proporção de classes)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Aplica oversampling se configurado (balanceia classes minoritárias)
    if USE_OVERSAMPLING:
        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        except Exception as exc:
            logging.warning("Oversampling indisponível: %s", exc)

    # Treina modelo de regressão logística com configurações otimizadas
    model = LogisticRegression(
        max_iter=1000,           # Máximo de iterações
        class_weight="balanced",  # Balanceamento automático de classes
        C=0.1                    # Regularização (menor = mais regularização)
    )
    model.fit(X_train, y_train)
    
    # Faz predições no conjunto de teste
    predicted = model.predict(X_test)

    # Gera relatório detalhado de classificação
    report = classification_report(y_test, predicted, zero_division=0)
    
    # Calcula métricas F1 adicionais
    try:
        macro_f1 = f1_score(y_test, predicted, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, predicted, average="weighted", zero_division=0)
        logging.info("Macro-F1 (embeddings): %.4f | Weighted-F1: %.4f", macro_f1, weighted_f1)
    except Exception:
        pass
    
    return report

# =============================
# TF-IDF baseline (opcional)
# =============================

def evaluate_tfidf_baseline(texts: List[str], labels: Sequence[str]) -> str:
    """
    Avalia um classificador baseline usando TF-IDF como alternativa aos embeddings.
    
    Esta função implementa um pipeline tradicional de NLP usando TF-IDF
    (Term Frequency-Inverse Document Frequency) para comparar com embeddings.
    
    Args:
        texts: Lista de textos para classificar
        labels: Rótulos correspondentes aos textos
        
    Returns:
        String com relatório de classificação do modelo TF-IDF
        
    Note:
        - Usa n-gramas (1,2) para capturar contexto local
        - Aplica stop words em português
        - Filtra termos muito raros (min_df=2) e muito comuns (max_df=0.95)
        - Serve como baseline para comparar com embeddings da OpenAI
    """
    # Divisão treino/teste com tamanho adaptativo baseado no dataset
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=min(0.3, max(0.1, 1.0 / max(1, len(labels)))),  # Entre 10% e 30%
        random_state=42,
        stratify=labels,
    )

    # Pipeline TF-IDF + Regressão Logística
    clf = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),      # Usa unigramas e bigramas
            min_df=2,                # Ignora termos que aparecem menos de 2 vezes
            max_df=0.95,             # Ignora termos que aparecem em mais de 95% dos docs
            stop_words="portuguese", # Remove stop words em português
        ),
        LogisticRegression(
            max_iter=1000,           # Máximo de iterações
            class_weight="balanced"   # Balanceamento automático de classes
        ),
    )
    
    # Treina o modelo
    clf.fit(X_train, y_train)
    
    # Faz predições
    pred = clf.predict(X_test)
    
    # Gera relatório de classificação
    report = classification_report(y_test, pred, zero_division=0)
    
    # Calcula F1 macro para comparação
    try:
        macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        logging.info("Macro-F1 (TF-IDF %s): %.4f", TFIDF_SOURCE, macro_f1)
    except Exception:
        pass
    
    return report

# =============================
# Ingestão multi-CSV
# =============================

def get_input_csv_paths() -> List[Path]:
    """
    Determina quais arquivos CSV devem ser processados baseado nas variáveis de ambiente.
    
    Esta função implementa uma estratégia flexível para especificar arquivos de entrada:
    1. INPUT_FILES: lista específica de arquivos separados por vírgula
    2. INPUT_DIR: diretório contendo arquivos CSV
    3. CONVERSATION_FILE: fallback para compatibilidade (pode ser lista)
    
    Returns:
        Lista de objetos Path com os arquivos CSV a processar
        
    Raises:
        SystemExit: Se nenhum arquivo for especificado
        
    Note:
        - Suporta caminhos absolutos e relativos
        - Caminhos relativos são resolvidos em relação ao diretório do script
        - Para INPUT_DIR, procura todos os arquivos *.csv no diretório
    """
    # Lê variáveis de ambiente
    files_env = os.getenv("INPUT_FILES", "").strip()      # Lista específica de arquivos
    dir_env = os.getenv("INPUT_DIR", "").strip()          # Diretório com CSVs
    paths: List[Path] = []

    # Diretório base para resolver caminhos relativos
    base_dir = Path(__file__).parent

    if files_env:
        # Opção 1: Lista específica de arquivos
        parts = [p.strip() for p in files_env.split(",") if p.strip()]
        paths = [Path(p) if Path(p).is_absolute() else base_dir / p for p in parts]
        
    elif dir_env:
        # Opção 2: Todos os CSVs de um diretório
        base = Path(dir_env)
        if not base.is_absolute():
            base = base_dir / base
        paths = sorted(base.glob("*.csv"))  # Busca todos os .csv
        
    else:
        # Opção 3: Fallback usando CONVERSATION_FILE (compatibilidade)
        single = os.getenv("CONVERSATION_FILE", DEFAULT_CONVERSATION_FILE)
        if single:
            parts = [p.strip() for p in single.split(",") if p.strip()]
            paths = [Path(p) if Path(p).is_absolute() else base_dir / p for p in parts]

    # Valida se pelo menos um arquivo foi especificado
    if not paths:
        raise SystemExit("Nenhum CSV informado. Use INPUT_FILES (lista separada por vírgulas) ou INPUT_DIR.")
    
    return paths


def load_and_normalize(paths: List[Path], message_column: str) -> pd.DataFrame:
    """
    Carrega e normaliza múltiplos arquivos CSV em um único DataFrame.
    
    Esta função lida com a heterogeneidade dos arquivos CSV, criando uma
    estrutura unificada e garantindo que a coluna de mensagens exista.
    
    Args:
        paths: Lista de caminhos para arquivos CSV
        message_column: Nome da coluna que deve conter o texto das interações
        
    Returns:
        DataFrame unificado com todos os dados normalizados
        
    Note:
        - Adiciona coluna __source_file para rastrear origem dos dados
        - Cria coluna de mensagens por concatenação se não existir
        - Usa colunas padrão: Data Evento, Descrição Evento, Contato, Pergunta, Input
        - Fallback: concatena todas as colunas se estrutura não reconhecida
    """
    frames: List[pd.DataFrame] = []
    
    for p in paths:
        # Carrega CSV com leitura robusta
        df = read_csv_com_fallback(p)
        
        # Adiciona coluna para rastrear arquivo de origem
        df["__source_file"] = p.name
        
        # Verifica se coluna de mensagens existe
        if message_column not in df.columns:
            # Lista de colunas esperadas no formato padrão
            required_cols = ["Data Evento", "Descrição Evento", "Contato", "Pergunta", "Input"]
            
            if all(col in df.columns for col in required_cols):
                # Cria coluna de mensagens concatenando colunas padrão
                df[message_column] = (
                    df["Data Evento"].astype(str).fillna("")
                    + " | " + df["Descrição Evento"].astype(str).fillna("")
                    + " | " + df["Contato"].astype(str).fillna("")
                    + " | " + df["Pergunta"].astype(str).fillna("")
                    + " | " + df["Input"].astype(str).fillna("")
                )
            else:
                # Fallback: concatena todas as colunas disponíveis
                # Isso evita crashes mas pode gerar dados menos úteis
                df[message_column] = df.astype(str).agg(" | ".join, axis=1)
        
        frames.append(df)
    
    # Combina todos os DataFrames em um único
    big = pd.concat(frames, ignore_index=True)
    return big


def find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifica automaticamente a coluna de data/hora em um DataFrame.
    
    Esta função procura por colunas que contenham informações temporais
    baseando-se no nome da coluna e na capacidade de parsing das datas.
    
    Args:
        df: DataFrame para analisar
        
    Returns:
        Nome da coluna de data/hora encontrada, ou None se não encontrada
        
    Note:
        - Procura por nomes contendo "data", "date", "hora", "time"
        - Testa parsing com formato brasileiro (dia primeiro)
        - Requer que pelo menos 40% dos valores sejam parseáveis como data
        - Retorna a primeira coluna que atende aos critérios
    """
    # Busca colunas com nomes relacionados a data/hora
    cands = [c for c in df.columns if re.search(r"data|date|hora|time", c, re.IGNORECASE)]
    
    # Testa cada candidata
    for c in cands:
        # Tenta converter para datetime (formato brasileiro: dia primeiro)
        s = pd.to_datetime(df[c], errors="coerce", dayfirst=False, format='mixed')
        
        # Verifica se pelo menos 40% dos valores são parseáveis
        if s.notna().mean() > 0.4:
            return c
    
    return None

# =============================
# GERAÇÃO DE RELATÓRIO PDF COMPLETO
# =============================

def gerar_relatorio_pdf(
    df: pd.DataFrame,
    output_path: Path,
    embeddings_report: str = "",
    tfidf_report: str = "",
    topic_summary_df: Optional[pd.DataFrame] = None,
    daily_pivot: Optional[pd.DataFrame] = None,
) -> None:
    """
    Gera um relatório PDF completo com análises visuais e estatísticas.
    
    Esta função cria um documento PDF multi-página contendo:
    - Resumo executivo com métricas principais
    - Análises temporais (tendências diárias)
    - Distribuição de tipos de erro
    - Relatórios de performance dos modelos ML
    - Resumos por tópico
    - Visualizações gráficas
    
    Args:
        df: DataFrame principal com os dados analisados
        output_path: Caminho base para o arquivo de saída
        embeddings_report: Relatório do classificador com embeddings
        tfidf_report: Relatório do classificador TF-IDF baseline
        topic_summary_df: DataFrame com resumos por tópico
        daily_pivot: DataFrame com análise temporal diária
        
    Note:
        - Usa matplotlib e seaborn para visualizações
        - Salva PDF no mesmo diretório do arquivo de saída
        - Implementa layout profissional com múltiplas páginas
        - Inclui timestamp de geração
    """
    pdf_path = output_path.parent / (output_path.stem + ".pdf")

    plt.style.use('default')
    sns.set_palette("husl")

    with PdfPages(pdf_path) as pdf:
        # Página 1: Título e Resumo Executivo
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.5, 0.95, 'Análise de Erros do Chatbot', fontsize=24, fontweight='bold', ha='center', va='top')
        ax.text(0.5, 0.90, 'Diagnóstico Detalhado (multi-arquivos)', fontsize=18, ha='center', va='top')
        data_atual = datetime.now().strftime("%d/%m/%Y %H:%M")
        ax.text(0.5, 0.85, f'Gerado em: {data_atual}', fontsize=12, ha='center', va='top')

        total_interacoes = len(df)
        if 'rotulo' in df.columns and total_interacoes:
            distribuicao_rotulos = df['rotulo'].value_counts()
            ax.text(0.1, 0.75, 'RESUMO EXECUTIVO', fontsize=16, fontweight='bold', va='top')
            erro_mais_comum = distribuicao_rotulos.index[0]
            tipos_de_erros = distribuicao_rotulos.index.tolist()
            resumo_executivo = f"""
**Resumo**

* **Total de interações analisadas:** {total_interacoes}
* **Erro mais comum:** {erro_mais_comum}
* **Tipos de erros:** {len(tipos_de_erros)}
"""
            ax.text(0.1, 0.70, resumo_executivo, fontsize=11, va='top', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # Página 2: Tendência Temporal (por dia)
        if daily_pivot is not None and not daily_pivot.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            daily_pivot.plot(ax=ax, marker='o', linewidth=2)
            ax.set_title('Tendência diária por tipo de erro', fontsize=14, fontweight='bold')
            ax.set_xlabel('Data'); ax.set_ylabel('Quantidade de Erros')
            ax.legend(title='Tipo de Erro', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # Página 3: Distribuição de Rótulos (geral)
        if 'rotulo' in df.columns:
            dist = df['rotulo'].value_counts().sort_values(ascending=True)
            if not dist.empty:
                fig, ax = plt.subplots(figsize=(12, 7))
                cores = plt.cm.Blues(np.linspace(0.4, 0.9, len(dist)))
                barras = ax.barh(dist.index, dist.values, color=cores, edgecolor='black')
                total = dist.sum(); max_valor = dist.max()
                for barra, quantidade in zip(barras, dist.values):
                    percentual = (quantidade / total * 100) if total else 0
                    ax.text(barra.get_width() + max_valor * 0.01,
                            barra.get_y() + barra.get_height() / 2,
                            f"{quantidade} ({percentual:.1f}%)",
                            va='center', fontsize=10, fontweight='bold')
                ax.set_title('Distribuição de Tipos de Erro (Geral)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Quantidade'); ax.set_ylabel('Tipo de Erro')
                ax.grid(axis='x', alpha=0.2, linestyle='--')
                plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # Página 4+: Resumo por Tópico
        if topic_summary_df is not None and not topic_summary_df.empty:
            ordenado = topic_summary_df.sort_values("quantidade_ocorrencias", ascending=False)

            def _nova_pagina_topicos():
                fig_local, ax_local = plt.subplots(figsize=(8.5, 11))
                ax_local.axis('off')
                ax_local.text(0.5, 0.95, 'Resumo por Tópico', fontsize=18, fontweight='bold', ha='center', va='top')
                return fig_local, ax_local, 0.9

            fig, ax, y_pos = _nova_pagina_topicos()
            for _, row in ordenado.iterrows():
                topo = str(row.get("topico", "Não informado"))
                quantidade = int(row.get("quantidade_ocorrencias", 0))
                plural = "s" if quantidade != 1 else ""
                cabecalho = f"{topo} — {quantidade} ocorrência{plural}"
                resumo = textwrap.wrap(str(row.get("resumo", "")), width=90)
                required_space = 0.05 + 0.035 * (len(resumo) or 1) + 0.02
                if y_pos - required_space < 0.05:
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
                    fig, ax, y_pos = _nova_pagina_topicos()
                ax.text(0.05, y_pos, cabecalho, fontsize=12, fontweight='bold', va='top'); y_pos -= 0.05
                if not resumo: resumo = ["Resumo indisponível."]
                for linha in resumo:
                    ax.text(0.05, y_pos, linha, fontsize=10, va='top'); y_pos -= 0.035
                y_pos -= 0.02
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Última: Performance dos Modelos
        if embeddings_report or tfidf_report:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.95, 'Performance dos Modelos de Classificação', fontsize=18, fontweight='bold', ha='center', va='top')
            y_pos = 0.85
            if embeddings_report:
                ax.text(0.1, y_pos, 'MODELO COM EMBEDDINGS:', fontsize=14, fontweight='bold', va='top'); y_pos -= 0.05
                report_text = '\n'.join(embeddings_report.strip().split('\n')[:20])
                ax.text(0.1, y_pos, report_text, fontsize=9, va='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
                y_pos -= 0.35
            if tfidf_report:
                ax.text(0.1, y_pos, 'MODELO TF-IDF (BASELINE):', fontsize=14, fontweight='bold', va='top'); y_pos -= 0.05
                report_text = '\n'.join(tfidf_report.strip().split('\n')[:20])
                ax.text(0.1, y_pos, report_text, fontsize=9, va='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
            pdf.savefig(fig, bbox_inches='tight'); plt.close()

    logging.info(f"Relatório PDF salvo em: {pdf_path.resolve()}")

# =============================
# Main
# =============================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    load_dotenv()

    client = build_client()  # pode ser None

    message_column = os.getenv("MESSAGE_COLUMN", DEFAULT_MESSAGE_COLUMN)
    label_column = os.getenv("LABEL_COLUMN", DEFAULT_LABEL_COLUMN)
    chat_model = os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)
    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    # Multi-CSV
    csv_paths = get_input_csv_paths()
    logging.info("Arquivos detectados: %s", ", ".join(str(p) for p in csv_paths))

    for csv_path in csv_paths:
        frame = load_and_normalize([csv_path], message_column)

        # Filtra mensagens válidas
        messages_series = frame[message_column].dropna().astype(str).str.strip()
        messages_series = messages_series[messages_series.str.len() > 0]
        if messages_series.empty:
            logging.warning(f"Nenhuma mensagem válida para análise em {csv_path}.")
            continue

        filtered_frame = frame.loc[messages_series.index].reset_index(drop=True)
        conversations = messages_series.reset_index(drop=True).tolist()

        # Resumo (com API se houver)
        summaries: List[str] = resumir_em_lote(client, conversations, chat_model)
        filtered_frame = filtered_frame.assign(resumo=summaries)

        # Tópicos
        filtered_frame = filtered_frame.assign(topico=filtered_frame[message_column].apply(extrair_topico))
        topic_text_map: Dict[str, List[str]] = filtered_frame.groupby("topico")["resumo"].apply(list).to_dict()
        topic_label_map: Dict[str, List[str]] = {}
        if label_column in filtered_frame.columns:
            topic_label_map = filtered_frame.groupby("topico")[label_column].apply(list).to_dict()

        topic_summary_map: Dict[str, str] = {}
        if topic_text_map:
            try:
                topic_summary_map = gerar_resumos_por_topico(
                    client, topic_text_map, chat_model, topico_para_rotulos=topic_label_map if topic_label_map else None
                )
            except Exception as exc:
                logging.error("Falha ao gerar resumos por tópico: %s", exc)
                topic_summary_map = {
                    topico: gerar_resumo_topico_fallback(topico, textos, topic_label_map.get(topico) if topic_label_map else None)
                    for topico, textos in topic_text_map.items()
                }

        filtered_frame = filtered_frame.assign(resumo_topico=filtered_frame["topico"].map(topic_summary_map) if topic_summary_map else "")
        topic_summary_df = (
            pd.DataFrame({
                "topico": list(topic_summary_map.keys()),
                "quantidade_ocorrencias": [len(topic_text_map[t]) for t in topic_summary_map.keys()],
                "resumo": list(topic_summary_map.values()),
            }).sort_values("quantidade_ocorrencias", ascending=False)
            if topic_summary_map else pd.DataFrame(columns=["topico", "quantidade_ocorrencias", "resumo"])
        )

        # Rótulos automáticos se não houver coluna
        if label_column not in filtered_frame.columns:
            filtered_frame = filtered_frame.assign(rotulo=filtered_frame[message_column].apply(atribuir_rotulo))

        # -----------------
        # Agregação diária
        # -----------------
        dt_col = find_datetime_column(filtered_frame)
        if not dt_col:
            logging.warning(f"Nenhuma coluna de data/hora identificada em {csv_path}; visão diária indisponível.")
            continue

        parsed = pd.to_datetime(filtered_frame[dt_col], errors="coerce", dayfirst=False, format='mixed')
        filtered_frame["__date"] = parsed.dt.date
        unique_dates = filtered_frame["__date"].dropna().unique()

        for day in unique_dates:
            day_mask = filtered_frame["__date"] == day
            day_frame = filtered_frame.loc[day_mask].reset_index(drop=True)
            if day_frame.empty:
                continue

            # Recalcula daily_pivot para o dia
            daily_pivot = None
            if "rotulo" in day_frame.columns:
                daily_pivot = (
                    day_frame.groupby(["__date", "rotulo"]).size().unstack(fill_value=0)
                )

            # Embeddings
            try:
                day_summaries = day_frame["resumo"].tolist()
                embeddings = generate_embeddings_for_texts(client, day_summaries, embedding_model, batch_size=128)
            except Exception as exc:
                logging.error("Falha ao gerar embeddings: %s", exc)
                embeddings = np.zeros((len(day_frame), 1), dtype=np.float32)

            # Treino
            report_emb = ""; report_tfidf = ""
            if label_column in day_frame.columns and not day_frame[label_column].isna().all():
                labels_series = day_frame[label_column].astype(str).str.strip()
                non_empty_mask = labels_series.str.len() > 0
                if not non_empty_mask.all():
                    labels_series = labels_series[non_empty_mask]
                    embeddings = embeddings[non_empty_mask.to_numpy()]
                    day_summaries = list(pd.Series(day_summaries)[non_empty_mask].astype(str))
                try:
                    report_emb = train_classifier_balanced(embeddings, labels_series.tolist())
                except ValueError as exc:
                    logging.warning("Treino com embeddings pulado: %s", exc)
                if USE_TFIDF_BASELINE:
                    if TFIDF_SOURCE == "resumo":
                        tfidf_texts = day_summaries
                    else:
                        col = TFIDF_SOURCE if TFIDF_SOURCE in day_frame.columns else message_column
                        tfidf_texts = day_frame.loc[non_empty_mask, col].astype(str).tolist()
                    try:
                        report_tfidf = evaluate_tfidf_baseline(tfidf_texts, labels_series.tolist())
                    except Exception as exc:
                        logging.warning("Erro no baseline TF-IDF: %s", exc)

            # Salva resultados CSVs
            output_dir = csv_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            date_str = str(day)
            base_name = csv_path.stem
            output_path = output_dir / f"{base_name}_{date_str}.csv"
            day_frame.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")
            topic_summary_filename = os.getenv("TOPIC_SUMMARY_FILE", DEFAULT_TOPIC_SUMMARY_FILE)
            topic_summary_path = output_dir / f"Analise_Erros_Chatbot_Diagnostico_{csv_path.stem}_{date_str}_{topic_summary_filename}"
            if not topic_summary_df.empty:
                topic_summary_df.to_csv(topic_summary_path, index=False, sep=";", encoding="utf-8-sig")
            # Gera PDF
            pdf_base = Path(DEFAULT_PDF_OUTPUT).stem
            pdf_name = f"{pdf_base}_{csv_path.stem}_{date_str}.pdf"
            pdf_path = output_dir / pdf_name
            gerar_relatorio_pdf(day_frame, pdf_path, report_emb, report_tfidf, topic_summary_df, daily_pivot)


if __name__ == "__main__":
    main()
    if USAGE_STATS["usd_total"] > 0:
        print("\n========== CUSTOS ==========")
        print(f"Tokens entrada: {USAGE_STATS['tokens_in']}")
        print(f"Tokens saída: {USAGE_STATS['tokens_out']}")
        print(f"Tokens total: {USAGE_STATS['tokens_total']}")
        print(f"Tokens embeddings: {USAGE_STATS['emb_tokens']}")
        print(f"💰 Custo estimado: ${USAGE_STATS['usd_total']:.4f} USD")