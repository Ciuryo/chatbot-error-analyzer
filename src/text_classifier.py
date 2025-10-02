"""
Sistema de classificação heurística de textos.
"""
import re
from typing import Dict, List
from collections import Counter


# Dicionário de padrões regex para classificação automática
KEYS = {
    "Solicita CPF": [r"\bcpf\b", r"solicit(a|ou).*cpf", r"inform(e|ar).*cpf"],
    "Pergunta Algo Mais": [r"algo mais", r"mais alguma coisa", r"posso ajudar em.*mais"],
    "Proposta Inicial": [r"\bproposta inicial\b", r"iniciar proposta", r"simular proposta"],
    "Proposta Inicial Creli": [r"\bproposta inicial creli\b", r"creli.*proposta"],
    "Lista Opções Parcelamento": [r"opç(ao|ões) de parcelamento\b", r"parcelas?\b"],
    "Lista Mais Opções Parcelamento": [r"mais opç(ao|ões) parcelamento"],
    "Lista Opções Parcelamento Réplica": [r"parcelamento.*r[eé]plica", r"r[eé]plica.*parcel"],
    "Mensagem Solicita Valor Entrada Creli": [r"valor de entrada.*creli", r"entrada creli"],
    "Lista Mais Opções Creli": [r"mais opç(ao|ões).*creli"],
    "Lista Opções Parcelamento Creli": [r"opç(ao|ões) de parcelamento.*creli", r"creli.*parcel"],
    "Lista Mais Opções Parcelamento Creli ": [r"mais opç(ao|ões) parcelamento.*creli"],
    "Lista Data Desejada": [r"data desejada", r"agendar data", r"escolher data"],
    "Lista Mais Opções": [r"mais opç(ao|ões)\b", r"ver mais"],
    "Menu CPF Não Localizado": [r"cpf n[aã]o localizado", r"n[aã]o encontrei.*cpf"],
    "Decisão Erro Persistente": [r"erro persistente", r"tentativa.*falha.*(de novo|novamente)"],
    "Botão Negociou Recentemente": [r"negociou recentemente", r"bot[aã]o.*negociou"],
    "Pergunta Inatividade Atendente": [r"inatividade.*atendente", r"sem resposta.*atendente"],
}

# Resumos padrão para cada tipo de classificação
FALLBACK_SUMMARIES = {
    "Solicita CPF": "Usuário não forneceu um CPF válido quando solicitado.",
    "Pergunta Algo Mais": "Usuário desviou do fluxo padrão e pediu outra ajuda.",
    "Proposta Inicial": "Usuário não informou os dados necessários para iniciar a proposta.",
    "Proposta Inicial Creli": "Usuário não completou as informações para a proposta Creli.",
    "Lista Opções Parcelamento": "Usuário teve dúvida sobre as opções de parcelamento disponíveis.",
    "Lista Mais Opções Parcelamento": "Usuário pediu alternativas extras de parcelamento.",
    "Lista Opções Parcelamento Réplica": "Usuário repetiu a solicitação de opções de parcelamento.",
    "Mensagem Solicita Valor Entrada Creli": "Usuário questionou o valor de entrada do produto Creli.",
    "Lista Mais Opções Creli": "Usuário pediu mais opções específicas do Creli.",
    "Lista Opções Parcelamento Creli": "Usuário consultou as parcelas do produto Creli.",
    "Lista Mais Opções Parcelamento Creli ": "Usuário busca mais alternativas de parcelamento do Creli.",
    "Lista Data Desejada": "Usuário solicitou negociar em outra data.",
    "Lista Mais Opções": "Usuário pediu para ver opções adicionais do fluxo.",
    "Menu CPF Não Localizado": "CPF informado não foi localizado no sistema.",
    "Decisão Erro Persistente": "Usuário relata erro recorrente sem solução.",
    "Botão Negociou Recentemente": "Usuário já negociou recentemente e não encontra o botão correspondente.",
    "Pergunta Inatividade Atendente": "Usuário pede atendimento humano após falta de resposta.",
    "Campo vazio ou nulo": "Interação vazia; nenhuma mensagem do usuário foi registrada.",
}


def atribuir_rotulo(texto: str) -> str:
    """
    Classifica um texto usando regras heurísticas baseadas em regex.
    
    Args:
        texto: Texto da interação a ser classificado
        
    Returns:
        String com o rótulo/categoria identificada
    """
    if not isinstance(texto, str) or not texto.strip():
        return "Campo vazio ou nulo"
    
    t = texto.lower()
    
    for label, pats in KEYS.items():
        for pat in pats:
            if re.search(pat, t):
                return label
    
    # Detecção especial para pedidos de atendimento humano
    if any(x in t for x in ["atendente", "humano", "suporte"]):
        return "Pergunta Inatividade Atendente"
    
    return "Pergunta Algo Mais"


def truncate(s: str, max_chars: int) -> str:
    """
    Trunca uma string para um número máximo de caracteres.
    
    Args:
        s: String a ser truncada
        max_chars: Número máximo de caracteres permitidos
        
    Returns:
        String truncada se necessário, ou string original se dentro do limite
    """
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    return s if len(s) <= max_chars else s[:max_chars]


def gerar_resumo_fallback(texto: str) -> str:
    """
    Gera um resumo usando heurísticas locais quando a API não está disponível.
    
    Args:
        texto: Texto original da interação
        
    Returns:
        String com resumo gerado localmente
    """
    texto = texto or ""
    rotulo = atribuir_rotulo(texto)
    base = FALLBACK_SUMMARIES.get(rotulo)
    trecho = truncate(texto, 140)
    
    if base:
        return f"{base} Trecho original: {trecho}" if trecho else base
    
    return f"Resumo automático indisponível. Revisar interação: {trecho}" if trecho else "Resumo automático indisponível. Texto original ausente."


def gerar_resumo_topico_fallback(topico: str, textos: List[str], rotulos: List[str] = None) -> str:
    """
    Gera resumo de tópico usando heurísticas locais.
    
    Args:
        topico: Nome do tópico
        textos: Lista de textos do tópico
        rotulos: Lista de rótulos correspondentes (opcional)
        
    Returns:
        String com resumo do tópico
    """
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