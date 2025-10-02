"""
Sistema de extração e agrupamento de tópicos.
"""
import re
from typing import Dict, List


# Regex para extrair tópicos do formato "pergunta: <conteúdo>"
TOPICO_REGEX = re.compile(r"pergunta:\s*([^|]+)", re.IGNORECASE)


def extrair_topico(texto: str) -> str:
    """
    Extrai o tópico principal de um texto baseado no padrão "pergunta: <conteúdo>".
    
    Args:
        texto: Texto da interação
        
    Returns:
        String com o tópico extraído ou "Outro" se não encontrado
    """
    if not isinstance(texto, str):
        return "Outro"
    
    match = TOPICO_REGEX.search(texto)
    if match:
        bruto = match.group(1)
        if bruto:
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
    """
    topicos: Dict[str, List[str]] = {}
    
    for pergunta in questions:
        if not isinstance(pergunta, str):
            continue
        
        topico = extrair_topico(pergunta)
        topicos.setdefault(topico, []).append(pergunta)
    
    return topicos