"""
Sistema de controle de custos da API OpenAI.
"""
from typing import Dict, Any
from .config import USAGE_STATS, MODEL_PRICES


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
    
    in_toks = usage.get("prompt_tokens", 0)
    out_toks = usage.get("completion_tokens", 0)
    total_toks = usage.get("total_tokens", in_toks + out_toks)
    
    USAGE_STATS["tokens_in"] += in_toks
    USAGE_STATS["tokens_out"] += out_toks
    USAGE_STATS["tokens_total"] += total_toks
    
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
    
    if model in MODEL_PRICES:
        p = MODEL_PRICES[model]
        custo = (tokens/1000.0)*p.get("in",0)
        USAGE_STATS["usd_total"] += custo


def print_usage_stats():
    """Imprime estatísticas de uso e custos."""
    if USAGE_STATS["usd_total"] > 0:
        print("\n========== CUSTOS ==========")
        print(f"Tokens entrada: {USAGE_STATS['tokens_in']}")
        print(f"Tokens saída: {USAGE_STATS['tokens_out']}")
        print(f"Tokens total: {USAGE_STATS['tokens_total']}")
        print(f"Tokens embeddings: {USAGE_STATS['emb_tokens']}")
        print(f"💰 Custo estimado: ${USAGE_STATS['usd_total']:.4f} USD")