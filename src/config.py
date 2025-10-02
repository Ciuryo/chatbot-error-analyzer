"""
Configurações e constantes do sistema de análise de erros do chatbot.
"""
import os
from typing import Dict, Any

# =============================
# CONFIGURAÇÕES PADRÃO DO SISTEMA
# =============================

# Arquivos CSV padrão para análise
DEFAULT_CONVERSATION_FILE = "21_filtrado_21.csv"

# Configurações de colunas dos dados
DEFAULT_MESSAGE_COLUMN = "Observação"
DEFAULT_LABEL_COLUMN = "rotulo"

# Arquivos de saída
DEFAULT_OUTPUT_FILE = "relatorio_erro_input.csv"
DEFAULT_PDF_OUTPUT = "Analise_Erros_Chatbot_Diagnostico.pdf"

# Modelos da OpenAI
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

# Configurações de API
MAX_API_RETRIES = 3

# =============================
# CONFIGURAÇÕES DE PROCESSAMENTO
# =============================

# Configurações de resumo em lote
SUMMARY_BATCH_SIZE = int(os.getenv("SUMMARY_BATCH_SIZE", "100"))
SUMMARY_MAX_CHARS_PER_ITEM = int(os.getenv("SUMMARY_MAX_CHARS_PER_ITEM", "800"))

# Configurações de resumo por tópico
DEFAULT_TOPIC_SUMMARY_FILE = "resumo_topicos.csv"
TOPIC_SUMMARY_BATCH_SIZE = int(os.getenv("TOPIC_SUMMARY_BATCH_SIZE", "8"))
TOPIC_SUMMARY_MAX_ITEMS = int(os.getenv("TOPIC_SUMMARY_MAX_ITEMS", "30"))
TOPIC_SUMMARY_MAX_CHARS = int(os.getenv("TOPIC_SUMMARY_MAX_CHARS", "500"))

# Opções de Machine Learning
USE_OVERSAMPLING = os.getenv("USE_OVERSAMPLING", "true").lower() == "true"
USE_TFIDF_BASELINE = os.getenv("USE_TFIDF_BASELINE", "true").lower() == "true"
TFIDF_SOURCE = os.getenv("TFIDF_SOURCE", "resumo")
MIN_SAMPLES_PER_CLASS = int(os.getenv("MIN_SAMPLES_PER_CLASS", "10"))

# =============================
# SISTEMA DE CONTROLE DE CUSTOS
# =============================

# Dicionário global para rastrear uso de tokens e custos
USAGE_STATS = {
    "tokens_in": 0,
    "tokens_out": 0,
    "tokens_total": 0,
    "emb_tokens": 0,
    "usd_total": 0.0
}

# Tabela de preços da OpenAI por 1000 tokens (valores aproximados em USD)
MODEL_PRICES = {
    "gpt-4.1-mini": {"in": 0.00015, "out": 0.0006},
    "text-embedding-3-large": {"in": 0.00013},
}

# =============================
# INSTRUÇÕES PARA API
# =============================

SUMMARY_SYS_INSTR = (
    "Você é um assistente especializado em análise de dados. "
    "Para CADA item fornecido, gere um resumo curto (2–3 frases) destacando "
    "os PRINCIPAIS erros de input do usuário (se houver). "
    "Saia ESTRITAMENTE em JSON como uma lista, onde cada elemento é {\"i\": <indice>, \"resumo\": <texto>}."
)

TOPIC_SUMMARY_SYS_INSTR = (
    "Você é um analista de atendimento especializado em encontrar padrões de erro. "
    "Para CADA tópico fornecido, identifique os principais problemas, causas recorrentes "
    "e impactos percebidos. "
    "Retorne ESTRITAMENTE em JSON como uma lista de objetos com o formato "
    "{\"topico\": <nome_exato>, \"resumo\": <texto>}. "
    "Use frases curtas ou bullet points para destacar os pontos críticos de forma objetiva."
)