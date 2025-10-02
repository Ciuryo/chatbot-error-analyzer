"""
Analisador principal de erros do chatbot.
Sistema modular para análise de interações, classificação automática e geração de relatórios.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Imports dos módulos locais
from .config import (
    DEFAULT_MESSAGE_COLUMN, DEFAULT_LABEL_COLUMN, DEFAULT_CHAT_MODEL, 
    DEFAULT_EMBEDDING_MODEL, DEFAULT_PDF_OUTPUT, DEFAULT_TOPIC_SUMMARY_FILE,
    USE_TFIDF_BASELINE, TFIDF_SOURCE
)
from .cost_tracker import print_usage_stats
from .file_utils import get_input_csv_paths, load_and_normalize, find_datetime_column
from .openai_client import build_client, resumir_em_lote, gerar_resumos_por_topico
from .text_classifier import atribuir_rotulo, gerar_resumo_topico_fallback
from .topic_extractor import extrair_topico
from .ml_models import generate_embeddings_for_texts, train_classifier_balanced, evaluate_tfidf_baseline
from .report_generator import gerar_relatorio_pdf


def main() -> None:
    """Função principal do analisador de erros do chatbot."""
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

        # Agregação diária
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
    print_usage_stats()