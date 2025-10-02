"""
Gerador de relatórios em PDF com visualizações.
"""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import textwrap


def gerar_relatorio_pdf(
    df: pd.DataFrame,
    output_path: Path,
    report_emb: str = "",
    report_tfidf: str = "",
    topic_summary_df: Optional[pd.DataFrame] = None,
    daily_pivot: Optional[pd.DataFrame] = None,
) -> None:
    """
    Gera um relatório completo em PDF com análises e visualizações.
    
    Args:
        df: DataFrame principal com os dados analisados
        output_path: Caminho para salvar o PDF
        report_emb: Relatório do modelo de embeddings
        report_tfidf: Relatório do modelo TF-IDF
        topic_summary_df: DataFrame com resumos por tópico
        daily_pivot: DataFrame com contagens diárias por rótulo
    """
    try:
        with PdfPages(output_path) as pdf:
            # Página 1: Resumo Executivo
            _criar_pagina_resumo(pdf, df, daily_pivot)
            
            # Página 2: Distribuição de Rótulos
            if "rotulo" in df.columns:
                _criar_pagina_rotulos(pdf, df)
            
            # Página 3: Análise de Tópicos
            if topic_summary_df is not None and not topic_summary_df.empty:
                _criar_pagina_topicos(pdf, topic_summary_df)
            
            # Página 4: Relatórios de ML
            if report_emb or report_tfidf:
                _criar_pagina_ml(pdf, report_emb, report_tfidf)
            
            # Página 5: Amostras de Dados
            _criar_pagina_amostras(pdf, df)
        
        logging.info(f"Relatório PDF gerado: {output_path}")
        
    except Exception as e:
        logging.error(f"Erro ao gerar PDF {output_path}: {e}")


def _criar_pagina_resumo(pdf: PdfPages, df: pd.DataFrame, daily_pivot: Optional[pd.DataFrame]):
    """Cria página de resumo executivo."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Análise de Erros do Chatbot - Resumo Executivo", fontsize=16, fontweight='bold')
    
    # Estatísticas básicas
    total_interacoes = len(df)
    ax1.text(0.1, 0.8, f"Total de Interações: {total_interacoes:,}", fontsize=14, fontweight='bold')
    
    if "rotulo" in df.columns:
        rotulos_unicos = df["rotulo"].nunique()
        ax1.text(0.1, 0.6, f"Tipos de Problemas: {rotulos_unicos}", fontsize=12)
        
        problema_mais_comum = df["rotulo"].value_counts().index[0]
        freq_mais_comum = df["rotulo"].value_counts().iloc[0]
        ax1.text(0.1, 0.4, f"Problema Mais Comum:\n{problema_mais_comum}", fontsize=10)
        ax1.text(0.1, 0.2, f"Frequência: {freq_mais_comum} ({freq_mais_comum/total_interacoes*100:.1f}%)", fontsize=10)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title("Estatísticas Gerais")
    
    # Gráfico de barras dos principais problemas
    if "rotulo" in df.columns:
        top_rotulos = df["rotulo"].value_counts().head(10)
        top_rotulos.plot(kind='bar', ax=ax2)
        ax2.set_title("Top 10 Problemas Mais Frequentes")
        ax2.set_xlabel("Tipo de Problema")
        ax2.set_ylabel("Frequência")
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, "Dados de rótulos não disponíveis", ha='center', va='center')
        ax2.set_title("Distribuição de Problemas")
    
    # Análise temporal se disponível
    if daily_pivot is not None and not daily_pivot.empty:
        daily_total = daily_pivot.sum(axis=1)
        daily_total.plot(ax=ax3)
        ax3.set_title("Evolução Temporal dos Problemas")
        ax3.set_xlabel("Data")
        ax3.set_ylabel("Total de Ocorrências")
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, "Dados temporais não disponíveis", ha='center', va='center')
        ax3.set_title("Análise Temporal")
    
    # Análise de tópicos
    if "topico" in df.columns:
        top_topicos = df["topico"].value_counts().head(8)
        colors = plt.cm.Set3(range(len(top_topicos)))
        ax4.pie(top_topicos.values, labels=None, autopct='%1.1f%%', colors=colors)
        ax4.set_title("Distribuição por Tópicos")
        
        # Legenda
        legend_labels = [f"{topico[:20]}..." if len(topico) > 20 else topico 
                        for topico in top_topicos.index]
        ax4.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        ax4.text(0.5, 0.5, "Dados de tópicos não disponíveis", ha='center', va='center')
        ax4.set_title("Análise de Tópicos")
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _criar_pagina_rotulos(pdf: PdfPages, df: pd.DataFrame):
    """Cria página de análise detalhada de rótulos."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Análise Detalhada de Rótulos/Problemas", fontsize=16, fontweight='bold')
    
    rotulo_counts = df["rotulo"].value_counts()
    
    # Gráfico de barras horizontal
    top_15 = rotulo_counts.head(15)
    top_15.plot(kind='barh', ax=ax1)
    ax1.set_title("Top 15 Problemas por Frequência")
    ax1.set_xlabel("Número de Ocorrências")
    
    # Distribuição percentual
    percentuais = (rotulo_counts / len(df) * 100).head(10)
    percentuais.plot(kind='bar', ax=ax2, color='orange')
    ax2.set_title("Distribuição Percentual (Top 10)")
    ax2.set_ylabel("Percentual (%)")
    ax2.tick_params(axis='x', rotation=45)
    
    # Análise de concentração
    cumsum = (rotulo_counts.cumsum() / rotulo_counts.sum() * 100)
    ax3.plot(range(1, len(cumsum) + 1), cumsum.values)
    ax3.axhline(y=80, color='r', linestyle='--', label='80% dos problemas')
    ax3.set_title("Curva de Concentração de Problemas")
    ax3.set_xlabel("Número de Tipos de Problemas")
    ax3.set_ylabel("Percentual Acumulado (%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Tabela com estatísticas
    ax4.axis('tight')
    ax4.axis('off')
    
    stats_data = [
        ["Total de Tipos", f"{len(rotulo_counts)}"],
        ["Mais Frequente", f"{rotulo_counts.index[0]} ({rotulo_counts.iloc[0]})"],
        ["Menos Frequente", f"{rotulo_counts.index[-1]} ({rotulo_counts.iloc[-1]})"],
        ["Mediana", f"{rotulo_counts.median():.1f}"],
        ["Desvio Padrão", f"{rotulo_counts.std():.1f}"],
    ]
    
    table = ax4.table(cellText=stats_data, colLabels=["Métrica", "Valor"],
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title("Estatísticas dos Rótulos")
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _criar_pagina_topicos(pdf: PdfPages, topic_summary_df: pd.DataFrame):
    """Cria página de análise de tópicos."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    fig.suptitle("Resumos por Tópico", fontsize=16, fontweight='bold', y=0.95)
    
    # Ordena por quantidade de ocorrências
    ordenado = topic_summary_df.sort_values("quantidade_ocorrencias", ascending=False)
    
    y_pos = 0.9
    for _, row in ordenado.head(15).iterrows():  # Top 15 tópicos
        topico = row["topico"]
        quantidade = row["quantidade_ocorrencias"]
        resumo = row["resumo"]
        
        # Título do tópico
        ax.text(0.05, y_pos, f"{topico} ({quantidade} ocorrências)", 
               fontsize=12, fontweight='bold', wrap=True)
        y_pos -= 0.03
        
        # Resumo (quebra de linha automática)
        resumo_wrapped = textwrap.fill(str(resumo), width=80)
        for linha in resumo_wrapped.split('\n'):
            ax.text(0.05, y_pos, linha, fontsize=10, wrap=True)
            y_pos -= 0.025
        
        y_pos -= 0.02  # Espaço entre tópicos
        
        if y_pos < 0.1:  # Se chegou no fim da página
            break
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _criar_pagina_ml(pdf: PdfPages, report_emb: str, report_tfidf: str):
    """Cria página com relatórios de Machine Learning."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle("Relatórios de Machine Learning", fontsize=16, fontweight='bold')
    
    # Relatório de Embeddings
    ax1.axis('off')
    ax1.set_title("Modelo com Embeddings", fontsize=14, fontweight='bold')
    
    if report_emb:
        # Quebra o texto em linhas
        linhas_emb = report_emb.split('\n')
        y_pos = 0.95
        for linha in linhas_emb:
            if y_pos < 0.05:
                break
            ax1.text(0.05, y_pos, linha, fontsize=8, fontfamily='monospace')
            y_pos -= 0.04
    else:
        ax1.text(0.5, 0.5, "Relatório de embeddings não disponível", 
                ha='center', va='center', fontsize=12)
    
    # Relatório TF-IDF
    ax2.axis('off')
    ax2.set_title("Modelo TF-IDF Baseline", fontsize=14, fontweight='bold')
    
    if report_tfidf:
        linhas_tfidf = report_tfidf.split('\n')
        y_pos = 0.95
        for linha in linhas_tfidf:
            if y_pos < 0.05:
                break
            ax2.text(0.05, y_pos, linha, fontsize=8, fontfamily='monospace')
            y_pos -= 0.04
    else:
        ax2.text(0.5, 0.5, "Relatório TF-IDF não disponível", 
                ha='center', va='center', fontsize=12)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _criar_pagina_amostras(pdf: PdfPages, df: pd.DataFrame):
    """Cria página com amostras dos dados."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    fig.suptitle("Amostras dos Dados Analisados", fontsize=16, fontweight='bold', y=0.95)
    
    # Seleciona algumas amostras representativas
    if len(df) > 0:
        amostras = df.head(10)  # Primeiras 10 linhas
        
        y_pos = 0.9
        for idx, row in amostras.iterrows():
            # Mostra informações principais de cada amostra
            if "rotulo" in df.columns:
                ax.text(0.05, y_pos, f"Rótulo: {row['rotulo']}", 
                       fontsize=10, fontweight='bold')
                y_pos -= 0.03
            
            if "topico" in df.columns:
                ax.text(0.05, y_pos, f"Tópico: {row['topico']}", fontsize=9)
                y_pos -= 0.025
            
            if "resumo" in df.columns:
                resumo_texto = str(row['resumo'])[:200] + "..." if len(str(row['resumo'])) > 200 else str(row['resumo'])
                resumo_wrapped = textwrap.fill(resumo_texto, width=80)
                ax.text(0.05, y_pos, f"Resumo: {resumo_wrapped}", fontsize=8)
                y_pos -= 0.04
            
            y_pos -= 0.02  # Espaço entre amostras
            
            if y_pos < 0.1:
                break
    else:
        ax.text(0.5, 0.5, "Nenhuma amostra disponível", ha='center', va='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)