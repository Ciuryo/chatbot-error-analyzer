"""
Utilitários para leitura e manipulação de arquivos CSV.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    logging.info("chardet não disponível. Detecção automática de encoding limitada.")


def read_csv_com_fallback(path: Path, sep: str = ";") -> pd.DataFrame:
    """
    Lê um arquivo CSV tentando diferentes encodings e separadores automaticamente.
    
    Args:
        path: Caminho para o arquivo CSV
        sep: Separador preferencial (padrão: ";")
        
    Returns:
        DataFrame do pandas com os dados do CSV
        
    Raises:
        RuntimeError: Se não conseguir ler o arquivo com nenhuma combinação
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "utf-16", "iso8859-15"]
    seps_to_try = [sep, ",", "\t", "|"]
    
    last_err: Optional[Exception] = None
    
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
                rawdata = f.read(50000)
            enc = chardet.detect(rawdata).get("encoding", "utf-8")
            df = pd.read_csv(path, sep=sep, encoding=enc, on_bad_lines="skip")
            logging.info(f"Lido {path} via chardet: encoding={enc}")
            return df
        except Exception as e:
            pass
    
    raise RuntimeError(f"Falha ao ler CSV {path}. Último erro: {last_err}. Instale 'chardet' para melhor detecção de encoding.")


def get_input_csv_paths() -> List[Path]:
    """
    Determina quais arquivos CSV devem ser processados baseado nas variáveis de ambiente.
    
    Returns:
        Lista de caminhos Path para arquivos CSV a serem processados
        
    Note:
        - Verifica INPUT_FILES para lista específica de arquivos
        - Verifica INPUT_DIR para processar todos CSVs de uma pasta
        - Usa arquivo padrão se nenhuma variável estiver definida
    """
    from .config import DEFAULT_CONVERSATION_FILE
    
    # Opção 1: Lista específica de arquivos
    input_files_env = os.getenv("INPUT_FILES")
    if input_files_env:
        file_list = [f.strip() for f in input_files_env.split(",") if f.strip()]
        paths = [Path(f) for f in file_list]
        existing_paths = [p for p in paths if p.exists()]
        if not existing_paths:
            logging.warning("Nenhum arquivo da lista INPUT_FILES foi encontrado.")
        return existing_paths
    
    # Opção 2: Todos os CSVs de um diretório
    input_dir_env = os.getenv("INPUT_DIR")
    if input_dir_env:
        input_dir = Path(input_dir_env)
        if input_dir.is_dir():
            csv_files = list(input_dir.glob("*.csv"))
            if csv_files:
                return csv_files
            else:
                logging.warning(f"Nenhum arquivo CSV encontrado em {input_dir}")
        else:
            logging.warning(f"Diretório INPUT_DIR não existe: {input_dir}")
    
    # Opção 3: Arquivo padrão
    default_path = Path(DEFAULT_CONVERSATION_FILE)
    if default_path.exists():
        return [default_path]
    
    # Busca na pasta data/input
    data_input_path = Path("data/input") / DEFAULT_CONVERSATION_FILE
    if data_input_path.exists():
        return [data_input_path]
    
    logging.error(f"Arquivo padrão não encontrado: {DEFAULT_CONVERSATION_FILE}")
    return []


def load_and_normalize(paths: List[Path], message_column: str) -> pd.DataFrame:
    """
    Carrega e normaliza múltiplos arquivos CSV em um único DataFrame.
    
    Args:
        paths: Lista de caminhos para arquivos CSV
        message_column: Nome da coluna que contém as mensagens
        
    Returns:
        DataFrame consolidado e normalizado
        
    Note:
        - Adiciona coluna 'arquivo_origem' para rastrear fonte dos dados
        - Remove duplicatas baseadas na coluna de mensagens
        - Filtra mensagens válidas (não vazias)
    """
    if not paths:
        return pd.DataFrame()
    
    frames = []
    for path in paths:
        try:
            df = read_csv_com_fallback(path)
            df["arquivo_origem"] = str(path)
            frames.append(df)
        except Exception as e:
            logging.error(f"Erro ao carregar {path}: {e}")
            continue
    
    if not frames:
        return pd.DataFrame()
    
    # Consolida todos os DataFrames
    combined = pd.concat(frames, ignore_index=True)
    
    # Normaliza coluna de mensagens se existir
    if message_column in combined.columns:
        combined[message_column] = combined[message_column].astype(str).str.strip()
        # Remove mensagens vazias ou inválidas
        valid_mask = (
            combined[message_column].notna() & 
            (combined[message_column] != "") & 
            (combined[message_column] != "nan")
        )
        combined = combined[valid_mask].reset_index(drop=True)
    
    # Remove duplicatas baseadas na coluna de mensagens
    if message_column in combined.columns and len(combined) > 0:
        combined = combined.drop_duplicates(subset=[message_column]).reset_index(drop=True)
    
    logging.info(f"Carregados {len(combined)} registros únicos de {len(paths)} arquivo(s)")
    return combined


def find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifica automaticamente a coluna de data/hora em um DataFrame.
    
    Args:
        df: DataFrame para analisar
        
    Returns:
        Nome da coluna de data/hora encontrada ou None se não encontrada
        
    Note:
        - Procura por nomes comuns de colunas de data/hora
        - Tenta converter colunas para datetime para validar
        - Retorna a primeira coluna válida encontrada
    """
    # Nomes comuns para colunas de data/hora
    datetime_candidates = [
        "data", "date", "datetime", "timestamp", "created_at", "updated_at",
        "Data", "Date", "DateTime", "Timestamp", "Created_At", "Updated_At",
        "data_hora", "data_criacao", "data_atualizacao"
    ]
    
    # Primeiro, verifica nomes conhecidos
    for col in datetime_candidates:
        if col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(100), errors='raise')
                return col
            except:
                continue
    
    # Se não encontrou por nome, tenta todas as colunas
    for col in df.columns:
        if df[col].dtype == 'object':  # Apenas colunas de texto
            try:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    pd.to_datetime(sample, errors='raise')
                    return col
            except:
                continue
    
    return None