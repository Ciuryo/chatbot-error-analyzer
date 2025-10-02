"""
Modelos de Machine Learning para classificação de textos.
"""
import logging
from typing import List, Sequence
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

from .config import USE_OVERSAMPLING, MIN_SAMPLES_PER_CLASS
from .cost_tracker import registrar_emb_usage


def generate_embeddings_for_texts(
    client, 
    texts: List[str], 
    model: str, 
    batch_size: int = 128
) -> np.ndarray:
    """
    Gera embeddings vetoriais para uma lista de textos usando a API OpenAI.
    
    Args:
        client: Cliente OpenAI
        texts: Lista de textos para gerar embeddings
        model: Nome do modelo de embedding
        batch_size: Tamanho do lote para processamento
        
    Returns:
        Array numpy com os embeddings gerados
    """
    if client is None:
        logging.warning("Cliente OpenAI não disponível; retornando embeddings zeros")
        return np.zeros((len(texts), 1), dtype=np.float32)
    
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Registra uso de tokens
            total_tokens = sum(len(text.split()) for text in batch)
            registrar_emb_usage(model, total_tokens)
            
        except Exception as e:
            logging.error(f"Erro ao gerar embeddings para lote {i//batch_size + 1}: {e}")
            # Fallback: embeddings zeros para este lote
            embedding_dim = 1536 if "large" in model else 1024
            batch_embeddings = [[0.0] * embedding_dim] * len(batch)
            all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings, dtype=np.float32)


def train_classifier_balanced(embeddings: np.ndarray, labels: Sequence[str]) -> str:
    """
    Treina um classificador de regressão logística usando embeddings.
    
    Args:
        embeddings: Array de embeddings
        labels: Lista de rótulos correspondentes
        
    Returns:
        String com relatório de classificação
    """
    if len(embeddings) == 0 or len(labels) == 0:
        return "Dados insuficientes para treinamento"
    
    if len(embeddings) != len(labels):
        return "Número de embeddings não corresponde ao número de rótulos"
    
    # Conta amostras por classe
    label_counts = Counter(labels)
    
    # Filtra classes com amostras suficientes
    valid_labels = [label for label, count in label_counts.items() 
                   if count >= MIN_SAMPLES_PER_CLASS]
    
    if len(valid_labels) < 2:
        return f"Insuficientes classes com >= {MIN_SAMPLES_PER_CLASS} amostras. Encontradas: {len(valid_labels)}"
    
    # Filtra dados para classes válidas
    valid_indices = [i for i, label in enumerate(labels) if label in valid_labels]
    filtered_embeddings = embeddings[valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    
    try:
        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            filtered_embeddings, filtered_labels, 
            test_size=0.2, random_state=42, stratify=filtered_labels
        )
        
        # Treina modelo
        if USE_OVERSAMPLING:
            # Implementação simples de oversampling
            from collections import defaultdict
            class_samples = defaultdict(list)
            
            for i, label in enumerate(y_train):
                class_samples[label].append(X_train[i])
            
            # Encontra classe majoritária
            max_samples = max(len(samples) for samples in class_samples.values())
            
            # Balanceia classes
            balanced_X, balanced_y = [], []
            for label, samples in class_samples.items():
                samples_array = np.array(samples)
                current_count = len(samples)
                
                # Adiciona amostras originais
                balanced_X.extend(samples)
                balanced_y.extend([label] * current_count)
                
                # Adiciona amostras duplicadas se necessário
                if current_count < max_samples:
                    needed = max_samples - current_count
                    indices = np.random.choice(current_count, needed, replace=True)
                    balanced_X.extend(samples_array[indices])
                    balanced_y.extend([label] * needed)
            
            X_train = np.array(balanced_X)
            y_train = balanced_y
        
        # Treina classificador
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train, y_train)
        
        # Avalia modelo
        y_pred = classifier.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        
        return f"F1-Score (weighted): {f1:.3f}\n\nRelatório detalhado:\n{report}"
        
    except Exception as e:
        return f"Erro durante treinamento: {str(e)}"


def evaluate_tfidf_baseline(texts: List[str], labels: Sequence[str]) -> str:
    """
    Avalia um classificador baseline usando TF-IDF.
    
    Args:
        texts: Lista de textos
        labels: Lista de rótulos correspondentes
        
    Returns:
        String com relatório de avaliação
    """
    if len(texts) != len(labels):
        return "Número de textos não corresponde ao número de rótulos"
    
    if len(texts) == 0:
        return "Dados insuficientes para avaliação TF-IDF"
    
    # Conta amostras por classe
    label_counts = Counter(labels)
    valid_labels = [label for label, count in label_counts.items() 
                   if count >= MIN_SAMPLES_PER_CLASS]
    
    if len(valid_labels) < 2:
        return f"TF-IDF: Insuficientes classes com >= {MIN_SAMPLES_PER_CLASS} amostras"
    
    # Filtra dados
    valid_indices = [i for i, label in enumerate(labels) if label in valid_labels]
    filtered_texts = [texts[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    
    try:
        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            filtered_texts, filtered_labels, 
            test_size=0.2, random_state=42, stratify=filtered_labels
        )
        
        # Pipeline TF-IDF + Logistic Regression
        pipeline = make_pipeline(
            TfidfVectorizer(max_features=5000, stop_words=None),
            LogisticRegression(random_state=42, max_iter=1000)
        )
        
        # Treina e avalia
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        
        return f"TF-IDF Baseline - F1-Score (weighted): {f1:.3f}\n\nRelatório:\n{report}"
        
    except Exception as e:
        return f"Erro na avaliação TF-IDF: {str(e)}"