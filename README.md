# 🤖 Analisador de Erros de Chatbot

Sistema inteligente para análise automática de interações de chatbot usando IA, classificação de erros e geração de relatórios detalhados.

## 🚀 Funcionalidades

- **Análise Automática**: Processa conversas de chatbot e identifica padrões de erro
- **IA Integrada**: Usa GPT-4 para gerar resumos inteligentes das interações
- **Classificação ML**: Treina modelos de machine learning para categorizar problemas
- **Relatórios Visuais**: Gera PDFs com gráficos e análises detalhadas
- **Processamento em Lote**: Suporta múltiplos arquivos CSV simultaneamente

## 📊 Tipos de Erro Detectados

- Problemas com CPF
- Questões de parcelamento
- Erros em propostas
- Problemas de navegação
- Solicitações de atendimento humano
- E muito mais...

## 🛠️ Instalação

### Pré-requisitos
- Python 3.8+
- Chave da API OpenAI

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/chatbot-error-analyzer.git
cd chatbot-error-analyzer
```

### 2. Crie um ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente
```bash
cp .env.example .env
# Edite o arquivo .env com sua chave da OpenAI
```

## 🚀 Como Usar

### 1. Prepare seus dados
Coloque seus arquivos CSV na pasta `data/input/` com as colunas:
- `Observação`: Texto da interação do usuário
- Data/hora (qualquer formato)

**Exemplo de formato CSV:**
```csv
Data;Observação
2024-01-15 10:30:00;pergunta: Como faço para solicitar meu CPF?
2024-01-15 11:15:00;pergunta: Não consigo ver as opções de parcelamento
```

Veja o arquivo `data/input/exemplo_conversas.csv` para referência.

### 2. Execute a análise
```bash
python src/analyzer.py
```

### 3. Veja os resultados
- **CSV processado**: `data/output/`
- **Relatório PDF**: `reports/`
- **Resumos por tópico**: `data/output/`

## ⚙️ Configuração

Edite o arquivo `.env` para personalizar:

```bash
# API OpenAI
OPENAI_API_KEY=sua_chave_aqui

# Modelos
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large

# Performance
SUMMARY_BATCH_SIZE=50
USE_OVERSAMPLING=true
```

## 📈 Métricas de Performance

O sistema reporta automaticamente:
- **F1-Score Macro**: Performance média em todas as classes
- **F1-Score Weighted**: Performance ponderada por frequência
- **Custo da API**: Estimativa em USD dos tokens utilizados
