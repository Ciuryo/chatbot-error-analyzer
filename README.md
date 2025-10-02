# 🤖 Analisador de Erros de Chatbot

Sistema inteligente para análise automática de interações de chatbot usando IA, classificação de erros e geração de relatórios detalhados.

## 🚀 Funcionalidades

- **Análise Automática**: Processa conversas de chatbot e identifica padrões de erro
- **IA Integrada**: Usa GPT-4 para gerar resumos inteligentes das interações
- **Classificação ML**: Treina modelos de machine learning para categorizar problemas
- **Relatórios Visuais**: Gera PDFs com gráficos e análises detalhadas
- **Processamento em Lote**: Suporta múltiplos arquivos CSV simultaneamente

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

## 📈 Métricas de Performance

O sistema reporta automaticamente:
- **F1-Score Macro**: Performance média em todas as classes
- **F1-Score Weighted**: Performance ponderada por frequência
- **Custo da API**: Estimativa em USD dos tokens utilizados
