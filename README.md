# 🤖 Analisador de Erros de Chatbot

Sistema modular para análise de interações de chatbot, classificação automática de problemas e geração de relatórios detalhados.

## 📁 Estrutura Modular

```
src/
├── __init__.py              # Pacote principal
├── analyzer.py              # Função principal e orquestração
├── config.py                # Configurações e constantes
├── cost_tracker.py          # Controle de custos da API OpenAI
├── file_utils.py            # Utilitários para leitura de arquivos CSV
├── text_classifier.py       # Sistema de classificação heurística
├── topic_extractor.py       # Extração e agrupamento de tópicos
├── openai_client.py         # Cliente e integração com OpenAI
├── ml_models.py             # Modelos de Machine Learning
└── report_generator.py      # Geração de relatórios em PDF
```

## 🚀 Funcionalidades

### 🔍 Análise Automática
- **Classificação heurística**: Identifica tipos de problemas usando regex
- **Extração de tópicos**: Agrupa interações por assunto
- **Resumos inteligentes**: Gera resumos usando OpenAI GPT ou fallback local

### 🤖 Machine Learning
- **Embeddings**: Vetorização semântica usando OpenAI
- **Classificação**: Regressão logística com balanceamento de classes
- **Baseline TF-IDF**: Comparação com métodos tradicionais

### 📊 Relatórios
- **CSV estruturados**: Dados processados com resumos e classificações
- **PDFs visuais**: Gráficos, estatísticas e análises detalhadas
- **Agregação temporal**: Análise por dia/período

### 💰 Controle de Custos
- **Monitoramento**: Rastreamento de tokens e custos da API
- **Cache inteligente**: Evita reprocessamento desnecessário
- **Fallback local**: Funciona sem API quando necessário

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
python run_analyzer.py
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

## 🏗️ Benefícios da Modularização

### ✅ Manutenibilidade
- Código organizado em módulos específicos
- Responsabilidades bem definidas
- Fácil localização de funcionalidades

### ✅ Testabilidade
- Módulos independentes podem ser testados isoladamente
- Mocking simplificado para testes unitários
- Cobertura de código mais eficiente

### ✅ Reutilização
- Módulos podem ser importados individualmente
- Funcionalidades reutilizáveis em outros projetos
- APIs bem definidas entre componentes

### ✅ Escalabilidade
- Fácil adição de novos classificadores
- Extensão de funcionalidades sem impacto
- Paralelização de processamento

## 💻 Exemplo de Uso Programático

```python
from src.openai_client import build_client, resumir_em_lote
from src.text_classifier import atribuir_rotulo
from src.file_utils import read_csv_com_fallback

# Carrega dados
df = read_csv_com_fallback("dados.csv")

# Classifica textos
df['rotulo'] = df['mensagem'].apply(atribuir_rotulo)

# Gera resumos (se tiver API)
client = build_client()
if client:
    resumos = resumir_em_lote(client, df['mensagem'].tolist(), "gpt-4.1-mini")
    df['resumo'] = resumos
```
## 📈 Métricas de Performance

O sistema reporta automaticamente:
- **F1-Score Macro**: Performance média em todas as classes
- **F1-Score Weighted**: Performance ponderada por frequência
- **Custo da API**: Estimativa em USD dos tokens utilizados

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🆘 Suporte

- 📧 Email: seu-email@exemplo.com
- 🐛 Issues: [GitHub Issues](https://github.com/Ciuryo/chatbot-error-analyzer/issues)
- 📖 Wiki: [Documentação completa](https://github.com/Ciuryo/chatbot-error-analyzer/wiki)

## 🏆 Créditos

Desenvolvido por [Ciuryo](https://github.com/Ciuryo)
