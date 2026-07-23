# 📈 Stock Price Prediction API with LSTM

Projeto de Machine Learning para previsão de preços de ações utilizando redes neurais LSTM (Long Short-Term Memory). O projeto contempla todo o pipeline de desenvolvimento, desde a coleta dos dados financeiros, engenharia de atributos e treinamento do modelo até a disponibilização das previsões por meio de uma API REST desenvolvida com FastAPI.

O objetivo do projeto é construir um pipeline completo de previsão de séries temporais financeiras utilizando boas práticas de organização de código, Data Science e Engenharia de Software.

---

## ▶️ Video Apresentação
``` code
...
```

---

## 🌐 API em Produção
``` code
Rota inicial:  https://stock-lstm-mx0c.onrender.com/
Documentação:  https://stock-lstm-mx0c.onrender.com/docs
Status da API: https://stock-lstm-mx0c.onrender.com/health
```

---

## 🚀 Funcionalidades

- Treinamento de modelos LSTM para ações da B3.
- Previsão do próximo preço de fechamento.
- API REST para consulta das previsões.

---

## 🧱 Arquitetura do Projeto

```
stock_lstm/
│
├── app/                  # API FastAPI
│   ├── api.py
│   ├── schemas.py
│   └── services.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   │ ACAO 
│   │   ├── model.keras
│   │   ├── scaler.pkl
│   │   └── metadata.json
│
│
├── reports/
│   │ ACAO 
│   │   ├── prediction.png
│   │   ├── training_history
│
├── src/
│   | data/
|   │    ├── dataset.py
│   │    ├── download.py
|   │    ├── prepare_dataset.py
│   │    └── preprocess.py   
│   | features/
|   │    ├── feature_engineering.py
│   │    └── indicators.py 
│   | model/
|   │    ├── evaluate.py
│   │    ├── lstm.py
|   │    ├── predict.py
│   │    └── train.py
│   | pipeline/
│   │    └── training_pipeline.py
│   | utils/
|   │    ├── metrics.py
│   │    └── plots.py
|   | config.py
│
├── requirements.txt
└── README.md
```

---

## 🔄 Fluxo do Projeto 

```text
Yahoo Finance
      │
      ▼
Download dos dados
      │
      ▼
Pré-processamento
      │
      ▼
Feature Engineering
      │
      ▼
Treinamento LSTM
      │
      ▼
Avaliação
      │
      ▼
Modelo salvo (.keras)
      │
      ▼
FastAPI
      │
      ▼
Predições
```

---

## 📦 Funcionalidades

- Download automático de ações via Yahoo Finance
- Criação de indicadores técnicos
- Pré-processamento dos dados
- Normalização
- Criação das sequências temporais
- Treinamento da rede LSTM
- Avaliação do modelo
- Salvamento do modelo treinado
- API REST para realizar previsões

---

### ▶️ Executando a API manualmente

Caso queira **Rodar a API em sua máquina local**

1. Clonar o repositório:
   ```bash
   git clone https://github.com/Gabriel-limadev/stock_lstm.git
   cd stock_lstm
   
2. Instalar dependências:
   ```bash
   pip install -r requirements.txt

3. Rodar API
   ```bash
   uvicorn app.main:app --reload

A API ficará disponivel localmente em: http://127.0.0.1:8000/docs

---

## 📄 Exemplos de Requests e Responses

### 🔹 Treinar alguma ação da B3
**Request**
```http
POST /train (passando ação)
```
**Response**
```code
{
    "message": "Treinamento concluído com sucesso.",
    "stock": "PETR4.SA"
}
```

### 🔹 Realizar predição
**Request**
```http
GET /predict/PETR4
```
**Response**
```code
{
    "stock": "PETR4.SA",
    "last_available_date": "2026-07-21",
    "prediction_date": "2026-07-22",
    "last_close": 41.65999984741211,
    "predicted_close": 40.36281201802706
}
```

---

## 📈 Escalabilidade Futura

- Docker
- Testes automatizados
- CI/CD com GitHub Actions
- Banco de dados para histórico de previsões
- Múltiplos modelos (GRU, Transformer)
- Dashboard em Streamlit

---

## 👨‍💻 Autor

Gabriel Lima  
Pós-graduação em Engenharia de Machine Learning — FIAP
