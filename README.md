#  Projeto de Previsão de Preços de Imóveis (MLOps)

Este repositório contém um projeto completo de Machine Learning para prever preços de imóveis, abrangendo desde o treinamento do modelo até ao seu deploy como uma API containerizada.

##  Descrição

O objetivo deste projeto é construir e servir um modelo de regressão para estimar os preços de venda de casas com base nas suas características (número de quartos, área, localização, etc.).

O projeto explora duas abordagens principais para o deploy:
1.  Uma **API RESTful** robusta construída com **FastAPI** e containerizada com **Docker**.
2.  Uma **aplicação web interativa** e standalone construída com **Streamlit**.

##  Funcionalidades

-   **Modelo de Machine Learning:** Utiliza um modelo **XGBoost Regressor** para alta performance.
-   **Otimização de Hiperparâmetros:** Usa a biblioteca **Optuna** para encontrar os melhores parâmetros para o modelo.
-   **Pré-processamento:** Emprega um `Pipeline` do Scikit-learn para encapsular a transformação de dados (StandardScaler para dados numéricos e OneHotEncoder para dados categóricos).
-   **Deploy como API:** A API é construída com FastAPI, garantindo alta velocidade e documentação automática (Swagger UI).
-   **Deploy como Web App:** Uma interface interativa e amigável criada com Streamlit para demonstrações rápidas.
-   **Containerização:** A API FastAPI é totalmente containerizada com **Docker** garantindo portabilidade e reprodutibilidade.
-   **Gestão de Ficheiros Grandes:** Utiliza **Git LFS** (Large File Storage) para gerir o ficheiro do modelo treinado.

##  Estrutura do Projeto

```
/projeto-mlops/
|
├── .dockerignore           # Ficheiros a serem ignorados pelo Docker
├── .gitignore              # Ficheiros a serem ignorados pelo Git
├── Dockerfile              # Receita para construir a imagem Docker da API
├── app.py                  # Código da aplicação web com Streamlit
├── main.py                 # Código da API com FastAPI
├── modelo_previsao_preco_v1.pkl # O pipeline de modelo treinado (gerido pelo Git LFS)
├── requirements.txt        # Dependências Python do projeto
└── README.md               # Este ficheiro
```

## ⚙ Instalação e Configuração Local

Siga os passos abaixo para executar o projeto na sua máquina.

**Pré-requisitos:**
* Python 3.9+
* Git e Git LFS
* Docker 

**Passos:**

1.  **Clonar o repositório:**
    ```bash
    git clone git@github.com:seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2.  **Instalar o Git LFS e baixar o modelo:**
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Criar e ativar um ambiente virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Linux/macOS
    # venv\Scripts\activate    # No Windows
    ```

4.  **Instalar as dependências Python:**
    ```bash
    pip install -r requirements.txt
    ```

##  Como Usar

Existem três maneiras de executar esta aplicação.

### 1. Executar a API FastAPI Diretamente

Esta opção inicia um servidor de desenvolvimento local que serve a sua API.

```bash
uvicorn main:app --reload
```
A API estará disponível em `http://127.0.0.1:8000`. A documentação interativa pode ser acedida em `http://127.0.0.1:8000/docs`.

### 2. Executar a Aplicação Web com Streamlit

Esta opção inicia uma interface web completa e interativa. Não requer que a API FastAPI esteja a ser executada.

```bash
streamlit run app.py
```
A aplicação abrirá automaticamente no seu navegador.

### 3. Executar a API com Docker

Esta é a forma recomendada para simular um ambiente de produção.

1.  **Construir a imagem Docker:**
    ```bash
    docker build -t api-precos .
    ```

2.  **Executar o contentor:**
    ```bash
    docker run -d -p 8000:8000 --name container-api-precos api-precos
    ```
A API estará disponível em `http://localhost:8000`.

##  Detalhes do Endpoint da API

### `POST /predict`

Este endpoint recebe as características de um imóvel e retorna o seu preço previsto.

**Corpo do Pedido (Request Body):**

O corpo do pedido deve ser um JSON com a seguinte estrutura:

```json
{
  "bedrooms": 3,
  "bathrooms": 2.5,
  "sqft_living": 1800,
  "sqft_lot": 5000,
  "floors": 2,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "sqft_above": 1800,
  "sqft_basement": 0,
  "yr_built": 1995,
  "yr_renovated": 0,
  "city": "Seattle"
}
```

**Resposta de Sucesso (200 OK):**

```json
{
  "predicted_price": 540345.75
}
```

##  Processo de Treinamento do Modelo

O modelo `modelo_previsao_preco_v1.pkl` foi gerado através de um processo que incluiu:
-   Limpeza e pré-processamento dos dados.
-   Transformação logarítmica da variável alvo (`price`) para normalizar a sua distribuição.
-   Busca de hiperparâmetros com Optuna para encontrar a melhor configuração para o modelo XGBoost.
-   Treinamento do `Pipeline` final (pré-processador + modelo) com todo o conjunto de dados.
-   Persistência do objeto `Pipeline` completo para garantir que as mesmas transformações sejam aplicadas em produção.

##  Tecnologias Utilizadas

-   **Linguagem:** Python 3.12
-   **Machine Learning:** Scikit-learn, XGBoost, Optuna, Pandas, NumPy
-   **API:** FastAPI
-   **Web App:** Streamlit
-   **Containerização:** Docker
-   **Controlo de Versão:** Git, Git LFS
