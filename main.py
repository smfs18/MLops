from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Criar a instância do FastAPI
app = FastAPI(title="API de Previsão de Preços de Casas")

# Carregar o PIPELINE completo
# Este arquivo contém o pré-processador E o modelo.
with open("modelo_previsao_preco_v1.pkl", "rb") as f:
    pipeline = joblib.load(f)

# Definir o esquema de entrada (como o usuário vai enviar os dados)
# Isso está perfeito, não precisa mudar.
class InputFeatures(BaseModel):
    bedrooms: float  
    bathrooms: float
    sqft_living: int 
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    city: str

@app.post("/predict")
def predict_price(features: InputFeatures):
    
    # 2. Converter os dados de entrada Pydantic para um Dicionário
    data = features.dict()
    
    # 3. Criar um DataFrame do Pandas a partir do dicionário
    # A estrutura (nomes das colunas) deve ser a mesma usada no treinamento!
    input_df = pd.DataFrame([data])
    
    # 4. Fazer a previsão usando o PIPELINE carregado
    # O pipeline irá aplicar o StandardScaler e o OneHotEncoder automaticamente
    predicted_price_log = pipeline.predict(input_df)
    
    # 5. Reverter a transformação logarítmica para obter o preço real
    # Lembre-se que seu modelo foi treinado em y_log!
    predicted_price = np.expm1(predicted_price_log)
    
    # 6. Retornar a previsão final como resposta JSON
    # Usamos [0] para pegar o primeiro (e único) valor do array de predição
    return {"predicted_price": predicted_price[0]}

# Endpoint raiz
@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de Previsão de Preços de Casas!"}
