import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURAÇÃO DA PÁGINA ---
# Usar st.set_page_config() como o primeiro comando do Streamlit
st.set_page_config(
    page_title="Previsor de Preços de Imóveis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNÇÃO PARA CARREGAR O MODELO ---
# O decorator @st.cache_resource garante que o modelo seja carregado apenas uma vez,
# otimizando a performance da aplicação.
@st.cache_resource
def load_model():
    """ Carrega o pipeline de machine learning a partir do arquivo. """
    try:
        pipeline = joblib.load('modelo_previsao_preco_v1.pkl')
        return pipeline
    except FileNotFoundError:
        return None

# Carregar o modelo
pipeline = load_model()

# --- TÍTULO E DESCRIÇÃO ---
st.title('🏠 Previsão de Preços de Imóveis')
st.markdown("""
    Bem-vindo à ferramenta de previsão de preços de imóveis! 
    Esta aplicação utiliza um modelo de Machine Learning (XGBoost) para estimar o valor de mercado de uma casa.
    **Preencha os campos na barra lateral à esquerda com as características do imóvel.**
""")

# --- BARRA LATERAL COM INPUTS DO USUÁRIO ---
st.sidebar.header('Características do Imóvel')

def get_user_inputs():
    """ Coleta os inputs do usuário a partir da barra lateral. """
    bedrooms = st.sidebar.slider('Quartos (bedrooms)', 1, 10, 3)
    bathrooms = st.sidebar.slider('Banheiros (bathrooms)', 1.0, 8.0, 2.0, 0.25)
    sqft_living = st.sidebar.number_input('Área Interna (sqft_living)', min_value=300, max_value=15000, value=1800)
    sqft_lot = st.sidebar.number_input('Área do Lote (sqft_lot)', min_value=500, max_value=1000000, value=5000)
    floors = st.sidebar.slider('Andares (floors)', 1.0, 4.0, 2.0, 0.5)
    waterfront = st.sidebar.selectbox('Vista para Água (waterfront)', options=[0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não')
    view = st.sidebar.slider('Vista (0-4)', 0, 4, 0)
    condition = st.sidebar.slider('Condição (1-5)', 1, 5, 3)
    sqft_above = st.sidebar.number_input('Área (sem porão)', min_value=300, max_value=10000, value=1800)
    sqft_basement = st.sidebar.number_input('Área do Porão', min_value=0, max_value=5000, value=0)
    yr_built = st.sidebar.number_input('Ano de Construção', min_value=1900, max_value=2025, value=1995)
    yr_renovated = st.sidebar.number_input('Ano de Renovação (0 se nunca)', min_value=0, max_value=2025, value=0)
    city = st.sidebar.text_input('Cidade (city)', 'Seattle').strip()

    # Criar um dicionário com os dados
    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'city': city
    }
    return data

user_data = get_user_inputs()

# --- EXIBIÇÃO DA PREVISÃO ---
st.write('---')
st.header('Resultado da Previsão')

# Botão para iniciar a predição
if st.sidebar.button('Estimar Preço', use_container_width=True):
    if pipeline is not None:
        if user_data['city']:
            try:
                # Converter os dados do usuário em um DataFrame
                input_df = pd.DataFrame([user_data])
                
                # Fazer a predição usando o pipeline
                predicted_price_log = pipeline.predict(input_df)
                
                # Reverter a transformação logarítmica
                predicted_price = np.expm1(predicted_price_log)
                
                # Formatar o preço para exibição
                preco_formatado = f"R$ {predicted_price[0]:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

                # Exibir o resultado com destaque
                st.metric(label="Preço Estimado do Imóvel", value=preco_formatado)
                
                # Mostrar os dados que foram usados para a previsão (opcional)
                with st.expander("Ver detalhes dos dados inseridos"):
                    st.dataframe(input_df)

            except Exception as e:
                st.error(f"Ocorreu um erro ao fazer a predição: {e}")
                st.warning("Verifique se o nome da cidade está correto e se foi um dos nomes usados no treinamento do modelo.")
        else:
            st.warning("Por favor, insira o nome da cidade.")
    else:
        st.error("O arquivo do modelo ('modelo_previsao_preco_v1.pkl') não foi encontrado.")
else:
    st.info("Aguardando o preenchimento dos dados na barra lateral para fazer a estimativa.")

    
