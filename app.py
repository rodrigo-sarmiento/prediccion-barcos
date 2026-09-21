import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Configuración de la página
st.set_page_config(page_title="Tasador de Embarcaciones", page_icon="⛵")
st.title("⛵ Calculadora de Precios de Embarcaciones")
st.write("Ingresá las características del barco para estimar su valor de mercado.")

# 2. Carga del modelo (usamos caché para que no lo cargue a cada rato)
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_barcos.joblib')

try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontró el archivo 'modelo_barcos.joblib'. Por favor, ejecutá primero tu pipeline de entrenamiento.")
    st.stop()

# 3. Interfaz de usuario: Entradas de datos agrupadas en columnas
st.subheader("Características de la Embarcación")

col1, col2, col3 = st.columns(3)

with col1:
    make = st.text_input("Marca (Make)", value="Sea Ray")
    model = st.text_input("Modelo", value="Sundancer")
    year = st.number_input("Año de fabricación", min_value=1900, max_value=2026, value=2015)
    length_ft = st.number_input("Eslora (pies)", min_value=5.0, max_value=200.0, value=30.0)
    beam_ft = st.number_input("Manga (pies)", min_value=2.0, max_value=50.0, value=9.5)
    dryWeight_lb = st.number_input("Peso en seco (lbs)", min_value=100.0, value=8000.0)

with col2:
    boatClass = st.selectbox("Clase", ["power-cruiser", "power-center", "sail-cruiser", "Otro"])
    type_boat = st.selectbox("Tipo", ["power", "sail"])
    condition = st.selectbox("Condición", ["used", "new"])
    hullMaterial = st.selectbox("Material del Casco", ["fiberglass", "aluminum", "wood"])
    city = st.text_input("Ciudad", value="Miami")
    state = st.text_input("Estado", value="FL")

with col3:
    numEngines = st.number_input("Cantidad de Motores", min_value=0, max_value=4, value=1)
    totalHP = st.number_input("Potencia Total (HP)", min_value=0.0, value=300.0)
    engineCategory = st.selectbox("Categoría de Motor", ["inboard", "outboard-4s", "outboard-2s"])
    fuelType = st.selectbox("Combustible", ["gasoline", "diesel", "other"])
    minEngineYear = st.number_input("Año Min. Motor", min_value=1900, max_value=2026, value=2015)
    maxEngineYear = st.number_input("Año Max. Motor", min_value=1900, max_value=2026, value=2015)

# 4. Botón de Inferencia
if st.button("Estimar Precio", type="primary"):
    # Empaquetamos los datos en un diccionario, igual que en tu script original
    datos_ingresados = {
        'type': type_boat, 'boatClass': boatClass, 'make': make, 'model': model,
        'year': year, 'condition': condition, 'length_ft': length_ft, 'beam_ft': beam_ft,
        'dryWeight_lb': dryWeight_lb, 'hullMaterial': hullMaterial, 'fuelType': fuelType,
        'numEngines': numEngines, 'totalHP': totalHP, 'maxEngineYear': maxEngineYear,
        'minEngineYear': minEngineYear, 'engineCategory': engineCategory,
        'city': city, 'state': state
    }
    
    # Creamos el DataFrame
    df_nuevo = pd.DataFrame([datos_ingresados])
    
    # Aplicamos la misma lógica de ingeniería de características que tenías en predecir_precio_barco
    CURRENT_YEAR = 2026
    df_nuevo['boat_age'] = CURRENT_YEAR - df_nuevo['year']
    df_nuevo['engine_age_diff'] = CURRENT_YEAR - df_nuevo['maxEngineYear']
    
    # Hacemos la predicción
    precio_estimado = modelo.predict(df_nuevo)[0]
    
    # Mostramos el resultado
    st.success(f"### Precio Estimado: USD {precio_estimado:,.2f}")
    st.info("Recordá que esto es una estimación del modelo basada en datos históricos, no una tasación profesional.")