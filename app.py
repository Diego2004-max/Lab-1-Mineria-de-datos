import streamlit as st
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Laboratorio Minería de Datos", page_icon="📊", layout="centered")

st.title("🔬 Laboratorio de Minería de Datos")
st.subheader("Modelos de Regresión Lineal Múltiple (CRISP-DM)")

# Menú lateral para seleccionar el ejercicio
opcion = st.sidebar.selectbox(
    "Seleccione el Escenario:",
    ("Precio del Dólar", "Nivel de Glucosa", "Consumo de Energía")
)

st.markdown("---")

# Cargar los modelos de forma segura
@st.cache_resource
def cargar_modelos():
    try:
        m_dolar = joblib.load('modelo_dolar.joblib')
        m_glucosa = joblib.load('modelo_glucosa.joblib')
        m_energia = joblib.load('modelo_energia.joblib')
        return m_dolar, m_glucosa, m_energia
    except Exception as e:
        return None, None, None

m_dolar, m_glucosa, m_energia = cargar_modelos()

if m_dolar is None:
    st.error(" No se encontraron los archivos `.joblib` de los modelos en la carpeta. Asegúrate de subirlos junto con `app.py`.")
else:
    if opcion == "Precio del Dólar":
        st.header(" Predicción del Precio del Dólar")
        st.write("Ingrese las variables macroeconómicas para estimar el precio del dólar.")
        
        dia = st.number_input("Número de Día (Dia)", min_value=1, max_value=1000, value=50)
        inflacion = st.number_input("Tasa de Inflación Diaria (Inflacion)", min_value=0.0, max_value=10.0, value=0.05, step=0.01)
        tasa_interes = st.number_input("Tasa de Interés Diaria (Tasa_interes)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)
        
        if st.button("Predecir Precio del Dólar"):
            entrada = np.array([[dia, inflacion, tasa_interes]])
            prediccion = m_dolar.predict(entrada)[0]
            st.success(f" **Precio del Dólar Estimado:** ${prediccion:,.2f}")

    elif opcion == "Nivel de Glucosa":
        st.header(" Predicción de Niveles de Glucosa en Sangre")
        st.write("Ingrese los datos del paciente para estimar su nivel de glucosa (mg/dL).")
        
        edad = st.number_input("Edad del Paciente", min_value=1, max_value=120, value=40)
        imc = st.number_input("Índice de Masa Corporal (IMC)", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
        actividad = st.number_input("Actividad Física (Horas semanales)", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
        
        if st.button("Predecir Nivel de Glucosa"):
            entrada = np.array([[edad, imc, actividad]])
            prediccion = m_glucosa.predict(entrada)[0]
            st.success(f" **Nivel de Glucosa Estimado:** {prediccion:.2f} mg/dL")

    elif opcion == "Consumo de Energía":
        st.header(" Predicción del Consumo de Energía Eléctrica")
        st.write("Ingrese las condiciones ambientales y temporales para estimar el consumo (kWh).")
        
        temperatura = st.number_input("Temperatura (°C)", min_value=-10.0, max_value=50.0, value=22.0, step=0.5)
        hora = st.slider("Hora del Día (1 a 24)", min_value=1, max_value=24, value=12)
        dia_semana = st.selectbox("Día de la Semana", options=[1, 2, 3, 4, 5, 6, 7], format_func=lambda x: ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][x-1])
        
        if st.button("Predecir Consumo de Energía"):
            entrada = np.array([[temperatura, hora, dia_semana]])
            prediccion = m_energia.predict(entrada)[0]
            st.success(f" **Consumo de Energía Estimado:** {prediccion:.2f} kWh")