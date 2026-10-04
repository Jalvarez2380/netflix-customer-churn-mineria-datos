"""Interfaz de inferencia del pipeline de abandono de clientes de Netflix."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mejor_modelo_netflix.joblib"
FEATURES = [
    "age", "gender", "subscription_type", "watch_hours", "last_login_days",
    "region", "device", "monthly_fee", "payment_method", "number_of_profiles",
    "avg_watch_time_per_day", "favorite_genre",
]

st.set_page_config(page_title="Netflix | Predicción de abandono", page_icon="🎬", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #0b0b0b; color: #ffffff; }
    [data-testid="stHeader"] { background: #0b0b0b; }
    .block-container { max-width: 1150px; padding-top: 2.5rem; }
    h1, h2, h3, p, label, [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] { color: #ffffff !important; }
    .brand { color: #e50914; font-weight: 900; letter-spacing: .2em; }
    [data-testid="stForm"] { background: #171717; border: 1px solid #333;
        border-top: 4px solid #e50914; border-radius: 12px; padding: 24px; }
    [data-testid="stNumberInputContainer"], [data-baseweb="select"] > div {
        background: #252525 !important; color: white !important; }
    input { color: white !important; }
    [data-testid="stFormSubmitButton"] button {
        background: #e50914; color: white; border: 0; font-weight: 700; }
    [data-testid="stFormSubmitButton"] button:hover { background: #b20710; color: white; }
    [data-testid="stProgress"] [role="progressbar"] > div > div { background: #e50914; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="brand">NETFLIX · CUSTOMER INSIGHTS</div>', unsafe_allow_html=True)
st.title("Predicción del abandono de clientes de Netflix")
st.write("Esta aplicación estima la probabilidad de que un cliente abandone el servicio para ayudar a priorizar acciones de retención.")


@st.cache_resource
def cargar_modelo(ruta: str, modificacion: int):
    """Carga el modelo una vez; invalida la caché si cambia el archivo."""
    return joblib.load(ruta)


try:
    modelo = cargar_modelo(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except FileNotFoundError:
    st.error("No se encontró el modelo. Coloca mejor_modelo_netflix.joblib en la carpeta models del proyecto y vuelve a abrir la aplicación.")
    st.stop()
except Exception:
    st.error("No se pudo cargar el modelo. Comprueba que el archivo sea válido y que las dependencias sean compatibles con el entorno donde se guardó.")
    st.stop()

with st.form("cliente"):
    st.subheader("Datos del cliente")
    st.caption("Introduce las horas según el mismo período de observación usado en los datos del modelo. La tarifa mensual se expresa en la moneda del conjunto de datos.")
    izquierda, derecha = st.columns(2)
    with izquierda:
        age = st.number_input("age", min_value=18, max_value=100, value=35, step=1, help="Edad en años.")
        gender = st.selectbox("gender", ["Female", "Male", "Other"])
        subscription_type = st.selectbox("subscription_type", ["Basic", "Premium", "Standard"])
        watch_hours = st.number_input("watch_hours", min_value=0.0, value=15.0, step=0.5, help="Horas totales de visualización.")
        last_login_days = st.number_input("last_login_days", min_value=0, value=7, step=1, help="Días desde el último inicio de sesión.")
        region = st.selectbox("region", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
    with derecha:
        device = st.selectbox("device", ["Desktop", "Laptop", "Mobile", "TV", "Tablet"])
        monthly_fee = st.number_input("monthly_fee", min_value=0.0, value=13.99, step=0.01, format="%.2f")
        payment_method = st.selectbox("payment_method", ["Credit Card", "Crypto", "Debit Card", "Gift Card", "PayPal"])
        number_of_profiles = st.number_input("number_of_profiles", min_value=1, max_value=5, value=2, step=1)
        avg_watch_time_per_day = st.number_input("avg_watch_time_per_day", min_value=0.0, value=1.5, step=0.1, help="Promedio diario de horas de visualización.")
        favorite_genre = st.selectbox("favorite_genre", ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"])
    enviado = st.form_submit_button("Realizar predicción", use_container_width=True)

if enviado:
    datos = pd.DataFrame([[
        age, gender, subscription_type, watch_hours, last_login_days,
        region, device, monthly_fee, payment_method, number_of_profiles,
        avg_watch_time_per_day, favorite_genre,
    ]], columns=FEATURES)
    try:
        with st.spinner("Analizando el perfil del cliente…"):
            prediccion = modelo.predict(datos)[0]
            probabilidades = modelo.predict_proba(datos)
            clases = list(modelo.classes_)
            if prediccion not in (0, 1) or 1 not in clases:
                raise ValueError("El modelo debe usar 0 = permanece y 1 = abandona.")
            probabilidad = float(probabilidades[0, clases.index(1)])
            if not np.isfinite(probabilidad) or not 0.0 <= probabilidad <= 1.0:
                raise ValueError("Probabilidad inválida.")
    except Exception:
        st.error("No se pudo realizar la predicción. Verifica que el modelo acepte las 12 variables del formulario, las clases 0 y 1 y las versiones de las dependencias instaladas.")
    else:
        if probabilidad < 0.30:
            riesgo = "Bajo"
            recomendacion = "Mantener la fidelización con recomendaciones de contenido personalizadas y seguimiento periódico de la actividad."
        elif probabilidad < 0.60:
            riesgo = "Medio"
            recomendacion = "Activar una campaña de reenganche, destacar novedades afines a sus gustos y revisar posibles dificultades de uso o pago."
        else:
            riesgo = "Alto"
            recomendacion = "Priorizar una acción de retención: contactar al cliente, investigar los motivos de insatisfacción y evaluar una oferta o un plan adecuado."
        st.subheader("Resultado del análisis")
        columna_prediccion, columna_probabilidad, columna_riesgo = st.columns(3)
        columna_prediccion.metric("Predicción", "Abandona" if prediccion == 1 else "Permanece")
        columna_probabilidad.metric("Probabilidad de abandono", f"{probabilidad:.2%}")
        columna_riesgo.metric("Nivel de riesgo", riesgo)
        st.progress(probabilidad, text=f"Probabilidad de abandono: {probabilidad:.2%}")
        st.info(f"Recomendación empresarial: {recomendacion}")
        st.caption("Riesgo bajo: < 30 % · Medio: 30 % a < 60 % · Alto: ≥ 60 %. Los niveles orientan la prioridad de retención; la predicción de clase la determina el modelo.")

st.caption("Herramienta de apoyo a la retención de clientes · Modelo previamente entrenado")
