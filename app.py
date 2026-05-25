import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN
# =========================

st.set_page_config(
    page_title="Titanic Dashboard",
    layout="wide"
)

# =========================
# CARGA DE DATOS
# =========================

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

df = pd.read_csv(url)

# =========================
# CARGAR MODELO
# =========================

model = pickle.load(
    open('modelo_titanic.pkl', 'rb')
)

scaler = pickle.load(
    open('scaler.pkl', 'rb')
)

# =========================
# TÍTULO
# =========================

st.title("🚢 Dashboard de Seguridad en Cruceros Turísticos")

st.markdown("""
Aplicación de Machine Learning para predicción de supervivencia
de pasajeros durante emergencias marítimas.
""")

# =========================
# KPIs
# =========================

st.subheader("📊 Indicadores Generales")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Pasajeros",
        len(df)
    )

with col2:
    st.metric(
        "Supervivientes",
        df['Survived'].sum()
    )

with col3:
    st.metric(
        "Tasa Supervivencia",
        f"{df['Survived'].mean()*100:.2f}%"
    )

# =========================
# GRÁFICOS
# =========================

st.subheader("📈 Análisis Exploratorio")

col1, col2 = st.columns(2)

# -------------------------
# Supervivencia
# -------------------------

with col1:

    fig, ax = plt.subplots()

    sns.countplot(
        x='Survived',
        data=df,
        ax=ax
    )

    ax.set_title(
        "Distribución de Supervivencia"
    )

    st.pyplot(fig)

# -------------------------
# Sexo
# -------------------------

with col2:

    fig, ax = plt.subplots()

    sns.countplot(
        x='Sex',
        hue='Survived',
        data=df,
        ax=ax
    )

    ax.set_title(
        "Supervivencia según Sexo"
    )

    st.pyplot(fig)

# =========================
# SEGUNDA FILA
# =========================

col3, col4 = st.columns(2)

# -------------------------
# Clase
# -------------------------

with col3:

    fig, ax = plt.subplots()

    sns.countplot(
        x='Pclass',
        hue='Survived',
        data=df,
        ax=ax
    )

    ax.set_title(
        "Supervivencia según Clase"
    )

    st.pyplot(fig)

# -------------------------
# Edad
# -------------------------

with col4:

    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x='Age',
        hue='Survived',
        bins=30,
        ax=ax
    )

    ax.set_title(
        "Distribución de Edad"
    )

    st.pyplot(fig)

# =========================
# PREDICCIÓN
# =========================

st.subheader("🤖 Predicción de Supervivencia")

col1, col2 = st.columns(2)

with col1:

    pclass = st.selectbox(
        "Clase",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Sexo",
        ["female", "male"]
    )

    age = st.slider(
        "Edad",
        1,
        80,
        30
    )

with col2:

    sibsp = st.slider(
        "SibSp",
        0,
        8,
        0
    )

    parch = st.slider(
        "Parch",
        0,
        6,
        0
    )

    fare = st.slider(
        "Fare",
        0,
        500,
        50
    )

# =========================
# PREPROCESAMIENTO
# =========================

sex_male = 1 if sex == "male" else 0

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Sex_male': [sex_male]
})

input_data[['Age', 'Fare']] = scaler.transform(
    input_data[['Age', 'Fare']]
)

# =========================
# BOTÓN
# =========================

if st.button("Predecir Supervivencia"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    if prediction[0] == 1:

        st.success(
            f"""
            ✅ Alta probabilidad de supervivencia

            Probabilidad:
            {probability[0][1]*100:.2f}%
            """
        )

    else:

        st.error(
            f"""
            ❌ Baja probabilidad de supervivencia

            Probabilidad:
            {probability[0][0]*100:.2f}%
            """
        )

# =========================
# MÉTRICAS
# =========================

st.subheader("📌 Métricas del Modelo")

st.write("""
- Accuracy ≥ 0.75
- Recall ≥ 0.80
- F1-Score ≥ 0.75
""")

# =========================
# CONCLUSIONES
# =========================

st.subheader("🚢 Conclusiones")

st.write("""
La Regresión Logística presentó el mejor desempeño
para la predicción de supervivencia de pasajeros.

Las variables más influyentes fueron:
- Sexo
- Clase
- Edad
- Tarifa
""")