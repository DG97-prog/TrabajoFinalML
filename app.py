import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================
# CONFIGURACIÓN GENERAL
# ============================================

st.set_page_config(
    page_title="Dashboard Titanic",
    page_icon="🚢",
    layout="wide"
)

# ============================================
# ESTILO PERSONALIZADO
# ============================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
    color: white;
}

.css-1d391kg {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# CARGA DE DATOS
# ============================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

# ============================================
# PREPROCESAMIENTO PARA KMEANS
# ============================================

df_kmeans = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].copy()

df_kmeans['Age'] = df_kmeans['Age'].fillna(
    df_kmeans['Age'].median()
)

scaler_kmeans = StandardScaler()

X_scaled = scaler_kmeans.fit_transform(df_kmeans)

# ============================================
# KMEANS
# ============================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

# ============================================
# CARGAR MODELO Y SCALER
# ============================================

model = pickle.load(
    open('modelo_titanic.pkl', 'rb')
)

scaler = pickle.load(
    open('scaler.pkl', 'rb')
)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🚢 Titanic ML")

st.sidebar.info("""
Dashboard de Machine Learning desarrollado
con Streamlit para predecir supervivencia
de pasajeros durante emergencias marítimas.
""")

st.sidebar.success("""
Modelos utilizados:
- Logistic Regression
- Decision Tree
""")

# ============================================
# TÍTULO
# ============================================

st.title("🚢 Dashboard de Seguridad en Cruceros Turísticos")

st.markdown("""
Aplicación de Machine Learning para predicción
de supervivencia de pasajeros durante
emergencias marítimas.
""")

# ============================================
# KPIs
# ============================================

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
        int(df['Survived'].sum())
    )

with col3:
    st.metric(
        "Tasa de Supervivencia",
        f"{df['Survived'].mean()*100:.2f}%"
    )

# ============================================
# PREPARACIÓN PARA TOP 10
# ============================================

df_top = df.copy()

# Limpiar nulos
df_top['Age'] = df_top['Age'].fillna(
    df_top['Age'].median()
)

# Crear variable Sex_male
df_top['Sex_male'] = df_top['Sex'].map({
    'male': 1,
    'female': 0
})

# Variables del modelo
features = [
    'Pclass',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Sex_male'
]

# Guardar valores originales
df_top['Age_Original'] = df_top['Age']
df_top['Fare_Original'] = df_top['Fare']

# Guardar valores originales
df_top['Age_Original'] = df_top['Age']
df_top['Fare_Original'] = df_top['Fare']

# Escalar variables
df_top[['Age', 'Fare']] = scaler.transform(
    df_top[['Age', 'Fare']]
)

# Predicción de probabilidades
df_top['Probabilidad'] = model.predict_proba(
    df_top[features]
)[:,1]

# ============================================
# TABS
# ============================================

tab1, tab2 = st.tabs([
    "🚢 Predicción",
    "📊 Análisis Exploratorio"
])

# ============================================
# TAB 1 - PREDICCIÓN
# ============================================

with tab1:

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
            "Tarifa",
            0,
            500,
            50
        )

    # ============================================
    # PREPROCESAMIENTO
    # ============================================

    sex_male = 1 if sex == "male" else 0

    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_male': [sex_male]
    })

    # Escalar variables numéricas

    input_data[['Age', 'Fare']] = scaler.transform(
        input_data[['Age', 'Fare']]
    )

    # ============================================
    # BOTÓN DE PREDICCIÓN
    # ============================================

    if st.button("Predecir Supervivencia"):

        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)[0][1]

        # ============================================
        # RESULTADO
        # ============================================

        st.subheader("📌 Resultado")

        if prediction[0] == 1:

            st.success(
                f"""
                ✅ Alta probabilidad de supervivencia

                Probabilidad:
                {probability*100:.2f}%
                """
            )

        else:

            st.error(
                f"""
                ❌ Baja probabilidad de supervivencia

                Probabilidad:
                {(1-probability)*100:.2f}%
                """
            )

        # ============================================
        # MÉTRICA
        # ============================================

        st.metric(
            "Probabilidad de Supervivencia",
            f"{probability*100:.2f}%"
        )

        # ============================================
        # GAUGE CHART
        # ============================================

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Nivel de Supervivencia"},
            gauge={
                'axis': {'range': [0, 100]}
            }
        ))

        st.plotly_chart(
            fig_gauge,
            use_container_width=True
        )

        # ============================================
        # ALERTAS
        # ============================================

        if probability >= 0.75:

            st.success("🟢 Riesgo Bajo")

        elif probability >= 0.50:

            st.warning("🟡 Riesgo Moderado")

        else:

            st.error("🔴 Riesgo Alto")

    # ============================================
    # MÉTRICAS DEL MODELO
    # ============================================

    st.subheader("📌 Métricas del Modelo")

    st.write("""
    - Accuracy ≥ 0.75
    - Recall ≥ 0.80
    - F1-Score ≥ 0.75
    """)

    # ============================================
    # CONCLUSIONES
    # ============================================

    st.subheader("🚢 Conclusiones")

    st.write("""
    La Regresión Logística presentó el mejor desempeño
    para la predicción de supervivencia de pasajeros.

    Variables más influyentes:
    - Sexo
    - Clase
    - Edad
    - Tarifa
    """)

# ============================================
# TAB 2 - ANÁLISIS EXPLORATORIO
# ============================================

with tab2:

    st.subheader("📊 Gráficas Exploratorias")

    st.markdown("## 👩 Mujeres con Mayor Probabilidad de Supervivencia")

    top_mujeres = df_top[
    df_top['Sex'] == 'female'
    ].sort_values(
        by='Probabilidad',
        ascending=False
    ).head(10)

    st.dataframe(
        top_mujeres[
            [
    'Name',
    'Age_Original',
    'Pclass',
    'Fare_Original',
    'Probabilidad'
    ]
        ]
    )

    st.markdown("## 👨 Hombres con Mayor Probabilidad de Supervivencia")

    top_hombres = df_top[
        df_top['Sex'] == 'male'
    ].sort_values(
        by='Probabilidad',
        ascending=False
    ).head(10)

    st.dataframe(
        top_hombres[
            [
    'Name',
    'Age_Original',
    'Pclass',
    'Fare_Original',
    'Probabilidad'
    ]
        ]
    )

    # ============================================
    # DISTRIBUCIÓN SUPERVIVENCIA
    # ============================================

    st.markdown("### Distribución de Supervivencia")

    fig1, ax1 = plt.subplots(figsize=(6,4))

    sns.countplot(
        x='Survived',
        data=df,
        ax=ax1
    )

    st.pyplot(fig1)

    # ============================================
    # SUPERVIVENCIA POR SEXO
    # ============================================

    st.markdown("### Supervivencia según Sexo")

    fig2, ax2 = plt.subplots(figsize=(6,4))

    sns.countplot(
        x='Sex',
        hue='Survived',
        data=df,
        ax=ax2
    )

    st.pyplot(fig2)

    # ============================================
    # SUPERVIVENCIA POR CLASE
    # ============================================

    st.markdown("### Supervivencia según Clase")

    fig3, ax3 = plt.subplots(figsize=(6,4))

    sns.countplot(
        x='Pclass',
        hue='Survived',
        data=df,
        ax=ax3
    )

    st.pyplot(fig3)

    # ============================================
    # HEATMAP
    # ============================================

    st.markdown("### Heatmap de Correlación")

    fig4, ax4 = plt.subplots(figsize=(10,6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm',
        ax=ax4
    )

    st.pyplot(fig4)

    # ============================================
    # CLUSTERS KMEANS
    # ============================================

    st.markdown("### Clústeres de Pasajeros")

    fig5, ax5 = plt.subplots(figsize=(8,5))

    sns.scatterplot(
        x=df['Age'],
        y=df['Fare'],
        hue=df['Cluster'],
        palette='viridis',
        ax=ax5
    )

    ax5.set_xlabel("Edad")
    ax5.set_ylabel("Tarifa")

    st.pyplot(fig5)

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.caption("""
Proyecto CRISP-DM — Machine Learning aplicado
a seguridad marítima utilizando el dataset Titanic.
""")