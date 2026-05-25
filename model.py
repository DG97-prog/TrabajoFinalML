import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================
# CARGA DE DATOS
# =========================

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

df = pd.read_csv(url)

# =========================
# SELECCIÓN DE VARIABLES
# =========================

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

X = df[features].copy()

y = df['Survived']

# =========================
# LIMPIEZA
# =========================

X['Age'] = X['Age'].fillna(
    X['Age'].median()
)

# One Hot Encoding
X = pd.get_dummies(
    X,
    columns=['Sex'],
    drop_first=True
)

# =========================
# ESCALAMIENTO
# =========================

scaler = StandardScaler()

X[['Age', 'Fare']] = scaler.fit_transform(
    X[['Age', 'Fare']]
)

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODELO
# =========================

model = LogisticRegression()

model.fit(X_train, y_train)

# =========================
# GUARDAR MODELO
# =========================

pickle.dump(model, open('modelo_titanic.pkl', 'wb'))

pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Modelo guardado correctamente")