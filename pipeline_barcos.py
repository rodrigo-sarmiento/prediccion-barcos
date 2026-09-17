import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. CARGA Y LIMPIEZA DE DATOS
# ==========================================
# Cargar el dataset
df = pd.read_csv('Boats_Cleaned_dataset.csv')

# Eliminar columna técnica de índice si existiera
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Filtrar outliers en el target ('price') usando percentiles (1% y 99%)
q_low = df['price'].quantile(0.01)
q_high = df['price'].quantile(0.99)
df_clean = df[(df['price'] >= q_low) & (df['price'] <= q_high)].copy()

# ==========================================
# 2. INGENIERÍA DE CARACTERÍSTICAS
# ==========================================
CURRENT_YEAR = 2026
df_clean['boat_age'] = CURRENT_YEAR - df_clean['year']
df_clean['engine_age_diff'] = df_clean['maxEngineYear'].apply(
    lambda x: CURRENT_YEAR - x if pd.notnull(x) else np.nan
)

# Definición de variables
target = 'price'
drop_cols = ['id', 'sellerId', 'zip', 'created_date', 'created_month', 'created_year', target]

X = df_clean.drop(columns=drop_cols)
y = df_clean[target]

# Grupos de columnas por tipo y cardinalidad
num_cols = ['year', 'length_ft', 'beam_ft', 'dryWeight_lb', 'numEngines', 
            'totalHP', 'maxEngineYear', 'minEngineYear', 'boat_age', 'engine_age_diff']

low_card_cat = ['type', 'condition', 'hullMaterial', 'fuelType', 'engineCategory']
high_card_cat = ['boatClass', 'make', 'model', 'city', 'state']

# ==========================================
# 3. DIVISIÓN DE DATOS (TRAIN / TEST)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ==========================================
# 4. CONSTRUCCIÓN DEL PIPELINE DE PREPROCESAMIENTO
# ==========================================
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

low_card_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

high_card_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
    ('target_enc', TargetEncoder(smooth="auto", cv=5))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('low_cat', low_card_transformer, low_card_cat),
    ('high_cat', high_card_transformer, high_card_cat)
])

# ==========================================
# 5. ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
# ==========================================

# A) Modelo Baseline: Random Forest
baseline_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
baseline_pipeline.fit(X_train, y_train)

# B) Modelo Avanzado: HistGradientBoostingRegressor
advanced_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# C) Optimización básica de Hiperparámetros con GridSearchCV
param_grid = {
    'regressor__max_iter': [100, 200],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [5, 10]
}

grid_search = GridSearchCV(
    advanced_pipeline, 
    param_grid, 
    cv=3, 
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# ==========================================
# 6. EVALUACIÓN DEL MODELO
# ==========================================
def calcular_metricas(y_true, y_pred, conjunto=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- Métricas ({conjunto}) ---")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"R²:   {r2:.4f}\n")

print("=== EVALUACIÓN MODELO OPTIMIZADO ===")
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

calcular_metricas(y_train, y_pred_train, "Entrenamiento")
calcular_metricas(y_test, y_pred_test, "Prueba (Test)")

# Visualización gráfica
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Real vs Predicho
axes[0].scatter(y_test, y_pred_test, alpha=0.3, color='b')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Precio Real ($)')
axes[0].set_ylabel('Precio Predicho ($)')
axes[0].set_title('Valores Reales vs. Predichos (Test)')

# Gráfico 2: Histograma de Residuos
residuos = y_test - y_pred_test
sns.histplot(residuos, kde=True, ax=axes[1], color='purple')
axes[1].set_xlabel('Error de Predicción (Residuo $)')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Distribución de Residuos')

plt.tight_layout()
plt.show()

# ==========================================
# 7. FUNCIÓN DE INFERENCIA
# ==========================================
def predecir_precio_barco(datos_nuevo_barco_dict, modelo_entrenado):
    """
    Toma un diccionario con los datos de un barco y devuelve el precio estimado.
    """
    df_nuevo = pd.DataFrame([datos_nuevo_barco_dict])
    
    # Aplicar transformaciones de características derivadas
    df_nuevo['boat_age'] = CURRENT_YEAR - df_nuevo['year']
    if 'maxEngineYear' in df_nuevo.columns and pd.notnull(df_nuevo['maxEngineYear'].iloc[0]):
        df_nuevo['engine_age_diff'] = CURRENT_YEAR - df_nuevo['maxEngineYear']
    else:
        df_nuevo['engine_age_diff'] = np.nan

    precio_estimado = modelo_entrenado.predict(df_nuevo)[0]
    return precio_estimado

# Ejemplo de uso de la función de inferencia
nuevo_barco = {
    'type': 'power',
    'boatClass': 'power-center',
    'make': 'Aquasport',
    'model': '210 CC',
    'year': 2020,
    'condition': 'used',
    'length_ft': 25.0,
    'beam_ft': 8.5,
    'dryWeight_lb': 3500.0,
    'hullMaterial': 'fiberglass',
    'fuelType': 'gasoline',
    'numEngines': 1,
    'totalHP': 200.0,
    'maxEngineYear': 2020.0,
    'minEngineYear': 2020.0,
    'engineCategory': 'outboard-4s',
    'city': 'Miami',
    'state': 'FL'
}

precio_predicho = predecir_precio_barco(nuevo_barco, best_model)
print(f"El precio estimado para el barco ingresado es: ${precio_predicho:,.2f}")