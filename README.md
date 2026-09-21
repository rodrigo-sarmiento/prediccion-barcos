# **⛵ Predicción de Precios de Embarcaciones**

### **Guía Pedagógica y Documentación de Proyecto**

**Carrera:** Ciencia de Datos e Inteligencia Artificial — **ISFDyT N.° 57 (Chascomús)**

## ---

**📌 1\. Introducción y Objetivo del Proyecto**

Modelo de estudio para la carrera de Ciencia de Datos e Inteligencia Artificial del Instituto Superior de Formación Docente y Tecnica N.° 57 de la ciudad de Chascomús.

Este modelo aborda un problema clásico de Aprendizaje Automático Supervisado en la categoría de **Regresión**: estimar el valor comercial continuo (en dólares) de una embarcación a partir de sus atributos técnicos (eslora, potencia, tipo de casco), comerciales (estado, marca, año) y geográficos (ciudad, estado).  
En lugar de definir reglas manuales fijas para tasar un barco, construimos un **Pipeline End-to-End** que:

> 1. Aprende patrones matemáticos a partir de un histórico de datos real.  
> 2. Evalúa objetivamente su margen de error.  
> 3. Se guarda en disco (**persistencia**) para hacer predicciones en tiempo real mediante una interfaz web interactiva (**inferencia**).

## **📂 2\. Estructura del Proyecto**

`prediccion_barcos/`  
`├── Boats_Cleaned_dataset.csv   # Dataset histórico con registros de embarcaciones`  
`├── pipeline_barcos.py          # Script principal: Limpieza, entrenamiento, evaluación y guardado`  
`├── modelo_barcos.joblib        # Archivo binario con el modelo entrenado y listo para usar`  
`├── app.py                      # Aplicación web interactiva (Streamlit) para consultas de usuarios`  
`└── README.md                   # Documentación didáctica del proyecto`

## **🧠 3\. Flujo del Pipeline Explicado Paso a Paso**

### **Paso 1: Carga y Limpieza de Datos**

> * **Eliminación de índices redundantes:** Se descartan columnas como Unnamed: 0 que no aportan valor predictivo.  
> * **Filtrado de valores extremos (Outliers):** Se acotan los precios entre el percentil 1 (q0.01) y el percentil 99 (q0.99).  
>   *¿Por qué hacemos esto?* Barcos con precio $0 (errores de carga) o yates de super lujo distorsionan la escala y confunden al algoritmo durante el aprendizaje.

### **Paso 2: Ingeniería de Características (Feature Engineering)**

Transformamos datos existentes para crear señales con mayor poder explicativo:

> * **boat\_age (Antigüedad del barco):** 2026 \- año de fabricación.  
> * **engine\_age\_diff (Diferencia de antigüedad del motor):** 2026 \- año del motor.

*Explicación pedagógica:* Para un modelo numérico es mucho más fácil interpretar "10 años de uso" que procesar el año "2016" como un número arbitrario.

### **Paso 3: División de Datos (Train / Test Split)**

> * **80% Entrenamiento (X\_train, y\_train):** El "material de estudio" donde el modelo aprende las relaciones.  
> * **20% Prueba (X\_test, y\_test):** El "examen final" con datos nunca antes vistos para medir el rendimiento real.

### **Paso 4: Preprocesamiento con ColumnTransformer**

> 1. **Variables Numéricas:**  
   * Imputación por Mediana.  
   * Estandarización (Z-score).  
> 2. **Categorías de Baja Cardinalidad:** One-Hot Encoding.  
> 3. **Categorías de Alta Cardinalidad:** Target Encoding.

### **Paso 5: Entrenamiento y Optimización (GridSearchCV)**

> * **Baseline (Referencia):** RandomForestRegressor.  
> * **Modelo Avanzado:** HistGradientBoostingRegressor.  
> * **Búsqueda en Malla (GridSearchCV):** Evalúa combinaciones de hiperparámetros con Validación Cruzada (k=3).

### **Paso 6: Evaluación Gráfica**

> * **Valores Reales vs. Predichos:** Muestra qué tan alineadas están las predicciones.  
> * **Distribución de Residuos:** Visualiza los errores de predicción.

### **Paso 7: Persistencia del Modelo (Guardado)**

Mediante la librería joblib, guardamos en modelo\_barcos.joblib todo el objeto entrenado.

### **Paso 8: Inferencia en Tiempo Real (Interfaz Web con Streamlit)**

La aplicación app.py carga modelo\_barcos.joblib y expone un formulario visual para hacer predicciones al instante.

## **📐 4\. Acotación de Fórmulas Matemáticas Clave**

### **1\. Error Absoluto Medio (MAE)**

**Fórmula:** MAE \= (1 / n) \* Σ |yi \- ŷi|  
Mide el promedio simple del margen de error en dólares (USD).

### **2\. Raíz del Error Cuadrático Medio (RMSE)**

**Fórmula:** RMSE \= √\[ (1 / n) \* Σ (yi \- ŷi)² \]  
Penaliza fuertemente los grandes errores al elevarlos al cuadrado.

### **3\. Coeficiente de Determinación (R²)**

**Fórmula:** R² \= 1 \- \[ Σ (yi \- ŷi)² / Σ (yi \- ȳ)² \]  
Indica la proporción de la variabilidad del precio explicada por el modelo.

### **4\. Estandarización de Variables (Z-Score)**

**Fórmula:** z \= (x \- μ) / σ  
Transforma los datos para que tengan media 0 y desviación estándar 1\.

## **🛠️ 5\. Guía de Instalación y Ejecución**

`# 1. Crear y activar entorno virtual`  
`python -m venv venv`  
`.\venv\Scripts\Activate.ps1  # Windows`

`# 2. Instalar librerías`  
`pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib`

`# 3. Entrenar y guardar el modelo`  
`python pipeline_barcos.py`

`# 4. Lanzar la aplicación interactiva`  
`streamlit run app.py`

## **🚀 6\. Próximos Pasos Sugeridos para Estudiantes**

> * Parametrizar el año de referencia dinámicamente.  
> * Analizar la importancia relativa de variables (Feature Importance).  
> * Desplegar la aplicación web en Streamlit Community Cloud.