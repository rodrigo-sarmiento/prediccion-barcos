# Prediccion de precios de embarcaciones

Modelo de estudio para la carrera de **Ciencia de Datos e Inteligencia Artificial** del **Instituto Superior de Formacion Docente y Tecnica N.° 57 de la ciudad de Chascomus**.

El proyecto aplica un pipeline de aprendizaje automatico para estimar el precio de una embarcacion a partir de sus caracteristicas tecnicas, comerciales y de ubicacion.

> Proyecto educativo y experimental. Las predicciones no deben interpretarse como una tasacion profesional ni como una recomendacion de compra o venta.

## Objetivos

- Practicar limpieza y preparacion de datos.
- Crear variables derivadas, como la antiguedad del barco y del motor.
- Comparar un modelo baseline con un modelo de gradient boosting.
- Optimizar hiperparametros mediante validacion cruzada.
- Evaluar el modelo con RMSE, MAE y R2.
- Realizar inferencias para nuevos registros.

## Estructura

```text
.
├── Boats_Cleaned_dataset.csv   # Dataset utilizado para entrenar y evaluar
├── pipeline_barcos.py          # Pipeline completo de entrenamiento e inferencia
└── README.md
```

## Flujo del pipeline

1. Carga `Boats_Cleaned_dataset.csv`.
2. Elimina una posible columna tecnica de indice.
3. Filtra valores extremos del precio entre los percentiles 1 y 99.
4. Genera `boat_age` y `engine_age_diff` usando 2026 como ano de referencia.
5. Separa los datos en entrenamiento y prueba con una proporcion 80/20.
6. Imputa y estandariza variables numericas.
7. Aplica one-hot encoding a variables categoricas de baja cardinalidad.
8. Aplica target encoding a variables categoricas de alta cardinalidad.
9. Entrena un Random Forest como baseline.
10. Busca la mejor configuracion de `HistGradientBoostingRegressor` con `GridSearchCV`.
11. Imprime las metricas y muestra graficos de valores reales contra predichos y residuos.
12. Expone `predecir_precio_barco` para estimar el precio de un nuevo barco.

## Requisitos

- Python 3.10 o superior recomendado.
- Dependencias:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Instalacion y ejecucion

Crear y activar un entorno virtual:

```bash
python -m venv venv
```

En Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

En Linux o macOS:

```bash
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Ejecutar el pipeline desde la raiz del proyecto:

```bash
python pipeline_barcos.py
```

El script imprime las metricas de entrenamiento y prueba, abre las visualizaciones y calcula una prediccion de ejemplo.

## Variables utilizadas

El objetivo es `price`. Entre las variables predictoras se incluyen:

- Tipo, clase, marca y modelo.
- Ano y condicion de la embarcacion.
- Eslora, manga y peso en seco.
- Cantidad de motores y potencia total.
- Ano minimo y maximo de motor.
- Material del casco y tipo de combustible.
- Ciudad y estado.

Por evitar fuga de informacion o variables administrativas, el pipeline descarta identificadores, codigo postal y campos derivados de la fecha de publicacion.

## Metricas

El modelo se evalua con:

- **RMSE**: penaliza especialmente los errores grandes.
- **MAE**: representa el error absoluto medio en unidades de precio.
- **R2**: indica la proporcion de variabilidad explicada por el modelo.

Los resultados pueden cambiar al modificar el dataset, la version de las librerias o los parametros del entrenamiento.

## Limitaciones y proximos pasos

- El ano de referencia esta fijado en 2026 y deberia parametrizarse para futuros usos.
- No se guarda un modelo entrenado; el entrenamiento se repite en cada ejecucion.
- Se puede agregar persistencia del modelo, validacion mas robusta, analisis de importancia de variables y una interfaz web o dashboard.
- Antes de usarlo fuera del contexto academico, conviene revisar sesgos, calidad de datos y representatividad del mercado.

## Autor y contexto academico

Trabajo practico desarrollado como modelo de estudio para la carrera de Ciencia de Datos e Inteligencia Artificial del Instituto 57 de Chascomus.
