# Predicción del abandono de clientes de Netflix

Proyecto de minería de datos para estimar si un cliente permanece o abandona el servicio y orientar campañas de retención. La aplicación Streamlit utiliza el pipeline previamente entrenado en `models/mejor_modelo_netflix.joblib`; no realiza entrenamiento.

## Estructura

```text
NetflixChurn/
├── data/netflix_customer_churn.csv
├── models/mejor_modelo_netflix.joblib
├── notebooks/Netflix_Customer_Churn_Analisis_Completo.ipynb
├── webapp/app.py
├── api/           # Carpeta reservada para una API
├── images/        # Recursos visuales
├── reports/       # Informes
├── requirements.txt
├── .gitignore
└── README.md
```

## Datos y modelos comparados

El cuaderno analiza 5.000 clientes y 14 columnas. Excluye `customer_id` de los predictores y utiliza `churned` como objetivo: **0 = permanece**, **1 = abandona**. La distribución registrada es 2.485 clientes que permanecen (49,7 %) y 2.515 que abandonan (50,3 %).

Se comparan regresión logística, árbol de decisión y Random Forest. El pipeline integra imputación y estandarización de variables numéricas, además de imputación y codificación one-hot de categorías. Se utiliza una partición estratificada de 80 % para entrenamiento y 20 % para prueba, con `random_state=42`, y validación cruzada de cinco particiones sobre entrenamiento. La selección final del cuaderno se realiza por F1 en prueba; también se evalúan exactitud, precisión, sensibilidad y ROC-AUC.

## Resultados principales

Las salidas guardadas del cuaderno seleccionan **Random Forest** y reportan, sobre 1.000 registros de prueba:

| Métrica | Resultado |
| --- | ---: |
| Exactitud | 97,70 % |
| Precisión de abandono | 0,9762 |
| Sensibilidad de abandono | 0,9781 |
| F1 de abandono | 0,9772 |

Las variables con mayor importancia registrada son `avg_watch_time_per_day` (0,404703), `watch_hours` (0,211182) y `last_login_days` (0,184796). Las cifras corresponden a la evaluación guardada del cuaderno, no a una nueva evaluación del archivo joblib. No se reproducen cifras de la comparación completa ni de ROC-AUC porque no están disponibles como texto en sus salidas guardadas.

## Instalación y ejecución

En un entorno que permita cargar las librerías de Python, desde la raíz del proyecto:

```powershell
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m streamlit run webapp/app.py
```

Abre la URL local que indique Streamlit, normalmente `http://localhost:8501`. El modelo se encuentra mediante `Path(__file__).resolve().parents[1]`, por lo que su carga no depende del directorio de trabajo.

Los rangos de `requirements.txt` son dependencias generales, no un registro del entorno de entrenamiento. Para cargar un modelo serializado con scikit-learn se deben usar las mismas versiones del entorno donde se guardó si aparecen incompatibilidades. El cuaderno también requiere matplotlib y seaborn, incluidos en los requisitos.

## Uso de la aplicación

Completa el formulario con las variables, en este orden exacto:

```text
age, gender, subscription_type, watch_hours, last_login_days,
region, device, monthly_fee, payment_method, number_of_profiles,
avg_watch_time_per_day, favorite_genre
```

Las categorías disponibles son:

| Variable | Opciones |
| --- | --- |
| gender | Female, Male, Other |
| subscription_type | Basic, Premium, Standard |
| region | Africa, Asia, Europe, North America, Oceania, South America |
| device | Desktop, Laptop, Mobile, TV, Tablet |
| payment_method | Credit Card, Crypto, Debit Card, Gift Card, PayPal |
| favorite_genre | Action, Comedy, Documentary, Drama, Horror, Romance, Sci-Fi |

La edad se introduce en años, `last_login_days` en días, `watch_hours` en horas del período observado y `avg_watch_time_per_day` en horas diarias. Utiliza para `monthly_fee` la misma moneda del conjunto de datos. El promedio diario se introduce independientemente porque el cuaderno no define una conversión fija desde las horas totales.

Al pulsar **Realizar predicción**, se construye un DataFrame de una fila con esos nombres y orden. El pipeline aplica su preprocesamiento y ejecuta `predict()` y `predict_proba()`. La probabilidad de abandono se toma de la columna correspondiente a la clase 1 según `classes_`.

Se muestran la predicción, el porcentaje de abandono, una barra de progreso, el riesgo y una recomendación empresarial:

| Riesgo | Probabilidad de abandono | Acción sugerida |
| --- | --- | --- |
| Bajo | Menor que 30 % | Fidelización y contenido personalizado |
| Medio | Desde 30 % hasta menos de 60 % | Campaña de reenganche y revisión de dificultades |
| Alto | Desde 60 % | Contacto prioritario y evaluación de una oferta de retención |

Estos umbrales son criterios empresariales orientativos, no umbrales optimizados en el cuaderno. La clase predicha depende del modelo y puede diferir del nivel de riesgo. Si falta el modelo, no puede cargarse o falla la inferencia, la interfaz muestra un mensaje comprensible.

## Validación local

Debido al bloqueo de librerías de Windows Application Control, la validación local se limita a analizar la sintaxis con la biblioteca estándar, sin ejecutar Streamlit ni importar pandas o scikit-learn:

```powershell
python -c "import ast; from pathlib import Path; ast.parse(Path('webapp/app.py').read_text(encoding='utf-8')); print('Sintaxis correcta')"
```

La carga del modelo y el comportamiento visual deben comprobarse en un entorno que permita ejecutar las dependencias. El CSV, el modelo y el cuaderno se conservan y no se excluyen mediante `.gitignore`.
