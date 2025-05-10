Análisis Predictivo de Condición de Productos en una Tienda de Comercio Electrónico
Introducción
Este proyecto utiliza aprendizaje automático para predecir si un producto listado en una plataforma de comercio electrónico es nuevo o usado (condition). A través de un conjunto de datos con información detallada sobre los listings, se implementa un RandomForestClassifier dentro de un pipeline que incluye preprocesamiento, entrenamiento y evaluación, identificando las características clave que influyen en la clasificación.
Estructura del Repositorio

02_model_training.ipynb: Notebook con el código para el análisis exploratorio de datos (EDA), preprocesamiento, entrenamiento y evaluación de modelos.
README.md: Este archivo, con la descripción del proyecto, análisis, resultados y conclusiones.

Conjunto de Datos
El conjunto de datos contiene 53 columnas con información sobre listings en la tienda de comercio electrónico, incluyendo:

Variables numéricas: base_price, price, initial_quantity, sold_quantity, available_quantity, entre otras.
Variables categóricas: warranty, condition, category_id, listing_type_id, etc.
Variables booleanas: shipping_local_pick_up, has_discount, accepts_online_payment.

La variable objetivo es condition, que indica si el producto es "new" (nuevo) o "used" (usado).
Metodología
Análisis Exploratorio de Datos (EDA)
Se realizó un análisis exploratorio para identificar patrones, valores faltantes y correlaciones:

Se verificó la distribución de condition para evaluar el balance de clases.
Se analizaron valores faltantes en columnas clave (price, initial_quantity, warranty).
Se generaron visualizaciones (histogramas, diagramas de caja, matrices de correlación) para entender relaciones entre variables.
Se excluyeron columnas irrelevantes o redundantes.

Preprocesamiento
Se utilizó un ColumnTransformer para manejar diferentes tipos de variables:

Numéricas: Imputación con la mediana (SimpleImputer) y escalado con StandardScaler.
Categóricas: Imputación con "missing" (SimpleImputer) y codificación con OneHotEncoder.
Booleanas: Imputación con el valor más frecuente (SimpleImputer).

Modelos Evaluados
Se compararon varios modelos de clasificación para predecir condition:

Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Tree
Random Forest
Multi-Layer Perceptron (MLP)
XGBoost

Métricas de Evaluación
Los modelos se evaluaron con:

Accuracy: Proporción de predicciones correctas.
Precision: Proporción de predicciones positivas correctas.
Recall: Proporción de casos positivos identificados.
F1 Score: Media armónica de precisión y recall.

Resultados
Comparación de Modelos
La siguiente tabla resume el desempeño de los modelos:



Model
Accuracy
Precision
Recall
F1 Score



Logistic Regression
0.816000
0.817911
0.816000
0.816249


KNN
0.814889
0.819502
0.814889
0.815120


SVM
0.805944
0.811678
0.805944
0.806142


Decision Tree
0.803333
0.807832
0.803333
0.803582


Random Forest
0.824000
0.829866
0.824000
0.824177


MLP
0.813667
0.817757
0.813667
0.813914


XGBoost
0.810611
0.827057
0.810611
0.810065


Observación: Random Forest obtuvo el mejor desempeño, con un Accuracy de 0.824, Precision de 0.829, Recall de 0.824 y F1 Score de 0.824.
Importancia de las Características (Random Forest)
El modelo Random Forest identificó las características más influyentes para predecir la condición del producto:

initial_quantity (11.66%): Cantidad inicial de productos listados.
available_quantity (11.07%): Cantidad disponible para la venta.
price (8.33%): Precio del producto.
base_price (7.73%): Precio base del producto.
sold_quantity (4.87%): Cantidad vendida.
warranty_Sin garantía (2.37%): Ausencia de garantía.
date_created_month (1.18%): Mes de creación del listing.
last_updated_month (1.02%): Mes de última actualización.
shipping_local_pick_up (0.69%): Opción de recolección local.
category_id_1227 (0.56%): Categoría específica (posiblemente revistas o coleccionables).

Estas características destacan la importancia de la disponibilidad, el precio y la garantía en la clasificación.
Análisis

Disponibilidad y Precio: initial_quantity, available_quantity, price y base_price sugieren que los productos nuevos tienen mayor stock y precios más estables, mientras que los usados presentan menor disponibilidad o precios variables.
Garantía: La ausencia de garantía (warranty_Sin garantía) es relevante, ya que los productos usados suelen carecer de ella.
Factores Temporales: date_created_month y last_updated_month indican tendencias estacionales o diferencias en la antigüedad de los listings.
Categorías y Logística: category_id_1227 y shipping_local_pick_up tienen un impacto menor, posiblemente ligado a tipos específicos de productos.

Conclusión
El modelo Random Forest es el más adecuado para predecir si un producto es nuevo o usado, con un Accuracy de 0.824 y un F1 Score de 0.824, superando a modelos como Logistic Regression y XGBoost. Las características clave (initial_quantity, available_quantity, price, base_price, warranty_Sin garantía) proporcionan información valiosa para clasificar productos y optimizar listings.
Limitaciones

Desbalance de Clases: Si condition está desbalanceada, el modelo podría sesgarse hacia la clase mayoritaria. Requiere verificación con una matriz de confusión.
Sobreajuste: Random Forest puede sobreajustar sin optimización de hiperparámetros.
Características Redundantes: Características con baja importancia podrían eliminarse para simplificar el modelo.

Recomendaciones

Optimización: Ajustar hiperparámetros de Random Forest con GridSearchCV.
Evaluación Adicional: Usar ROC-AUC y matrices de confusión para clases desbalanceadas.
Selección de Características: Eliminar características de baja importancia.
Estrategias Comerciales: Ajustar precios según la condición, mejorar información de garantías y garantizar stock.
Validación Externa: Probar el modelo con nuevos datos para confirmar generalización.

Requisitos
Instala las dependencias:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

Instrucciones de Uso

Clona el repositorio:git clone <URL_DEL_REPOSITORIO>


Instala las dependencias:pip install -r requirements.txt


Abre el notebook 02_model_training.ipynb en Jupyter Notebook o JupyterLab.
Ejecuta las celdas para reproducir el análisis.

Autor

Luz Ángela Carabalí Mulato (@luzangelacarabli)

