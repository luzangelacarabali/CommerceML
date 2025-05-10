# 🛍️ Análisis Predictivo de Condición de Productos en una Tienda de Comercio Electrónico

## 📌 Introducción

Este proyecto aplica **aprendizaje automático** para predecir si un producto listado en una tienda de comercio electrónico es **nuevo** o **usado** (`condition`). A partir de un conjunto de datos con información detallada de los listings, se construye un `RandomForestClassifier` dentro de un pipeline con:

* Preprocesamiento de datos
* Entrenamiento y evaluación de modelos
* Identificación de variables clave

---

## 📂 Estructura del Repositorio

```bash
├── cuderno/
│   ├── 01_EDA_and_cleaning.ipynb     # Análisis exploratorio y limpieza de datos
│   ├── 02_model_training.ipynb       # Análisis exploratorio, preprocesamiento y modelado
├── model/                           # Carpeta para modelos entrenados
├── drive/                           # Carpeta para datos y modelos en Google Drive
│   ├── data/                        # Datos originales y procesados
│   ├── model/                       # Modelos entrenados en Google Drive
├── README.md                        # Descripción completa del proyecto
├── .gitignore                       # gitignore
```

---

## 💻 Drive

Los datos del proyecto están disponibles en Google Drive. Puedes acceder a ellos a través del siguiente enlace:

[**Accede al proyecto en Google Drive**](https://drive.google.com/drive/folders/1EO0Ey5X1gbV8skhZj6cEYFcPZkwrY3Nk?usp=sharing)

Dentro de Google Drive encontrarás:

* **Carpeta `data/`**: Contiene los archivos de datos originales y procesados.
* **Carpeta `model/`**: Contiene los modelos entrenados.

---

## 🧾 Conjunto de Datos

El dataset contiene **53 columnas** que describen cada publicación de producto. Algunos ejemplos:

* **Numéricas**: `base_price`, `price`, `initial_quantity`, `sold_quantity`
* **Categóricas**: `warranty`, `condition`, `category_id`, `listing_type_id`
* **Booleanas**: `shipping_local_pick_up`, `has_discount`, `accepts_online_payment`

> 🎯 Variable objetivo: `condition` (`new` o `used`)

Los archivos `model/datos_limpios.csv` y `datos/MLA_100k.jsonlines` fueron eliminados debido a restricciones de tamaño en GitHub (>100 MB). Los datos están disponibles en Google Drive para su uso y manejo consciente.

---

## 🔎 Metodología

### 1. Análisis Exploratorio de Datos (EDA)

* Evaluación del balance de clases (`condition`)
* Detección de valores faltantes y outliers
* Visualizaciones: histogramas, diagramas de caja, matriz de correlación

![image](https://github.com/user-attachments/assets/2a9faa28-c983-467d-93dd-4a3a7ac96b85)

---

### 2. Preprocesamiento

Uso de `ColumnTransformer` para tratar los distintos tipos de datos:

| Tipo        | Imputación            | Transformación   |
| ----------- | --------------------- | ---------------- |
| Numéricas   | Mediana               | `StandardScaler` |
| Categóricas | Relleno con "missing" | `OneHotEncoder`  |
| Booleanas   | Valor más frecuente   | Ninguna          |

---

### 3. Modelos Evaluados

Se entrenaron y compararon los siguientes algoritmos:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest ✅
* Multi-Layer Perceptron (MLP)
* XGBoost

---

## 📈 Evaluación de Modelos

| Modelo              | Accuracy  | Precision  | Recall    | F1 Score   |
| ------------------- | --------- | ---------- | --------- | ---------- |
| Logistic Regression | 0.816     | 0.8179     | 0.816     | 0.8162     |
| KNN                 | 0.8149    | 0.8195     | 0.8149    | 0.8151     |
| SVM                 | 0.8059    | 0.8116     | 0.8059    | 0.8061     |
| Decision Tree       | 0.8033    | 0.8078     | 0.8033    | 0.8035     |
| **Random Forest**   | **0.824** | **0.8298** | **0.824** | **0.8241** |
| MLP                 | 0.8137    | 0.8178     | 0.8137    | 0.8139     |
| XGBoost             | 0.8106    | 0.8270     | 0.8106    | 0.8100     |

---

## 🌟 Importancia de las Características

Las 10 variables más importantes según Random Forest:

1. `initial_quantity` (11.66%)
2. `available_quantity` (11.07%)
3. `price` (8.33%)
4. `base_price` (7.73%)
5. `sold_quantity` (4.87%)
6. `warranty_Sin garantía` (2.37%)
7. `date_created_month` (1.18%)
8. `last_updated_month` (1.02%)
9. `shipping_local_pick_up` (0.69%)
10. `category_id_1227` (0.56%)

![image](https://github.com/user-attachments/assets/ac70d029-387f-4450-821c-0c5002e48e1a)

---

## 🧠 Análisis

* **Disponibilidad y Precio**: Mayor disponibilidad y precios más estables en productos nuevos.
* **Garantía**: Su ausencia es más común en productos usados.
* **Temporalidad**: Listings más recientes tienden a ser nuevos.
* **Categoría y Logística**: Variables con menor impacto.

---

## ✅ Conclusión

El modelo **Random Forest** se desempeña mejor para clasificar productos por condición, logrando:

* **Accuracy:** 0.824
* **F1 Score:** 0.824

Este modelo puede aplicarse para:

* Mejorar las búsquedas del usuario
* Optimizar la visibilidad de productos
* Ajustar precios y estrategias de venta

![image](https://github.com/user-attachments/assets/eb115944-3e21-4dce-9a23-1341fe3d64b2)

---

## ⚠️ Limitaciones

* Posible **desbalance de clases**
* **Sobreajuste** sin optimización
* Algunas variables podrían ser eliminadas

---

## 💡 Recomendaciones

* Usar `GridSearchCV` para afinar hiperparámetros
* Incluir `ROC-AUC` y matriz de confusión
* Evaluar con nuevos datos externos
* Simplificar el modelo eliminando variables poco relevantes

---

## 🧪 Requisitos

Instala las dependencias con:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 🚀 Instrucciones de Uso

1. Clona el repositorio:

```bash
git clone https://github.com/luzangelacarabali/CommerceML.git
```

2. Entra al directorio:

```bash
cd CommerceML
```

3. Abre el notebook `02_model_training.ipynb` en Jupyter Notebook o JupyterLab.

---

## 👩‍💻 Autor

**Luz Ángela Carabalí Mulato**
📧 [@luzangelacarabli](https://github.com/luzangelacarabali)


