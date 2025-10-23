# Proyecto 1: Análisis de Sentimientos en Texto

## Descripción
Construye tu primer modelo de procesamiento de lenguaje natural (NLP) que clasifica opiniones de películas como positivas o negativas. Aprenderás los fundamentos del análisis de texto y la clasificación binaria.

## Duración estimada
3-5 horas

## Dataset
- **IMDB Movie Reviews Dataset** (25,000 reseñas etiquetadas)
- Fuente: Disponible en Kaggle y directamente en TensorFlow Datasets
- URL: `tensorflow_datasets` - `imdb_reviews`

## Tecnologías
- Scikit-learn
- NLTK
- Pandas
- Matplotlib

## Objetivos de Aprendizaje
- Preprocesamiento de texto (tokenización, limpieza)
- Técnicas de vectorización (TF-IDF, Bag of Words)
- Entrenamiento de clasificadores (Naive Bayes, Logistic Regression)
- Evaluación de modelos (accuracy, precision, recall, F1-score)

## Métricas de Éxito
- Accuracy > 85%
- Comprender la matriz de confusión
- Visualizar las palabras más importantes

## Estructura del Proyecto
```
01-analisis-sentimientos-basico/
├── README.md                 # Este archivo
├── notebook.ipynb           # Notebook principal con todo el análisis
├── data/                    # Datos descargados (gitignore)
├── models/                  # Modelos entrenados guardados
└── src/
    ├── preprocessing.py     # Funciones de preprocesamiento
    ├── train_model.py       # Script de entrenamiento
    └── predict.py           # Script de predicción
```

## Pasos a Seguir

### Paso 1: Cargar y Explorar los Datos
- [ ] Importar librerías necesarias
- [ ] Cargar el dataset IMDB desde TensorFlow Datasets
- [ ] Explorar la distribución de clases
- [ ] Visualizar ejemplos de reseñas positivas y negativas

### Paso 2: Preprocesamiento de Texto
- [ ] Convertir texto a minúsculas
- [ ] Eliminar caracteres especiales y puntuación
- [ ] Tokenización del texto
- [ ] Eliminar stopwords
- [ ] Aplicar stemming o lemmatization

### Paso 3: Vectorización
- [ ] Implementar Bag of Words (CountVectorizer)
- [ ] Implementar TF-IDF (TfidfVectorizer)
- [ ] Comparar ambos métodos

### Paso 4: División de Datos
- [ ] Dividir datos en train/test (80/20)
- [ ] Verificar distribución balanceada

### Paso 5: Entrenamiento de Modelos
- [ ] Entrenar Naive Bayes
- [ ] Entrenar Logistic Regression
- [ ] Entrenar un tercer clasificador (opcional: SVM o Random Forest)

### Paso 6: Evaluación
- [ ] Calcular accuracy, precision, recall, F1-score
- [ ] Crear matriz de confusión
- [ ] Comparar performance de modelos
- [ ] Identificar palabras más importantes

### Paso 7: Predicción en Nuevos Textos
- [ ] Crear función de predicción
- [ ] Probar con reseñas personalizadas
- [ ] Analizar casos de error

### Paso 8: Visualización y Conclusiones
- [ ] Visualizar matriz de confusión
- [ ] Graficar las palabras más relevantes
- [ ] Documentar hallazgos

## Archivos a Implementar

### `src/preprocessing.py`
```python
# Funciones para:
# - clean_text()
# - tokenize()
# - remove_stopwords()
# - lemmatize_text()
```

### `src/train_model.py`
```python
# Script para:
# - Cargar datos
# - Preprocesar
# - Entrenar modelo
# - Guardar modelo
```

### `src/predict.py`
```python
# Script para:
# - Cargar modelo guardado
# - Predecir sentimiento de texto nuevo
```

## Recursos Útiles
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

## Notas
- Guarda los modelos entrenados en la carpeta `models/`
- Los datos descargados van en `data/` (esta carpeta está en .gitignore)
- Documenta todos tus experimentos en el notebook
