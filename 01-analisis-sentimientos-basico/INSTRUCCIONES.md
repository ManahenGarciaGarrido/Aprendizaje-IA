# INSTRUCCIONES RÁPIDAS - Proyecto 1

## ¿Dónde ejecutar?

**TU PORTÁTIL es PERFECTO** (no necesitas GPU)
- Los modelos son ligeros (ML tradicional, no deep learning)
- Tiempo estimado: 5-15 minutos
- Requiere: ~4GB RAM, ~500MB disco

**Alternativa:** Google Colab (solo si tu PC tiene <4GB RAM)

---

## Instalación y Ejecución

### Paso 1: Instalar dependencias

```bash
# Navegar a la carpeta del proyecto
cd 01-analisis-sentimientos-basico

# Instalar paquetes necesarios
pip install numpy pandas scikit-learn nltk tensorflow-datasets joblib matplotlib seaborn
```

### Paso 2: Entrenar el modelo

```bash
# Desde la carpeta src/
cd src
python train_model.py
```

**¿Qué hace este script?**
- Descarga dataset IMDB (25k reviews, ~80MB) - solo la 1ª vez
- Preprocesa textos (limpieza, tokenización, lemmatización)
- Entrena 2 modelos (Naive Bayes y Logistic Regression)
- Guarda el mejor modelo en `models/`

**Tiempo:** 2-5 minutos en portátil normal

### Paso 3: Probar predicciones

```bash
python predict.py
```

**¿Qué hace?**
- Carga el modelo entrenado
- Muestra ejemplos de predicción
- Permite modo interactivo para tus propios textos

---

## Opciones de uso

### Opción A: Scripts Python (Recomendado para aprender)

```bash
# 1. Probar preprocesamiento
cd src/
python preprocessing.py

# 2. Entrenar modelo completo
python train_model.py

# 3. Hacer predicciones
python predict.py
```

### Opción B: Notebook Jupyter

```bash
# Instalar Jupyter si no lo tienes
pip install jupyter

# Abrir notebook
jupyter notebook notebook.ipynb
```

---

## Configuración personalizada

### Cambiar cantidad de datos

Edita `train_model.py` línea 315:
```python
MAX_SAMPLES = 10000  # Cambiar a 25000 para usar todo el dataset
```

### Usar CSV propio

Si tienes tu propio dataset CSV:
```python
# En train_model.py, reemplaza load_data_from_tfds() por:
texts, labels = load_data_from_csv('ruta/a/tu/archivo.csv')
```

### 🌍 IMPORTANTE: Idioma del modelo

**El modelo está entrenado en INGLÉS** (dataset IMDB de reviews en inglés).

**Para usar textos en inglés** (recomendado):
- ✅ Ya está configurado por defecto
- Los textos deben estar en inglés para obtener buenos resultados
- Ejemplos: "This movie is amazing!", "Terrible film, waste of time"

**Para entrenar con textos en español:**

1. Necesitas un dataset en español (el IMDB es solo inglés)
2. Cambia el idioma del preprocesador en `train_model.py` línea 320:
   ```python
   processed_texts = batch_preprocess(texts, language='spanish')
   ```
3. Cambia el idioma en `predict.py` línea 64:
   ```python
   self.preprocessor = TextPreprocessor(language='spanish')
   ```
4. Opciones de datasets en español:
   - Kaggle: "Spanish Sentiment Analysis"
   - Twitter datasets
   - Reviews de productos en español

**¿Por qué el idioma es importante?**
- Las stopwords cambian: "the, a, is" (inglés) vs "el, la, es" (español)
- El lemmatizer funciona diferente según el idioma
- Si el idioma no coincide, las predicciones serán incorrectas

---

## Solución de problemas

### Error: "No module named 'nltk'"
```bash
pip install nltk
```

### Error: NLTK resources not found
```python
# Ejecutar en Python:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### Error: "tensorflow_datasets not found"
```bash
pip install tensorflow-datasets
```

### Dataset IMDB no descarga
- Verifica conexión a internet
- Alternativa: descarga manual desde Kaggle

---

## Resultados esperados

**Accuracy esperado:** 85-90%
- Naive Bayes: ~84-87%
- Logistic Regression: ~87-90%

**Si obtienes menor accuracy:**
- Aumenta `MAX_SAMPLES` a 25000
- Aumenta `max_features` en TfidfVectorizer

---

## Próximos pasos

1. Ejecuta los scripts y entiende cada paso
2. Experimenta con el código:
   - Cambia hiperparámetros
   - Prueba otros modelos (SVM)
   - Analiza palabras más importantes
3. Lee los comentarios en el código
4. Pasa al Proyecto 2

---

## Conceptos clave aprendidos

- Preprocesamiento de texto (tokenización, lemmatización, stopwords)
- Vectorización (TF-IDF vs Bag of Words)
- Modelos de clasificación (Naive Bayes, Logistic Regression)
- Evaluación (Accuracy, Precision, Recall, F1-Score)
- Pipeline completo de ML

---

¿Dudas? Revisa los comentarios en el código - cada línea está explicada!
