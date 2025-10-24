"""
Script para entrenar el modelo de análisis de sentimientos

Este script implementa el PIPELINE COMPLETO de entrenamiento:
1. Carga de datos
2. Preprocesamiento
3. Vectorización (convertir texto a números)
4. División train/test
5. Entrenamiento de modelos
6. Evaluación
7. Guardado del mejor modelo

Autor: Manahen García Garrido
Fecha: 2025
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import tensorflow_datasets as tfds
from preprocessing import TextPreprocessor, batch_preprocess


def load_data_from_tfds():
    """
    Carga el dataset IMDB desde TensorFlow Datasets

    ¿POR QUÉ usar TensorFlow Datasets?
    - Dataset público y limpio
    - Descarga automática
    - Ya viene etiquetado (0=negativo, 1=positivo)

    Returns:
        tuple: (texts, labels) - listas de textos y etiquetas
    """
    print("Cargando dataset IMDB desde TensorFlow Datasets...")
    print("(Primera vez puede tardar ~5 min en descargar ~80MB)")

    # Cargar solo el split de entrenamiento (25,000 reviews)
    # as_supervised=True devuelve (texto, label) directamente
    dataset = tfds.load('imdb_reviews', split='train', as_supervised=True)

    texts = []
    labels = []

    # Convertir TensorFlow dataset a listas Python
    # Esto es más fácil de manejar con scikit-learn
    print("   Convirtiendo datos...")
    for text, label in dataset:
        # .numpy() convierte tensores TF a numpy arrays
        # .decode() convierte bytes a string
        texts.append(text.numpy().decode('utf-8'))
        labels.append(label.numpy())

    print(f"Dataset cargado: {len(texts)} reviews")
    print(f"Distribución: {sum(labels)} positivas, {len(labels)-sum(labels)} negativas")

    return texts, labels


def load_data_from_csv(csv_path):
    """
    Carga datos desde archivo CSV (alternativa al método anterior)

    El CSV debe tener columnas: 'review' y 'sentiment'

    Args:
        csv_path (str): Ruta al CSV

    Returns:
        tuple: (texts, labels)
    """
    print(f"Cargando datos desde {csv_path}...")

    df = pd.read_csv(csv_path)

    # Convertir sentimientos a números: 'positive' -> 1, 'negative' -> 0
    sentiment_map = {'positive': 1, 'negative': 0}
    df['label'] = df['sentiment'].map(sentiment_map)

    texts = df['review'].tolist()
    labels = df['label'].tolist()

    print(f"Dataset cargado: {len(texts)} reviews")
    return texts, labels


def vectorize_texts(texts_train, texts_test, method='tfidf', max_features=5000):
    """
    Convierte textos a vectores numéricos

    ¿POR QUÉ necesitamos vectorizar?
    - Los modelos de ML solo entienden números
    - Necesitamos convertir palabras a features numéricas

    ¿QUÉ métodos hay?
    1. Bag of Words (BoW): Cuenta frecuencia de cada palabra
    2. TF-IDF: Pondera palabras por importancia (mejor para sentimientos)

    TF-IDF (Term Frequency - Inverse Document Frequency):
    - Palabras frecuentes en UN documento pero raras en general = importantes
    - Ejemplo: "spectacular" aparece mucho en reviews positivas = high TF-IDF
    - "the", "and" aparecen en TODAS las reviews = low TF-IDF

    Args:
        texts_train (list): Textos de entrenamiento
        texts_test (list): Textos de prueba
        method (str): 'tfidf' o 'bow'
        max_features (int): Número máximo de palabras a considerar

    Returns:
        tuple: (X_train_vec, X_test_vec, vectorizer)
    """
    print(f"\nVectorizando textos con {method.upper()}...")
    print(f"   Vocabulario máximo: {max_features} palabras")

    if method == 'tfidf':
        # TF-IDF Vectorizer
        # - max_features: limita vocabulario a las N palabras más comunes
        # - min_df: ignora palabras que aparecen en menos de X documentos
        # - max_df: ignora palabras que aparecen en más de X% documentos
        # - sublinear_tf: usa escala logarítmica (ayuda con palabras muy frecuentes)
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,           # Palabra debe aparecer en al menos 2 documentos
            max_df=0.8,         # Ignorar palabras en más del 80% de docs
            sublinear_tf=True   # Usar log(TF) en vez de TF
        )
    elif method == 'bow':
        # Bag of Words simple (cuenta frecuencias)
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8
        )
    else:
        raise ValueError(f"Método desconocido: {method}")

    # fit_transform: aprende vocabulario Y transforma textos
    # Esto solo se hace en TRAIN
    X_train_vec = vectorizer.fit_transform(texts_train)

    # transform: solo transforma (usa vocabulario ya aprendido)
    # En TEST usamos el vocabulario de TRAIN (¡nunca al revés!)
    X_test_vec = vectorizer.transform(texts_test)

    print(f"Vectorización completada!")
    print(f"   Shape train: {X_train_vec.shape}")
    print(f"   Shape test: {X_test_vec.shape}")
    print(f"   Vocabulario aprendido: {len(vectorizer.vocabulary_)} palabras")

    return X_train_vec, X_test_vec, vectorizer


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='logistic_regression'):
    """
    Entrena y evalúa un modelo

    ¿QUÉ modelos usar para clasificación de texto?
    1. Naive Bayes: Rápido, simple, funciona bien con texto
    2. Logistic Regression: Más potente, suele dar mejor accuracy
    3. SVM: Muy bueno pero más lento

    Args:
        X_train: Features de entrenamiento (vectorizadas)
        y_train: Labels de entrenamiento
        X_test: Features de prueba
        y_test: Labels de prueba
        model_type (str): Tipo de modelo

    Returns:
        tuple: (modelo entrenado, métricas)
    """
    print(f"\nEntrenando modelo: {model_type}")

    # Seleccionar modelo
    if model_type == 'naive_bayes':
        # Naive Bayes Multinomial: Asume que palabras siguen distribución multinomial
        # Muy rápido y sorprendentemente efectivo para texto
        model = MultinomialNB(alpha=1.0)  # alpha: suavizado (evita prob=0)

    elif model_type == 'logistic_regression':
        # Logistic Regression: Modelo lineal con función sigmoide
        # C: inverso de regularización (más bajo = más regularización)
        # max_iter: iteraciones máximas para convergencia
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )

    elif model_type == 'svm':
        # Support Vector Machine lineal
        # Muy poderoso pero más lento de entrenar
        model = LinearSVC(
            C=1.0,
            max_iter=1000,
            random_state=42
        )

    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    # ENTRENAR el modelo
    print("   Entrenando...")
    model.fit(X_train, y_train)

    # PREDECIR en test set
    print("   Evaluando...")
    y_pred = model.predict(X_test)

    # CALCULAR MÉTRICAS
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nResultados {model_type}:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=['Negativo', 'Positivo']))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("   Matriz de Confusión:")
    print(f"   [[TN={cm[0,0]:>4}  FP={cm[0,1]:>4}]")
    print(f"    [FN={cm[1,0]:>4}  TP={cm[1,1]:>4}]]")

    # ¿Qué significa la matriz de confusión?
    # TN (True Negative): Reviews negativas correctamente clasificadas
    # TP (True Positive): Reviews positivas correctamente clasificadas
    # FP (False Positive): Reviews negativas clasificadas como positivas (ERROR)
    # FN (False Negative): Reviews positivas clasificadas como negativas (ERROR)

    metrics = {
        'model_type': model_type,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

    return model, metrics


def save_model(model, vectorizer, model_dir='../models'):
    """
    Guarda el modelo y vectorizador para uso posterior

    ¿POR QUÉ guardar ambos?
    - Modelo: para hacer predicciones
    - Vectorizer: para convertir texto nuevo al mismo formato

    Args:
        model: Modelo entrenado
        vectorizer: Vectorizador entrenado
        model_dir (str): Directorio donde guardar
    """
    # Crear directorio si no existe
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'sentiment_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')

    # joblib es mejor que pickle para objetos de scikit-learn
    # Es más eficiente y maneja arrays grandes mejor
    print(f"\nGuardando modelo...")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Modelo guardado en: {model_path}")
    print(f"Vectorizer guardado en: {vectorizer_path}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    """
    Pipeline completo de entrenamiento

    PASOS:
    1. Cargar datos
    2. Preprocesar textos
    3. Dividir train/test
    4. Vectorizar
    5. Entrenar múltiples modelos
    6. Comparar y guardar el mejor
    """
    print("=" * 70)
    print("  ENTRENAMIENTO DE MODELO DE ANÁLISIS DE SENTIMIENTOS")
    print("=" * 70)

    # ========================================
    # PASO 1: CARGAR DATOS
    # ========================================
    try:
        texts, labels = load_data_from_tfds()
    except Exception as e:
        print(f"Error cargando desde TFDS: {e}")
        print("   Intenta con CSV si tienes el dataset descargado")
        return

    # ========================================
    # PASO 2: PREPROCESAR TEXTOS
    # ========================================
    print("\n🧹 Preprocesando textos...")
    print("   (Esto puede tardar 2-5 minutos para 25k reviews)")

    # Usar máximo 10000 textos para entrenamiento rápido
    # Para producción, usa todos (25000)
    MAX_SAMPLES = 10000  # Cambiar a len(texts) para usar todos
    texts = texts[:MAX_SAMPLES]
    labels = labels[:MAX_SAMPLES]

    # Preprocesar usando inglés (idioma del dataset IMDB)
    processed_texts = batch_preprocess(texts, language='english')

    # ========================================
    # PASO 3: DIVIDIR TRAIN/TEST
    # ========================================
    print("\nDividiendo datos...")

    # 80% entrenamiento, 20% prueba
    # random_state=42: hace que la división sea reproducible
    texts_train, texts_test, y_train, y_test = train_test_split(
        processed_texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels  # Mantiene proporción de clases
    )

    print(f"   Train: {len(texts_train)} samples")
    print(f"   Test:  {len(texts_test)} samples")

    # ========================================
    # PASO 4: VECTORIZAR
    # ========================================
    X_train, X_test, vectorizer = vectorize_texts(
        texts_train, texts_test,
        method='tfidf',
        max_features=5000
    )

    # ========================================
    # PASO 5: ENTRENAR MÚLTIPLES MODELOS
    # ========================================
    models = ['naive_bayes', 'logistic_regression']

    results = []
    trained_models = []

    for model_type in models:
        model, metrics = train_and_evaluate_model(
            X_train, y_train,
            X_test, y_test,
            model_type=model_type
        )
        results.append(metrics)
        trained_models.append(model)

    # ========================================
    # PASO 6: SELECCIONAR MEJOR MODELO
    # ========================================
    print("\n" + "=" * 70)
    print("  COMPARACIÓN DE MODELOS")
    print("=" * 70)

    for i, metrics in enumerate(results):
        print(f"{metrics['model_type']:20s}: Accuracy = {metrics['accuracy']:.4f}")

    # Encontrar mejor modelo (mayor accuracy)
    best_idx = max(range(len(results)), key=lambda i: results[i]['accuracy'])
    best_model = trained_models[best_idx]
    best_metrics = results[best_idx]

    print(f"\nMEJOR MODELO: {best_metrics['model_type']}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")

    # ========================================
    # PASO 7: GUARDAR MEJOR MODELO
    # ========================================
    save_model(best_model, vectorizer)

    print("\n" + "=" * 70)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print("\nPara hacer predicciones, usa: python predict.py")


if __name__ == "__main__":
    main()
