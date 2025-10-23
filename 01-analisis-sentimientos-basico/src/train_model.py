"""
Script para entrenar el modelo de análisis de sentimientos
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocessing import TextPreprocessor


def load_data(data_path):
    """
    Carga el dataset

    Args:
        data_path (str): Ruta al archivo de datos

    Returns:
        pd.DataFrame: Dataset cargado
    """
    # TODO: Implementar carga de datos
    # Puede ser desde CSV, TensorFlow Datasets, etc.
    pass


def prepare_data(df, preprocessor):
    """
    Prepara los datos para entrenamiento

    Args:
        df (pd.DataFrame): Dataset
        preprocessor (TextPreprocessor): Instancia del preprocesador

    Returns:
        tuple: X (textos), y (etiquetas)
    """
    # TODO: Implementar preparación de datos
    pass


def train_model(X_train, y_train, model_type='logistic_regression'):
    """
    Entrena el modelo

    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        model_type (str): Tipo de modelo ('naive_bayes' o 'logistic_regression')

    Returns:
        model: Modelo entrenado
    """
    # TODO: Implementar entrenamiento
    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    # TODO: Entrenar modelo
    # model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo

    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba

    Returns:
        dict: Métricas de evaluación
    """
    # TODO: Implementar evaluación
    # predictions = model.predict(X_test)
    # accuracy = accuracy_score(y_test, predictions)
    # report = classification_report(y_test, predictions)

    pass


def save_model(model, vectorizer, model_path, vectorizer_path):
    """
    Guarda el modelo y el vectorizador

    Args:
        model: Modelo entrenado
        vectorizer: Vectorizador entrenado
        model_path (str): Ruta para guardar el modelo
        vectorizer_path (str): Ruta para guardar el vectorizador
    """
    # TODO: Guardar modelo y vectorizador
    # joblib.dump(model, model_path)
    # joblib.dump(vectorizer, vectorizer_path)
    pass


def main():
    """
    Función principal
    """
    # TODO: Implementar pipeline completo
    # 1. Cargar datos
    # 2. Preprocesar
    # 3. Vectorizar
    # 4. Dividir train/test
    # 5. Entrenar modelo
    # 6. Evaluar
    # 7. Guardar modelo

    print("Iniciando entrenamiento...")

    # Configuración
    DATA_PATH = "../data/IMDB-Dataset.csv"
    MODEL_PATH = "../models/sentiment_model.pkl"
    VECTORIZER_PATH = "../models/vectorizer.pkl"

    # Pipeline de entrenamiento
    # ...

    print("Entrenamiento completado!")


if __name__ == "__main__":
    main()
