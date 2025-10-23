"""
Script para hacer predicciones con el modelo entrenado
"""

import joblib
from preprocessing import TextPreprocessor


class SentimentPredictor:
    """
    Clase para predecir sentimientos en textos nuevos
    """

    def __init__(self, model_path, vectorizer_path):
        """
        Inicializa el predictor cargando el modelo y vectorizador

        Args:
            model_path (str): Ruta al modelo guardado
            vectorizer_path (str): Ruta al vectorizador guardado
        """
        # TODO: Cargar modelo y vectorizador
        # self.model = joblib.load(model_path)
        # self.vectorizer = joblib.load(vectorizer_path)
        self.preprocessor = TextPreprocessor()

    def predict(self, text):
        """
        Predice el sentimiento de un texto

        Args:
            text (str): Texto a analizar

        Returns:
            str: Sentimiento predicho ('positive' o 'negative')
        """
        # TODO: Implementar predicción
        # 1. Preprocesar texto
        # 2. Vectorizar
        # 3. Predecir
        # 4. Retornar resultado

        pass

    def predict_proba(self, text):
        """
        Predice la probabilidad de cada sentimiento

        Args:
            text (str): Texto a analizar

        Returns:
            dict: Probabilidades de cada clase
        """
        # TODO: Implementar predicción de probabilidades
        pass

    def predict_batch(self, texts):
        """
        Predice sentimientos para múltiples textos

        Args:
            texts (list): Lista de textos

        Returns:
            list: Lista de sentimientos predichos
        """
        # TODO: Implementar predicción por lotes
        pass


def main():
    """
    Función de prueba
    """
    MODEL_PATH = "../models/sentiment_model.pkl"
    VECTORIZER_PATH = "../models/vectorizer.pkl"

    # TODO: Crear predictor
    # predictor = SentimentPredictor(MODEL_PATH, VECTORIZER_PATH)

    # Textos de prueba
    test_texts = [
        "This movie was absolutely amazing! Best film I've ever seen.",
        "Terrible movie, complete waste of time and money.",
        "It was okay, nothing special but not bad either.",
    ]

    # TODO: Hacer predicciones
    # for text in test_texts:
    #     sentiment = predictor.predict(text)
    #     probabilities = predictor.predict_proba(text)
    #     print(f"Text: {text}")
    #     print(f"Sentiment: {sentiment}")
    #     print(f"Probabilities: {probabilities}")
    #     print("-" * 50)

    print("Predictor listo para usar!")


if __name__ == "__main__":
    main()
