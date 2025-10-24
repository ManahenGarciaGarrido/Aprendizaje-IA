"""
Script para hacer predicciones con el modelo entrenado

Este script carga el modelo ya entrenado y permite hacer predicciones
sobre textos nuevos. Es útil para:
- Probar el modelo con ejemplos personalizados
- Integrar en aplicaciones
- Analizar reviews en tiempo real

Autor: Tu nombre
Fecha: 2025
"""

import os
import joblib
import numpy as np
from preprocessing import TextPreprocessor


class SentimentPredictor:
    """
    Clase para predecir sentimientos en textos nuevos

    ¿CÓMO FUNCIONA?
    1. Carga modelo y vectorizador guardados
    2. Preprocesa texto nuevo (igual que en entrenamiento)
    3. Vectoriza el texto
    4. Usa el modelo para predecir

    ¡IMPORTANTE! Debe usar el MISMO preprocesamiento y vectorizador
    que se usaron durante el entrenamiento.
    """

    def __init__(self, model_path, vectorizer_path):
        """
        Inicializa el predictor cargando el modelo y vectorizador

        Args:
            model_path (str): Ruta al modelo guardado (.pkl)
            vectorizer_path (str): Ruta al vectorizador guardado (.pkl)
        """
        print("Cargando modelo y vectorizador...")

        # Verificar que los archivos existen
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo no encontrado: {model_path}\n"
                "   Ejecuta primero: python train_model.py"
            )

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizador no encontrado: {vectorizer_path}\n"
                "   Ejecuta primero: python train_model.py"
            )

        # Cargar modelo y vectorizador
        # joblib es el formato usado por scikit-learn
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        # Inicializar preprocesador (MISMO que en entrenamiento)
        self.preprocessor = TextPreprocessor()

        # Mapeo de etiquetas a nombres legibles
        # Ahora incluye 'Neutral' para predicciones inciertas
        self.label_names = {0: 'Negativo', 1: 'Positivo', 2: 'Neutral'}

        # Umbral para clasificar como neutral
        # Si la probabilidad está entre NEUTRAL_THRESHOLD_LOW y NEUTRAL_THRESHOLD_HIGH,
        # se clasifica como neutral
        self.NEUTRAL_THRESHOLD_LOW = 0.35
        self.NEUTRAL_THRESHOLD_HIGH = 0.65

        print("Modelo cargado y listo!")

    def predict(self, text):
        """
        Predice el sentimiento de un texto

        PIPELINE:
        1. Preprocesar (limpiar, tokenizar, lemmatizar)
        2. Vectorizar (convertir a números con TF-IDF)
        3. Predecir con el modelo
        4. Evaluar confianza para detectar neutrales
        5. Convertir número a etiqueta

        Args:
            text (str): Texto a analizar

        Returns:
            str: Sentimiento predicho ('Positivo', 'Negativo' o 'Neutral')
        """
        # 1. Preprocesar el texto
        # Esto lo limpia y normaliza igual que en entrenamiento
        processed_text = self.preprocessor.preprocess(text)

        # 2. Vectorizar
        # El vectorizador transforma el texto a la MISMA representación
        # que vio durante el entrenamiento
        text_vector = self.vectorizer.transform([processed_text])

        # 3. Obtener probabilidades
        # Si el modelo no está seguro (probabilidades cercanas a 0.5),
        # clasificamos como neutral
        probas = self.predict_proba(text)

        # 4. Detectar neutral basado en incertidumbre
        # Si la probabilidad del sentimiento positivo está entre los umbrales,
        # significa que el modelo no está seguro -> clasificar como neutral
        prob_positive = probas['Positivo']

        if self.NEUTRAL_THRESHOLD_LOW < prob_positive < self.NEUTRAL_THRESHOLD_HIGH:
            return 'Neutral'

        # 5. Si no es neutral, usar predicción normal
        prediction = self.model.predict(text_vector)[0]
        return self.label_names[prediction]

    def predict_proba(self, text):
        """
        Predice las probabilidades de cada sentimiento

        ¿POR QUÉ es útil?
        - Ver qué tan seguro está el modelo
        - "Positivo (98%)" es más confiable que "Positivo (51%)"
        - Útil para filtrar predicciones inciertas

        Args:
            text (str): Texto a analizar

        Returns:
            dict: Probabilidades de cada clase
                  {'Negativo': 0.2, 'Positivo': 0.8}
        """
        # Preprocesar y vectorizar
        processed_text = self.preprocessor.preprocess(text)
        text_vector = self.vectorizer.transform([processed_text])

        # predict_proba devuelve probabilidades [prob_neg, prob_pos]
        # No todos los modelos tienen este método (SVM lineal no)
        try:
            probabilities = self.model.predict_proba(text_vector)[0]
        except AttributeError:
            # Si el modelo no tiene predict_proba, usamos decision_function
            scores = self.model.decision_function(text_vector)[0]
            # Convertir a "pseudo-probabilidades" con sigmoide
            prob_pos = 1 / (1 + np.exp(-scores))
            probabilities = [1 - prob_pos, prob_pos]

        # Crear diccionario con nombres de clases
        result = {
            'Negativo': float(probabilities[0]),
            'Positivo': float(probabilities[1])
        }

        return result

    def predict_batch(self, texts):
        """
        Predice sentimientos para múltiples textos de una vez

        ¿POR QUÉ usar batch prediction?
        - Más eficiente que predecir uno por uno
        - Útil para analizar muchas reviews a la vez

        Args:
            texts (list): Lista de textos

        Returns:
            list: Lista de predicciones
        """
        predictions = []

        for text in texts:
            pred = self.predict(text)
            predictions.append(pred)

        return predictions

    def analyze(self, text, show_confidence=True):
        """
        Análisis completo de un texto con formato bonito

        Args:
            text (str): Texto a analizar
            show_confidence (bool): Mostrar probabilidades

        Returns:
            dict: Resultado completo
        """
        # Obtener predicción y probabilidades
        sentiment = self.predict(text)
        probas = self.predict_proba(text)

        # Calcular confianza
        # Para neutral, la confianza es qué tan cerca está de 0.5
        # Para positivo/negativo, es la probabilidad máxima
        if sentiment == "Neutral":
            # Confianza en neutral = qué tan cerca está de 0.5 (máxima incertidumbre)
            confidence = 1.0 - abs(probas['Positivo'] - 0.5) * 2
        else:
            confidence = max(probas.values())

        # Emoji según sentimiento
        if sentiment == "Positivo":
            emoji = "😊"
        elif sentiment == "Negativo":
            emoji = "😞"
        else:  # Neutral
            emoji = "😐"

        result = {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': probas,
            'emoji': emoji
        }

        # Imprimir resultado formateado
        print(f"\n{emoji} Sentimiento: {sentiment}")
        if show_confidence:
            print(f"   Confianza: {confidence*100:.1f}%")
            print(f"   Probabilidades:")
            for label, prob in probas.items():
                bar = "█" * int(prob * 20)
                print(f"     {label:10s}: {prob:>5.1%} {bar}")

        return result


# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================

def interactive_mode(predictor):
    """
    Modo interactivo: el usuario escribe textos y ve predicciones

    Args:
        predictor (SentimentPredictor): Predictor inicializado
    """
    print("\n" + "=" * 70)
    print("  MODO INTERACTIVO - Análisis de Sentimientos")
    print("=" * 70)
    print("Escribe un texto para analizar su sentimiento.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        # Leer input del usuario
        text = input("Tu texto: ").strip()

        # Salir si el usuario escribe 'salir'
        if text.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\n¡Hasta luego!")
            break

        # Saltar si está vacío
        if not text:
            continue

        # Analizar y mostrar resultado
        predictor.analyze(text)


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():
    """
    Función principal con ejemplos de uso
    """
    print("=" * 70)
    print("  PREDICTOR DE SENTIMIENTOS")
    print("=" * 70)

    # Rutas a modelo y vectorizador
    MODEL_PATH = "../models/sentiment_model.pkl"
    VECTORIZER_PATH = "../models/vectorizer.pkl"

    try:
        # Crear predictor
        predictor = SentimentPredictor(MODEL_PATH, VECTORIZER_PATH)

    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPrimero debes entrenar el modelo:")
        print("   python train_model.py")
        return

    # ========================================
    # EJEMPLOS DE PREDICCIÓN
    # ========================================
    print("\n" + "=" * 70)
    print("  EJEMPLOS DE PREDICCIÓN")
    print("=" * 70)

    # Textos de ejemplo (incluye casos positivos, negativos y neutrales)
    test_texts = [
        "This movie was absolutely amazing! Best film I've ever seen.",
        "Terrible movie, complete waste of time and money.",
        "It was okay, nothing special but not bad either.",
        "The movie is average, has some good and bad parts.",
        "I loved the acting but the plot was confusing.",
        "Worst movie ever. I want my money back!",
        "Brilliant masterpiece! Oscar-worthy performance.",
        "The film has its moments but overall it's just fine.",
    ]

    # Analizar cada texto
    for i, text in enumerate(test_texts, 1):
        print(f"\nEjemplo {i}:")
        print(f"   \"{text}\"")
        predictor.analyze(text)

    # ========================================
    # MODO INTERACTIVO (opcional)
    # ========================================
    print("\n" + "=" * 70)

    # Preguntar si quiere modo interactivo
    response = input("\n¿Quieres probar el modo interactivo? (s/n): ").strip().lower()

    if response in ['s', 'si', 'sí', 'yes', 'y']:
        interactive_mode(predictor)

    print("\n¡Gracias por usar el predictor de sentimientos!")


if __name__ == "__main__":
    main()
