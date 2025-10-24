"""
Script para hacer predicciones con el clasificador MNIST entrenado

Este script permite:
- Cargar modelo entrenado
- Hacer predicciones en imágenes del test set
- Visualizar predicciones con probabilidades
- Analizar errores del modelo

Autor: Manahen García Garrido
Fecha: 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Importar funciones de visualización
from visualize import (
    plot_predictions,
    plot_misclassified_images,
    plot_sample_images
)


class MNISTPredictor:
    """
    Clase para hacer predicciones con el modelo MNIST entrenado

    ¿CÓMO FUNCIONA?
    1. Carga el modelo guardado
    2. Preprocesa nuevas imágenes (mismo formato que training)
    3. Hace predicciones
    4. Interpreta resultados

    ¡IMPORTANTE! Debe usar el MISMO preprocesamiento
    que se usó durante el entrenamiento.
    """

    def __init__(self, model_path='../models/mnist_model_best.h5'):
        """
        Inicializa el predictor cargando el modelo

        Args:
            model_path (str): Ruta al modelo guardado (.h5)
        """
        print("=" * 70)
        print("  INICIALIZANDO PREDICTOR MNIST")
        print("=" * 70)

        # Verificar que el modelo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo no encontrado: {model_path}\n"
                "   Ejecuta primero: python train.py"
            )

        # Cargar modelo
        print(f"\nCargando modelo desde: {model_path}")
        self.model = keras.models.load_model(model_path)

        print("✓ Modelo cargado exitosamente!")

        # Mostrar información del modelo
        print(f"\nInformación del modelo:")
        print(f"  - Arquitectura: {len(self.model.layers)} capas")
        print(f"  - Parámetros: {self.model.count_params():,}")
        print(f"  - Input shape: {self.model.input_shape}")
        print(f"  - Output shape: {self.model.output_shape}")

        # Nombres de las clases
        self.class_names = [str(i) for i in range(10)]

    def preprocess_image(self, image):
        """
        Preprocesa una imagen para predicción

        PASOS:
        1. Normalizar píxeles a [0, 1]
        2. Añadir dimensión de canal si es necesario
        3. Añadir dimensión de batch

        Args:
            image (numpy.ndarray): Imagen de 28x28 píxeles

        Returns:
            numpy.ndarray: Imagen preprocesada
        """
        # Normalizar si es necesario
        if image.max() > 1.0:
            image = image.astype('float32') / 255.0

        # Verificar si necesita dimensión de canal
        # Modelo denso: (28, 28)
        # Modelo CNN: (28, 28, 1)
        expected_shape = self.model.input_shape[1:]  # Sin batch dimension

        if len(expected_shape) == 3 and len(image.shape) == 2:
            # Añadir dimensión de canal
            image = np.expand_dims(image, axis=-1)

        # Añadir dimensión de batch
        # [28, 28, 1] -> [1, 28, 28, 1]
        image = np.expand_dims(image, axis=0)

        return image

    def predict(self, image):
        """
        Predice la clase de una imagen

        Args:
            image (numpy.ndarray): Imagen de 28x28 píxeles

        Returns:
            int: Clase predicha (0-9)
        """
        # Preprocesar
        processed_image = self.preprocess_image(image)

        # Predecir
        prediction = self.model.predict(processed_image, verbose=0)[0]

        # Obtener clase con mayor probabilidad
        predicted_class = np.argmax(prediction)

        return predicted_class

    def predict_with_probabilities(self, image):
        """
        Predice con probabilidades de cada clase

        Args:
            image (numpy.ndarray): Imagen de 28x28 píxeles

        Returns:
            tuple: (predicted_class, probabilities_dict)
        """
        # Preprocesar
        processed_image = self.preprocess_image(image)

        # Predecir
        probabilities = self.model.predict(processed_image, verbose=0)[0]

        # Clase predicha
        predicted_class = np.argmax(probabilities)

        # Crear diccionario de probabilidades
        prob_dict = {
            str(i): float(probabilities[i]) for i in range(10)
        }

        return predicted_class, prob_dict

    def predict_batch(self, images):
        """
        Predice múltiples imágenes a la vez

        ¿POR QUÉ batch prediction?
        - Más eficiente que predecir una por una
        - Aprovecha mejor GPU/CPU

        Args:
            images (numpy.ndarray): Array de imágenes

        Returns:
            numpy.ndarray: Array de predicciones
        """
        # Normalizar si es necesario
        if images.max() > 1.0:
            images = images.astype('float32') / 255.0

        # Verificar shape
        expected_shape = self.model.input_shape[1:]

        if len(expected_shape) == 3 and len(images.shape) == 3:
            # Añadir dimensión de canal
            images = np.expand_dims(images, axis=-1)

        # Predecir
        predictions = self.model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes

    def analyze_image(self, image, true_label=None):
        """
        Análisis completo de una imagen con visualización

        Args:
            image (numpy.ndarray): Imagen a analizar
            true_label (int): Etiqueta verdadera (opcional)

        Returns:
            dict: Resultado del análisis
        """
        # Obtener predicción y probabilidades
        pred_class, probabilities = self.predict_with_probabilities(image)
        confidence = probabilities[str(pred_class)]

        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # SUBPLOT 1: Imagen
        ax1.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)

        # Título con etiqueta verdadera si existe
        if true_label is not None:
            color = 'green' if pred_class == true_label else 'red'
            title = f'Real: {true_label} | Predicción: {pred_class}'
            ax1.set_title(title, fontsize=14, color=color, fontweight='bold')
        else:
            ax1.set_title(f'Predicción: {pred_class}', fontsize=14, fontweight='bold')

        ax1.axis('off')

        # SUBPLOT 2: Probabilidades
        classes = list(range(10))
        probs = [probabilities[str(i)] for i in classes]

        bars = ax2.bar(classes, probs, color='skyblue', edgecolor='navy')
        # Resaltar la clase predicha
        bars[pred_class].set_color('red')

        ax2.set_ylim([0, 1])
        ax2.set_xlabel('Clase (Dígito)', fontsize=12)
        ax2.set_ylabel('Probabilidad', fontsize=12)
        ax2.set_title(f'Confianza: {confidence:.1%}', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Resultado
        result = {
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': probabilities
        }

        if true_label is not None:
            result['true_label'] = true_label
            result['correct'] = (pred_class == true_label)

        return result


# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================

def interactive_prediction(predictor, test_images, test_labels):
    """
    Modo interactivo: el usuario elige imágenes aleatorias

    Args:
        predictor (MNISTPredictor): Predictor inicializado
        test_images: Imágenes de test
        test_labels: Etiquetas de test
    """
    print("\n" + "=" * 70)
    print("  MODO INTERACTIVO - Predicción de Dígitos")
    print("=" * 70)
    print("\nPresiona Enter para analizar una imagen aleatoria")
    print("Escribe 'salir' para terminar\n")

    while True:
        user_input = input(">>> ").strip().lower()

        if user_input in ['salir', 'exit', 'quit', 'q']:
            print("\n¡Hasta luego!")
            break

        # Seleccionar imagen aleatoria
        idx = np.random.randint(0, len(test_images))
        image = test_images[idx]
        true_label = test_labels[idx]

        # Analizar
        print(f"\nAnalizando imagen #{idx}...")
        result = predictor.analyze_image(image, true_label)

        # Mostrar resultado
        if result['correct']:
            print(f"✓ ¡Predicción correcta! ({result['confidence']:.1%} confianza)")
        else:
            print(f"✗ Error. Real: {result['true_label']}, Predicho: {result['predicted_class']}")


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():
    """
    Función principal con ejemplos de predicción
    """
    print("\n" + "=" * 70)
    print("  PREDICTOR DE DÍGITOS MNIST")
    print("=" * 70)

    # ========================================
    # CARGAR PREDICTOR
    # ========================================
    try:
        predictor = MNISTPredictor()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPrimero debes entrenar el modelo:")
        print("  python train.py")
        return

    # ========================================
    # CARGAR DATOS DE TEST
    # ========================================
    print("\n" + "=" * 70)
    print("  CARGANDO DATOS DE TEST")
    print("=" * 70)

    (_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # Normalizar
    test_images = test_images.astype('float32') / 255.0

    print(f"\nDatos de test cargados: {len(test_images)} imágenes")

    # ========================================
    # EJEMPLOS DE PREDICCIÓN
    # ========================================
    print("\n" + "=" * 70)
    print("  EJEMPLOS DE PREDICCIÓN")
    print("=" * 70)

    # Predecir algunas imágenes aleatorias
    print("\nVisualizando predicciones...")
    plot_predictions(predictor.model, test_images, test_labels, num_samples=5)

    # ========================================
    # ANÁLISIS DE ERRORES
    # ========================================
    print("\n" + "=" * 70)
    print("  ANÁLISIS DE ERRORES")
    print("=" * 70)

    print("\nBuscando y visualizando errores...")
    plot_misclassified_images(
        predictor.model,
        test_images,
        test_labels,
        num_samples=20
    )

    # ========================================
    # EVALUAR EN TODO EL TEST SET
    # ========================================
    print("\n" + "=" * 70)
    print("  EVALUACIÓN COMPLETA EN TEST SET")
    print("=" * 70)

    print("\nHaciendo predicciones en todo el test set...")

    # Preparar imágenes según el modelo
    if len(predictor.model.input_shape) == 4:  # CNN
        test_images_processed = np.expand_dims(test_images, axis=-1)
    else:  # Dense
        test_images_processed = test_images

    # Evaluar
    test_loss, test_accuracy = predictor.model.evaluate(
        test_images_processed,
        test_labels,
        verbose=0
    )

    print(f"\nResultados finales:")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # ========================================
    # MODO INTERACTIVO (Opcional)
    # ========================================
    print("\n" + "=" * 70)

    response = input("\n¿Quieres probar el modo interactivo? (s/n): ").strip().lower()

    if response in ['s', 'si', 'sí', 'yes', 'y']:
        interactive_prediction(predictor, test_images, test_labels)

    print("\n¡Gracias por usar el predictor de dígitos MNIST!")
    print("=" * 70)


if __name__ == "__main__":
    main()
