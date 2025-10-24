"""
Definición de arquitecturas de modelos para clasificación MNIST

Este módulo contiene diferentes arquitecturas de redes neuronales
para clasificar dígitos escritos a mano. Aprenderás sobre:
- Modelos densos (Fully Connected)
- Redes Convolucionales (CNN)
- Activaciones y regularización

Autor: Manahen García Garrido
Fecha: 2025
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


def create_simple_dense_model(input_shape=(28, 28), num_classes=10):
    """
    Crea un modelo simple con capas densas (fully connected)

    ¿QUÉ es una red densa?
    - Cada neurona está conectada a TODAS las neuronas de la capa anterior
    - Simple pero no aprovecha la estructura espacial de las imágenes
    - Buen punto de partida para entender redes neuronales

    Arquitectura:
    Input (28x28) -> Flatten -> Dense(128) -> Dropout -> Dense(64) -> Output(10)

    Args:
        input_shape (tuple): Dimensiones de entrada (altura, ancho)
        num_classes (int): Número de clases a clasificar (10 dígitos)

    Returns:
        keras.Model: Modelo compilado
    """
    print("Creando modelo denso simple...")

    model = models.Sequential([
        # Capa de entrada: especifica el shape de los datos
        layers.Input(shape=input_shape),

        # Flatten: convierte imagen 28x28 en vector de 784 elementos
        # ¿POR QUÉ? Las capas densas solo aceptan vectores 1D
        # [28, 28] -> [784]
        layers.Flatten(),

        # Primera capa densa: 128 neuronas
        # Cada neurona está conectada a las 784 entradas
        # Total parámetros: 784 * 128 + 128 (bias) = 100,480
        layers.Dense(
            128,
            activation='relu',  # ReLU: max(0, x) - activa solo valores positivos
            name='dense_1'
        ),

        # Dropout: apaga aleatoriamente 30% de neuronas durante entrenamiento
        # ¿POR QUÉ? Evita overfitting (que el modelo memorice en vez de aprender)
        # Solo activo durante training, no durante predicción
        layers.Dropout(0.3, name='dropout_1'),

        # Segunda capa densa: 64 neuronas
        layers.Dense(
            64,
            activation='relu',
            name='dense_2'
        ),

        # Dropout adicional
        layers.Dropout(0.3, name='dropout_2'),

        # Capa de salida: 10 neuronas (una por cada dígito 0-9)
        # Softmax: convierte outputs a probabilidades que suman 1
        # Ej: [0.1, 0.05, 0.7, 0.02, ...] -> el dígito 2 tiene 70% probabilidad
        layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        )
    ])

    # Resumen del modelo
    print("\nArquitectura del modelo:")
    model.summary()

    return model


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Crea una Red Neuronal Convolucional (CNN)

    ¿POR QUÉ usar CNN para imágenes?
    - Capas convolucionales detectan patrones locales (bordes, curvas)
    - Usa muchos menos parámetros que redes densas
    - Preserva la estructura espacial de la imagen
    - MUCHO más efectivas para visión por computadora

    ¿CÓMO funcionan las convoluciones?
    - Filtros pequeños (3x3, 5x5) se deslizan por la imagen
    - Cada filtro aprende a detectar un patrón específico
    - Primeras capas: bordes simples
    - Capas profundas: formas complejas (curvas de dígitos)

    Arquitectura:
    Input -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool ->
    Flatten -> Dense(128) -> Dropout -> Output(10)

    Args:
        input_shape (tuple): Dimensiones (altura, ancho, canales)
        num_classes (int): Número de clases

    Returns:
        keras.Model: Modelo CNN compilado
    """
    print("Creando modelo CNN...")

    model = models.Sequential([
        # Capa de entrada
        layers.Input(shape=input_shape),

        # ========================================
        # BLOQUE CONVOLUCIONAL 1
        # ========================================

        # Conv2D: aplica 32 filtros de 3x3 a la imagen
        # - 32 filtros = aprende 32 patrones diferentes
        # - kernel_size=(3,3) = cada filtro es una matriz 3x3
        # - padding='same' = mantiene el tamaño de la imagen
        # - activation='relu' = activa solo valores positivos
        layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2d_1'
        ),

        # MaxPooling2D: reduce dimensiones tomando el máximo de cada región 2x2
        # ¿POR QUÉ? Reduce parámetros y hace el modelo más robusto
        # [28, 28, 32] -> [14, 14, 32]
        layers.MaxPooling2D(
            pool_size=(2, 2),
            name='maxpool_1'
        ),

        # ========================================
        # BLOQUE CONVOLUCIONAL 2
        # ========================================

        # Segunda capa convolucional: 64 filtros
        # Aprende patrones más complejos basados en los de la capa anterior
        layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2d_2'
        ),

        # Segundo MaxPooling
        # [14, 14, 64] -> [7, 7, 64]
        layers.MaxPooling2D(
            pool_size=(2, 2),
            name='maxpool_2'
        ),

        # ========================================
        # BLOQUE CONVOLUCIONAL 3 (Opcional pero mejora accuracy)
        # ========================================

        # Tercera capa convolucional: 64 filtros adicionales
        layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2d_3'
        ),

        # ========================================
        # CLASIFICADOR DENSO
        # ========================================

        # Flatten: convierte feature maps 3D en vector 1D
        # [7, 7, 64] -> [3136]
        layers.Flatten(name='flatten'),

        # Capa densa para clasificación final
        layers.Dense(
            128,
            activation='relu',
            name='dense_1'
        ),

        # Dropout para evitar overfitting
        layers.Dropout(0.5, name='dropout'),

        # Capa de salida: 10 clases con softmax
        layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        )
    ])

    # Resumen
    print("\nArquitectura CNN:")
    model.summary()

    return model


def create_advanced_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Crea una CNN avanzada con técnicas modernas

    Mejoras incluidas:
    - Batch Normalization: normaliza activaciones entre capas
    - L2 Regularization: penaliza pesos muy grandes
    - Más capas convolucionales
    - Mejor estructura

    Esta arquitectura debería alcanzar >99% accuracy en MNIST

    Args:
        input_shape (tuple): Dimensiones (altura, ancho, canales)
        num_classes (int): Número de clases

    Returns:
        keras.Model: Modelo CNN avanzado
    """
    print("Creando modelo CNN avanzado...")

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),  # Normaliza activaciones
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),

        # Clasificador
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    print("\nArquitectura CNN Avanzada:")
    model.summary()

    return model


def compile_model(model, learning_rate=0.001):
    """
    Compila el modelo con optimizer, loss y métricas

    ¿QUÉ significa compilar?
    - Define CÓMO el modelo aprenderá
    - Optimizer: algoritmo para ajustar pesos
    - Loss: qué tan lejos están las predicciones de la realidad
    - Metrics: qué medir durante entrenamiento

    Args:
        model (keras.Model): Modelo a compilar
        learning_rate (float): Tasa de aprendizaje

    Returns:
        keras.Model: Modelo compilado listo para entrenar
    """
    # Adam: optimizer adaptativo, funciona bien en la mayoría de casos
    # learning_rate: qué tan rápido aprende (muy alto = inestable, muy bajo = lento)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Sparse Categorical Crossentropy: loss para clasificación multiclase
    # "sparse" porque las etiquetas son números (0-9), no one-hot vectors
    loss = keras.losses.SparseCategoricalCrossentropy()

    # Métricas a monitorear durante entrenamiento
    metrics = [
        'accuracy',  # Porcentaje de predicciones correctas
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    print(f"\nModelo compilado con:")
    print(f"  - Optimizer: Adam (lr={learning_rate})")
    print(f"  - Loss: Sparse Categorical Crossentropy")
    print(f"  - Metrics: {metrics}")

    return model


# ============================================================
# FUNCIÓN DE PRUEBA
# ============================================================

def main():
    """
    Función de prueba para ver las arquitecturas
    """
    print("=" * 70)
    print("  ARQUITECTURAS DE MODELOS MNIST")
    print("=" * 70)

    # Probar modelo denso
    print("\n\n1. MODELO DENSO SIMPLE")
    print("-" * 70)
    model_dense = create_simple_dense_model()
    compile_model(model_dense)

    # Probar CNN
    print("\n\n2. MODELO CNN")
    print("-" * 70)
    model_cnn = create_cnn_model()
    compile_model(model_cnn)

    # Probar CNN avanzada
    print("\n\n3. MODELO CNN AVANZADO")
    print("-" * 70)
    model_advanced = create_advanced_cnn_model()
    compile_model(model_advanced)

    print("\n" + "=" * 70)
    print("¡Arquitecturas creadas con éxito!")
    print("=" * 70)


if __name__ == "__main__":
    main()
