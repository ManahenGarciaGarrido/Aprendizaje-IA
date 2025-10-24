"""
Script para entrenar el clasificador de dígitos MNIST

Este script implementa el PIPELINE COMPLETO de entrenamiento:
1. Carga de datos MNIST
2. Preprocesamiento (normalización, reshape)
3. División train/validation/test
4. Construcción del modelo
5. Entrenamiento con callbacks
6. Evaluación y visualización
7. Guardado del modelo

Autor: Manahen García Garrido
Fecha: 2025
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Importar módulos locales
from model import (
    create_simple_dense_model,
    create_cnn_model,
    create_advanced_cnn_model,
    compile_model
)
from visualize import (
    plot_sample_images,
    plot_training_history,
    plot_confusion_matrix
)


def load_mnist_data():
    """
    Carga el dataset MNIST desde Keras

    MNIST (Modified National Institute of Standards and Technology):
    - 70,000 imágenes de dígitos escritos a mano
    - 28x28 píxeles en escala de grises (0-255)
    - 60,000 para entrenamiento, 10,000 para test
    - 10 clases (dígitos 0-9)

    ¿POR QUÉ MNIST?
    - Dataset estándar para aprender Deep Learning
    - Ligero y rápido de entrenar
    - Bien balanceado (similar número de cada dígito)

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    print("=" * 70)
    print("  CARGANDO DATASET MNIST")
    print("=" * 70)

    # Cargar datos desde Keras
    # La primera vez descargará ~11MB
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print(f"\nDataset cargado exitosamente!")
    print(f"  Train set: {x_train.shape[0]} imágenes")
    print(f"  Test set:  {x_test.shape[0]} imágenes")
    print(f"  Dimensiones: {x_train.shape[1]}x{x_train.shape[2]} píxeles")

    # Mostrar distribución de clases
    print(f"\n  Distribución de clases (train):")
    for digit in range(10):
        count = np.sum(y_train == digit)
        print(f"    Dígito {digit}: {count} imágenes")

    return x_train, y_train, x_test, y_test


def preprocess_data(x_train, y_train, x_test, y_test, model_type='cnn'):
    """
    Preprocesa los datos para entrenamiento

    PASOS:
    1. Normalización: convertir píxeles de [0, 255] a [0, 1]
    2. Reshape: añadir dimensión de canal para CNN
    3. División: separar validation set del training set

    ¿POR QUÉ normalizar?
    - Redes neuronales funcionan mejor con valores pequeños
    - Facilita convergencia durante entrenamiento
    - Evita problemas numéricos

    Args:
        x_train, y_train: Datos de entrenamiento
        x_test, y_test: Datos de prueba
        model_type (str): 'dense' o 'cnn'

    Returns:
        tuple: Datos preprocesados
    """
    print("\n" + "=" * 70)
    print("  PREPROCESANDO DATOS")
    print("=" * 70)

    # ========================================
    # PASO 1: NORMALIZACIÓN
    # ========================================
    print("\n1. Normalizando píxeles...")

    # Convertir a float32 (más eficiente que float64)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalizar de [0, 255] a [0, 1]
    # ¿Por qué dividir entre 255? Es el valor máximo de un píxel
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(f"   Rango original: [0, 255]")
    print(f"   Rango normalizado: [{x_train.min():.1f}, {x_train.max():.1f}]")

    # ========================================
    # PASO 2: RESHAPE (solo para CNN)
    # ========================================
    if model_type == 'cnn':
        print("\n2. Añadiendo dimensión de canal...")

        # CNN espera formato: [batch, height, width, channels]
        # MNIST es escala de grises = 1 canal
        # [60000, 28, 28] -> [60000, 28, 28, 1]
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        print(f"   Shape train: {x_train.shape}")
        print(f"   Shape test:  {x_test.shape}")
    else:
        print("\n2. Modelo denso - manteniendo shape [batch, height, width]")

    # ========================================
    # PASO 3: DIVISIÓN TRAIN/VALIDATION
    # ========================================
    print("\n3. Dividiendo en train/validation...")

    # Separar últimas 10,000 imágenes para validación
    # Validación: monitorear overfitting durante entrenamiento
    # Test: evaluación final (NUNCA usar durante entrenamiento)
    validation_split = 10000

    x_val = x_train[-validation_split:]
    y_val = y_train[-validation_split:]

    x_train = x_train[:-validation_split]
    y_train = y_train[:-validation_split]

    print(f"   Train:      {x_train.shape[0]} imágenes")
    print(f"   Validation: {x_val.shape[0]} imágenes")
    print(f"   Test:       {x_test.shape[0]} imágenes")

    return x_train, y_train, x_val, y_val, x_test, y_test


def create_callbacks(model_dir='../models'):
    """
    Crea callbacks para el entrenamiento

    ¿QUÉ son callbacks?
    - Funciones que se ejecutan durante el entrenamiento
    - Permiten monitorear, guardar, y controlar el proceso

    Callbacks incluidos:
    1. ModelCheckpoint: guarda el mejor modelo
    2. EarlyStopping: detiene si no hay mejora
    3. ReduceLROnPlateau: reduce learning rate si se estanca

    Args:
        model_dir (str): Directorio donde guardar modelos

    Returns:
        list: Lista de callbacks
    """
    print("\n" + "=" * 70)
    print("  CONFIGURANDO CALLBACKS")
    print("=" * 70)

    # Crear directorio si no existe
    os.makedirs(model_dir, exist_ok=True)

    callbacks_list = []

    # ========================================
    # CALLBACK 1: ModelCheckpoint
    # ========================================
    # Guarda el modelo cuando mejora la validation accuracy

    checkpoint_path = os.path.join(model_dir, 'mnist_model_best.h5')

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',      # Métrica a monitorear
        save_best_only=True,          # Solo guardar si mejora
        mode='max',                   # Queremos maximizar accuracy
        verbose=1                     # Mostrar mensaje cuando guarde
    )
    callbacks_list.append(checkpoint)

    print(f"\n✓ ModelCheckpoint configurado")
    print(f"  - Guardará mejor modelo en: {checkpoint_path}")
    print(f"  - Monitoreando: val_accuracy")

    # ========================================
    # CALLBACK 2: EarlyStopping
    # ========================================
    # Detiene el entrenamiento si no hay mejora después de N epochs

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',           # Monitorear pérdida de validación
        patience=5,                   # Esperar 5 epochs sin mejora
        restore_best_weights=True,    # Restaurar mejores pesos al final
        verbose=1
    )
    callbacks_list.append(early_stopping)

    print(f"\n✓ EarlyStopping configurado")
    print(f"  - Paciencia: 5 epochs sin mejora")
    print(f"  - Restaurará mejores pesos")

    # ========================================
    # CALLBACK 3: ReduceLROnPlateau
    # ========================================
    # Reduce learning rate cuando el entrenamiento se estanca

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',           # Monitorear pérdida
        factor=0.5,                   # Reducir LR a la mitad
        patience=3,                   # Después de 3 epochs sin mejora
        min_lr=1e-7,                  # LR mínimo
        verbose=1
    )
    callbacks_list.append(reduce_lr)

    print(f"\n✓ ReduceLROnPlateau configurado")
    print(f"  - Factor de reducción: 0.5x")
    print(f"  - Paciencia: 3 epochs")

    return callbacks_list


def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=128):
    """
    Entrena el modelo

    ¿QUÉ sucede durante el entrenamiento?
    1. Forward pass: calcular predicciones
    2. Calcular loss: qué tan lejos de la realidad
    3. Backward pass: calcular gradientes
    4. Actualizar pesos: usando el optimizer

    Este ciclo se repite para cada batch, cada epoch

    Args:
        model: Modelo a entrenar
        x_train, y_train: Datos de entrenamiento
        x_val, y_val: Datos de validación
        epochs (int): Número de epochs
        batch_size (int): Tamaño del batch

    Returns:
        History: Historial del entrenamiento
    """
    print("\n" + "=" * 70)
    print("  ENTRENANDO MODELO")
    print("=" * 70)

    print(f"\nParámetros de entrenamiento:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Steps por epoch: {len(x_train) // batch_size}")
    print(f"  - Total de parámetros: {model.count_params():,}")

    # Crear callbacks
    callbacks = create_callbacks()

    # ENTRENAR
    print("\n" + "-" * 70)
    print("Iniciando entrenamiento...")
    print("-" * 70 + "\n")

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1  # Mostrar barra de progreso
    )

    print("\n" + "-" * 70)
    print("¡Entrenamiento completado!")
    print("-" * 70)

    return history


def evaluate_model(model, x_test, y_test):
    """
    Evalúa el modelo en el test set

    ¿POR QUÉ evaluar en test set?
    - Test set nunca se usó en entrenamiento
    - Da una estimación realista del rendimiento
    - Detecta si hubo overfitting

    Args:
        model: Modelo entrenado
        x_test, y_test: Datos de prueba

    Returns:
        tuple: (test_loss, test_accuracy, predictions)
    """
    print("\n" + "=" * 70)
    print("  EVALUANDO MODELO EN TEST SET")
    print("=" * 70)

    # Evaluar
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"\nResultados en Test Set:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Hacer predicciones para análisis posterior
    print("\nGenerando predicciones...")
    predictions = model.predict(x_test, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)

    # Calcular errores
    errors = np.sum(pred_labels != y_test)
    print(f"  Total de errores: {errors} de {len(y_test)} ({errors/len(y_test)*100:.2f}%)")

    return test_loss, test_accuracy, pred_labels


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    """
    Pipeline completo de entrenamiento

    PASOS:
    1. Cargar datos
    2. Preprocesar
    3. Crear modelo
    4. Entrenar
    5. Evaluar
    6. Visualizar resultados
    """
    print("\n" + "=" * 70)
    print("  ENTRENAMIENTO CLASIFICADOR MNIST")
    print("=" * 70)

    # ========================================
    # PASO 1: CARGAR DATOS
    # ========================================
    x_train, y_train, x_test, y_test = load_mnist_data()

    # Visualizar algunas muestras
    print("\nVisualizando muestras del dataset...")
    plot_sample_images(x_train, y_train, num_samples=25)

    # ========================================
    # PASO 2: PREPROCESAR
    # ========================================

    # Elegir tipo de modelo: 'dense' o 'cnn'
    MODEL_TYPE = 'cnn'  # Cambiar a 'dense' para probar modelo denso

    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test,
        model_type=MODEL_TYPE
    )

    # ========================================
    # PASO 3: CREAR MODELO
    # ========================================
    print("\n" + "=" * 70)
    print("  CREANDO MODELO")
    print("=" * 70)

    if MODEL_TYPE == 'dense':
        print("\nCreando modelo denso simple...")
        model = create_simple_dense_model()
    elif MODEL_TYPE == 'cnn':
        print("\nCreando modelo CNN...")
        model = create_cnn_model()
        # Descomentar para usar CNN avanzada:
        # model = create_advanced_cnn_model()

    # Compilar modelo
    model = compile_model(model, learning_rate=0.001)

    # ========================================
    # PASO 4: ENTRENAR
    # ========================================
    history = train_model(
        model,
        x_train, y_train,
        x_val, y_val,
        epochs=20,
        batch_size=128
    )

    # ========================================
    # PASO 5: EVALUAR
    # ========================================
    test_loss, test_accuracy, pred_labels = evaluate_model(
        model, x_test, y_test
    )

    # ========================================
    # PASO 6: VISUALIZAR RESULTADOS
    # ========================================
    print("\n" + "=" * 70)
    print("  VISUALIZANDO RESULTADOS")
    print("=" * 70)

    # Curvas de entrenamiento
    print("\nGraficando curvas de entrenamiento...")
    plot_training_history(history)

    # Matriz de confusión
    print("\nGenerando matriz de confusión...")
    plot_confusion_matrix(y_test, pred_labels)

    # ========================================
    # PASO 7: GUARDAR MODELO FINAL
    # ========================================
    print("\n" + "=" * 70)
    print("  GUARDANDO MODELO FINAL")
    print("=" * 70)

    final_model_path = '../models/mnist_model_final.h5'
    model.save(final_model_path)
    print(f"\nModelo final guardado en: {final_model_path}")

    # También guardar en formato SavedModel (más moderno)
    saved_model_path = '../models/mnist_saved_model'
    model.save(saved_model_path, save_format='tf')
    print(f"Modelo SavedModel guardado en: {saved_model_path}")

    # ========================================
    # RESUMEN FINAL
    # ========================================
    print("\n" + "=" * 70)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 70)

    print(f"\nResultados finales:")
    print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Total parámetros: {model.count_params():,}")

    print(f"\nArchivos generados:")
    print(f"  - {final_model_path}")
    print(f"  - {saved_model_path}/")
    print(f"  - ../models/mnist_model_best.h5")

    print("\nPara hacer predicciones, ejecuta:")
    print("  python predict.py")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
