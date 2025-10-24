"""
Funciones de visualización para el clasificador MNIST

Este módulo contiene funciones para visualizar:
- Muestras del dataset
- Predicciones del modelo
- Matrices de confusión
- Curvas de entrenamiento
- Filtros convolucionales

Autor: Manahen García Garrido
Fecha: 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf


def plot_sample_images(images, labels, num_samples=25, predictions=None):
    """
    Visualiza una cuadrícula de imágenes del dataset

    ¿POR QUÉ visualizar datos?
    - Verificar que los datos se cargaron correctamente
    - Entender la dificultad del problema
    - Ver ejemplos de cada clase

    Args:
        images (numpy.ndarray): Array de imágenes [num_images, height, width]
        labels (numpy.ndarray): Array de etiquetas
        num_samples (int): Número de imágenes a mostrar (debe ser cuadrado perfecto)
        predictions (numpy.ndarray): Predicciones del modelo (opcional)
    """
    # Calcular dimensiones de la cuadrícula
    grid_size = int(np.sqrt(num_samples))
    num_samples = grid_size * grid_size

    # Crear figura
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle('Muestras del Dataset MNIST', fontsize=16, y=0.995)

    # Aplanar array de axes para iterar fácilmente
    axes = axes.flatten()

    # Seleccionar muestras aleatorias
    indices = np.random.choice(len(images), num_samples, replace=False)

    for idx, ax in enumerate(axes):
        # Obtener imagen y etiqueta
        img_idx = indices[idx]
        image = images[img_idx]
        true_label = labels[img_idx]

        # Mostrar imagen
        # cmap='gray' para escala de grises
        # vmin=0, vmax=1 para normalizar el rango
        ax.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)

        # Título con etiqueta verdadera y predicción (si existe)
        if predictions is not None:
            pred_label = predictions[img_idx]
            color = 'green' if pred_label == true_label else 'red'
            title = f'Real: {true_label}\nPred: {pred_label}'
            ax.set_title(title, fontsize=9, color=color)
        else:
            ax.set_title(f'Dígito: {true_label}', fontsize=10)

        # Quitar ejes para imagen más limpia
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Visualiza las curvas de aprendizaje (loss y accuracy)

    ¿QUÉ muestran estas curvas?
    - Loss: qué tan lejos están las predicciones (menor = mejor)
    - Accuracy: porcentaje de aciertos (mayor = mejor)
    - Train vs Validation: detecta overfitting

    ¿Qué buscar?
    - Loss bajando: el modelo está aprendiendo
    - Val loss subiendo mientras train loss baja: OVERFITTING
    - Ambas curvas estables: buen entrenamiento

    Args:
        history (keras.callbacks.History): Historial del entrenamiento
    """
    # Extraer métricas del historial
    train_loss = history.history['loss']
    val_loss = history.history.get('val_loss', None)
    train_acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy', None)

    # Número de epochs
    epochs = range(1, len(train_loss) + 1)

    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ========================================
    # SUBPLOT 1: Loss
    # ========================================
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    if val_loss:
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Loss durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ========================================
    # SUBPLOT 2: Accuracy
    # ========================================
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    if val_acc:
        ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Accuracy durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Imprimir resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("=" * 60)
    print(f"Training Loss final:      {train_loss[-1]:.4f}")
    if val_loss:
        print(f"Validation Loss final:    {val_loss[-1]:.4f}")
    print(f"Training Accuracy final:  {train_acc[-1]:.4f} ({train_acc[-1]*100:.2f}%)")
    if val_acc:
        print(f"Validation Accuracy final: {val_acc[-1]:.4f} ({val_acc[-1]*100:.2f}%)")
    print("=" * 60)


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Visualiza la matriz de confusión

    ¿QUÉ es una matriz de confusión?
    - Muestra qué clases se confunden entre sí
    - Diagonal: predicciones correctas
    - Fuera de diagonal: errores

    Ejemplo: Si el modelo confunde mucho 3 con 8,
    habrá un valor alto en cm[3,8]

    Args:
        y_true (numpy.ndarray): Etiquetas verdaderas
        y_pred (numpy.ndarray): Predicciones del modelo
        class_names (list): Nombres de las clases (opcional)
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    # Nombres de clases por defecto (0-9)
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # Crear figura
    plt.figure(figsize=(10, 8))

    # Usar seaborn para visualización bonita
    # annot=True: muestra números en cada celda
    # fmt='d': formato entero
    # cmap: mapa de colores
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Número de muestras'}
    )

    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicción', fontsize=13, fontweight='bold')
    plt.ylabel('Etiqueta Verdadera', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Análisis de errores más comunes
    print("\n" + "=" * 60)
    print("ANÁLISIS DE CONFUSIONES MÁS COMUNES")
    print("=" * 60)

    # Copiar matriz y poner diagonal en 0 (no nos interesan aciertos)
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)

    # Encontrar los 5 errores más frecuentes
    flat_indices = np.argsort(cm_errors.flatten())[::-1][:5]
    error_coords = [(idx // 10, idx % 10) for idx in flat_indices]

    print("\nTop 5 confusiones:")
    for i, (true_class, pred_class) in enumerate(error_coords, 1):
        count = cm[true_class, pred_class]
        if count > 0:
            print(f"{i}. El dígito '{true_class}' confundido con '{pred_class}': {count} veces")

    print("=" * 60)


def plot_predictions(model, images, labels, num_samples=10):
    """
    Visualiza predicciones del modelo con probabilidades

    Muestra:
    - Imagen original
    - Etiqueta verdadera
    - Predicción del modelo
    - Probabilidades de cada clase (gráfico de barras)

    Args:
        model (keras.Model): Modelo entrenado
        images (numpy.ndarray): Imágenes a predecir
        labels (numpy.ndarray): Etiquetas verdaderas
        num_samples (int): Número de muestras a mostrar
    """
    # Seleccionar muestras aleatorias
    indices = np.random.choice(len(images), num_samples, replace=False)

    # Crear figura
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 2))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Predicciones del Modelo', fontsize=16, fontweight='bold')

    for idx, ax_pair in enumerate(axes):
        img_idx = indices[idx]
        image = images[img_idx]
        true_label = labels[img_idx]

        # Hacer predicción
        # Añadir dimensión de batch: [28, 28, 1] -> [1, 28, 28, 1]
        img_input = np.expand_dims(image, axis=0)
        prediction = model.predict(img_input, verbose=0)[0]
        pred_label = np.argmax(prediction)

        # SUBPLOT 1: Imagen
        ax_pair[0].imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)

        # Color del título según si es correcto o no
        color = 'green' if pred_label == true_label else 'red'
        title = f'Real: {true_label} | Predicción: {pred_label}'
        ax_pair[0].set_title(title, fontsize=11, color=color, fontweight='bold')
        ax_pair[0].axis('off')

        # SUBPLOT 2: Probabilidades
        ax_pair[1].bar(range(10), prediction, color='skyblue', edgecolor='navy')
        ax_pair[1].set_ylim([0, 1])
        ax_pair[1].set_xticks(range(10))
        ax_pair[1].set_xlabel('Clase', fontsize=10)
        ax_pair[1].set_ylabel('Probabilidad', fontsize=10)
        ax_pair[1].axhline(y=prediction[pred_label], color='red',
                           linestyle='--', alpha=0.5, label=f'Max: {prediction[pred_label]:.2f}')
        ax_pair[1].legend(fontsize=9)
        ax_pair[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_misclassified_images(model, images, labels, num_samples=20):
    """
    Visualiza solo las imágenes mal clasificadas

    ¿POR QUÉ es útil?
    - Entender qué tipo de errores comete el modelo
    - Identificar patrones difíciles
    - Mejorar el modelo basándose en los errores

    Args:
        model (keras.Model): Modelo entrenado
        images (numpy.ndarray): Imágenes
        labels (numpy.ndarray): Etiquetas verdaderas
        num_samples (int): Número máximo de errores a mostrar
    """
    print("Buscando imágenes mal clasificadas...")

    # Hacer predicciones en todo el conjunto
    predictions = model.predict(images, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)

    # Encontrar índices de errores
    error_indices = np.where(pred_labels != labels)[0]

    if len(error_indices) == 0:
        print("¡No hay errores! El modelo tiene 100% accuracy.")
        return

    print(f"Encontrados {len(error_indices)} errores.")

    # Limitar al número solicitado
    num_samples = min(num_samples, len(error_indices))
    selected_errors = np.random.choice(error_indices, num_samples, replace=False)

    # Calcular grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    # Crear figura
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'Imágenes Mal Clasificadas ({num_samples} de {len(error_indices)} errores)',
                 fontsize=14, fontweight='bold')

    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < num_samples:
            img_idx = selected_errors[idx]
            image = images[img_idx]
            true_label = labels[img_idx]
            pred_label = pred_labels[img_idx]
            confidence = predictions[img_idx][pred_label]

            # Mostrar imagen
            ax.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)
            title = f'Real: {true_label}\nPred: {pred_label} ({confidence:.1%})'
            ax.set_title(title, fontsize=9, color='red')
        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def visualize_filters(model, layer_name='conv2d_1'):
    """
    Visualiza los filtros aprendidos por una capa convolucional

    ¿QUÉ son los filtros?
    - Pequeñas matrices que detectan patrones
    - Primeras capas: detectan bordes, líneas
    - Capas profundas: detectan formas complejas

    Args:
        model (keras.Model): Modelo entrenado
        layer_name (str): Nombre de la capa convolucional
    """
    try:
        # Obtener la capa
        layer = model.get_layer(layer_name)

        # Obtener pesos (filtros)
        filters, biases = layer.get_weights()

        # filters shape: [kernel_height, kernel_width, in_channels, out_channels]
        # Para MNIST: [3, 3, 1, 32] (32 filtros de 3x3)

        num_filters = filters.shape[-1]
        num_channels = filters.shape[-2]

        print(f"\nVisualizando {num_filters} filtros de la capa '{layer_name}'")
        print(f"  Shape de filtros: {filters.shape}")

        # Calcular grid
        grid_size = int(np.ceil(np.sqrt(num_filters)))

        # Crear figura
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f'Filtros de la Capa: {layer_name}', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i in range(grid_size * grid_size):
            if i < num_filters:
                # Obtener filtro
                f = filters[:, :, 0, i]  # Primer canal de entrada

                # Normalizar para visualización
                f_min, f_max = f.min(), f.max()
                f_normalized = (f - f_min) / (f_max - f_min + 1e-8)

                # Mostrar
                axes[i].imshow(f_normalized, cmap='viridis')
                axes[i].set_title(f'Filtro {i}', fontsize=8)
            else:
                axes[i].axis('off')

            axes[i].set_xticks([])
            axes[i].set_yticks([])

        plt.tight_layout()
        plt.show()

    except ValueError:
        print(f"Error: Capa '{layer_name}' no encontrada en el modelo")
        print("Capas disponibles:")
        for layer in model.layers:
            print(f"  - {layer.name}")


# ============================================================
# FUNCIÓN DE PRUEBA
# ============================================================

def main():
    """
    Función de prueba con datos sintéticos
    """
    print("=" * 60)
    print("DEMO: Funciones de Visualización")
    print("=" * 60)

    # Crear datos sintéticos para demostración
    print("\nGenerando datos sintéticos...")

    # Imágenes aleatorias
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, 100)

    # Mostrar muestras
    print("\n1. Visualizando muestras del dataset...")
    plot_sample_images(images, labels, num_samples=16)

    print("\n✓ Funciones de visualización listas para usar!")


if __name__ == "__main__":
    main()
