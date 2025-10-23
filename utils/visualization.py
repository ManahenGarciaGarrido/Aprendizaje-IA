"""
Utilidades para visualización de datos
Gráficos y plots reutilizables
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class DataVisualizer:
    """
    Clase para visualización de datos
    """

    @staticmethod
    def plot_distribution(data, columns=None, bins=30):
        """
        Grafica distribuciones de variables

        Args:
            data (pd.DataFrame): Datos
            columns (list): Columnas a graficar (None = todas)
            bins (int): Número de bins para histogramas
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns

        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(columns):
            axes[idx].hist(data[col].dropna(), bins=bins, edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')

        # Ocultar ejes sobrantes
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_matrix(data, figsize=(12, 10)):
        """
        Grafica matriz de correlación

        Args:
            data (pd.DataFrame): Datos
            figsize (tuple): Tamaño de la figura
        """
        # Seleccionar solo columnas numéricas
        numeric_data = data.select_dtypes(include=[np.number])

        corr = numeric_data.corr()

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_training_history(history):
        """
        Grafica historial de entrenamiento (para modelos de deep learning)

        Args:
            history: History object de Keras/TensorFlow
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # Accuracy (si existe)
        if 'accuracy' in history.history:
            axes[1].plot(history.history['accuracy'], label='Train Accuracy')
            if 'val_accuracy' in history.history:
                axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
            axes[1].set_title('Model Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_importance(feature_names, importances, top_n=20):
        """
        Grafica importancia de features

        Args:
            feature_names (list): Nombres de features
            importances (array): Importancias
            top_n (int): Top N features a mostrar
        """
        # Ordenar por importancia
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_time_series(data, date_column, value_column, title='Time Series'):
        """
        Grafica serie temporal

        Args:
            data (pd.DataFrame): Datos
            date_column (str): Columna de fechas
            value_column (str): Columna de valores
            title (str): Título del gráfico
        """
        plt.figure(figsize=(15, 6))
        plt.plot(data[date_column], data[value_column])
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_image_grid(images, labels=None, predictions=None, n_cols=5):
    """
    Muestra grid de imágenes

    Args:
        images: Array de imágenes
        labels: Etiquetas reales
        predictions: Predicciones
        n_cols (int): Número de columnas
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    axes = axes.flatten() if n_images > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < n_images:
            ax.imshow(images[idx], cmap='gray')
            title = ""
            if labels is not None:
                title += f"True: {labels[idx]}"
            if predictions is not None:
                title += f"\nPred: {predictions[idx]}"
            ax.set_title(title)
            ax.axis('off')
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Pruebas
    print("Utilidades de visualización")
