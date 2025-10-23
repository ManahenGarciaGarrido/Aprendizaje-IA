"""
Utilidades para evaluación de modelos
Métricas y visualizaciones reutilizables
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


class ModelEvaluator:
    """
    Clase para evaluar modelos de ML
    """

    @staticmethod
    def evaluate_classification(y_true, y_pred, labels=None):
        """
        Evalúa un modelo de clasificación

        Args:
            y_true: Valores reales
            y_pred: Predicciones
            labels: Nombres de las clases

        Returns:
            dict: Diccionario con métricas
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

        print("=== Classification Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\n" + classification_report(y_true, y_pred, target_names=labels))

        return metrics

    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """
        Evalúa un modelo de regresión

        Args:
            y_true: Valores reales
            y_pred: Predicciones

        Returns:
            dict: Diccionario con métricas
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

        print("=== Regression Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False):
        """
        Visualiza matriz de confusión

        Args:
            y_true: Valores reales
            y_pred: Predicciones
            labels: Nombres de las clases
            normalize (bool): Normalizar matriz
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                    cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def plot_prediction_vs_actual(y_true, y_pred):
        """
        Gráfico de predicciones vs valores reales (regresión)

        Args:
            y_true: Valores reales
            y_pred: Predicciones
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.show()

    @staticmethod
    def plot_residuals(y_true, y_pred):
        """
        Gráfico de residuos

        Args:
            y_true: Valores reales
            y_pred: Predicciones
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')

        # Distribution of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Pruebas
    print("Utilidades de evaluación de modelos")
