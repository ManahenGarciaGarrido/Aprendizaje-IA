"""
Utilidades para preprocesamiento de datos
Funciones reutilizables en múltiples proyectos
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Clase para preprocesamiento de datos genérico
    """

    def __init__(self):
        self.scaler = None

    @staticmethod
    def handle_missing_values(df, strategy='mean'):
        """
        Maneja valores faltantes en un DataFrame

        Args:
            df (pd.DataFrame): DataFrame con datos
            strategy (str): Estrategia ('mean', 'median', 'mode', 'drop')

        Returns:
            pd.DataFrame: DataFrame sin valores faltantes
        """
        # TODO: Implementar manejo de valores faltantes
        pass

    @staticmethod
    def remove_outliers(df, columns, method='iqr', threshold=1.5):
        """
        Elimina outliers de columnas especificadas

        Args:
            df (pd.DataFrame): DataFrame
            columns (list): Columnas a procesar
            method (str): Método ('iqr', 'zscore')
            threshold (float): Umbral para detección

        Returns:
            pd.DataFrame: DataFrame sin outliers
        """
        # TODO: Implementar eliminación de outliers
        pass

    def normalize_data(self, X, method='standard'):
        """
        Normaliza datos numéricos

        Args:
            X (array-like): Datos a normalizar
            method (str): 'standard' o 'minmax'

        Returns:
            array: Datos normalizados
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Método no soportado: {method}")

        return self.scaler.fit_transform(X)

    @staticmethod
    def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Divide datos en train, validation y test

        Args:
            X: Features
            y: Labels
            test_size (float): Proporción de test
            val_size (float): Proporción de validación
            random_state (int): Semilla aleatoria

        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # TODO: Implementar división tripartita
        pass


def load_and_prepare_data(file_path, target_column=None, test_size=0.2):
    """
    Carga y prepara datos desde un archivo

    Args:
        file_path (str): Ruta al archivo
        target_column (str): Nombre de la columna objetivo
        test_size (float): Tamaño del conjunto de test

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # TODO: Implementar carga y preparación
    pass


if __name__ == "__main__":
    # Pruebas
    print("Utilidades de preprocesamiento de datos")
