"""
Módulo de preprocesamiento de texto para análisis de sentimientos
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios (ejecutar una vez)
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


class TextPreprocessor:
    """
    Clase para preprocesar texto
    """

    def __init__(self, language='english'):
        """
        Inicializa el preprocesador

        Args:
            language (str): Idioma para stopwords
        """
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Limpia el texto eliminando caracteres especiales y normalizando

        Args:
            text (str): Texto a limpiar

        Returns:
            str: Texto limpio
        """
        # TODO: Implementar limpieza
        # 1. Convertir a minúsculas
        # 2. Eliminar HTML tags
        # 3. Eliminar URLs
        # 4. Eliminar caracteres especiales
        # 5. Eliminar números
        # 6. Eliminar espacios extra

        pass

    def tokenize(self, text):
        """
        Tokeniza el texto

        Args:
            text (str): Texto a tokenizar

        Returns:
            list: Lista de tokens
        """
        # TODO: Implementar tokenización
        pass

    def remove_stopwords(self, tokens):
        """
        Elimina stopwords de la lista de tokens

        Args:
            tokens (list): Lista de tokens

        Returns:
            list: Tokens sin stopwords
        """
        # TODO: Implementar eliminación de stopwords
        pass

    def lemmatize(self, tokens):
        """
        Aplica lematización a los tokens

        Args:
            tokens (list): Lista de tokens

        Returns:
            list: Tokens lematizados
        """
        # TODO: Implementar lematización
        pass

    def preprocess(self, text):
        """
        Pipeline completo de preprocesamiento

        Args:
            text (str): Texto original

        Returns:
            str: Texto preprocesado
        """
        # TODO: Implementar pipeline completo
        # 1. Limpiar
        # 2. Tokenizar
        # 3. Eliminar stopwords
        # 4. Lematizar
        # 5. Unir tokens

        pass


def main():
    """
    Función de prueba
    """
    # TODO: Probar el preprocesador con texto de ejemplo
    sample_text = "This movie was absolutely AMAZING! I loved every minute of it."
    preprocessor = TextPreprocessor()

    print("Original:", sample_text)
    # print("Procesado:", preprocessor.preprocess(sample_text))


if __name__ == "__main__":
    main()
