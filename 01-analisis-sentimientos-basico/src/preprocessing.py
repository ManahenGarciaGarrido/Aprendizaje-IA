"""
Módulo de preprocesamiento de texto para análisis de sentimientos

Este módulo contiene todas las funciones necesarias para limpiar y preparar
texto para análisis de sentimientos. El preprocesamiento es CRUCIAL en NLP
porque los modelos de ML no entienden texto crudo, solo números.

Autor: Tu nombre
Fecha: 2025
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios de NLTK (ejecutar una vez)
# Estos recursos contienen diccionarios y reglas lingüísticas
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Descargando recursos de NLTK...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("Recursos descargados!")


class TextPreprocessor:
    """
    Clase para preprocesamiento de texto.

    ¿Por qué una clase y no solo funciones?
    - Permite mantener estado (ej: el lemmatizer)
    - Más fácil de reutilizar
    - Sigue patrones de diseño comunes en ML
    """

    def __init__(self, language='english'):
        """
        Inicializa el preprocesador

        Args:
            language (str): Idioma para stopwords ('english' o 'spanish')
                          Por defecto 'english' porque IMDB está en inglés

        ¿Qué son las stopwords?
        - Palabras muy comunes que aportan poco significado: "the", "a", "is"
        - Las eliminamos porque no ayudan a distinguir sentimientos

        IMPORTANTE: El idioma debe coincidir con el dataset de entrenamiento:
        - IMDB dataset → 'english'
        - Dataset español → 'spanish'
        """
        self.stop_words = set(stopwords.words(language))

        # Lemmatizer: convierte palabras a su forma base
        # Ej: "running" -> "run", "better" -> "good"
        self.lemmatizer = WordNetLemmatizer()

        print(f"Preprocesador inicializado con {len(self.stop_words)} stopwords")

    def clean_text(self, text):
        """
        Limpia el texto eliminando caracteres especiales y normalizando

        ¿POR QUÉ es importante limpiar?
        - Los modelos de ML son sensibles a inconsistencias
        - "Movie" y "MOVIE" deben ser tratados igual
        - Los símbolos extraños confunden al modelo

        Args:
            text (str): Texto a limpiar

        Returns:
            str: Texto limpio
        """
        # 1. Convertir a minúsculas
        # ¿Por qué? Para que "Good" y "good" se traten igual
        text = text.lower()

        # 2. Eliminar HTML tags (a veces vienen en reviews web)
        # Ej: <br> <p> etc.
        text = re.sub(r'<[^>]+>', '', text)

        # 3. Eliminar URLs
        # Las URLs no aportan sentimiento: http://example.com
        text = re.sub(r'http\S+|www\S+', '', text)

        # 4. Eliminar menciones de usuarios y hashtags
        # @usuario #hashtag -> no relevantes para sentimiento
        text = re.sub(r'@\w+|#\w+', '', text)

        # 5. Mantener solo letras y espacios
        # Eliminamos números, puntuación, símbolos especiales
        # ¿Por qué? Simplifican el vocabulario sin perder mucho significado
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # 6. Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)

        # 7. Eliminar espacios al inicio/final
        text = text.strip()

        return text

    def tokenize(self, text):
        """
        Convierte texto en lista de palabras (tokens)

        ¿QUÉ es tokenización?
        - Separar texto en unidades mínimas (palabras)
        - "I love this" -> ["I", "love", "this"]

        ¿POR QUÉ necesitamos tokenizar?
        - Los modelos procesan palabras individuales
        - Permite contar frecuencias de palabras

        Args:
            text (str): Texto a tokenizar

        Returns:
            list: Lista de tokens (palabras)
        """
        # word_tokenize es más inteligente que split()
        # Maneja puntuación, contracciones, etc. mejor
        tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, tokens):
        """
        Elimina stopwords de la lista de tokens

        ¿POR QUÉ eliminar stopwords?
        - "the", "is", "at" aparecen en TODO texto
        - No ayudan a distinguir si una review es positiva o negativa
        - Reducen el tamaño del vocabulario (más eficiente)

        Ejemplo:
        ["this", "is", "a", "great", "movie"]
        -> ["great", "movie"]

        Args:
            tokens (list): Lista de tokens

        Returns:
            list: Tokens sin stopwords
        """
        # Filtramos tokens que NO están en la lista de stopwords
        filtered_tokens = [token for token in tokens
                          if token not in self.stop_words]
        return filtered_tokens

    def lemmatize(self, tokens):
        """
        Aplica lematización a los tokens

        ¿QUÉ es lemmatización?
        - Reducir palabras a su forma base/diccionario
        - "running", "runs", "ran" -> "run"
        - "better" -> "good"

        ¿POR QUÉ es útil?
        - Reduce vocabulario (menos features = más eficiente)
        - Agrupa palabras relacionadas
        - "loved", "loving", "loves" -> "love" (mismo sentimiento)

        Diferencia con STEMMING:
        - Stemming: corta sufijos ("running" -> "run")
        - Lemmatizing: usa diccionario ("better" -> "good")
        - Lemmatizing es más preciso pero más lento

        Args:
            tokens (list): Lista de tokens

        Returns:
            list: Tokens lematizados
        """
        # Aplicar lemmatization a cada token
        lemmatized_tokens = [self.lemmatizer.lemmatize(token)
                            for token in tokens]
        return lemmatized_tokens

    def preprocess(self, text):
        """
        Pipeline completo de preprocesamiento

        Este método aplica TODOS los pasos en orden:
        1. Limpiar
        2. Tokenizar
        3. Eliminar stopwords
        4. Lematizar
        5. Unir de nuevo en string

        ¿POR QUÉ en este orden?
        - Primero limpiamos caracteres extraños
        - Luego separamos en palabras
        - Filtramos palabras irrelevantes
        - Normalizamos a forma base

        Args:
            text (str): Texto original

        Returns:
            str: Texto preprocesado listo para vectorización
        """
        # Paso 1: Limpiar texto
        text = self.clean_text(text)

        # Paso 2: Tokenizar
        tokens = self.tokenize(text)

        # Paso 3: Eliminar stopwords
        tokens = self.remove_stopwords(tokens)

        # Paso 4: Lematizar
        tokens = self.lemmatize(tokens)

        # Paso 5: Unir tokens de nuevo en string
        # ¿Por qué? Porque TfidfVectorizer espera strings, no listas
        processed_text = ' '.join(tokens)

        return processed_text


def batch_preprocess(texts, language='english'):
    """
    Preprocesa múltiples textos de forma eficiente

    Esta función es útil para procesar datasets completos

    Args:
        texts (list): Lista de textos
        language (str): Idioma (por defecto 'english' para IMDB)

    Returns:
        list: Lista de textos preprocesados
    """
    preprocessor = TextPreprocessor(language=language)

    processed_texts = []
    for i, text in enumerate(texts):
        # Mostrar progreso cada 1000 textos
        if i % 1000 == 0:
            print(f"Procesados {i}/{len(texts)} textos...")

        processed = preprocessor.preprocess(text)
        processed_texts.append(processed)

    print(f"¡Preprocesamiento completado! Total: {len(processed_texts)} textos")
    return processed_texts


# ============================================================
# FUNCIÓN DE PRUEBA
# ============================================================

def main():
    """
    Función de prueba para ver el preprocesamiento en acción
    """
    print("=" * 60)
    print("DEMO: Preprocesamiento de Texto")
    print("=" * 60)

    # Texto de ejemplo (simulando una review de película)
    sample_text = """
    This movie was ABSOLUTELY AMAZING!!! I loved every minute of it.
    The acting was superb and the plot kept me on the edge of my seat.
    Best film I've seen in years! 10/10 would watch again!!!
    Check it out at http://example.com #mustwatch @director
    """

    print("\nTEXTO ORIGINAL:")
    print("-" * 60)
    print(sample_text)

    # Crear preprocessor
    preprocessor = TextPreprocessor()

    # Aplicar preprocesamiento
    processed = preprocessor.preprocess(sample_text)

    print("\nTEXTO PROCESADO:")
    print("-" * 60)
    print(processed)

    print("\n🔍 ANÁLISIS:")
    print("-" * 60)
    original_words = len(sample_text.split())
    processed_words = len(processed.split())
    print(f"Palabras originales: {original_words}")
    print(f"Palabras después del procesamiento: {processed_words}")
    print(f"Reducción: {100 * (1 - processed_words/original_words):.1f}%")

    print("\nEl texto está listo para ser vectorizado!")


if __name__ == "__main__":
    main()
