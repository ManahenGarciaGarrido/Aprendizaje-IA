# Proyecto 5: Chatbot de Preguntas Frecuentes (FAQ)

## Descripción
Construye un chatbot basado en similitud de texto que responda preguntas frecuentes.

## Duración estimada: 6-8 horas

## Dataset
- **Customer Support Dataset**
- Kaggle: "Customer Support on Twitter"
- Alternativa: Crear dataset propio de FAQs

## Tecnologías
- NLTK/spaCy
- Sentence-BERT para embeddings
- FAISS para búsqueda de similitud
- Streamlit para interfaz

## Pasos a Seguir

### Paso 1: Preparar Dataset de FAQs
- [ ] Crear/cargar preguntas y respuestas
- [ ] Limpiar y preprocesar texto
- [ ] Generar embeddings

### Paso 2: Sistema de Similitud
- [ ] Implementar cosine similarity
- [ ] Usar FAISS para búsqueda eficiente
- [ ] Establecer threshold de confianza

### Paso 3: Lógica del Chatbot
- [ ] Detectar intención
- [ ] Encontrar respuesta más similar
- [ ] Manejar casos sin match

### Paso 4: Interfaz
- [ ] Crear UI con Streamlit
- [ ] Logging de conversaciones
- [ ] Manejo de errores

### Paso 5: Mejoras
- [ ] Corrección ortográfica
- [ ] Respuestas contextuales
- [ ] Análisis de sentimiento
