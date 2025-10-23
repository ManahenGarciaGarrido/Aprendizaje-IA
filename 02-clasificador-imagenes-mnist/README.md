# Proyecto 2: Clasificador de Imágenes MNIST

## Descripción
Crea tu primera red neuronal para reconocer dígitos escritos a mano. Este es el "Hello World" del deep learning y te introducirá a las redes neuronales convolucionales.

## Duración estimada
4-6 horas

## Dataset
- **MNIST Handwritten Digits**
- 70,000 imágenes de dígitos (0-9) en escala de grises de 28x28 píxeles
- URL: `tensorflow.keras.datasets.mnist`

## Tecnologías
- TensorFlow/Keras
- NumPy
- Matplotlib

## Objetivos de Aprendizaje
- Arquitectura básica de redes neuronales
- Conceptos de capas densas y activaciones
- Entrenamiento y validación
- Visualización de predicciones

## Métricas de Éxito
- Accuracy > 97%
- Entender el proceso de backpropagation
- Visualizar filtros aprendidos

## Estructura del Proyecto
```
02-clasificador-imagenes-mnist/
├── README.md
├── notebook.ipynb
├── data/                    # Datos descargados automáticamente
├── models/                  # Modelos guardados (.h5)
└── src/
    ├── model.py            # Arquitectura del modelo
    ├── train.py            # Script de entrenamiento
    ├── predict.py          # Script de predicción
    └── visualize.py        # Funciones de visualización
```

## Pasos a Seguir

### Paso 1: Cargar y Explorar los Datos
- [ ] Cargar MNIST desde Keras
- [ ] Visualizar muestras de cada dígito
- [ ] Verificar forma de los datos
- [ ] Analizar distribución de clases

### Paso 2: Preprocesamiento
- [ ] Normalizar píxeles (0-1)
- [ ] Reshape si es necesario
- [ ] Convertir etiquetas a one-hot encoding
- [ ] Dividir train/validation/test

### Paso 3: Construcción del Modelo
- [ ] Crear modelo secuencial simple
- [ ] Agregar capas densas
- [ ] Probar con capas convolucionales
- [ ] Compilar modelo

### Paso 4: Entrenamiento
- [ ] Configurar callbacks (EarlyStopping, ModelCheckpoint)
- [ ] Entrenar modelo
- [ ] Visualizar curvas de learning
- [ ] Prevenir overfitting

### Paso 5: Evaluación
- [ ] Evaluar en test set
- [ ] Crear matriz de confusión
- [ ] Analizar errores
- [ ] Visualizar predicciones

### Paso 6: Mejoras
- [ ] Data augmentation
- [ ] Diferentes arquitecturas
- [ ] Optimización de hiperparámetros
- [ ] Ensemble de modelos

### Paso 7: Exportar Modelo
- [ ] Guardar modelo entrenado
- [ ] Guardar pesos
- [ ] Documentar arquitectura

## Desafíos Adicionales
- Implementar Data Augmentation
- Probar diferentes arquitecturas (CNN profundas)
- Exportar el modelo para TensorFlow Lite
- Crear una interfaz para dibujar dígitos

## Recursos Útiles
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
