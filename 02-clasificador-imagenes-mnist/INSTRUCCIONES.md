# INSTRUCCIONES RÁPIDAS - Proyecto 2: Clasificador MNIST

## ¿Dónde ejecutar?

**TU PORTÁTIL es SUFICIENTE** (GPU opcional)
- Los modelos son relativamente ligeros
- Tiempo estimado: 10-30 minutos (dependiendo de hardware)
- Requiere: ~4GB RAM, ~1GB disco
- Con GPU: 5-10 minutos de entrenamiento
- Sin GPU: 20-30 minutos de entrenamiento

**Alternativa:** Google Colab (recomendado si no tienes GPU)

---

## Instalación y Ejecución

### Paso 1: Instalar dependencias

```bash
# Navegar a la carpeta del proyecto
cd 02-clasificador-imagenes-mnist

# Instalar paquetes necesarios
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

**Nota:** TensorFlow puede tardar algunos minutos en instalarse (~500MB)

### Paso 2: Entrenar el modelo

```bash
# Desde la carpeta src/
cd src
python train.py
```

**¿Qué hace este script?**
- Descarga dataset MNIST (60k imágenes de entrenamiento + 10k de test, ~11MB) - solo la 1ª vez
- Preprocesa imágenes (normalización, reshape)
- Entrena modelo CNN (Red Neuronal Convolucional)
- Guarda el mejor modelo en `models/`
- Genera visualizaciones (curvas de entrenamiento, matriz de confusión)

**Tiempo:**
- Con GPU: 5-10 minutos
- Sin GPU: 20-30 minutos
- Resultado esperado: >97% accuracy

### Paso 3: Probar predicciones

```bash
python predict.py
```

**¿Qué hace?**
- Carga el modelo entrenado
- Muestra ejemplos de predicción con probabilidades
- Analiza errores del modelo
- Permite modo interactivo para probar imágenes aleatorias

---

## Opciones de uso

### Opción A: Scripts Python (Recomendado para aprender)

```bash
cd src/

# 1. Probar arquitecturas de modelos
python model.py

# 2. Entrenar modelo completo
python train.py

# 3. Hacer predicciones
python predict.py

# 4. Probar funciones de visualización
python visualize.py
```

### Opción B: Notebook Jupyter

```bash
# Instalar Jupyter si no lo tienes
pip install jupyter

# Abrir notebook
jupyter notebook notebook.ipynb
```

---

## Configuración personalizada

### Cambiar tipo de modelo

Edita `train.py` línea 209:
```python
# Opciones: 'dense' (simple) o 'cnn' (recomendado)
MODEL_TYPE = 'cnn'
```

Para usar CNN avanzada (>99% accuracy), edita línea 220:
```python
# Descomentar esta línea:
model = create_advanced_cnn_model()
```

### Ajustar hiperparámetros

En `train.py`:

```python
# Número de epochs (línea 230)
epochs=20  # Aumentar a 30-50 para mejor accuracy

# Batch size (línea 231)
batch_size=128  # Reducir a 64 si tienes poca RAM

# Learning rate (al compilar modelo)
learning_rate=0.001  # Reducir si el training es inestable
```

### Visualizar filtros convolucionales

En `predict.py`, añade al final del main():

```python
from visualize import visualize_filters
visualize_filters(predictor.model, layer_name='conv2d_1')
```

---

## Solución de problemas

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Error: Out of Memory (OOM)
- Reduce el batch_size a 64 o 32 en `train.py`
- Usa modelo denso en vez de CNN (más ligero)
- Cierra otras aplicaciones

### Training muy lento sin GPU
- Considera usar Google Colab (GPU gratis)
- Reduce epochs a 10
- Usa modelo denso simple

### Error: "Model file not found"
```bash
# Asegúrate de haber entrenado primero
cd src
python train.py
```

### Accuracy muy baja (<90%)
- Verifica que los datos se normalizaron correctamente
- Aumenta el número de epochs
- Usa modelo CNN en vez de denso
- Intenta con CNN avanzada

---

## Resultados esperados

**Accuracy esperado:**
- Modelo Denso: ~95-96%
- Modelo CNN básico: ~98-99%
- Modelo CNN avanzado: >99%

**Señales de buen entrenamiento:**
- Loss bajando consistentemente
- Validation accuracy cercana a training accuracy
- Pocas confusiones en la matriz de confusión

**Si obtienes menor accuracy:**
- Aumenta epochs a 30-50
- Usa modelo CNN avanzado
- Verifica que el preprocesamiento es correcto
- Revisa los logs por errores

---

## Estructura del proyecto

```
02-clasificador-imagenes-mnist/
├── README.md                 # Descripción general del proyecto
├── INSTRUCCIONES.md         # Este archivo
├── notebook.ipynb           # Jupyter Notebook
├── models/                  # Modelos entrenados (generado)
│   ├── mnist_model_best.h5      # Mejor modelo durante training
│   ├── mnist_model_final.h5     # Modelo final
│   └── mnist_saved_model/       # Formato SavedModel
└── src/
    ├── model.py            # Definición de arquitecturas
    ├── train.py            # Script de entrenamiento
    ├── predict.py          # Script de predicción
    └── visualize.py        # Funciones de visualización
```

---

## Próximos pasos

1. **Ejecuta los scripts y entiende cada paso**
   - Lee los comentarios en el código
   - Experimenta cambiando hiperparámetros

2. **Experimenta con el modelo:**
   - Prueba diferentes arquitecturas
   - Ajusta número de capas y filtros
   - Añade Batch Normalization
   - Prueba Data Augmentation

3. **Análisis profundo:**
   - Estudia qué dígitos se confunden más
   - Visualiza los filtros aprendidos
   - Analiza errores específicos

4. **Desafíos adicionales:**
   - Implementa Data Augmentation (rotaciones, traslaciones)
   - Exporta modelo para TensorFlow Lite
   - Crea interfaz gráfica para dibujar dígitos
   - Prueba con tus propios dígitos escritos a mano

5. **Pasa al Proyecto 3**

---

## Conceptos clave aprendidos

- **Redes Neuronales Densas:** Fully connected layers
- **Redes Convolucionales (CNN):** Filtros, pooling, arquitectura
- **Activaciones:** ReLU, Softmax
- **Regularización:** Dropout, L2
- **Optimización:** Adam optimizer, learning rate
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Evaluación:** Accuracy, Loss, Matriz de Confusión
- **Visualización:** Curvas de entrenamiento, predicciones, filtros

---

## Recursos adicionales

- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [Understanding CNNs](https://cs231n.github.io/convolutional-networks/)

---

¿Dudas? Revisa los comentarios en el código - cada línea está explicada detalladamente!
