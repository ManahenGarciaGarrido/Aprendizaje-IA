# Proyecto 4: Clasificador de Flores con Transfer Learning

## Descripción
Utiliza transfer learning con modelos pre-entrenados para clasificar especies de flores.

## Duración estimada: 6-8 horas

## Dataset
- **Oxford 102 Flower Dataset**
- 102 categorías de flores, 8,189 imágenes
- URL: `tensorflow_datasets` - `oxford_flowers102`

## Tecnologías
- TensorFlow/Keras
- Pre-trained models (MobileNet, VGG16, ResNet)
- PIL/OpenCV

## Pasos a Seguir

### Paso 1: Cargar Datos
- [ ] Descargar Oxford Flowers dataset
- [ ] Explorar clases y distribución
- [ ] Visualizar muestras

### Paso 2: Preprocesamiento
- [ ] Redimensionar imágenes
- [ ] Normalizar píxeles
- [ ] Data augmentation
- [ ] Dividir train/val/test

### Paso 3: Transfer Learning
- [ ] Cargar modelo pre-entrenado (MobileNet/VGG16)
- [ ] Congelar capas base
- [ ] Agregar capas custom
- [ ] Compilar modelo

### Paso 4: Fine-Tuning
- [ ] Descongelar algunas capas
- [ ] Entrenar con learning rate bajo
- [ ] Monitorear overfitting

### Paso 5: Evaluación
- [ ] Accuracy > 90%
- [ ] Comparar diferentes arquitecturas
- [ ] Visualizar activaciones

### Paso 6: Desafíos
- [ ] Probar 3 arquitecturas diferentes
- [ ] Implementar ensemble learning
- [ ] Crear interfaz de predicción
