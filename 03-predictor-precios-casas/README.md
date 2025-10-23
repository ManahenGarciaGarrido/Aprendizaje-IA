# Proyecto 3: Predictor de Precios de Casas

## Descripción
Desarrolla un modelo de regresión para predecir precios de viviendas basándote en características como tamaño, ubicación y número de habitaciones.

## Duración estimada
5-7 horas

## Dataset
- **California Housing Dataset**
- Incluido en scikit-learn
- 20,640 muestras con 8 características
- URL: `sklearn.datasets.fetch_california_housing()`

## Tecnologías
- Scikit-learn
- Pandas
- Seaborn
- XGBoost

## Objetivos de Aprendizaje
- Análisis exploratorio de datos (EDA)
- Feature engineering y selección
- Modelos de regresión (Linear, Ridge, Lasso, Random Forest)
- Validación cruzada
- Optimización de hiperparámetros

## Métricas de Éxito
- R² Score > 0.80
- RMSE optimizado
- Identificar features más importantes

## Pasos a Seguir

### Paso 1: EDA (Análisis Exploratorio)
- [ ] Cargar dataset
- [ ] Análisis estadístico descriptivo
- [ ] Visualizar distribuciones
- [ ] Analizar correlaciones
- [ ] Detectar outliers

### Paso 2: Feature Engineering
- [ ] Crear nuevas features
- [ ] Normalización/Estandarización
- [ ] Encoding de variables categóricas
- [ ] Manejo de outliers

### Paso 3: Modelado
- [ ] Linear Regression baseline
- [ ] Ridge y Lasso Regression
- [ ] Random Forest
- [ ] XGBoost

### Paso 4: Evaluación y Optimización
- [ ] Validación cruzada
- [ ] Grid Search/Random Search
- [ ] Feature importance
- [ ] Comparar modelos

### Paso 5: Análisis Final
- [ ] Residual analysis
- [ ] Visualizar predicciones vs reales
- [ ] Interpretar resultados
- [ ] Guardar mejor modelo
