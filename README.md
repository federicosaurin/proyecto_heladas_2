# ❄️ Predicción de Heladas Tardías mediante Redes Neuronales

Este proyecto implementa un pipeline industrializado de **Data Engineering** y **Machine Learning** para predecir eventos de heladas en la región del Alto Valle (Villa Regina/Guerrico), Argentina. La arquitectura permite predecir con hasta 24 horas de antelación (lead time) utilizando modelos de aprendizaje profundo.

## 🚀 Características Principales
- **Arquitectura Modular**: Código desacoplado en procesadores, analizadores y modelos.
- **Configuración Dinámica**: Control total del experimento mediante archivos `YAML` (sin tocar el código).
- **Contenerización**: Entorno reproducible al 100% mediante **Docker** y **Docker Compose**.
- **Análisis Multivariable**: Correlaciones estáticas y dinámicas para entender el comportamiento climático.

## 🛠️ Tecnologías Utilizadas
- **Python 3.10**
- **Scikit-learn**: MLPClassifier y métricas de rendimiento.
- **Pandas/NumPy**: Procesamiento de series temporales.
- **Seaborn/Matplotlib**: Visualización de datos y comparativa de arquitecturas.
- **Docker**: Despliegue y portabilidad.

## 📋 Estructura del Proyecto
```text
├── config/             # Parámetros del modelo (YAML)
├── data/               # Datos crudos y procesados
├── src/                # Lógica central (Procesamiento, Análisis, ANN)
├── reports/            # Gráficos y métricas finales (Generado por el modelo)
├── main.py             # Orquestador del pipeline
└── Dockerfile          # Configuración del contenedor

## ⚙️ Pipeline de Datos y Entrenamiento
El sistema está diseñado para operar de forma autónoma siguiendo estas etapas:
1. **Ingesta y Limpieza**: Procesamiento de **+500k registros** meteorológicos con imputación de datos faltantes (KNNImputer).
2. **Ingeniería de Variables**: Análisis de multicolinealidad y cálculo de correlaciones estáticas/dinámicas (Pearson, Spearman, Kendall).
3. **Entrenamiento Predictivo**: Ejecución de una ventana deslizante que entrena y valida el modelo para cada **Lead Time (0-23h)**, permitiendo un seguimiento real de la probabilidad de helada con un día de antelación.
4. **Validación**: Implementación de validación cruzada para asegurar la estabilidad del modelo frente a la variabilidad climática.

## 📦 Ejecución rápida (Docker)
Para replicar el experimento completo con todas sus dependencias:
```bash
docker compose up --build