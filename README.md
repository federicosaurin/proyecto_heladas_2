# ❄️ Plataforma AgroTech: Pipeline End-to-End para la Predicción de Heladas Tardías

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15--Alpine-blue)](https://www.postgresql.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-MLOps-orange)](https://scikit-learn.org/)

---

### 🎯 El Problema que Resuelve
En la región del **Alto Valle (Río Negro, Argentina)**, las heladas tardías primaverales son un enemigo silencioso pero devastador para la producción de fruta de pepita (peras y manzanas). Un descenso brusco e imprevisto de la temperatura a niveles críticos puede destruir cosechas enteras en unas pocas horas. El control activo de heladas (como el riego por aspersión o los quemadores) requiere mucha energía y recursos, por lo que **cada hora de anticipación vale oro**.

Esta plataforma resuelve de raíz la falta de herramientas de predicción localizadas de corto plazo. Convierte datos meteorológicos crudos en un **sistema de alerta temprana automatizado**, capaz de predecir la probabilidad exacta de heladas con hasta **24 horas de antelación** (*Lead Times* de 0 a 23h), permitiendo a los productores tomar decisiones defensivas críticas con un margen de acción real y respaldado por datos.

---

## 🚀 De la Academia a la Industria (MLOps)

Este producto representa la **industrialización y pase a producción** de mi tesis de grado en colaboración con el **INTA**. El sistema original fue completamente migrado desde entornos de experimentación monolíticos y volátiles (Google Colab) hacia una arquitectura de software empresarial: robusta, modular, tolerante a fallos e instalable con un solo comando.

## 🏗️ Arquitectura de la Solución (Clean Code)

El ecosistema está desacoplado bajo principios **SOLID**, aislando las responsabilidades de ingeniería de datos de las del modelado matemático:

```text
├── config/              # Parámetros del experimento controlados por YAML (sin tocar código)
├── database/            # Scripts de inicialización DDL y DML (Seeds) del Data Warehouse
├── data/                # Volúmenes locales para datos crudos e intermedios (Ignorados en Git)
├── models/              # Artefactos binarios (.joblib) serializados listos para producción
├── reports/             # Reportes analíticos y grillas de correlación autogeneradas
│   ├── figures/         # Gráficos de salida (Estáticos y Dinámicos por Lag)
│   └── raw_results/     # Tablas estadísticas en formato CSV para auditoría climática
├── src/                 # La cocina del proyecto
│   ├── processors.py    # ETL, parseo de cabeceras dobles de WeatherLink y limpieza de ruido
│   ├── database_manager.py # Capa de persistencia y abstracción relacional (PostgreSQL)
│   ├── analyzer.py      # Motor estadístico inferencial (Pearson, Spearman, Kendall)
│   ├── visualizer.py    # Automatización de reportería gráfica corporativa
│   └── model.py         # Pipeline de entrenamiento, validación y MLOps (ANN)
├── main.py              # Orquestador del Pipeline Principal
├── Dockerfile           # Receta de la capa de aplicación Python
└── docker-compose.yml   # Orquestador de la infraestructura (App + Base de Datos)
