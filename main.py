import os
import yaml
import logging
import pandas as pd
from sklearn.impute import KNNImputer

# Módulos del pipeline original
from src.processors import DataProcessor
from src.database_manager import DatabaseManager
from src.model import FrostPredictor

# Módulos de soporte analítico incorporados 🚀
from src.analyzer import DataAnalyzer
from src.visualizer import Visualizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== Iniciando Plataforma Comercial AgroTech ===")

    # 1. Cargar Configuración del YAML
    config_path = os.path.join("config", "parameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. ETAPA A: Procesar TXT crudo e ingestar en el Data Warehouse
    processor = DataProcessor(config)
    df_raw = processor.clean_raw_data(config['paths']['raw_data'])
    
    db_manager = DatabaseManager(config)
    # Volcamos todo en Postgres (mantiene nulos históricos de forma segura)
    db_manager.populate_pipeline(df_raw, station_name='Villa Regina Centro')

    # 3. ETAPA B: Extraer de la Base de Datos para Machine Learning
    # ¡De acá en adelante, el pipeline ya no depende de archivos locales!
    df_db = db_manager.get_data_for_training(station_name='Villa Regina Centro')

    if df_db.empty:
        logger.error("No se encontraron datos en la base de datos para entrenar el modelo. Abortando.")
        return

    # 4. APLICAR KNNIMPUTER EN CALIENTE (Sobre los nulos extraídos de la BD)
    logger.info("Aplicando algoritmo KNNImputer para rellenar nulos en memoria...")
    knn_neighbors = config['preprocessing'].get('knn_neighbors', 1)
    imputer = KNNImputer(n_neighbors=knn_neighbors)
    
    # Rellenamos nulos manteniendo la estructura de columnas y el índice temporal
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_db),
        columns=df_db.columns,
        index=df_db.index
    )
    logger.info("Imputación completada con éxito. Dataset listo para análisis y entrenamiento.")

    # 5. ANÁLISIS ESTADÍSTICO EXPLORATORIO Y VISUALIZACIÓN AUTOMATIZADA 📊
    logger.info("Iniciando fase analítica intermedia sobre el dataset imputado...")
    
    # Recuperamos rutas dinámicas desde el YAML o usamos los defaults de tus módulos
    target_col = config['model'].get('target_column', 'LowTemp')
    raw_results_dir = config['paths'].get('raw_results_dir', 'reports/raw_results')
    figures_dir = config['paths'].get('figures_dir', 'reports/figures')
    
    # Instanciamos y corremos el analizador (guarda CSVs internamente en reports/raw_results)
    analyzer = DataAnalyzer(target_column=target_col, output_raw_dir=raw_results_dir)
    static_corr = analyzer.get_static_correlations(df_imputed)
    dynamic_corr_results = analyzer.calculate_dynamic_analysis(df_imputed)
    
    # Instanciamos y corremos el visualizador para actualizar gráficos (.png)
    visualizer = Visualizer(output_dir=figures_dir)
    visualizer.plot_static_correlations(static_corr)
    visualizer.plot_dynamic_results(dynamic_corr_results)
    
    logger.info("Fase analítica finalizada con éxito. Reportes y gráficos actualizados.")

    # 6. ENTRENAMIENTO Y PERSISTENCIA DE MODELOS COMERCIALES (MLP)
    predictor = FrostPredictor(config)
    metrics_report = predictor.train_hourly_models(df_imputed)

    # 7. Guardar Reporte General de Métricas de rendimiento
    results_csv_path = config['paths'].get('results_csv', 'reports/results_metrics.csv')
    predictor.save_results(metrics_report, results_csv_path)
    
    logger.info("=== Pipeline AgroTech finalizado exitosamente ===")

if __name__ == "__main__":
    main()