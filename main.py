import os
import yaml
import logging
import pandas as pd  # 👈 AGREGADO ACÁ
from sklearn.impute import KNNImputer

from src.processors import DataProcessor
from src.database_manager import DatabaseManager
from src.model import FrostPredictor

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
    # Volcamos todo en Postgres (mantiene nulos)
    db_manager.populate_pipeline(df_raw, station_name='Villa Regina Centro')

    # 3. ETAPA B: Extraer de la Base de Datos para Machine Learning
    # De acá en adelante, tu pipeline ya no depende de archivos de texto locales!
    df_db = db_manager.get_data_for_training(station_name='Villa Regina Centro')

    if df_db.empty:
        logger.error("No se encontraron datos en la base de datos para entrenar el modelo. Abortando.")
        return

    # 4. APLICAR KNNIMPUTER EN CALIENTE (Sobre los nulos extraídos de la BD)
    logger.info("Aplicando algoritmo KNNImputer para rellenar nulos en memoria...")
    knn_neighbors = config['preprocessing'].get('knn_neighbors', 1)
    imputer = KNNImputer(n_neighbors=knn_neighbors)
    
    # Rellenamos nulos manteniendo la estructura de columnas y el índice
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_db),
        columns=df_db.columns,
        index=df_db.index
    )
    logger.info("Imputación completada con éxito. Dataset listo para entrenamiento.")

    # 5. ENTRENAMIENTO Y PERSISTENCIA DE MODELOS COMERCIALES
    predictor = FrostPredictor(config)
    metrics_report = predictor.train_hourly_models(df_imputed)

    # 6. Guardar Reporte General de Métricas
    predictor.save_results(metrics_report, config['paths']['results_csv'])
    logger.info("=== Proceso Finalizado Exitosamente ===")

if __name__ == "__main__":
    main()