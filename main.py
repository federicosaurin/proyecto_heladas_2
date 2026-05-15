import logging
import os
import yaml
import pandas as pd
from src.processors import DataProcessor
from src.analyzer import DataAnalyzer
from src.visualizer import Visualizer
from src.model import FrostPredictor

# Configuración de logging profesional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open("config/parameters.yaml", 'r') as file:
        return yaml.safe_load(file)

def main():
    # 1. Carga de Configuración y Setup
    config = load_config()
    
    # Inicializamos componentes pasando el diccionario de configuración pertinente
    processor = DataProcessor()
    
    analyzer = DataAnalyzer(
        target_column=config['model']['target_column'],
        output_raw_dir=config['paths']['raw_results_dir']
    )
    
    visualizer = Visualizer(output_dir=config['paths']['figures_dir'])
    
    # El predictor ahora toma todo el bloque 'config' para acceder a 'model' y 'paths'
    predictor = FrostPredictor(config)
    
    try:
        # --- FASE 1: PROCESAMIENTO ---
        logger.info("--- Iniciando Fase de Procesamiento ---")
        df_clean = processor.run_pipeline(
            raw_path=config['paths']['raw_data'],
            months=config['preprocessing']['months_to_keep'],
            drops=config['preprocessing']['columns_to_drop']
        )
        
        # Guardamos el procesado por si quieres auditarlo
        os.makedirs(os.path.dirname(config['paths']['processed_output']), exist_ok=True)
        df_clean.to_csv(config['paths']['processed_output'])
        
        # --- FASE 2: ANÁLISIS ESTADÍSTICO ---
        logger.info("--- Iniciando Fase de Análisis Estadístico ---")
        # Correlación estática
        static_corr = analyzer.get_static_correlations(df_clean)
        visualizer.plot_static_correlations(static_corr)
        
        # Correlación dinámica (Shifting 1-24) y guardado automático de CSVs crudos
        dynamic_results = analyzer.calculate_dynamic_analysis(df_clean)
        visualizer.plot_dynamic_results(dynamic_results)

        # --- FASE 3: MODELADO ANN ---
        logger.info("--- Iniciando Fase de Entrenamiento ANN (Clasificación por Hora) ---")
        
        # Ejecutamos el entrenamiento masivo (Arquitecturas x Horas x Folds)
        # El método ya itera internamente sobre las arquitecturas del YAML
        df_results = predictor.train_hourly_models(df_clean)
        
        # Guardamos el reporte maestro de métricas
        predictor.save_results(df_results, config['paths']['results_csv'])
        
        # --- FASE 4: VISUALIZACIÓN DE MODELOS ---
        logger.info("--- Generando comparativas de rendimiento ---")
        
        # Preparamos un diccionario por arquitectura para que el visualizer compare
        # (Separamos los resultados por el campo 'Arch' que añadimos en model.py)
        results_map = {str(arch): df_results[df_results['Arch'] == str(arch)] 
                       for arch in config['model']['architectures']}
        
        # Si tu visualizer tiene estos métodos, graficarán las curvas de F1 y Especificidad
        try:
            visualizer.plot_model_comparison(results_map, metric='F1')
            visualizer.plot_model_comparison(results_map, metric='Specificity')
        except AttributeError:
            logger.warning("El visualizador no tiene implementado plot_model_comparison. Saltando...")

        logger.info("--- Pipeline ejecutado con éxito total ---")
        logger.info(f"Reporte final disponible en: {config['paths']['results_csv']}")

    except Exception as e:
        logger.error(f"Falla en el pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()