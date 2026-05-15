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
    processor = DataProcessor()
    analyzer = DataAnalyzer(target_column=config['model']['target_column'])
    visualizer = Visualizer(output_dir=config['paths']['figures_dir'])
    
    try:
        # --- FASE 1: PROCESAMIENTO ---
        logger.info("--- Iniciando Fase de Procesamiento ---")
        df_clean = processor.run_pipeline(
            raw_path=config['paths']['raw_data'],
            months=config['preprocessing']['months_to_keep'],
            drops=config['preprocessing']['columns_to_drop']
        )
        
        # --- FASE 2: ANÁLISIS ESTADÍSTICO ---
        logger.info("--- Iniciando Fase de Análisis Estadístico ---")
        static_corr = analyzer.get_static_correlations(df_clean)
        visualizer.plot_static_correlations(static_corr)
        
        dynamic_results = analyzer.calculate_dynamic_analysis(df_clean)
        visualizer.plot_dynamic_results(dynamic_results)

        # --- FASE 3: MODELADO ANN (COMENTADO PARA VALIDACIÓN RÁPIDA) ---
        # Si deseas ejecutarlo, descomenta las líneas de abajo.
        """
        logger.info("--- Iniciando Fase de Entrenamiento ANN (24h Lead Time) ---")
        results_map = {}
        
        for arch in config['model']['architectures']:
            arch_label = f"ANN_{'_'.join(map(str, arch))}"
            logger.info(f"Entrenando arquitectura: {arch_label}")
            
            # Configuramos el predictor con la arquitectura actual
            model_cfg = config['model'].copy()
            model_cfg['hidden_units'] = tuple(arch)
            
            predictor = FrostPredictor(model_cfg)
            df_results = predictor.train_hourly_models(df_clean)
            results_map[arch_label] = df_results
            
        # Generación de Reportes Comparativos de Modelos
        visualizer.plot_model_comparison(results_map, metric='F1')
        visualizer.plot_model_comparison(results_map, metric='Specificity')
        
        # Exportación de resultados maestros
        final_report = pd.concat(results_map.values(), keys=results_map.keys())
        final_report.to_csv(config['paths']['results_csv'])
        """

        logger.info("--- Pipeline ejecutado con éxito ---")

    except Exception as e:
        logger.error(f"Falla en el pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()