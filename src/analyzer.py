import pandas as pd
import numpy as np
import logging
import os

class DataAnalyzer:
    def __init__(self, target_column='LowTemp', output_raw_dir="reports/raw_results"):
        self.target = target_column
        self.logger = logging.getLogger(__name__)
        self.output_raw_dir = output_raw_dir
        # Creamos la carpeta de resultados crudos si no existe
        os.makedirs(self.output_raw_dir, exist_ok=True)

    def get_static_correlations(self, df):
        """
        Calcula la matriz de correlación estática estándar.
        """
        self.logger.info("Calculando correlaciones estáticas...")
        return df.corr()

    def calculate_dynamic_analysis(self, df):
        """
        Calcula la evolución temporal de correlaciones (Lags 1-24h)
        y guarda los resultados en archivos CSV.
        """
        self.logger.info("Calculando evolución temporal de correlaciones (Lags 1-24h)...")
        
        # Aseguramos que solo trabajamos con datos numéricos
        df_numeric = df.select_dtypes(include=[np.number])
        
        lags = range(1, 25)
        variables = [c for c in df_numeric.columns if c != self.target]
        
        # Inicializamos los DataFrames de resultados con la columna 'Hora'
        results = {
            'Pearson': pd.DataFrame({'Hora': list(lags)}),
            'Spearman': pd.DataFrame({'Hora': list(lags)}),
            'Kendall': pd.DataFrame({'Hora': list(lags)})
        }

        for var in variables:
            p_list, s_list, k_list = [], [], []
            for lag in lags:
                # Creamos el desfase (shift) temporal
                # Desplazamos la variable independiente hacia atrás para predecir el target
                temp_df = df_numeric[[var, self.target]].copy()
                temp_df[var] = temp_df[var].shift(lag)
                temp_df = temp_df.dropna()
                
                # Calculamos correlaciones si hay suficientes datos
                if not temp_df.empty:
                    p_list.append(temp_df.corr(method='pearson').iloc[0, 1])
                    s_list.append(temp_df.corr(method='spearman').iloc[0, 1])
                    k_list.append(temp_df.corr(method='kendall').iloc[0, 1])
                else:
                    p_list.append(np.nan)
                    s_list.append(np.nan)
                    k_list.append(np.nan)
            
            results['Pearson'][var] = p_list
            results['Spearman'][var] = s_list
            results['Kendall'][var] = k_list

        # --- FASE DE GUARDADO AUTOMÁTICO ---
        self.save_raw_results(results)

        return results

    def save_raw_results(self, results):
        """
        Guarda los diccionarios de resultados en archivos CSV individuales.
        """
        try:
            for metodo, data in results.items():
                file_name = f"resultados_{metodo.lower()}.csv"
                full_path = os.path.join(self.output_raw_dir, file_name)
                
                # Guardamos el formato estándar (Hora como fila)
                data.to_csv(full_path, index=False)
                
                # Opcional: Guardamos una versión transpuesta (Invertida) 
                # que suele ser útil para las tablas de la tesis
                inverted_path = os.path.join(self.output_raw_dir, f"tabla_tesis_{metodo.lower()}.csv")
                data.set_index('Hora').T.to_csv(inverted_path)
                
                self.logger.info(f"Datos crudos de {metodo} guardados en: {full_path}")
        except Exception as e:
            self.logger.error(f"Error al guardar los datos crudos del análisis: {e}")