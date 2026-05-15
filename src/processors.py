import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging

class DataProcessor:
    def __init__(self, n_neighbors: int = 1):
        self.imputer = SimpleImputer(strategy='median')
        self.logger = logging.getLogger(__name__)

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        self.logger.info(f"Cargando datos desde {file_path}")
        return pd.read_csv(file_path, delimiter="\t", header=None, low_memory=False)

    def format_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        df_clean.iloc[0] = df_clean.iloc[0].fillna('')
        df_clean.columns = [str(c).strip() for c in (df_clean.iloc[0] + df_clean.iloc[1])]
        df_clean = df_clean.iloc[2:].reset_index(drop=True)
        return df_clean

    def process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df_dates = df.copy()
        date_col = df_dates.columns[0]
        df_dates[['dia', 'mes', 'anio']] = df_dates[date_col].str.split("/", expand=True)
        df_dates = df_dates.drop(columns=[date_col])
        return df_dates.apply(pd.to_numeric, errors='coerce')

    def filter_by_season(self, df: pd.DataFrame, months_to_keep: list) -> pd.DataFrame:
        self.logger.info(f"Filtrando meses críticos: {months_to_keep}")
        return df[df['mes'].isin(months_to_keep)].copy()
    
    def clean_station_data(self, df, station_type="regina"):
        self.logger.info(f"Aplicando selección de variables para: {station_type}")
        cols_to_keep = [
            'TempOut', 'HiTemp', 'LowTemp', 'OutHum', 'Dew', 
            'WindSpeed', 'WindChill', 'HeatIndex', 'Bar'
        ]
        available_cols = [c for c in cols_to_keep if c in df.columns]
        return df[available_cols].copy()

    def impute_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Iniciando imputación de datos faltantes...")
        cols_with_data = df.columns[df.notna().any()].tolist()
        if not cols_with_data: return df
        df_to_impute = df[cols_with_data]
        imputed_array = self.imputer.fit_transform(df_to_impute)
        return pd.DataFrame(imputed_array, columns=cols_with_data)

    def run_pipeline(self, raw_path: str, months: list, drops: list) -> pd.DataFrame:
        """
        Este es el método que Docker no estaba encontrando.
        """
        # 1. Carga y Headers
        df = self.load_raw_data(raw_path)
        df = self.format_headers(df)
        df = self.process_dates(df)
        
        # 2. Filtro de meses
        df = self.filter_by_season(df, months)
        
        # 3. Selección de variables físicas (Limpia las 30+ sobrantes)
        df = self.clean_station_data(df)
        
        # 4. Asegurar numéricos y eliminar basura
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 5. Imputación
        df = self.impute_data(df)
        
        # 6. Drop manual (opcional desde YAML)
        if drops:
            existing_drops = [c for c in drops if c in df.columns]
            df = df.drop(columns=existing_drops)

        self.logger.info(f"Pipeline completado. Columnas finales: {list(df.columns)}")
        return df