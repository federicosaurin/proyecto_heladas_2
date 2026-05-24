import pandas as pd
import numpy as np
import logging

class DataProcessor:
    def __init__(self, config: dict):
        prep_cfg = config.get('preprocessing', {})
        self.station_type = prep_cfg.get('station_type', 'regina')
        self.columns_to_drop = prep_cfg.get('columns_to_drop', ['dia', 'mes', 'anio'])
        self.logger = logging.getLogger(__name__)

    def clean_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Lee el archivo usando la lógica original de combinación de cabeceras dobles por TAB,
        extrae los componentes temporales y prepara el DataFrame.
        """
        self.logger.info(f"Cargando datos crudos desde {file_path}")
        
        # 1. Volvemos al delimitador de tabulación nativo que usabas antes
        # on_bad_lines='skip' es el escudo definitivo si alguna línea se deforma
        df_raw = pd.read_csv(
            file_path, 
            delimiter="\t", 
            header=None, 
            low_memory=False, 
            on_bad_lines='skip'
        )
        
        # 2. Tu lógica original para fusionar el encabezado doble
        df_raw.iloc[0] = df_raw.iloc[0].fillna('')
        df_raw.columns = [str(c).strip() for c in (df_raw.iloc[0].astype(str) + df_raw.iloc[1].astype(str))]
        
        # Cortamos las dos primeras filas de encabezado y reseteamos el índice
        df = df_raw.iloc[2:].reset_index(drop=True)
        
        self.logger.info(f"Columnas combinadas detectadas exitosamente: {list(df.columns[:5])}...")

        # 3. Procesar fechas basándonos en tus columnas combinadas
        try:
            # En tus logs vimos que las columnas se llaman exactamente "Date" y "Time"
            datetime_str = df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip()
            df['fecha'] = pd.to_datetime(datetime_str, format='%d/%m/%y %H:%M', errors='coerce')
            
            # Recreamos las columnas estructurales por si las usás en filtros
            df['dia'] = df['fecha'].dt.day
            df['mes'] = df['fecha'].dt.month
            df['anio'] = df['fecha'].dt.year
            
        except Exception as e:
            self.logger.error(f"Error al procesar el tiempo con las cabeceras unificadas: {e}")
            raise e

        # Limpieza de nulos temporales e indexación
        df = df.dropna(subset=['fecha'])
        df = df.set_index('fecha').sort_index()

        # 4. Convertir a numérico y mapear variables físicas
        for col in df.columns:
            if col not in ['Date', 'Time']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Mapeo exacto según los nombres fusionados de tu WeatherLink
        # "Hi" + "Temp" = "HiTemp", "Low" + "Temp" = "LowTemp", etc.
        rename_dict = {
            'HiTemp': 'temperature_max',
            'LowTemp': 'temperature_min',
            'OutHum': 'humidity',
            'Dew': 'dew_point',
            'Bar': 'pressure',
            'HiSolarRad': 'solar_radiation_max',
            'Rain': 'rain'
        }
        
        df = df.rename(columns=lambda x: rename_dict.get(x, x))
        df = df.replace([np.inf, -np.inf], np.nan)

        self.logger.info(f"Estructura unificada con éxito. Registros procesados: {len(df)}")
        return df