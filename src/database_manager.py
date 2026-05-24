import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import logging
import datetime

class DatabaseManager:
    def __init__(self, config: dict):
        db_cfg = config.get('database', {})
        self.host = db_cfg.get('host', 'tesis-db')
        self.port = db_cfg.get('port', 5432)
        self.dbname = db_cfg.get('name', 'agrotech_frost_db')
        self.user = db_cfg.get('user', 'saurin_admin')
        self.password = db_cfg.get('password', 'frost_password_2026')
        self.logger = logging.getLogger(__name__)

    def get_connection(self):
        """Establece una conexión directa con el Data Warehouse en PostgreSQL."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.dbname,
            user=self.user,
            password=self.password
        )

    def populate_pipeline(self, df_processed: pd.DataFrame, station_name: str = 'Villa Regina Centro'):
        """
        Toma el DataFrame ya preprocesado por tu pipeline de Python, 
        extrae los componentes temporales e ingesta todo en el Esquema Estrella.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 1. Obtener el ID de la estación semilla
            cursor.execute("SELECT pk_estacion_id FROM dim_estacion WHERE nombre_estacion = %s;", (station_name,))
            estacion_res = cursor.fetchone()
            if not estacion_res:
                raise ValueError(f"La estación '{station_name}' no existe en la base de datos.")
            fk_estacion_id = estacion_res[0]

            self.logger.info("Iniciando migración de datos al Data Warehouse...")

            # Aseguramos que el índice sea de tipo datetime por si acaso
            if not isinstance(df_processed.index, pd.DatetimeIndex):
                # Si 'fecha' es una columna, la usamos, sino intentamos convertir el índice
                if 'fecha' in df_processed.columns:
                    times = pd.to_datetime(df_processed['fecha'])
                else:
                    times = pd.to_datetime(df_processed.index)
            else:
                times = df_processed.index

            # --- 2. POBLAR DIM_TIEMPO (Sin duplicados) ---
            self.logger.info("Preparando datos para dim_tiempo...")
            time_rows = []
            seen_time_ids = set()

            for t in times:
                # Generamos la clave inteligente AAAAMMDDHHMM
                pk_tiempo_id = int(t.strftime('%Y%m%d%H%M'))
                
                if pk_tiempo_id in seen_time_ids:
                    continue
                seen_time_ids.add(pk_tiempo_id)

                is_weekend = t.weekday() >= 5
                is_critical = t.month in [6, 7, 8, 9] # Temporada de heladas tardías

                time_rows.append((
                    pk_tiempo_id, t, t.year, t.month, t.day, t.hour, t.minute,
                    t.weekday() + 1, is_weekend, is_critical
                ))

            # Inserción masiva ultra-rápida en la dimensión tiempo
            sql_time = """
                INSERT INTO dim_tiempo 
                (pk_tiempo_id, fecha_completa, anio, mes, dia, hora, minuto, dia_semana, es_fin_semana, es_temporada_critica)
                VALUES %s ON CONFLICT (pk_tiempo_id) DO NOTHING;
            """
            execute_values(cursor, sql_time, time_rows)
            self.logger.info(f"Dimensión Tiempo sincronizada ({len(time_rows)} registros procesados).")

            # --- 3. POBLAR FACT_MEDICIONES ---
            self.logger.info("Preparando datos para fact_mediciones...")
            fact_rows = []

            # Mapeo exacto de las columnas reales del DataFrame
            for idx, row in df_processed.iterrows():
                t = idx if isinstance(idx, datetime.datetime) else pd.to_datetime(idx)
                pk_tiempo_id = int(t.strftime('%Y%m%d%H%M'))

                # --- Conversión Segura permitiendo NULL en la variable objetivo ---
                temp_actual = float(row['temperature_min']) if pd.notna(row['temperature_min']) else None
                
                # Para la bandera de helada, si la temperatura es NULL, asumimos 0 (o manejalo según tu criterio)
                es_helada = 1 if (temp_actual is not None and temp_actual < 0.0) else 0

                # Columnas auxiliares seguras
                punto_rocio = float(row['dew_point']) if 'dew_point' in row and pd.notna(row['dew_point']) else None
                humedad = float(row['humidity']) if 'humidity' in row and pd.notna(row['humidity']) else None
                presion = float(row['pressure']) if 'pressure' in row and pd.notna(row['pressure']) else None
                viento = float(row['WindSpeed']) if 'WindSpeed' in row and pd.notna(row['WindSpeed']) else None
                radiacion = float(row['SolarRad.']) if 'SolarRad.' in row and pd.notna(row['SolarRad.']) else None

                fact_rows.append((
                    fk_estacion_id, pk_tiempo_id, temp_actual, punto_rocio,
                    humedad, presion, viento, radiacion, es_helada
                ))

            sql_fact = """
                INSERT INTO fact_mediciones 
                (fk_estacion_id, fk_tiempo_id, temperatura_actual, punto_rocio, humedad_relativa, presion_atmosferica, velocidad_viento, radiacion_solar, es_helada_actual)
                VALUES %s;
            """
            
            # Limpiamos mediciones previas para evitar duplicación masiva si re-corremos el script de carga
            cursor.execute("TRUNCATE TABLE fact_mediciones CASCADE;")
            
            execute_values(cursor, sql_fact, fact_rows)
            conn.commit()
            self.logger.info(f"Tabla de Hechos poblada con éxito ({len(fact_rows)} mediciones meteorológicas ingresadas).")

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error crítico durante la ingesta en el Data Warehouse: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

    def get_data_for_training(self, station_name: str = 'Villa Regina Centro') -> pd.DataFrame:
        """
        Hace el query dimensional (SELECT con JOIN), filtrando únicamente por la 
        temporada crítica (meses 6,7,8,9) para alimentar el pipeline de Machine Learning.
        """
        conn = self.get_connection()
        self.logger.info(f"Extrayendo datos históricos desde el Data Warehouse para: {station_name}")
        
        # Traemos los hechos cruzados con la dimensión tiempo filtrada por temporada crítica
        query = """
            SELECT 
                t.fecha_completa,
                f.temperatura_actual AS "LowTemp",
                f.punto_rocio AS "DewPoint",
                f.humedad_relativa AS "Hum",
                f.presion_atmosferica AS "Press",
                f.velocidad_viento AS "WindSpeed",
                f.radiacion_solar AS "SolarRad"
            FROM fact_mediciones f
            JOIN dim_tiempo t ON f.fk_tiempo_id = t.pk_tiempo_id
            JOIN dim_estacion e ON f.fk_estacion_id = e.pk_estacion_id
            WHERE e.nombre_estacion = %s AND t.es_temporada_critica = TRUE
            ORDER BY t.fecha_completa ASC;
        """
        
        try:
            # Usamos pandas para transformar el SQL directo a un DataFrame limpio con nulos
            df = pd.read_sql_query(query, conn, params=(station_name,), parse_dates=['fecha_completa'])
            df = df.set_index('fecha_completa')
            
            self.logger.info(f"Datos listos extraídos de la BD. Filas recuperadas: {len(df)}")
            return df
        except Exception as e:
            self.logger.error(f"Error al leer datos desde PostgreSQL: {e}")
            raise e
        finally:
            conn.close()