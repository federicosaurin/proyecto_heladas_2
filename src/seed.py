import logging
from sqlalchemy import text
# Reemplazá 'src.database_manager' por la ruta real de tu manejador
from src.database_manager import DatabaseManager 
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_seeding():
    logger.info("Iniciando el proceso de Seeding del Data Warehouse...")
    
    # Cargar configuración para la base de datos
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    db_manager = DatabaseManager(config['database'])
    
    # Definimos las estaciones maestras de tu tesis
    stations_to_seed = [
        {
            "id": 1, # O dejar que lo maneje el SERIAL si tu tabla está estructurada así
            "name": "Villa Regina Centro",
            "location": "Villa Regina, Río Negro",
            "latitude": -39.1000,
            "longitude": -67.0833,
            "station_type": "regina"
        },
        # Podés agregar acá más estaciones si tu tesis analiza otras zonas
    ]
    
    # Query SQL con soporte para no duplicar si ya existen (Idempotencia)
    query = """
        INSERT INTO stations (id, name, location, latitude, longitude, station_type)
        VALUES (:id, :name, :location, :latitude, :longitude, :station_type)
        ON CONFLICT (id) DO NOTHING;
    """
    
    try:
        with db_manager.engine.connect() as conn:
            for station in stations_to_seed:
                conn.execute(text(query), station)
            conn.commit()
        logger.info("¡Seeding completado con éxito! Estaciones maestras cargadas.")
    except Exception as e:
        logger.error(f"Error crítico durante el seeding: {e}")
        raise e

if __name__ == "__main__":
    run_seeding()