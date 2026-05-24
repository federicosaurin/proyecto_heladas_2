INSERT INTO dim_estacion (nombre_estacion, localidad, provincia, latitud, longitud)
VALUES ('Villa Regina Centro', 'Villa Regina', 'Río Negro', -39.1000, -67.0833)
ON CONFLICT (nombre_estacion) DO NOTHING;