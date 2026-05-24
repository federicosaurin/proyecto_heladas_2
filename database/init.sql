CREATE TABLE dim_estacion (
    pk_estacion_id SERIAL PRIMARY KEY,
    nombre_estacion VARCHAR(100) NOT NULL UNIQUE,
    localidad VARCHAR(100) NOT NULL,
    provincia VARCHAR(100) NOT NULL,
    latitud NUMERIC(9,6),
    longitud NUMERIC(9,6)
);

CREATE TABLE dim_tiempo (
    pk_tiempo_id BIGINT PRIMARY KEY, -- formato AAAAMMDDHHMM
    
    fecha_completa TIMESTAMP NOT NULL,
    
    anio INT NOT NULL,
    mes INT NOT NULL,
    dia INT NOT NULL,
    hora INT NOT NULL,
    minuto INT NOT NULL,
    
    dia_semana INT NOT NULL, -- 1 (lunes) a 7 (domingo)
    es_fin_semana BOOLEAN NOT NULL,
    
    es_temporada_critica BOOLEAN NOT NULL
);

CREATE TABLE dim_modelo (
    pk_modelo_id SERIAL PRIMARY KEY,
    
    nombre_modelo VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    
    algoritmo VARCHAR(50), -- XGBoost, RandomForest, etc.
    
    fecha_entrenamiento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_entrenamiento VARCHAR(100),
    
    features_utilizadas TEXT,
    
    UNIQUE (nombre_modelo, version)
);

CREATE TABLE fact_mediciones (
    pk_medicion_id BIGSERIAL PRIMARY KEY,
    
    fk_estacion_id INT NOT NULL,
    fk_tiempo_id BIGINT NOT NULL,
    
    -- Variables meteorológicas
    temperatura_actual NUMERIC(4,1) NULL,
    punto_rocio NUMERIC(4,1),
    humedad_relativa NUMERIC(4,1),
    presion_atmosferica NUMERIC(6,1),
    velocidad_viento NUMERIC(4,1),
    radiacion_solar NUMERIC(6,1),
    
    -- Target real
    es_helada_actual INT NOT NULL,
    
    FOREIGN KEY (fk_estacion_id) REFERENCES dim_estacion(pk_estacion_id),
    FOREIGN KEY (fk_tiempo_id) REFERENCES dim_tiempo(pk_tiempo_id)
);

CREATE TABLE fact_predicciones (
    pk_prediccion_id BIGSERIAL PRIMARY KEY,
    
    fk_estacion_id INT NOT NULL,
    fk_tiempo_id BIGINT NOT NULL,
    fk_modelo_id INT NOT NULL,
    
    probabilidad_helada NUMERIC(5,4),
    prediccion_helada INT,
    
    fecha_prediccion TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (fk_estacion_id) REFERENCES dim_estacion(pk_estacion_id),
    FOREIGN KEY (fk_tiempo_id) REFERENCES dim_tiempo(pk_tiempo_id),
    FOREIGN KEY (fk_modelo_id) REFERENCES dim_modelo(pk_modelo_id)
);

CREATE TABLE fact_metricas_modelo (
    pk_metrica_id BIGSERIAL PRIMARY KEY,
    
    fk_modelo_id INT NOT NULL,
    
    accuracy NUMERIC(5,4),
    precision NUMERIC(5,4),
    recall NUMERIC(5,4),
    f1_score NUMERIC(5,4),
    specificity NUMERIC(5,4),
    
    -- Matriz de confusión
    true_positives INT,
    true_negatives INT,
    false_positives INT,
    false_negatives INT,
    
    fecha_evaluacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (fk_modelo_id) REFERENCES dim_modelo(pk_modelo_id)
);

-- Fact mediciones
CREATE INDEX idx_fact_mediciones_tiempo 
ON fact_mediciones(fk_tiempo_id);

CREATE INDEX idx_fact_mediciones_estacion 
ON fact_mediciones(fk_estacion_id);

-- Fact predicciones
CREATE INDEX idx_fact_predicciones_tiempo 
ON fact_predicciones(fk_tiempo_id);

CREATE INDEX idx_fact_predicciones_modelo 
ON fact_predicciones(fk_modelo_id);

-- Dim tiempo (muy útil para filtros)
CREATE INDEX idx_dim_tiempo_mes 
ON dim_tiempo(mes);

CREATE INDEX idx_dim_tiempo_temporada 
ON dim_tiempo(es_temporada_critica);