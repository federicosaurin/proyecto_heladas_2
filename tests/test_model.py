import pytest
import pandas as pd
import numpy as np
from src.model import MetricsCalculator, FrostPredictor

def test_metrics_calculator():
    # Caso perfecto
    acc, prec, rec, f1, spec = MetricsCalculator.compute_all(10, 0, 0, 10)
    assert acc == 1.0
    assert prec == 1.0
    
    # Caso división por cero
    acc, prec, rec, f1, spec = MetricsCalculator.compute_all(0, 0, 0, 0)
    assert acc == 0

def test_predictor_logic_shifteo():
    config = {'n_hours': 2, 'records_per_hour': 1}
    predictor = FrostPredictor(config)
    # Datos de prueba: temperatura bajando
    df = pd.DataFrame({'Temp': [5, 4, 3, 2, 1, 0, -1, -2]})
    
    # Solo verificamos que el flujo no rompa y genere resultados
    results = predictor.train_hourly_models(df)
    assert 'Hour' in results.columns
    assert len(results['Hour'].unique()) == 2