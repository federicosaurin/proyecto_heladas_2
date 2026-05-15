import pytest
import pandas as pd
from src.analyzer import DataAnalyzer

def test_dynamic_correlation_shape():
    # Creamos datos sintéticos
    data = {
        'LowTemp': [10, 8, 5, 2, 0, -2, -1, 3],
        'WindSpeed': [5, 10, 15, 20, 25, 30, 35, 40]
    }
    df = pd.DataFrame(data)
    analyzer = DataAnalyzer(target_column='LowTemp')
    
    # Probamos 2 pasos con shift de -1
    p, s, k = analyzer.get_dynamic_correlations(df, steps=2, shift_size=-1)
    
    # Debe tener 2 columnas (Step_0 y Step_1)
    assert p.shape[1] == 2
    # La correlación de Step_0 con WindSpeed debe ser un float válido
    assert not pd.isna(p.loc['WindSpeed', 'Step_0'])