import pytest
import pandas as pd
import numpy as np
from src.processors import DataProcessor

@pytest.fixture
def sample_df():
    """Crea un dataframe dummy que simula el formato de la tesis."""
    data = [
        ['Date', 'Temp', None], # Fila 0 cabecera
        ['', 'Max', 'Min'],      # Fila 1 cabecera
        ['01/06/2023', '10.5', '2.1'],
        ['15/07/2023', '8.0', None], # Fila con nulo para probar imputer
    ]
    return pd.DataFrame(data)

def test_format_headers(sample_df):
    processor = DataProcessor()
    df_fixed = processor.format_headers(sample_df)
    assert 'Date' in df_fixed.columns
    assert 'TempMax' in df_fixed.columns
    assert len(df_fixed) == 2

def test_filter_by_season():
    processor = DataProcessor()
    df = pd.DataFrame({'mes': [1, 6, 12]})
    filtered = processor.filter_by_season(df, [6, 7, 8])
    assert len(filtered) == 1
    assert filtered.iloc[0]['mes'] == 6