import pandas as pd
import numpy as np
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

class FrostPredictor:
    def __init__(self, config: dict):
        self.hidden_units = config.get('hidden_units', (32,))
        self.alpha = config.get('alpha', 0.0001)
        self.random_state = config.get('random_state', 42)
        self.n_hours = config.get('n_hours', 24)
        self.records_per_hour = config.get('records_per_hour', 6)
        self.n_splits = config.get('n_splits', 5)
        self.logger = logging.getLogger(__name__)

    def train_hourly_models(self, df: pd.DataFrame):
        """Ejecuta el entrenamiento para cada hora de antelación usando métricas nativas."""
        all_results = []
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Asumimos que la columna 1 es la temperatura de referencia
        temp_col = df.columns[1] 

        for h in range(self.n_hours):
            self.logger.info(f"Procesando Lead Time: {h} horas")
            
            # Lógica de Shifteo: 6 registros = 1 hora
            shift_val = -self.records_per_hour * h
            df['target_temp'] = df[temp_col].shift(shift_val)
            
            # Definimos Helada (1) si la temperatura es < 0
            df['Helada'] = np.where(df['target_temp'] < 0, 1, 0)
            
            # Limpieza de nulos por el shift y preparación de datos
            df_hour = df.dropna(subset=['target_temp']).copy()
            X = df_hour.drop(columns=['Helada', 'target_temp'])
            y = df_hour['Helada']
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Pipeline de entrenamiento
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                clf = MLPClassifier(
                    hidden_layer_sizes=self.hidden_units, 
                    activation='relu', 
                    max_iter=1000, 
                    random_state=self.random_state, 
                    alpha=self.alpha
                )
                
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                
                # CÁLCULO DE MÉTRICAS NATIVAS
                # Usamos zero_division=0 para evitar warnings si no hay predicciones positivas
                result = {
                    'Hour': h,
                    'Fold': fold,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1': f1_score(y_test, y_pred, zero_division=0),
                    # Especificidad = Recall de la clase 0
                    'Specificity': recall_score(y_test, y_pred, pos_label=0, zero_division=0)
                }

                # Matriz de confusión para auditoría (opcional pero profesional)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                result.update({'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
                
                # Importancia de variables (pesos promedio de la primera capa)
                avg_weights = np.mean(clf.coefs_[0], axis=1)
                for j, col in enumerate(X.columns):
                    result[col] = round(avg_weights[j], 6)
                    
                all_results.append(result)
            
            # Limpieza para la siguiente iteración
            df.drop(columns=['Helada', 'target_temp'], inplace=True, errors='ignore')

        return pd.DataFrame(all_results)