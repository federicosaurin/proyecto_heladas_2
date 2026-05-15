import pandas as pd
import numpy as np
import logging
import os
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
        # Mapeamos los nombres exactos de tu parameters.yaml
        model_cfg = config.get('model', {})
        self.target_col = model_cfg.get('target_column', 'LowTemp')
        self.n_hours = model_cfg.get('n_hours', 24)
        self.records_per_hour = model_cfg.get('records_per_hour', 6)
        self.n_splits = model_cfg.get('n_splits', 5)
        self.random_state = model_cfg.get('random_state', 42)
        
        # Parámetros de la red (tomamos la primera arquitectura por defecto)
        self.architectures = model_cfg.get('architectures', [[32]])
        self.alpha = model_cfg.get('hyperparameters', {}).get('alpha', 0.0001)
        self.max_iter = model_cfg.get('hyperparameters', {}).get('max_iter', 1000)
        
        self.logger = logging.getLogger(__name__)

    def train_hourly_models(self, df: pd.DataFrame):
        """
        Ejecuta el entrenamiento para cada hora de antelación.
        Itera sobre las arquitecturas definidas en el YAML.
        """
        final_report = []
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Iteramos por cada arquitectura definida (ej: [32], [64], [32, 16])
        for arch in self.architectures:
            arch_tuple = tuple(arch)
            self.logger.info(f"--- Iniciando Entrenamiento Arquitectura: {arch_tuple} ---")

            for h in range(self.n_hours):
                self.logger.info(f"Procesando Lead Time: {h} horas")
                
                # Lógica de Shifteo original (6 registros = 1 hora)
                shift_val = -self.records_per_hour * h
                
                # Trabajamos sobre una copia para no ensuciar el DF original
                df_work = df.copy()
                df_work['target_temp'] = df_work[self.target_col].shift(shift_val)
                
                # Definimos Helada (1) si la temperatura futura es < 0
                df_work['Helada'] = np.where(df_work['target_temp'] < 0, 1, 0)
                
                # Limpieza de nulos por el shift
                df_hour = df_work.dropna(subset=['target_temp']).copy()
                
                X = df_hour.drop(columns=['Helada', 'target_temp'])
                y = df_hour['Helada']
                
                # Validación si hay suficientes clases para el split
                if len(y.unique()) < 2:
                    self.logger.warning(f"Saltando hora {h}: Solo una clase presente.")
                    continue

                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # Escalado
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Red Neuronal
                    clf = MLPClassifier(
                        hidden_layer_sizes=arch_tuple, 
                        activation='relu', 
                        max_iter=self.max_iter, 
                        random_state=self.random_state, 
                        alpha=self.alpha
                    )
                    
                    clf.fit(X_train_scaled, y_train)
                    y_pred = clf.predict(X_test_scaled)
                    
                    # Métricas
                    result = {
                        'Arch': str(arch_tuple),
                        'Hour': h,
                        'Fold': fold,
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred, zero_division=0),
                        'Recall': recall_score(y_test, y_pred, zero_division=0),
                        'F1': f1_score(y_test, y_pred, zero_division=0),
                        'Specificity': recall_score(y_test, y_pred, pos_label=0, zero_division=0)
                    }

                    # Auditoría de Matriz de Confusión
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                    result.update({'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
                    
                    # Importancia de variables (Pesos)
                    avg_weights = np.mean(np.abs(clf.coefs_[0]), axis=1) # Usamos valor absoluto para importancia real
                    for j, col in enumerate(X.columns):
                        result[f'W_{col}'] = round(avg_weights[j], 6)
                        
                    final_report.append(result)

        return pd.DataFrame(final_report)

    def save_results(self, results_df: pd.DataFrame, path: str):
        """Guarda el CSV final de métricas."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results_df.to_csv(path, index=False)
        self.logger.info(f"Resultados de entrenamiento guardados en: {path}")