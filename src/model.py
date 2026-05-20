import pandas as pd
import numpy as np
import logging
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler  # ¡Activado para solucionar el desbalance!

class FrostPredictor:
    def __init__(self, config: dict):
        model_cfg = config.get('model', {})
        self.target_col = model_cfg.get('target_column', 'LowTemp')
        self.n_hours = model_cfg.get('n_hours', 24)
        self.records_per_hour = model_cfg.get('records_per_hour', 6)
        self.n_splits = model_cfg.get('n_splits', 5)
        self.random_state = model_cfg.get('random_state', 42)
        
        self.architectures = model_cfg.get('architectures', [[32]])
        self.alpha = model_cfg.get('hyperparameters', {}).get('alpha', 0.0001)
        self.max_iter = model_cfg.get('hyperparameters', {}).get('max_iter', 1000)
        
        self.logger = logging.getLogger(__name__)

    def train_hourly_models(self, df: pd.DataFrame):
        """
        Entrena los modelos corrigiendo la matriz de confusión y aplicando
        RandomOverSampler para resolver el desbalance de clases de raíz.
        """
        final_report = []
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for arch in self.architectures:
            arch_tuple = tuple(arch)
            self.logger.info(f"--- Iniciando Entrenamiento Arquitectura: {arch_tuple} ---")

            for h in range(self.n_hours):
                self.logger.info(f"Procesando Lead Time: {h} horas")
                
                shift_val = -self.records_per_hour * h
                df_work = df.copy()
                df_work['target_temp'] = df_work[self.target_col].shift(shift_val)
                
                # Clase 1 = Helada (< 0°C), Clase 0 = Normal
                df_work['Helada'] = np.where(df_work['target_temp'] < 0, 1, 0)
                df_hour = df_work.dropna(subset=['target_temp']).copy()
                
                X = df_hour.drop(columns=['Helada', 'target_temp'])
                y = df_hour['Helada']
                
                if len(y.unique()) < 2:
                    self.logger.warning(f"Saltando hora {h}: Solo una clase presente.")
                    continue

                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # --- BALANCEO REAL DE CLASES ---
                    # Duplicamos estratégicamente las heladas en el set de ENTRENAMIENTO
                    ros = RandomOverSampler(random_state=self.random_state)
                    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

                    # Escalado fiteado con los datos balanceados
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_res)
                    X_test_scaled = scaler.transform(X_test)

                    clf = MLPClassifier(
                        hidden_layer_sizes=arch_tuple, 
                        activation='relu', 
                        max_iter=self.max_iter, 
                        random_state=self.random_state, 
                        alpha=self.alpha
                    )
                    
                    clf.fit(X_train_scaled, y_train_res)
                    y_pred = clf.predict(X_test_scaled)
                    
                    # --- MATRIZ DE CONFUSIÓN
                    # Estructura oficial de scikit-learn:
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

                    result = {
                        'Arch': str(arch_tuple),
                        'Lead Time': h,
                        'Fold': fold,
                        'Accuracy': round(accuracy_score(y_test, y_pred), 6),
                        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 6),
                        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 6), # Sensibilidad real a heladas
                        'F1': round(f1_score(y_test, y_pred, zero_division=0), 6),
                        'Specificity': round(tn / (tn + fp) if (tn + fp) != 0 else 0, 6), # Capacidad de descartar días normales
                        'TN': int(tn),
                        'FP': int(fp),
                        'FN': int(fn),
                        'TP': int(tp)
                    }
                    
                    # Importancia de variables (Pesos de la red)
                    avg_weights = np.mean(np.abs(clf.coefs_[0]), axis=1)
                    for j, col in enumerate(X.columns):
                        result[f'W_{col}'] = round(avg_weights[j], 6)
                        
                    final_report.append(result)

        return pd.DataFrame(final_report)

    def save_results(self, results_df: pd.DataFrame, path: str):
        """Guarda el archivo CSV con todas las métricas reales."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results_df.to_csv(path, index=False)
        self.logger.info(f"Resultados de entrenamiento guardados en: {path}")