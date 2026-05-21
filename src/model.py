import pandas as pd
import numpy as np
import logging
import os
import joblib  # Librería profesional para guardar/cargar modelos de ML
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler

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
        
        # Carpeta para el modelo de producción definitivo (1 por hora)
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)

    def train_hourly_models(self, df: pd.DataFrame):
        """
        Calcula las métricas de validación por fold en memoria y persiste UN ÚNICO
        modelo definitivo de producción por cada hora de antelación.
        """
        final_report = []
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for arch in self.architectures:
            arch_tuple = tuple(arch)
            self.logger.info(f"--- Iniciando Procesamiento Arquitectura: {arch_tuple} ---")

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

                # Definimos el nombre del modelo comercial definitivo de esta hora
                model_filename = f"mlp_{str(arch_tuple).replace(' ', '')}_h{h}.joblib"
                model_path = os.path.join(self.models_dir, model_filename)

                # --- 1. EVALUACIÓN CRUZADA (FOLDS) EN MEMORIA ---
                # Esto se ejecuta siempre para poder calcular las curvas y armar el CSV de métricas
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # Balanceo temporal para la evaluación del pliegue
                    ros = RandomOverSampler(random_state=self.random_state)
                    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_res)
                    X_test_scaled = scaler.transform(X_test)

                    # Red de validación (Muere al terminar el fold, no se guarda)
                    clf_fold = MLPClassifier(
                        hidden_layer_sizes=arch_tuple, 
                        activation='relu', 
                        max_iter=self.max_iter, 
                        random_state=self.random_state, 
                        alpha=self.alpha
                    )
                    clf_fold.fit(X_train_scaled, y_train_res)
                    y_pred = clf_fold.predict(X_test_scaled)
                    
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

                    result = {
                        'Arch': str(arch_tuple),
                        'Lead Time': h,
                        'Fold': fold,
                        'Accuracy': round(accuracy_score(y_test, y_pred), 6),
                        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 6),
                        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 6), 
                        'F1': round(f1_score(y_test, y_pred, zero_division=0), 6),
                        'Specificity': round(tn / (tn + fp) if (tn + fp) != 0 else 0, 6),
                        'TN': int(tn),
                        'FP': int(fp),
                        'FN': int(fn),
                        'TP': int(tp)
                    }
                    
                    avg_weights = np.mean(np.abs(clf_fold.coefs_[0]), axis=1)
                    for j, col in enumerate(X.columns):
                        result[f'W_{col}'] = round(avg_weights[j], 6)
                        
                    final_report.append(result)

                # --- 2. PERSISTENCIA INTELIGENTE DEL MODELO COMERCIAL ---
                # Una vez evaluada la hora por los folds, generamos o salteamos el modelo definitivo
                if os.path.exists(model_path):
                    self.logger.info(f"Modelo final de producción ya existente en: {model_filename} (Omitiendo entrenamiento final)")
                else:
                    self.logger.info(f"Generando modelo definitivo de producción: {model_filename}")
                    
                    # Para el modelo comercial usamos el 100% de los datos disponibles de esta hora
                    ros_final = RandomOverSampler(random_state=self.random_state)
                    X_resampled, y_resampled = ros_final.fit_resample(X, y)
                    
                    scaler_final = StandardScaler()
                    X_scaled = scaler_final.fit_transform(X_resampled)
                    
                    clf_production = MLPClassifier(
                        hidden_layer_sizes=arch_tuple, 
                        activation='relu', 
                        max_iter=self.max_iter, 
                        random_state=self.random_state, 
                        alpha=self.alpha
                    )
                    
                    # Entrenamos con todo el potencial de datos balanceados
                    clf_production.fit(X_scaled, y_resampled)
                    
                    # Guardamos en disco un solo archivo por hora
                    joblib.dump(clf_production, model_path)
                    self.logger.info(f"-> Guardado con éxito: {model_path}")

        return pd.DataFrame(final_report)

    def save_results(self, results_df: pd.DataFrame, path: str):
        """Guarda el archivo CSV general con todas las métricas reales."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results_df.to_csv(path, index=False)
        self.logger.info(f"Resultados de entrenamiento guardados en: {path}")