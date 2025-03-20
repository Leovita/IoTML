import os
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Predictor:
    """Classe per la predizione usando un modello salvato"""
    
    def __init__(self, model_path, scaler_path):
        self.logger = logging.getLogger(__name__)
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(model_path.replace('model.pkl', 'label_encoder.pkl'))
            self.feature_names = joblib.load(model_path.replace('model.pkl', 'feature_names.pkl'))
            self.logger.info(f"Modello caricato da {model_path}")
            self.logger.info(f"Etichette possibili: {self.label_encoder.classes_}")
        except Exception as e:
            self.logger.error(f"Errore nel caricamento del modello o dei relativi file: {e}")
            self.model = None
    
    def predict(self, df):
        """Effettua predizioni su dati non etichettati"""
        if self.model is None:
            self.logger.error("Nessun modello caricato. Impossibile fare predizioni.")
            return None, None
            
        try:
            from AnomalyDetectorModel import DataPreprocessor
            
            preprocessor = DataPreprocessor()
            processed_df = preprocessor.preprocess_data(df, is_training=False)
            
            if processed_df is None:
                return None, None
                
            features, _, _ = preprocessor.prepare_features_target(processed_df, is_training=False)
            
            if features is None:
                return None, None
                
            X_scaled = self.scaler.transform(features)
            
            y_pred_encoded = self.model.predict(X_scaled)
            y_pred_proba = self.model.predict_proba(X_scaled)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded)
            
            self.logger.info(f"Previsioni generate per {len(y_pred_labels)} campioni.")
            self.logger.info(f"Distribuzione delle predizioni: {pd.Series(y_pred_labels).value_counts()}")
            
            return y_pred_labels, y_pred_proba
            
        except Exception as e:
            self.logger.error(f"Errore durante la predizione: {e}")
            return None, None
    
    def save_predictions(self, original_df, predictions, probabilities, output_path):
        """Salva le predizioni in un file CSV"""
        try:
            result_df = original_df.copy()
            result_df['predicted_status'] = predictions
            
            for i, class_name in enumerate(self.label_encoder.classes_):
                result_df[f'prob_{class_name}'] = probabilities[:, i]
                
            self.logger.info(result_df.head())

            output_dir = os.path.dirname(output_path)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            result_df.to_csv(output_path, index=False)
            self.logger.info(f"Previsioni salvate in {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio delle previsioni: {e}")
            return False
