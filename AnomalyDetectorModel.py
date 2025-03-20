import os, joblib
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Classe per il caricamento e la validazione dei dati"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path):
        """Carica i dati da un file CSV"""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Il file {file_path} non esiste.")
                return None
                
            df = pd.read_csv(file_path)
            
            # Verifica se esiste la colonna timestamp e la converte in datetime se presente
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            self.logger.info(f"Dati caricati con successo da {file_path}. Forma: {df.shape}")
            
            if df.empty:
                self.logger.error(f"Il file {file_path} contiene un DataFrame vuoto.")
                return None
                
            return df
        except Exception as e:
            self.logger.error(f"Errore nel caricamento dei dati da {file_path}: {e}")
            return None
    
    def validate_data(self, df, check_status=True):
        """Valida il dataframe prima del preprocessing"""
        if df is None:
            self.logger.error("Dataframe nullo.")
            return False
        
        required_columns = ['timestamp']
        if check_status:
            required_columns.append('machine_status')
            
        # check missing cols
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            self.logger.error(f"Colonne mancanti: {missing}")
            return False
            
        sensor_columns = [col for col in df.columns if col.startswith('sensor_')]
        if len(sensor_columns) == 0:
            self.logger.error("Nessuna colonna trovata (sensors)")
            return False
            
        return True
    
    def load_and_combine_datasets(self, main_data_path, supplement_data_path=None):
        """Carica il dataset principale e, se fornito, lo combina con un dataset supplementare"""
        main_df = self.load_data(main_data_path)
        
        if main_df is None:
            self.logger.error("Impossibile caricare il dataset principale")
            return None
        
        # check integrity
        if not self.validate_data(main_df):
            self.logger.error("Validazione fallita per il dataset principale")
            return None
        
        if supplement_data_path is not None:
            supplement_df = self.load_data(supplement_data_path)
            
            if supplement_df is None:
                self.logger.warning("Impossibile caricare il dataset supplementare. Continuo solo con il dataset principale")
                return main_df
                
            if not self.validate_data(supplement_df):
                self.logger.warning("Validazione fallita per il dataset supplementare. Continuo solo con il dataset principale")
                return main_df
                
            if set(main_df.columns) != set(supplement_df.columns):
                self.logger.error("I dataset principale e supplementare hanno colonne diverse")
                return main_df
                
            combined_df = pd.concat([main_df, supplement_df], axis=0)
            self.logger.info(f"Dati combinati. Nuove dimensioni: {combined_df.shape}")
            return combined_df
        
        return main_df

    def load_unlabeled_data(self, file_path):
        """Carica dati senza etichette da un file CSV"""
        df = self.load_data(file_path)
        
        if df is None:
            self.logger.error("Impossibile caricare i dati senza etichette")
            return None
        
        # Verifica la presenza delle colonne necessarie senza richiedere machine_status
        if not self.validate_data(df, check_status=False):
            self.logger.error("Validazione fallita per i dati senza etichette")
            return None
        
        return df

class DataPreprocessor:
    """Classe per il preprocessing dei dati"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.le = LabelEncoder()
        self.scaler = StandardScaler()

        
    def preprocess_data(self, df, is_training=True):
        """Preprocessa i dati per l'analisi"""
        if df is None:
            self.logger.error("Impossibile prepxessare un dataFrame nullo")
            return None
            
        processed_df = df.copy()
        
        if is_training:
            duplicates = processed_df.duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Trovate {duplicates} righe duplicate!! Rimozione...")
                processed_df = processed_df.drop_duplicates()
        
        try:
            if 'timestamp' in processed_df.columns:
                processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
                processed_df['hour'] = processed_df['timestamp'].dt.hour
                processed_df['day'] = processed_df['timestamp'].dt.day
                processed_df['month'] = processed_df['timestamp'].dt.month
                processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
                processed_df['year'] = processed_df['timestamp'].dt.year
            else:
                self.logger.warning("Colonna timestamp non trovata. Le features temporali non saranno generate.")
        except Exception as e:
            self.logger.error(f"Errore nella gestione dei timestamp: {e}")
            return None
        
        sensor_columns = [col for col in processed_df.columns if col.startswith('sensor_')]
        for col in sensor_columns:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                
                missing_values = processed_df[col].isna().sum()
                if missing_values > 0:
                    missing_percent = (missing_values / len(processed_df)) * 100
                    self.logger.warning(f"colona {col}: {missing_values} valori mancanti ({missing_percent:.2f}%).")
                    # if missing_percent > 50:
                    #     self.logger.warning(f"Colonna {col} ha piu del 50 percento di valori mancanti!")
                    
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())

                # separazione dati 25 e 75 percentile e calcolo IQR
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    outlier_percent = (outliers / len(processed_df)) * 100
                    self.logger.warning(f"Colonna {col}: {outliers} outliers ({outlier_percent:.2f}%).")
                    
            except Exception as e:
                self.logger.error(f"Errore nel processamento della colonna {col}: {e}")
        
        return processed_df
    
    def prepare_features_target(self, processed_df, is_training=True):
        """Prepara feature e target per il modello."""
        try:
            if is_training:
                if 'machine_status' not in processed_df.columns:
                    self.logger.error("Colonna 'machine_status' non trovata.")
                    return None, None, None
                
                processed_df['status_encoded'] = self.le.fit_transform(processed_df['machine_status'])
                
                class_distribution = processed_df['machine_status'].value_counts()
                self.logger.info(f"Distribuzione delle classi:\n{class_distribution}")
                target = processed_df['status_encoded']
            else:
                target = None

            feature_columns = [col for col in processed_df.columns if col.startswith('sensor_') or 
                                col in ['hour', 'day', 'month', 'day_of_week', 'year']]
            if len(feature_columns) == 0:
                    self.logger.error("Nessuna feature trovata.")
                    return None, None, None
            
            features = processed_df[feature_columns]

            return features, target, self.le
            
        except Exception as e:
            self.logger.error(f"Errore nella preparazione di features e target: {e}")
            return None, None, None
    
    def split_and_scale_data(self, features, target, test_size=0.2):
        """Divide i dati in train e test set e standardizza le features."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=42, stratify=target
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            joblib.dump(self.scaler, './model/scaler.pkl')
            X_test_scaled = self.scaler.transform(X_test)

            
            return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns
            
        except Exception as e:
            self.logger.error(f"Errore nella divisione e standardizzazione dei dati: {e}")
            return None, None, None, None, None

class ModelTrainer:
    """Classe per l'addestramento e la valutazione del modello"""
    
    def __init__(self, output_dir='/imgs'):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def apply_smote(self, X_train, y_train):
        """Applica SMOTE per bilanciare le classi"""
        try:
            smote = SMOTEENN(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            self.logger.info(f"Applicato SMOTEENN. Nuova dimensione del training set: {X_train_resampled.shape}")
            
            unique, counts = np.unique(y_train_resampled, return_counts=True)
            self.logger.info(f"Nuova distribuzione delle classi dopo SMOTE: {dict(zip(unique, counts))}")
            
            return X_train_resampled, y_train_resampled
        except Exception as e:
            self.logger.error(f"Errore nell'applicazione di SMOTE: {e}")
            return X_train, y_train
    
    def train_model(self, X_train, y_train, use_smote=True):
        """Addestra un modello RandomForest."""
        if use_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)
        
        model = BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        
        model.fit(X_train, y_train)
        self.logger.info("Modello RandomForest addestrato con successo.")
        
        return model

    def evaluate_model(self, model, X_test, y_test, feature_names, label_encoder):
        """Valuta il modello e genera visualizzazioni"""
        if not model:
            self.logger.error("Impossibile valutare un modello nullo.")
            return
       
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        
        self.logger.info("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        self.logger.info(f"\n{report}")
        
        # plotting
        self.plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
        self.plot_feature_importance(model, feature_names)
        
        self.perform_cross_validation(model, X_test, y_test)  
   
    def predict(self, model, X, label_encoder):
        """Predice le classi per i dati di input."""
        try:
            y_pred_encoded = model.predict(X)
            y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
            
            # Ottieni anche le probabilità per ogni classe
            y_pred_proba = model.predict_proba(X)
            
            self.logger.info(f"Previsioni generate per {len(y_pred_labels)} campioni.")
            return y_pred_labels, y_pred_proba
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione delle previsioni: {e}")
            return None, None

    def save_model(self, model, filename="model.pkl"):
        """Salva il modello su disco"""
        self.output_dir = "./model"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(f"{self.output_dir}/{filename}", 'wb') as file:
            pickle.dump(model, file)
        self.logger.info(f"Modello salvato in {self.output_dir}/{filename}")
    
    def plot_confusion_matrix(self, y_test, y_pred, class_names):
        """Genera e salva la matrix di confusione"""
        self.output_dir = "./imgs"
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title("Matrice di Confusione")
        plt.ylabel('Etichetta Vera')
        plt.xlabel('Etichetta Predetta')
        
        plt.savefig(f"{self.output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Matrice di confusione salvata in {self.output_dir}/confusion_matrix.png")
    
    def plot_feature_importance(self, model, feature_names):
        """Genera e salva il grafico dell'importanza delle feature"""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Importanza delle Feature")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
        plt.xlim([-1, len(importances)])
        plt.tight_layout()
        
        plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Grafico dell'importanza delle feature salvato in {self.output_dir}/feature_importance.png")
        
    def perform_cross_validation(self, model, X, y, cv=5):
        """Esegue e salva i risultati della cross-validation"""
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        self.logger.info(f"Cross-validation scores (CV={cv}): {cv_scores}")
        self.logger.info(f"Media CV: {cv_scores.mean():.4f}, Deviazione standard CV: {cv_scores.std():.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, cv+1), cv_scores)
        plt.axhline(y=cv_scores.mean(), color='r', linestyle='-', label=f'Media: {cv_scores.mean():.4f}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('Risultati Cross-Validation')
        plt.xticks(range(1, cv+1))
        plt.legend()
        
        plt.savefig(f"{self.output_dir}/cross_validation.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Grafico della cross-validation salvato in {self.output_dir}/cross_validation.png")



def main():
    parser = argparse.ArgumentParser(description='Addestramento modello e previsione dati non etichettati')
    parser.add_argument('--train', action='store_true', help='Addestra il modello')
    parser.add_argument('--predict', action='store_true', help='Effettua previsioni su dati non etichettati')
    parser.add_argument('--input', type=str, help='Percorso del file di input per la previsione')
    parser.add_argument('--output', type=str, help='Percorso del file di output per le previsioni',
                        default='predictions/predictions.csv')
    parser.add_argument('--main_data', type=str, default='../datasets/sensor.csv',
                        help='Percorso del dataset principale')
    parser.add_argument('--supplement_data', type=str, default='../datasets/broken_sensor_data.csv',
                        help='Percorso del dataset supplementare')
    parser.add_argument('--model_path', type=str, default='./model/model.pkl',
                        help='Percorso del modello salvato')
    
    args = parser.parse_args()
    
    output_dir = "./imgs"
    model_path = args.model_path
    
    main_data_path = args.main_data
    supplement_data_path = args.supplement_data
    
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    model_trainer = ModelTrainer(output_dir=output_dir)

    
    model = None
    feature_names = None
    le = None
    
    if args.train or not os.path.exists(model_path):
        logger.info("Addestramento del modello...")
        combined_df = data_loader.load_and_combine_datasets(main_data_path, supplement_data_path)
        
        if combined_df is None:
            logger.error("Impossibile procedere senza dati validi.")
            return
        
        processed_df = preprocessor.preprocess_data(combined_df)
        
        if processed_df is None:
            logger.error("Errore nel preprocessing dei dati.")
            return
        
        features, target, le = preprocessor.prepare_features_target(processed_df)
        
        if features is None or target is None:
            logger.error("Errore nella preparazione di features e target.")
            return
        
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names = preprocessor.split_and_scale_data(
            features, target)
        
        if X_train_scaled is None:
            logger.error("Errore nella divisione e standardizzazione dei dati.")
            return
        
        model = model_trainer.train_model(X_train_scaled, y_train, use_smote=True)
        
        if model is None:
            logger.error("Errore nell'addestramento del modello.")
            return
        
        model_trainer.save_model(model)
        
        with open('./model/feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        with open('./model/label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
        
        model_trainer.evaluate_model(model, X_test_scaled, y_test, feature_names, le)
        
        logger.info("Addestramento e valutazione completati con successo.")
    
    if args.predict:
        logger.info("Modalità di previsione...")
        
        if not args.input:
            logger.error("Nessun file di input specificato per la previsione. Uso --input <percorso_file>")
            return
    
        unlabeled_df = data_loader.load_unlabeled_data(args.input)
        
        if unlabeled_df is None:
            logger.error("Impossibile caricare i dati non etichettati.")
            return
        try:
            from predictor import Predictor
        except ImportError:
            logger.error("Impossibile importare la classe Predictor. Assicurati che il file predictor.py sia nella directory corrente.")
            return
        
        predictor = Predictor(model_path, "./model/scaler.pkl")
        predizioni, probabilita = predictor.predict(unlabeled_df)
        
        if predizioni is None:
            logger.error("Errore nella generazione delle previsioni.")
            return
        
        output_path = args.output
        if predictor.save_predictions(unlabeled_df, predizioni, probabilita, output_path):
            logger.info(f"Previsioni salvate con successo in {output_path}")
        else:
            logger.error("Errore nel salvataggio delle previsioni.")
    
    if not args.train and not args.predict:
        logger.info("Nessuna opzione specificata. Usa --train per addestrare il modello o --predict --input <file> per fare previsioni.")
        parser.print_help()


if __name__ == '__main__':
    main()