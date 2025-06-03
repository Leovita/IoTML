# 🧠 Pump Sensor Failure Detection - Time Series Analysis

Un progetto di **analisi e previsione guasti su sensori industriali** basato su dati temporali. Utilizza un dataset reale di sensori e un set generato di sensori guasti per migliorare la fase di addestramento.

---

## 📦 Dataset

- **Fonte principale**:  
  Dataset originale disponibile su Kaggle:  
  [Pump Sensor Data - Time Series Analysis](https://www.kaggle.com/code/shawkyelgendy/pump-sensor-data-timeseriesanalysis/input)

- **Estensione AI-Generated**:  
  Dato il numero limitato di esempi di guasti reali (solo 8 nel dataset originale), sono stati generati dati sintetici di sensori "BROKEN" per addestrare efficacemente il modello.

---

## 🧪 Funzionalità principali

- Analisi di serie temporali su dati da sensori
- Rilevamento guasti (fault detection)
- Integrazione di dati sintetici per migliorare la generalizzazione
- Supporto a modalità **training**, **prediction** e **combined**

---

## 🚀 Utilizzo

### 🔧 Addestramento del modello pre-trained

```bash 
python script.py --train
```

### 🔧 Predizione del modello con dataset non etichettato

```bash
python script.py --predict --input data/unlabeled.csv --output preds/pred.csv
```
