Dataset utlizzato:

https://www.kaggle.com/code/shawkyelgendy/pump-sensor-data-timeseriesanalysis/input

+ in aggiunta -> AI generated BROKEN sensors dataset per trainare il modello per via della scarsita' dei dati "corrotti" (solo 8 nel df originale). 

Per addestrare il modello:
    python script.py --train

Per fare previsioni su un file CSV senza etichette:
    python script.py --predict --input path/to/unlabeled_data.csv --output preds/predictions.csv

Per entrambe le operazioni:
    python script.py --train --predict --input path/to/unlabeled_data.csv
