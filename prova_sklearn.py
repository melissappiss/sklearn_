import sklearn 
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
irisPandas = pd.read_csv(url, names = ['Lunghezza Sepalo', 'Larghezza Sepalo', 'Lunghezza Petalo', 'Larghezza Petalo'])
irisPandas.head()

labEnc = preprocessing.LabelEncoder()
irisPandasTrasformato = irisPandas.apply(labEnc.fit_transform)

# Visualizzazione delle classi presenti prima e dopo la trasformazione
print(irisPandas.Specie)

# MODELLO DI MACHINE LEARNING DI TIPO FEEDFORWARD
# Dato che quello a cui si è interessati è generare modelli che si comportino bene su dati reali,
# è buona norma dividere il dataset di partenza in tre gruppi: training set, usato per addestrare il modello, validation set,
# usato per il tuning dei parametri del modello, e il test set, usato per calcolare una stima dell'errore di generalizzazione

