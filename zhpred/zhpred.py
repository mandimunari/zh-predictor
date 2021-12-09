import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from xgboost import XGBClassifier

class Classifier:
    def X_y_data(self, df):
        X = df.drop(['PERFIL'], axis=1)
        y = df['PERFIL']

        return X, y

    def train_test_df(self, X, y):
        
        """       
        splid do dataframe em teste e treino
        """

        X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.25)

        return X_train, X_test, y_train, y_test

    def model(self, X_train, y_train):

        """       
        Modelo xgboost e o ajuste do mesmo 
        """

        model = XGBClassifier()
        model.fit(X_train, y_train)

        return model

    def predict(self, model, X_test, y_test):

        """       
        Gera uma previsão com base no modelo ajustado 
        """

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return accuracy, recall, f1

    def predict_prob(self, model, X_test):
        """       
        Gera a probabilidade de classe
        """

        prob = model.predict_proba(X_test)

        return prob
         
df = pd.read_csv('../data/processed/data.csv')


