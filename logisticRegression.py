import preprocessing
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def model():
    extracted = preprocessing.column_selection("data/train.csv")
    X = extracted.drop(columns="Survived", axis=1)
    Y = extracted["Survived"]

    pol = PolynomialFeatures(2) # Added 

    X_train, X_test, Y_train, Y_test = train_test_split(pol.fit_transform(X), Y, test_size=0.2)

    # rescale data in trainnig stage (decrease sensitivity of the model)
    sc = StandardScaler()
    scaler = sc.fit(X_train)
    trainX_scaled = scaler.transform(X_train)
    testX_scaled = scaler.transform(X_test)

    model = LogisticRegression(penalty="l2", max_iter=1000)
    model.fit(trainX_scaled,Y_train)

    Y_pred = model.predict(testX_scaled)

    print(classification_report(Y_pred,Y_test))
    print("Accuracy: {:.2f}".format(accuracy_score(Y_test, Y_pred)))

if __name__ == "__main__":
    model()
