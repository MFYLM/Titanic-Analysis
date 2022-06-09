import preprocessing
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings


def logRegression(X, Y):
    # rescale data in trainnig stage (decrease sensitivity of the model)
    sc = StandardScaler()
    scaler = sc.fit(X)
    trainX_scaled = scaler.transform(X)

    # parameters = {
    #     'penalty' : ['l1','l2'],
    #     'C' : np.logspace(-3,3,7),
    #     'solver' : ['newton-cg','lbfgs','liblinear'],
    # }
    # warnings.filterwarnings('ignore')
    
    model = LogisticRegression(C=0.1,penalty='l2',solver='liblinear',max_iter=5000)
    # clf = GridSearchCV(model, param_grid=parameters, scoring='accuracy',cv=10)
    #clf.fit(trainX_scaled,Y)

    # print(clf.best_params_)
    # print(clf.best_score_)

    model.fit(trainX_scaled, Y)
    return model


def neuralNetwork(X, Y, size=(6, 2)):
    sc = StandardScaler()
    scaler = sc.fit(X)
    trainX_scaled = scaler.transform(X)

    MLP = MLPClassifier(hidden_layer_sizes=size, max_iter=2000, activation="relu", solver="adam")     # increasing layer to three could lead to overfitting
    MLP.fit(trainX_scaled, Y)

    return MLP

def randforestclassf(X, Y, est_num=50):
    rf = RandomForestClassifier(n_estimators=est_num, criterion="entropy")
    rf.fit(X, Y)

    return rf



if __name__ == "__main__":
    extracted = preprocessing.column_selection("data/train.csv")
    X = extracted.drop(columns="Survived", axis=1)
    Y = extracted["Survived"]
    pol = PolynomialFeatures(2)

    X_train, X_test, Y_train, Y_test = train_test_split(pol.fit_transform(X), Y, test_size=0.2)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model = logRegression(X_train, Y_train)
    # # rescale data in trainnig stage (decrease sensitivity of the model)
    sc = StandardScaler()
    scaler = sc.fit(X_train)
    testX_scaled = scaler.transform(X_test)
    trainX_scaled = scaler.transform(X_train)

    Y_pred_train = model.predict(trainX_scaled)
    Y_pred = model.predict(testX_scaled)

    # # figure = plot_confusion_matrix(model, X_test, Y_test, display_labels=model.classes_)
    # # plt.show()

    print(classification_report(Y_pred,Y_test))
    print("Accuracy on Train data: {:.2f}".format(accuracy_score(Y_train, Y_pred_train)))
    print("Accuracy on Test data: {:.2f}".format(accuracy_score(Y_test, Y_pred)))


    df2 = preprocessing.column_selection("data/test.csv")
    df2 = pol.fit_transform(df2)
    df2 = scaler.transform(df2)
    test_prediction = model.predict(df2)
    test_prediction = pd.DataFrame(test_prediction)
    test_prediction.columns = ['Survived']
    test_prediction.insert(loc=0, column="PassengerId", value=[892 + i for i in range(len(test_prediction["Survived"]))])
    test_prediction.to_csv("data/log_submission.csv", index=False)