# data analysis and wrangling
import pandas as pd
import numpy as np

# data vizualization
import matplotlib.pyplot as plt

# machine learning packages
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

from model_training import train_model
from data_preprocessing import extract_title_from_name, add_is_alone_column, continous_category, add_categorical_columns, fill_nans

def prepare_and_test_data():
    df = pd.read_csv("/Users/aayushpuranik/Python Scripts/Dataset/Titanic Dataset/train.csv")
    
    df = extract_title_from_name(df)
    df = add_categorical_columns(df)
    df = add_is_alone_column(df)
    
    df["AgeBand"] = continous_category(df["Age"], n_bias = 5)
    df["FareBand"] = continous_category(df["Fare"], n_bias = 4)
        
    df = fill_nans(df, median_columns=["Title", "Embarked", "AgeBand", "FareBand"])    
    df.drop(columns = ['Ticket','Cabin','Name','Age','PassengerId','Parch','SibSp','Fare'], inplace=True)   
    
    X = df.drop("Survived", axis=1)
    Y = df["Survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    
    svc, acc_svc = train_model(SVC, X_train, X_test, y_train, y_test, gamma='auto')
    sgd, acc_sgd = train_model(SGDClassifier, X_train, X_test, y_train, y_test)
    knn, acc_knn = train_model(KNeighborsClassifier, X_train, X_test, y_train, y_test, n_neighbors=3)
    gaussian, acc_gaussian = train_model(GaussianNB, X_train, X_test, y_train, y_test)
    perceptron, acc_perceptron = train_model(Perceptron, X_train, X_test, y_train, y_test)
    decision_tree, acc_decision_tree = train_model(DecisionTreeClassifier,X_train, X_test, y_train, y_test)
    random_forest, acc_random_forest = train_model(RandomForestClassifier, X_train, X_test, y_train, y_test)
    
    return acc_svc, acc_knn, acc_random_forest, acc_gaussian, acc_perceptron, acc_sgd, acc_decision_tree
