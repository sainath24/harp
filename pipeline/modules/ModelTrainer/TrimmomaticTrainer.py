import pandas as pd
import numpy as np
import re, os, math, copy

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def MAPE(gold, predict):
    overall=0
    for g, p in zip(gold, predict):
        overall+=abs(g-p)/g
    overall=overall/len(gold)
    return overall

def linear_regression(X_train_pca, X_test_pca, y_train_pca, y_test_pca):
    linear_regress_pca = LinearRegression()
    linear_regress_pca.fit(X_train_pca,y_train_pca)
    y_pred_pca = linear_regress_pca.predict(X_test_pca)

    res_val_pca = []
    for p, a in zip(y_pred_pca, y_test_pca):
        res_val_pca.append([p, a])
    res_pca = pd.DataFrame(res_val_pca, columns=["predic", "actual"])

    print("\n--------------------LINEAR REGRESSION-----------------------")
    print("Mean Squared Error:", mean_squared_error(y_test_pca, y_pred_pca))
    print("Mean Absolute Error:", mean_absolute_error(y_test_pca, y_pred_pca))
    print("MAPE:", MAPE(y_test_pca, y_pred_pca))
    print("\n-----------------------------------------------------------")

    return linear_regress_pca

def evaluate_neural_net(model, X_train_pca, X_test_pca, y_train_pca, y_test_pca):
    model.evaluate(X_train_pca, y_train_pca, batch_size=20)
    model.evaluate(X_test_pca, y_test_pca, batch_size=20)

    y_pred_pca_2 = model.predict(X_test_pca).flatten()

    print("\n--------------------NEURAL NET------------------------------")
    print("Mean Squared Error:", mean_squared_error(y_test_pca, y_pred_pca_2))
    print("Mean Absolute Error:", mean_absolute_error(y_test_pca, y_pred_pca_2))
    print("MAPE:", MAPE(y_test_pca, y_pred_pca_2))
    print("\n-----------------------------------------------------------")

def neural_net(X_train_pca, X_test_pca, y_train_pca, y_test_pca):
    # NN Model with one dimention
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=X_train_pca.shape[1], activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(tf.keras.layers.Dense(32, activation='relu'))
    # model.add(tf.keras.layers.Dense(16, activation='relu'))
    # model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
    history = model.fit(X_train_pca, y_train_pca, epochs=300, validation_split=0.2, batch_size=20)

    evaluate_neural_net(model, X_train_pca, X_test_pca, y_train_pca, y_test_pca)

    return model

def decision_tree_regressor(X_train_pca, X_test_pca, y_train_pca, y_test_pca):
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state = 0) 
    
    # fit the regressor with X and Y data
    regressor.fit(X_train_pca, y_train_pca)

    y_pred_pca_3 = regressor.predict(X_test_pca)

    print("\n--------------------DECISION TREE REGRESSOR-----------------")
    print("Mean Squared Error:", mean_squared_error(y_test_pca, y_pred_pca_3))
    print("Mean Absolute Error:", mean_absolute_error(y_test_pca, y_pred_pca_3))
    print("MAPE:", MAPE(y_test_pca, y_pred_pca_3))
    print("\n-----------------------------------------------------------")

    return regressor


def train(dataset: str):
    df_pca = dataset

    feature_cols_pcs = [str(i) for i in range(0,12)]
    label_pca='walltime'
    X_pca = df_pca[feature_cols_pcs] # Features
    y_pca = df_pca[label_pca] # Target variable

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, test_size=0.3, random_state=1) 

    #TODO: return models
    lr_model = linear_regression(X_train_pca, X_test_pca, y_train_pca, y_test_pca)

    nn_model = neural_net(X_train_pca, X_test_pca, y_train_pca, y_test_pca)

    dtr_model = decision_tree_regressor(X_train_pca, X_test_pca, y_train_pca, y_test_pca)

    return {
        "linear_regression": lr_model,
        "dnn": nn_model,
        "decision_tree_regressor": dtr_model
    }
    