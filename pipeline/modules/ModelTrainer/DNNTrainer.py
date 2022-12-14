import pandas as pd
import numpy as np
import re, os, math, copy

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import time


def linear_regression(X_train_pca, X_test_pca, y_train_pca, y_test_pca):
    linear_regress_pca = LinearRegression()
    linear_regress_pca.fit(X_train_pca,y_train_pca)
    y_pred_pca = linear_regress_pca.predict(X_test_pca)

    res_val_pca = []
    for p, a in zip(y_pred_pca, y_test_pca):
        res_val_pca.append([p, a])
    res_pca = pd.DataFrame(res_val_pca, columns=["predic", "actual"])
    print("-----------------------Linear Regression------------------------")
    print("Mean Squared Error:", mean_squared_error(y_test_pca, y_pred_pca))
    print("Mean Absolute Error:", mean_absolute_error(y_test_pca, y_pred_pca))
    print("----------------------------------------------------------------")

    upn, upp, mae, mse, mape, maeO, mseO, mapeO = get_other_stats(y_test_pca, y_pred_pca)
    adj_fact = get_adjust_factor(y_test_pca, y_pred_pca)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"lr_{timestr}"

    data = model_name, adj_fact, mae, mse, mape, upn, upp, maeO, mseO, mapeO

    return linear_regress_pca, data

def evaluate_neural_net(model, X_train_pca, X_test_pca, y_train_pca, y_test_pca):
    model.evaluate(X_train_pca, y_train_pca, batch_size=20)
    model.evaluate(X_test_pca, y_test_pca, batch_size=20)

    y_pred_pca_2 = model.predict(X_test_pca).flatten()
    testy_np = y_test_pca.to_numpy()
    
    print("-----------------------Neural Net-------------------------------")
    print("Mean Squared Error:", mean_squared_error(y_test_pca, y_pred_pca_2))
    print("Mean Absolute Error:", mean_absolute_error(y_test_pca, y_pred_pca_2))
    print("----------------------------------------------------------------")

    upn, upp, mae, mse, mape, maeO, mseO, mapeO = get_other_stats(y_test_pca, y_pred_pca_2)
    adj_fact = get_adjust_factor(y_test_pca, y_pred_pca_2)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"dnn_{timestr}"
    data = model_name, adj_fact, mae, mse, mape, upn, upp, maeO, mseO, mapeO

    return data

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
    history = model.fit(X_train_pca, y_train_pca, epochs=300, validation_split=0.2, batch_size=20, verbose=0)

    data = evaluate_neural_net(model, X_train_pca, X_test_pca, y_train_pca, y_test_pca)
    
    return model, data

def decision_tree_regressor(X_train_pca, X_test_pca, y_train_pca, y_test_pca):
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state = 0) 
    
    # fit the regressor with X and Y data
    regressor.fit(X_train_pca, y_train_pca)

    y_pred_pca_3 = regressor.predict(X_test_pca)
    print("-----------------------Decision Tree Regressor-------------------")
    print("Mean Squared Error:", mean_squared_error(y_test_pca, y_pred_pca_3))
    print("Mean Absolute Error:", mean_absolute_error(y_test_pca, y_pred_pca_3))
    print("----------------------------------------------------------------")

    upn, upp, mae, mse, mape, maeO, mseO, mapeO = get_other_stats(y_test_pca, y_pred_pca_3)
    adj_fact = get_adjust_factor(y_test_pca, y_pred_pca_3)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"dtr_{timestr}"
    data = model_name, adj_fact, mae, mse, mape, upn, upp, maeO, mseO, mapeO


    return regressor, data

def get_other_stats(yf_test_pca, y_pred_pca):
    under_predict=0
    for g, p in zip(yf_test_pca, y_pred_pca):
        if p < g:
            under_predict+=1
    under_predict_P = under_predict*100/len(yf_test_pca)
    mae = mean_absolute_error(yf_test_pca, y_pred_pca)
    mse = mean_squared_error(yf_test_pca, y_pred_pca)
    mape = mean_absolute_percentage_error(yf_test_pca, y_pred_pca)
    
    # Only Over Predictions
    over_g=[]
    over_p=[]
    for g, p in zip(yf_test_pca, y_pred_pca):
        if p>=g:
            over_g.append(g)
            over_p.append(p)
    if len(over_p) > 0:
        maeP = mean_absolute_error(over_g, over_p)
        mseP = mean_squared_error(over_g, over_p)
        mapeP = mean_absolute_percentage_error(over_g, over_p)
        ret_val = under_predict, round(under_predict_P, 2), mae.round(2), mse.round(2), (mape*100).round(2), maeP.round(2), mseP.round(2), (mapeP*100).round(2)
    else:
        ret_val = under_predict, round(under_predict_P, 3), mae.round(2), mse.round(2), (mape*100).round(2), 0.0, 0.0, 0.0
    return ret_val

def get_adjust_factor(yf_test_pca, y_pred_pca):
    under_predict=0
    for g, p in zip(yf_test_pca, y_pred_pca):
        if p < g:
            under_predict+=1
    under_predict_P = under_predict*100/len(yf_test_pca)
    # Only Under Predictions
    under_g=[]
    under_p=[]
    for g, p in zip(yf_test_pca, y_pred_pca):
        if p < g:
            under_g.append(g)
            under_p.append(p)
    adjustF = 0.0
    if len(under_p) > 0:
        adjustF = mean_absolute_percentage_error(under_g, under_p).round(2)
    return adjustF


def train(dataset):
    df_pca = dataset
    pca_number = -1
    for key in df_pca.keys():
        if key.isnumeric():
            pca_number = max(pca_number, int(key))
    feature_cols_pcs = [str(i) for i in range(0, pca_number + 1)] #NOTE MADE THIS DYNAMIC WITH PCA VALUE
    targrt_column_list= ['epochs_times_one', 'epochs_times_avg', 'fit_time']
    X_pca = df_pca[feature_cols_pcs] # Features
    y_pca = df_pca[targrt_column_list[1]] # Target variable
    yf_pca = df_pca[targrt_column_list[2]]

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, test_size=0.3, random_state=1) 

    Xf_train_pca, Xf_test_pca, yf_train_pca, yf_test_pca = train_test_split(X_pca, yf_pca, test_size=0.3, random_state=1) 

    # lr_model1, lr_data1 = linear_regression(X_train_pca, X_test_pca, y_train_pca, y_test_pca)
    #TODO: allow multiple models
    # lr_model2 = linear_regression(Xf_train_pca, Xf_test_pca, yf_train_pca, yf_test_pca)

    # dnn_1, dnn_data_1 = neural_net(X_train_pca, X_test_pca, y_train_pca, y_test_pca)
    # dnn_2 = neural_net(Xf_train_pca, Xf_test_pca, yf_train_pca, yf_test_pca)

    # dt_1, dt_data_1 = decision_tree_regressor(X_train_pca, X_test_pca, y_train_pca, y_test_pca)
    # dt_2 = decision_tree_regressor(Xf_train_pca, Xf_test_pca, yf_train_pca, yf_test_pca)
    # TODO: write model info to dataset
    #TODO: save models
    train_models = [
        linear_regression(X_train_pca, X_test_pca, y_train_pca, y_test_pca),
        neural_net(X_train_pca, X_test_pca, y_train_pca, y_test_pca),
        decision_tree_regressor(X_train_pca, X_test_pca, y_train_pca, y_test_pca)
    ]
    output = {}
    df = []
    for model, data in train_models:
        output[data[0]] = model
        df.append([*data])
    
    pd_df = pd.DataFrame(df)

    return output, pd_df



    
