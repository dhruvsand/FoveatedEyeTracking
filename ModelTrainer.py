import random

import xgboost
import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle
from pymouse import PyMouse



def main():
    m = PyMouse()

    # x_dim = 1280, y_dim = 800

    print("python main function")

    data = pd.read_csv("dataMouse.csv.csv")

    predictors = ['left_x', 'left_y', 'right_x', 'right_y', 'horizontalRatio', 'verticalRatio']
    # outputFeatures = [ 'Centre', 'Left','Right', 'Up', 'Down','cursor_x','cursor_y']
    outputFeatures = [ 'cursor_x']

    input = data[predictors]
    output = data[outputFeatures]
    output= np.ravel(output)

    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2,shuffle=True)




    xgb1 = XGBRegressor(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    xgb1.fit(X_train,y_train/1280)

    # dtrain_predprob = xgb1.predict_proba(X_test)[:,1]
    dtrain_predprob = xgb1.predict(X_test)
    print("AUC Score (Train): %f" % metrics.r2_score(y_test/1280, dtrain_predprob))


    # save model to file
    pickle.dump(xgb1, open("cursor_x.pickle.dat", "wb"))




if __name__ == '__main__':
    main()