import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer





def NMSE(y_true, y_pred):
    return round((((y_true-y_pred)**2)/(y_true.mean()*y_pred.mean())).mean(), 2)

def RMSE(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)

def MAE(y_true, y_pred):
    return round(mean_absolute_error(y_true, y_pred), 2)

def MAPE(y_true, y_pred):
    try:
        return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
    except:
        return None 

def SMAPE(y_true, y_pred):
    try:
        return round(np.mean(np.abs(y_true-y_pred)/((np.abs(y_true)+np.abs(y_pred))/2)) * 100, 2)
    except:
        return None

def MASE(y_true, y_pred, seasons=1):
    try:
        return round(mean_absolute_error(y_true, y_pred) / mean_absolute_error(y_true[seasons:], y_true[:-seasons]), 2)
    except:
        return None
skloss = {
        'NMSE': make_scorer(NMSE),
        'RMSE': make_scorer(RMSE),
        'MAE': make_scorer(MAE),
        'MAPE': make_scorer(MAPE),
        'SMAPE': make_scorer(SMAPE),
        }

def loss(y_true, y_pred, seasons=1):

    NMSE = round((((y_true-y_pred)**2)/(y_true.mean()*y_pred.mean())).mean(), 2)
    RMSE = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
    MAE = round(mean_absolute_error(y_true, y_pred), 2)
    try:
        MAPE = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
    except:
        MAPE = None 
    try:
        SMAPE = round(np.mean(np.abs(y_true-y_pred)/((np.abs(y_true)+np.abs(y_pred))/2)) * 100, 2)
    except:
        SMAPE = None
    try:
        MASE = round(mean_absolute_error(y_true, y_pred) / mean_absolute_error(y_true[seasons:], y_true[:-seasons]), 2)
    except:
        MASE = None

    return [NMSE, RMSE, MAE, MAPE, SMAPE, MASE]
