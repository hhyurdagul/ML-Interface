import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


holiday_dates = [
                 {"month":1, "day":1},
                 {"month":4, "day":23},
                 {"month":5, "day":1},
                 {"month":5, "day":19},
                 {"month":7, "day":15},
                 {"month":8, "day":30},
                 {"month":10, "day":29}
]

def isHoliday(d):
    for i in holiday_dates:
        if(i["day"] == d.day) and (i["month"] == d.month):
            return 1
    return 0

def isWeekend(d):
    return 1 if d.weekday() == 5 or d.weekday() == 6 else 0

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


