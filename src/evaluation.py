import numpy as np

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mape(y_true, y_pred, eps=1e-9):
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def evaluate_all(y_true, predictions_dict):
    results = {}
    for name, y_pred in predictions_dict.items():
        results[name] = {
            "MAE": round(mae(y_true, y_pred), 2),
            "RMSE": round(rmse(y_true, y_pred), 2),
            "MAPE": round(mape(y_true, y_pred), 2)
        }
    return results
