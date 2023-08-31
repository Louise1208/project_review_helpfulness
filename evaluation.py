from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.metrics import ndcg_score

def rmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse

def pcc(y_true, y_pred):
    pcc = pearsonr(y_true, y_pred)[0]
    return pcc

def ndcg(y_true, y_pred):
    ndcg = ndcg_score([y_true], [y_pred])
    return ndcg

