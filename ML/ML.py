from datetime import datetime as dt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


source_ds = 'full_rep_dataset.csv'

def get_train_val(val_size, x_scaled=False):
    # load dataset
    ds = pd.read_csv(source_ds)
    ds = ds.where(ds['year'] < 2022).dropna()

    # format start and end times in datetime objects
    dt_format = '%Y-%m-%d %H:%M:%S'  # set dt_format to the format printed above
    ds['temp_start_time'] = [dt.strptime(startDate, dt_format) for startDate in ds['start_time']]
    ds['start_time'] = ds['temp_start_time']

    ds['temp_end_time'] = [dt.strptime(endDate, dt_format) for endDate in ds['end_time']]
    ds['end_time'] = ds['temp_end_time']

    ds = ds.drop(['temp_end_time', 'temp_start_time'], axis=1)

    # initial x/y split
    x_tv = ds.drop(['event', 'severe', 'year'], axis=1)
    y_tv = ds[['start_time', 'cape', 'LD', 'hail_size', 'severe']]

    if x_scaled:
        scaler = preprocessing.StandardScaler().fit(x_tv)
        x_tv = scaler.transform(x_tv)

    # x/y train/val split
    x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv, test_size=val_size)

    # clean training data
    # x_train = x_train.where(x_train['year'] < 2022)
    x_train = x_train.where(x_train['cape'] > 0).where((x_train['LD'] > 0)).where((x_train['hail_size'] > 0)).where(
        (x_train['start_time'].dt.month < 11)).where((x_train['start_time'].dt.month > 4)).dropna()
    x_train = x_train.drop(['start_time', 'end_time', 'hail_size'], axis=1)
    x_val = x_val.drop(['start_time', 'end_time', 'hail_size'], axis=1)

    # y_train = y_train.where(y_train['year'] < 2022)
    y_train = y_train.where(y_train['cape'] > 0).where((y_train['LD'] > 0)).where((y_train['hail_size'] > 0)).where(
        (y_train['start_time'].dt.month < 11)).where((y_train['start_time'].dt.month > 4)).dropna()
    y_train = y_train.drop(['start_time', 'cape', 'LD'], axis=1)
    y_val = y_val.drop(['start_time', 'cape', 'LD'], axis=1)

    test_ds = ds.where(ds['year'] == 2022).dropna()
    x_test = test_ds.drop(['start_time', 'hail_size', 'event', 'severe', 'year'], axis=1)
    y_test = test_ds[['hail_size', 'severe']]

    return x_train, x_val, y_train, y_val, x_tv, x_test, y_test

def test_model(model, map, dest, source_ds=source_ds):
    # load test data
    x_test = pd.read_csv(source_ds)
    x_test = x_test.where(x_test['year'] > 2021).dropna()

    print()
    print('x_test:')
    print()
    print(x_test)

    y_test = x_test[['hail_size', 'severe']]

    print()
    print('y_test:')
    print()
    print(y_test)


    x_test = x_test.drop(['event', 'severe', 'year', 'start_time', 'end_time', 'hail_size'], axis=1)

    y_pred = model.predict(X=x_test)

    if map == 'classification':
        col = 'severe'
        y_true = y_test[col]
        H = np.sum((y_true.ravel() == 1) & (y_pred.ravel() == 1))
        M = np.sum((y_true.ravel() == 1) & (y_pred.ravel() == 0))
        F = np.sum((y_true.ravel() == 0) & (y_pred.ravel() == 1))
        C = np.sum((y_true.ravel() == 0) & (y_pred.ravel() == 0))

        test_res_dict = {'hits': [H], 'misses': [M], 'false_alarms': [F], 'correct_negative': [C]}
        test_res = pd.DataFrame(data=test_res_dict)

        pod = H / (H + M)
        pofd = F / (F + C)

        num = np.log(pofd) - np.log(pod)
        den = np.log(pofd) + np.log(pod)
        edi = num / den

        test_res['POD'] = pod
        test_res['POFD'] = pofd
        test_res['EDI'] = edi
    elif map == 'regression':
        col = 'hail_size'
        y_true = y_test[col]
        y_pred = pd.DataFrame(data=y_pred, columns=['predicted'])
        y_pred['y_true'] = y_test['hail_size'].values

        test_res = y_pred

    else:
        raise Exception("map must be in ['classification', 'regression']")


    test_res.to_csv(dest + 'test_predictions.csv', index=False)
    # mean accuracy on the given test data and labels
    return model.score(x_test, y_test[col])

def feature_importance(rf, x_tv, dest):
    for i, column in enumerate(x_tv):

        fi = pd.DataFrame({'Variable': [column], 'Feature Importance Score': [rf.feature_importances_[i]]})

        try:
            final_fi = pd.concat([final_fi, fi], ignore_index=True)
        except:
            final_fi = fi

    # Ordering the data
    final_fi = final_fi.sort_values('Feature Importance Score', ascending=False).reset_index(drop=True)
    final_fi.to_csv(dest + 'feature_importance.csv', index=False)

    return final_fi

def evaluate(model, test_features, test_labels, map):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    if map == 'classification':
        mape = 100 * (np.sum(errors) / len(test_labels))
    else:
        mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape

    return np.mean(errors), accuracy

def EDI(model, x_test, y_test):
    y_pred = model.predict(X=x_test)
    y_true = y_test
    H = np.sum((y_true.ravel() == 1) & (y_pred.ravel() == 1))
    M = np.sum((y_true.ravel() == 1) & (y_pred.ravel() == 0))
    F = np.sum((y_true.ravel() == 0) & (y_pred.ravel() == 1))
    C = np.sum((y_true.ravel() == 0) & (y_pred.ravel() == 0))

    pod = H / (H + M)
    pofd = F / (F + C)

    num = np.log(pofd) - np.log(pod)
    den = np.log(pofd) + np.log(pod)
    edi = num / den

    return H, M, F, C, pod, pofd, edi



















