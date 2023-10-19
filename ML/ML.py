import os
from datetime import datetime as dt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import xgboost
import random

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

#TODO: remove plot_confusion_matrix
def plot_confusion_matrix(cm, model, vmin=0.0, vmax=1.0,classes=None, fn='max_confusion_matrix.png', destination_dir=None):
    # cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=vmin, vmax=vmax, annot=True,
                    fmt='d', annot_kws={'size': 50})
    else:
        sns.heatmap(cm, fmt='d', vmin=0., vmax=1.)

    hits = cm[1][1]
    fa = cm[0][1]
    misses = cm[1][0]
    tn = cm[0][0]

    csi = np.round(hits / (hits + fa + misses), 2)
    pod = np.round(hits / (hits + misses), 2)
    pofd = np.round(hits / (fa + tn), 2)
    far = np.round(fa / (fa + hits), 2)
    acc = np.round((hits + tn) / (hits + fa + misses + tn), 2)
    freqbias = np.round((hits + fa) / (hits + misses), 2)

    title = '{} \n CSI={}, POD={}, POFD={}, FAR={}, Acc.={}, Freq.bias={}'.format(model, csi, pod, pofd, far, acc, freqbias)

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(destination_dir + fn)
    plt.close()

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

def MLP_clf_model(x_train, x_val, y_train, y_val, n_iter=1000):
    max_accuracy_prod = 0
    max_cm = []
    max_pred = []
    for i in range(n_iter):
        print('Working on MLP_clf {} of {}. Current accuracy is {}'.format((i + 1), n_iter, max_accuracy_prod), end='\r')

        hidden_layer_sizes = random.randint(1, 500)
        max_iter = random.randint(2000, 20000)
        n_iter_no_change = random.randint(5, 50)

        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                            solver='lbfgs',
                            activation='relu',
                            max_iter=max_iter,
                            random_state=42,
                            n_iter_no_change=n_iter_no_change
                            )

        mlp.fit(x_train, y_train['severe'])
        pred_val = mlp.predict(X=x_val)

        cm = confusion_matrix(y_val['severe'], pred_val)
        cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
        accuracy_prod = cm_norm[0][0] * cm_norm[1][1]
        if accuracy_prod > max_accuracy_prod:
            max_accuracy_prod = accuracy_prod
            max_cm = cm
            max_pred = pred_val
            max_mlp = mlp

        com_sc = pd.DataFrame(
            {'hidden_layer_sizes':[hidden_layer_sizes],
             'solver':['lbfgs'],
             'activation':['relu'],
             'max_iter':[max_iter],
             'random_state':[42],
             'n_iter_no_change':[n_iter_no_change],
             'train_score': [mlp.score(x_train, y_train['severe'])],
             'val_score': [mlp.score(x_val, y_val['severe'])],
             'cm_score': [accuracy_prod],
             'true_positive': [cm[0][0]],
             'false_positive': [cm[0][1]],
             'false_negative': [cm[1][0]],
             'true_negative': [cm[1][1]]
             }
        )

        try:
            final_com_sc = pd.concat([final_com_sc, com_sc], ignore_index=True)
        except:
            final_com_sc = com_sc

    final_com_sc = final_com_sc.sort_values('val_score', ascending=False)
    final_com_sc.to_csv(mlp_clf_destination_dir + 'mlp_combination_scores.csv', index=False)

    classes = max_mlp.classes_
    plot_confusion_matrix(max_cm, classes, destination_dir=mlp_clf_destination_dir)

    print('Final validation accuracy: ', max_accuracy_prod)

    return max_mlp, max_pred

def MLP_reg_model(x_train, x_val, y_train, y_val, n_iter=1000):
    max_accuracy_prod = -10000
    max_pred = []
    for i in range(n_iter):
        print('Working on MLP_reg {} of {}. Current accuracy is {}'.format((i + 1), n_iter, max_accuracy_prod), end='\r')

        hidden_layer_sizes = random.randint(1, 500)
        max_iter = random.randint(2000, 20000)
        n_iter_no_change = random.randint(5, 50)

        mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                            solver='lbfgs',
                            activation='relu',
                            max_iter=max_iter,
                            random_state=42,
                            n_iter_no_change=n_iter_no_change
                            )

        mlp.fit(x_train, y_train['hail_size'])
        pred_val = mlp.predict(X=x_val)

        accuracy_prod = mlp.score(x_val, y_val['hail_size'])
        if accuracy_prod > max_accuracy_prod:
            max_accuracy_prod = accuracy_prod
            max_pred = pred_val
            max_mlp = mlp

        com_sc = pd.DataFrame(
            {'hidden_layer_sizes':[hidden_layer_sizes],
             'solver':['lbfgs'],
             'activation':['relu'],
             'max_iter':[max_iter],
             'random_state':[42],
             'n_iter_no_change':[n_iter_no_change],
             'train_score': [mlp.score(x_train, y_train['hail_size'])],
             'val_score': [mlp.score(x_val, y_val['hail_size'])],
             'cm_score': [accuracy_prod]
             }
        )

        try:
            final_com_sc = pd.concat([final_com_sc, com_sc], ignore_index=True)
        except:
            final_com_sc = com_sc

    final_com_sc = final_com_sc.sort_values('val_score', ascending=False)
    final_com_sc.to_csv(mlp_reg_destination_dir + 'mlp_combination_scores.csv', index=False)

    print('Final validation accuracy: ', max_accuracy_prod)

    return max_mlp, max_pred

def MLP(n_iter=1000, val_size=0.25, map='classification'):
    x_train, x_val, y_train, y_val, x_tv = get_train_val(val_size=val_size, x_scaled=True)

    # x_tv = x_tv.drop(['start_time', 'end_time', 'hail_size'], axis=1)

    # x_tv, y_tv = get_train_val()
    #
    # # scale input data
    # scaler = preprocessing.StandardScaler().fit(x_tv)
    # x_scaled = scaler.transform(x_tv)
    #
    # # train/test split
    # x_train, x_val, y_train, y_val = train_test_split(x_scaled, y_tv, test_size=val_size)

    # set model mapping
    model_name = '.{}{}{}{}{}'.format(f'{dt.today().year:02}', f'{dt.today().month:02}', f'{dt.today().day:02}',
                                      f'{dt.today().hour:02}', f'{dt.today().minute:02}')
    # find optimal rf model
    if map == 'classification':
        model_name = 'mlp_clf' + model_name
        dest = mlp_clf_destination_dir + '/' + model_name + '/'
        os.mkdir(dest)
        mlp, pred = MLP_clf_model(x_train, x_val, y_train, y_val, n_iter=n_iter)
    else:
        model_name = 'mlp_reg' + model_name
        dest = mlp_reg_destination_dir + '/' + model_name + '/'
        os.mkdir(dest)
        mlp, pred = MLP_reg_model(x_train, x_val, y_train, y_val, n_iter=n_iter)

    # print mlp parameters
    params = mlp.get_params()
    keys = list(params.keys())
    vals = list(params.values())
    print('MLP parameters: ', str(params))
    with open(dest + "mlp_params.txt", "w") as text_file:
        text_file.write('MLP {}: {}'.format(map, '{'))
        for i in range(len(keys)):
            text_file.write(str(keys[i]) + ': ' + str(vals[i]))
        text_file.write('}')
        text_file.close()

    # save training and testing datasets to csv files
    x_train_df = pd.DataFrame(data=x_train)
    y_train_df = pd.DataFrame(data=y_train)
    train_data = pd.concat([x_train_df, y_train_df], axis=1, join='inner')
    train_data.to_csv(dest + 'train_data.csv', index=False)

    pred = pd.DataFrame(data=pred, columns=['predicted'])
    x_val_df = pd.DataFrame(data=x_val)
    y_val_df = pd.DataFrame(data=y_val)
    val_data = pd.concat([x_val_df, y_val_df, pred], axis=1, join='inner')
    val_data.to_csv(dest + 'val_data.csv', index=False)

    return mlp


















