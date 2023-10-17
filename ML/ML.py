import os
from datetime import datetime as dt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import xgboost
import random

# source_dir ='/space/hall5/sitestore/eccc/mrd/rpnarmp/alc005/ML/ml_datasets/'
# rep_source_dir = '/space/hall5/sitestore/eccc/mrd/rpnarmp/alc005/ML/ml_rep_datasets/'
# source_ds = '/space/hall5/sitestore/eccc/mrd/rpnarmp/alc005/ML/ml_rep_datasets/full_rep_dataset.csv'
#
# rf_clf_destination_dir = '/space/hall5/sitestore/eccc/mrd/rpnarmp/alc005/ML/RF_clf/'
# rf_reg_destination_dir = '/space/hall5/sitestore/eccc/mrd/rpnarmp/alc005/ML/RF_reg/'
#
# xgb_clf_destination_dir = '/space/hall5/sitestore/eccc/mrd/rpnarmp/alc005/ML/XGB_clf/'
# xgb_reg_destination_dir = '/space/hall5/sitestore/eccc/mrd/rpnarmp/alc005/ML/XGB_reg/'
#
# mlp_clf_destination_dir = '/space/hall5/sitestore/eccc/mrd/rpnarmp/alc005/ML/MLP_clf/'
# mlp_reg_destination_dir = '/space/hall5/sitestore/eccc/mrd/rpnarmp/alc005/ML/MLP_reg/'

def get_train_val(val_size, source_ds, x_scaled=False):
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

    return x_train, x_val, y_train, y_val, x_tv

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

def test_model(model, map, dest, source_ds):
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

def rf_clf_model(x_train, x_val, y_train, y_val, dest, max_depths=[None, 10, 20, 50, 75, 100, 125, 150, 175, 200], n_jobs=-1):
    max_accuracy_prod = 0
    max_rf = RandomForestClassifier()
    max_pred = []
    for d in max_depths:
        rf = RandomForestClassifier(n_estimators=random.randint(100, 1000),
                                    criterion='entropy',
                                    max_features=None,
                                    max_depth=d,
                                    n_jobs=n_jobs,
                                    random_state=42)
        rf.fit(x_train, y_train['severe'])
        pred_val = rf.predict(X=x_val)

        cm = confusion_matrix(y_val['severe'], pred_val)

        C = cm[0][0]
        M = cm[1][0]
        F = cm[0][1]
        H = cm[1][1]

        pod = H / (H + M)
        pofd = F / (F + C)

        num = np.log(pofd) - np.log(pod)
        den = np.log(pofd) + np.log(pod)
        edi = num / den

        accuracy_prod = edi
        if accuracy_prod > max_accuracy_prod:
            print('new max: ', accuracy_prod)
            max_accuracy_prod = accuracy_prod
            max_pred = pred_val
            max_rf = rf

        com_sc = pd.DataFrame(
            {
             'max_depth': [d],
             'train_score': [rf.score(x_train, y_train['severe'])],
             'val_score': [rf.score(x_val, y_val['severe'])],
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

    final_com_sc = final_com_sc.sort_values('cm_score', ascending=False)
    final_com_sc.to_csv(dest + 'rf_combination_scores.csv', index=False)

    # test scores
    x_test = pd.read_csv(rep_source_dir + 'x_dirty_rep_test_dataset.csv')
    y_test = pd.read_csv(rep_source_dir + 'y_dirty_rep_test_dataset.csv')
    pred_test = max_rf.predict(X=x_test)
    pred_test_df = y_test
    pred_test_df['prediction'] = pred_test
    pred_test_df.to_csv(dest + 'rf_clf_test_pred.csv', index=False)

    test_cm = confusion_matrix(y_test['severe'], pred_test)

    classes = max_rf.classes_
    plot_confusion_matrix(test_cm, 'Random Forest Classifier', classes, fn='rf_clf_rep_confusion_matrix.png', destination_dir=dest)


    # save rf parameters in human readable format
    params = max_rf.get_params()
    keys = list(params.keys())
    vals = list(params.values())
    with open(rf_clf_destination_dir + "rf_clf_params.txt", "w") as text_file:
        text_file.write('RandomForestClassifier : {')
        for i in range(len(keys)):
            text_file.write(str(keys[i]) + ': ' + str(vals[i]))
        text_file.write('}')
        text_file.close()

    return max_rf, max_pred

def rf_reg_model(x_train, x_val, y_train, y_val, max_depths=[None, 10, 20, 50, 75, 100], n_jobs=-1):
    max_accuracy_prod = 0
    max_rf = RandomForestRegressor()
    max_pred = []
    for d in max_depths:
        rf = RandomForestRegressor(n_estimators=random.randint(100, 1000),
                                    max_features=None,
                                    max_depth=d,
                                    n_jobs=n_jobs,
                                    random_state=42)
        try:
            rf.fit(x_train, y_train['hail_size'])
        except TypeError:
            print('Value error for max_depth = {}'.format(d))
        try:
            pred_val = rf.predict(X=x_val)
        except IndexError:
            print('IndexError. train cols = {}, val vols = {}'.format(len(x_train.columns), len(x_val.columns)))

        accuracy_prod = rf.score(x_val, y_val['hail_size'])
        if accuracy_prod > max_accuracy_prod:
            print('new max: ', accuracy_prod)
            max_accuracy_prod = accuracy_prod
            max_pred = pred_val
            max_rf = rf

        com_sc = pd.DataFrame(
            {
             'max_depth': [d],
             'train_score': [rf.score(x_train, y_train['hail_size'])],
             'val_score': [rf.score(x_val, y_val['hail_size'])],
             'cm_score': [accuracy_prod]
             }
        )

        try:
            final_com_sc = pd.concat([final_com_sc, com_sc], ignore_index=True)
        except:
            final_com_sc = com_sc

    final_com_sc = final_com_sc.sort_values('val_score', ascending=False)
    final_com_sc.to_csv(rf_reg_destination_dir + 'rf_combination_scores.csv', index=False)

    # save rf parameters in human readable format
    params = max_rf.get_params()
    keys = list(params.keys())
    vals = list(params.values())
    with open(rf_reg_destination_dir + "rf_reg_params.txt", "w") as text_file:
        text_file.write('RandomForestRegressor : {')
        for i in range(len(keys)):
            text_file.write(str(keys[i]) + ': ' + str(vals[i]))
        text_file.write('}')
        text_file.close()

    return max_rf, max_pred

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

    # test_res['POD'] = pod
    # test_res['POFD'] = pofd
    # test_res['EDI'] = edi

    # test_res.to_csv(dest + 'test_results.csv')

    return H, M, F, C, pod, pofd, edi

def rf_kfold(n_iter=5, cv=5, clean_features=True, map='classification', n_jobs=[-1]):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
    # Number of features to consider at every split
    # max_features = ['auto', 'sqrt', None]
    max_features = [None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 300, num=50)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'n_jobs':n_jobs}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    model_name = '.{}{}{}{}{}'.format(f'{dt.today().year:02}', f'{dt.today().month:02}', f'{dt.today().day:02}',
                                      f'{dt.today().hour:02}', f'{dt.today().minute:02}')
    if map == 'classification':
        model_name = 'rf_10_kfold_clf' + model_name
        dest = rf_clf_destination_dir + '/' + model_name + '/'
        os.mkdir(dest)
        rf = RandomForestClassifier()
        base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        label = 'severe'
    elif map == 'regression':
        model_name = 'rf_10_kfold_reg' + model_name
        dest = rf_reg_destination_dir + '/' + model_name + '/'
        os.mkdir(dest)
        rf = RandomForestRegressor()
        base_model = RandomForestRegressor(n_estimators=10, random_state=42)
        label = 'hail_size'
    else:
        print('map must be "classification" or "regression"')

    # Random search of parameters, using 10 fold cross validation,
    # search across 1000 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=n_iter, cv=cv, verbose=4,
                                   random_state=42, n_jobs=-1)
    df = pd.read_csv(source_ds)

    # format start and end times in datetime objects
    dt_format = '%Y-%m-%d %H:%M:%S'  # set dt_format to the format printed above
    df['temp_start_time'] = [dt.strptime(startDate, dt_format) for startDate in df['start_time']]
    df['start_time'] = df['temp_start_time']

    df['temp_end_time'] = [dt.strptime(endDate, dt_format) for endDate in df['end_time']]
    df['end_time'] = df['temp_end_time']

    df = df.drop(['temp_end_time', 'temp_start_time'], axis=1)

    if clean_features:
        x_train = df.where(df['year'] < 2022).where(df['cape'] > 0).where((df['LD'] > 0)).where((df['hail_size'] > 0)).where(
            (df['start_time'].dt.month < 11)).where((df['start_time'].dt.month > 4)).dropna()
    else:
        x_train = df.where(df['year'] < 2022).dropna()

    y_train = x_train[['hail_size', 'severe']]
    x_train = x_train.drop(['event', 'year', 'start_time', 'end_time', 'hail_size', 'severe'], axis=1)

    x_test = df.where(df['year'] > 2021).dropna()
    y_test = x_test[['hail_size', 'severe']]
    x_test = x_test.drop(['event', 'year', 'start_time', 'end_time', 'hail_size', 'severe'], axis=1)

    # Fit the random search model
    rf_random.fit(x_train, y_train[label])
    params = rf_random.best_params_
    if map == 'classification':
        max_rf = RandomForestClassifier(n_jobs= -1,
                                        n_estimators=params['n_estimators'],
                                        min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        max_features=params['max_features'],
                                        max_depth=params['max_depth'],
                                        bootstrap=params['bootstrap'])
    else:
        max_rf = RandomForestRegressor(n_jobs= -1,
                                        n_estimators=params['n_estimators'],
                                        min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        max_features=params['max_features'],
                                        max_depth=params['max_depth'],
                                        bootstrap=params['bootstrap'])
    max_rf.fit(x_train, y_train[label])

    test_model(max_rf, map, dest)

    mean_error, accuracy = evaluate(max_rf, x_test, y_test[label], map)
    print('Tuned Model Performance')
    if map == 'regression':
        print('Average Error: {:0.4f} cm.'.format(mean_error))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    if map == 'classification':
        H, M, F, C, pod, pofd, edi = EDI(max_rf, x_test, y_test[label])
        print('EDI = {:0.4f}'.format(edi))
        test_res_dict = {'hits': [H], 'misses': [M], 'false_alarms': [F], 'correct_negative': [C], 'pod':[pod], 'pofd':[pofd], 'edi':edi}
        test_res = pd.DataFrame(data=test_res_dict)
        test_res.to_csv(dest + 'test_results.csv')
    print()

    base_model.fit(x_train, y_train[label])
    base_mean_error, base_accuracy = evaluate(base_model, x_test, y_test[label], map)
    print('Base Model Performance')
    if map == 'regression':
        print('Base Average Error: {:0.4f} cm.'.format(base_mean_error))
    print('Base Accuracy = {:0.2f}%.'.format(base_accuracy))
    if map == 'classification':
        H, M, F, C, pod, pofd, base_edi = EDI(base_model, x_test, y_test[label])
        print('Base EDI = {:0.4f}'.format(base_edi))
    print()

    improvement = 100 * (accuracy - base_accuracy) / base_accuracy
    print('Accuracy Improvement of {:0.2f}%'.format(improvement))

    s = open(dest + 'model_performance.txt', 'w')
    if map =='classification':
        s.write('Base Model Performance \n'
                'Base EDI: {:0.4f} \n'
                'Base Accuracy = {:0.2f}%. \n'
                '\n'
                'Tuned Model Performance \n'
                'EDI: {:0.4f} \n'
                'Accuracy = {:0.2f}%. \n'
                '\n'
                'Accuracy Improvement of {:0.2f}%.'.format(base_edi, base_accuracy, edi, accuracy,
                                                           improvement))
    else:
        s.write('Base Model Performance \n'
                'Base Average Error: {:0.4f} cm. \n'
                'Base Accuracy = {:0.2f}%. \n'
                '\n'
                'Tuned Model Performance \n'
                'Average Error: {:0.4f} cm. \n'
                'Accuracy = {:0.2f}%. \n'
                '\n'
                'Accuracy Improvement of {:0.2f}%.'.format(base_mean_error, base_accuracy, mean_error, accuracy, improvement))
    s.close()

    s = open(dest + 'best_params.txt', 'w')
    s.write(str(rf_random.best_params_))
    s.close()

    feature_importance(max_rf, x_train, dest)

def rf_kfold_fs(n_iter=5, cv=5, clean_features=True, map='classification', n_jobs=[-1], feat_list = [], dest=''):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=20)]
    # Number of features to consider at every split
    # max_features = ['auto', 'sqrt', None]
    max_features = [None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 300, num=50)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'n_jobs':n_jobs}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    model_name = '.{}{}{}{}{}'.format(f'{dt.today().year:02}', f'{dt.today().month:02}', f'{dt.today().day:02}',
                                      f'{dt.today().hour:02}', f'{dt.today().minute:02}')
    if map == 'classification':
        if len(feat_list) == 0:
            model_name = 'rf_10_kfold_clf' + model_name
            dest = rf_clf_destination_dir + '/' + model_name + '/'
            os.mkdir(dest)
        rf = RandomForestClassifier()
        base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        label = 'severe'
    elif map == 'regression':
        if len(feat_list) == 0:
            model_name = 'rf_10_kfold_reg' + model_name
            dest = rf_reg_destination_dir + '/' + model_name + '/'
            os.mkdir(dest)
        rf = RandomForestRegressor()
        base_model = RandomForestRegressor(n_estimators=10, random_state=42)
        label = 'hail_size'
    else:
        print('map must be "classification" or "regression"')

    # Random search of parameters, using 10 fold cross validation,
    # search across 1000 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=n_iter, cv=cv, verbose=4,
                                   random_state=42, n_jobs=-1)
    df = pd.read_csv(source_ds)

    # format start and end times in datetime objects
    dt_format = '%Y-%m-%d %H:%M:%S'  # set dt_format to the format printed above
    df['temp_start_time'] = [dt.strptime(startDate, dt_format) for startDate in df['start_time']]
    df['start_time'] = df['temp_start_time']

    df['temp_end_time'] = [dt.strptime(endDate, dt_format) for endDate in df['end_time']]
    df['end_time'] = df['temp_end_time']

    df = df.drop(['temp_end_time', 'temp_start_time'], axis=1)

    if clean_features:
        x_train = df.where(df['year'] < 2022).where(df['cape'] > 0).where((df['LD'] > 0)).where((df['hail_size'] > 0)).where(
            (df['start_time'].dt.month < 11)).where((df['start_time'].dt.month > 4)).dropna()
    else:
        x_train = df.where(df['year'] < 2022).dropna()

    y_train = x_train[['hail_size', 'severe']]
    x_train = x_train.drop(['event', 'year', 'start_time', 'end_time', 'hail_size', 'severe'], axis=1)
    for feat in feat_list:
       x_train = x_train.drop([feat], axis=1)

    x_test = df.where(df['year'] > 2021).dropna()
    y_test = x_test[['hail_size', 'severe']]
    x_test = x_test.drop(['event', 'year', 'start_time', 'end_time', 'hail_size', 'severe'], axis=1)

    print('feat_list:')
    print()
    print(feat_list)
    print()

    for feat in feat_list:
        try:
            x_test = x_test.drop([feat])
        except KeyError:
            print('KeyError: feat={}'.format(feat))
            raise

    # Fit the random search model
    rf_random.fit(x_train, y_train[label])
    params = rf_random.best_params_
    if map == 'classification':
        max_rf = RandomForestClassifier(n_jobs= -1,
                                        n_estimators=params['n_estimators'],
                                        min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        max_features=params['max_features'],
                                        max_depth=params['max_depth'],
                                        bootstrap=params['bootstrap'])
    else:
        max_rf = RandomForestRegressor(n_jobs= -1,
                                        n_estimators=params['n_estimators'],
                                        min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        max_features=params['max_features'],
                                        max_depth=params['max_depth'],
                                        bootstrap=params['bootstrap'])
    max_rf.fit(x_train, y_train[label])

    mean_error, accuracy = evaluate(max_rf, x_test, y_test[label], map)
    print('Tuned Model Performance')
    if map == 'regression':
        print('Average Error: {:0.4f} cm.'.format(mean_error))
        test_model(model=max_rf, map=map, dest=dest + 'feat_{}.'.format(len(x_train.columns)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    if map == 'classification':
        H, M, F, C, pod, pofd, edi = EDI(max_rf, x_test, y_test[label])
        print('EDI = {:0.4f}'.format(edi))
        test_res_dict = {'hits': [H], 'misses': [M], 'false_alarms': [F], 'correct_negative': [C], 'pod':[pod], 'pofd':[pofd], 'edi':edi}
        test_res = pd.DataFrame(data=test_res_dict)
        test_res.to_csv(dest + 'test_results.feat_{}.csv'.format(len(x_train.columns)))
    print()

    base_model.fit(x_train, y_train[label])
    base_mean_error, base_accuracy = evaluate(base_model, x_test, y_test[label], map)
    print('Base Model Performance')
    if map == 'regression':
        print('Base Average Error: {:0.4f} cm.'.format(base_mean_error))
    print('Base Accuracy = {:0.2f}%.'.format(base_accuracy))
    if map == 'classification':
        H, M, F, C, pod, pofd, base_edi = EDI(base_model, x_test, y_test[label])
        print('Base EDI = {:0.4f}'.format(base_edi))
    print()

    improvement = 100 * (accuracy - base_accuracy) / base_accuracy
    print('Accuracy Improvement of {:0.2f}%'.format(improvement))

    s = open(dest + 'model_performance.txt', 'w')
    if map =='classification':
        s.write('Base Model Performance \n'
                'Base EDI: {:0.4f} \n'
                'Base Accuracy = {:0.2f}%. \n'
                '\n'
                'Tuned Model Performance \n'
                'EDI: {:0.4f} \n'
                'Accuracy = {:0.2f}%. \n'
                '\n'
                'Accuracy Improvement of {:0.2f}%.'.format(base_edi, base_accuracy, edi, accuracy,
                                                           improvement))
    else:
        s.write('Base Model Performance \n'
                'Base Average Error: {:0.4f} cm. \n'
                'Base Accuracy = {:0.2f}%. \n'
                '\n'
                'Tuned Model Performance \n'
                'Average Error: {:0.4f} cm. \n'
                'Accuracy = {:0.2f}%. \n'
                '\n'
                'Accuracy Improvement of {:0.2f}%.'.format(base_mean_error, base_accuracy, mean_error, accuracy, improvement))
    s.close()

    s = open(dest + 'best_params.feat_{}.txt'.format(len(x_train.columns)), 'w')
    s.write(str(rf_random.best_params_))
    s.close()

    fi = feature_importance(max_rf, x_train, dest + 'feat_{}.'.format(len(x_train.index)))
    print()
    print('fi:')
    print()
    print(fi)
    print()
    if len(fi.index) > 10:
        feat_list = [fi.iloc[g].name for g in range(int(np.floor(len(fi.index) / 2)), len(fi.index))]
        return rf_kfold_fs(n_iter=n_iter, cv=cv, clean_features=clean_features, map=map, n_jobs=n_jobs, feat_list=feat_list, dest=dest)

def rf(source_dataset, val_size=0.25, map='classification', dest=''):
    x_train, x_val, y_train, y_val, x_tv = get_train_val(source_ds=source_dataset, val_size=val_size)

    x_tv = x_tv.drop(['start_time', 'end_time', 'hail_size'], axis=1)

    # find optimal rf model and create unique model name
    model_name = '.{}{}{}{}{}'.format(f'{dt.today().year:02}', f'{dt.today().month:02}', f'{dt.today().day:02}', f'{dt.today().hour:02}', f'{dt.today().minute:02}')
    if map == 'classification':
        model_name = 'rf_clf' + model_name
        dest = os.path.join(dest, model_name)
        os.mkdir(dest)
        rf, pred = rf_clf_model(x_train, x_val, y_train, y_val, dest)

    else:
        model_name = 'rf_reg' + model_name
        dest = os.path.join(dest, model_name)
        os.mkdir(dest)
        rf, pred = rf_reg_model(x_train, x_val, y_train, y_val, dest)

    score = test_model(model=rf, map=map, dest=dest, source_ds=source_dataset)
    s = open(os.path.join(dest, 'test_score.txt'), 'w')
    s.write('Test score: {}'.format(score))
    s.close()

    # calculate feature importance
    feature_importance(rf, x_tv, os.path.join(dest, 'feature_importance.csv'))

    # save training and validation datasets to csv files
    train_data = pd.concat([x_train, y_train], axis=1, join='inner')
    train_data.to_csv(os.path.join(dest, 'training_data.csv'), index=False)

    pred_df = pd.DataFrame(data=pred, columns=['predicted'])
    val_data = pd.concat([x_val, y_val, pred_df], axis=1, join='inner')
    val_data.to_csv(os.path.join(dest, 'validation_data.csv'), index=False)

    return rf

def xgb_clf_rand_params(x_train, x_val, y_train, y_val, col, dest, n_iter=1000, n_jobs=-1):
    max_accuracy_prod = 0
    max_cm = []
    max_xgb = xgboost.XGBClassifier()
    max_pred = []
    for i in range(n_iter):
        print('Working on XGB {} of {}. Current accuracy is {}'.format((i+1), n_iter, max_accuracy_prod), end='\r')
        learning_rate = random.uniform(0.001, 0.3)
        max_depth = random.randint(5, 120)
        min_child_weight = random.randint(1, 20)
        xgb = xgboost.XGBClassifier(tree_method='hist',
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    min_child_weight=min_child_weight,
                                    random_state=42,
                                    n_jobs=n_jobs,
                                    use_label_encoder=False,
                                    eval_metric='error',
                                    objective='binary:logistic')

        xgb.fit(x_train, y_train[col])
        pred_val = xgb.predict(X=x_val)

        cm = confusion_matrix(y_val[col], pred_val)

        C = cm[0][0]
        M = cm[0][1]
        F = cm[1][0]
        H = cm[1][1]

        pod = H / (H + M)
        pofd = F / (F + C)

        num = np.log(pofd) - np.log(pod)
        den = np.log(pofd) + np.log(pod)
        edi = num / den

        accuracy_prod = edi



        # cm = confusion_matrix(y_val[col], pred_val)
        # cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
        # accuracy_prod = cm_norm[0][0] * cm_norm[1][1]
        if accuracy_prod > max_accuracy_prod:
            max_accuracy_prod = accuracy_prod
            max_cm = cm
            max_pred = pred_val
            max_xgb = xgb

        com_sc = pd.DataFrame(
            {'learning_rate': [learning_rate],
             'max_depth': [max_depth],
             'min_child_weight':[min_child_weight],
             'train_score': [xgb.score(x_train, y_train[col])],
             'val_score': [xgb.score(x_val, y_val[col])],
             'cm_score': [accuracy_prod],
             'true_positive': [cm[1][1]],
             'false_positive': [cm[0][1]],
             'false_negative': [cm[1][0]],
             'true_negative': [cm[0][0]]
             }
        )

        try:
            final_com_sc = pd.concat([final_com_sc, com_sc], ignore_index=True)
        except:
            final_com_sc = com_sc

    final_com_sc = final_com_sc.sort_values('cm_score', ascending=False)
    final_com_sc.to_csv(dest + 'xgb_combination_scores.csv', index=False)

    classes = max_xgb.classes_
    plot_confusion_matrix(max_cm, classes, destination_dir=dest)

    # save model
    max_xgb.save_model(dest + 'xgb_clf_model.txt')

    return max_xgb, max_pred

def xgb_reg_rand_params(x_train, x_val, y_train, y_val, col, dest, n_iter=1000, n_jobs=-1):
    max_accuracy_prod = 0
    max_xgb = xgboost.XGBRegressor()
    max_pred = []
    for i in range(n_iter):
        print('Working on XGB {} of {}. Current accuracy is {}'.format((i+1), n_iter, max_accuracy_prod), end='\r')
        learning_rate = random.uniform(0.001, 0.3)
        max_depth = random.randint(5, 120)
        min_child_weight = random.randint(1, 20)
        xgb = xgboost.XGBRegressor(tree_method='hist',
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    min_child_weight=min_child_weight,
                                    random_state=42,
                                    n_jobs=n_jobs,
                                    use_label_encoder=False,
                                    eval_metric='error'
                                   )


        xgb.fit(x_train, y_train[col])
        pred_val = xgb.predict(X=x_val)

        accuracy_prod = xgb.score(x_val, y_val[col])
        if accuracy_prod > max_accuracy_prod:
            max_accuracy_prod = accuracy_prod
            max_pred = pred_val
            max_xgb = xgb

        com_sc = pd.DataFrame(
            {'learning_rate': [learning_rate],
             'max_depth': [max_depth],
             'min_child_weight':[min_child_weight],
             'train_score': [xgb.score(x_train, y_train[col])],
             'val_score': [xgb.score(x_val, y_val[col])]
             }
        )

        try:
            final_com_sc = pd.concat([final_com_sc, com_sc], ignore_index=True)
        except:
            final_com_sc = com_sc

    final_com_sc = final_com_sc.sort_values('val_score', ascending=False)
    final_com_sc.to_csv(dest + 'xgb_combination_scores.csv', index=False)

    # save model
    max_xgb.save_model(dest + 'xgb_reg_model.txt')

    return max_xgb, max_pred

def XGB_rand(n_iter=1000, val_size=0.25, map='classification'):
    # load data
    x_train, x_val, y_train, y_val, x_tv = get_train_val(val_size=val_size)

    x_tv = x_tv.drop(['start_time', 'end_time', 'hail_size'], axis=1)

    # set model mapping
    model_name = '.{}{}{}{}{}'.format(f'{dt.today().year:02}', f'{dt.today().month:02}', f'{dt.today().day:02}',
                                      f'{dt.today().hour:02}', f'{dt.today().minute:02}')
    if map == 'regression':
        model_name = 'xgb_reg' + model_name
        col = 'hail_size'
        dest = xgb_reg_destination_dir + '/' + model_name + '/'
        os.mkdir(dest)
        xgb, pred = xgb_reg_rand_params(x_train, x_val, y_train, y_val, col, dest, n_iter=n_iter)
    else:
        model_name = 'xgb_clf_edi' + model_name
        col = 'severe'
        dest = xgb_clf_destination_dir  + '/' + model_name + '/'
        os.mkdir(dest)
        xgb, pred = xgb_clf_rand_params(x_train, x_val, y_train, y_val, col, dest, n_iter=n_iter)

    # save xgb parameters in human readable format
    params = xgb.get_params()
    keys = list(params.keys())
    vals = list(params.values())
    with open(dest + "xgb_params.txt", "w") as text_file:
        text_file.write('XGB : {')
        for i in range(len(keys)):
            text_file.write(str(keys[i]) + ': ' + str(vals[i]))
        text_file.write('}')
        text_file.close()

    # save training and validation datasets to csv files
    train_data = pd.concat([x_train, y_train], axis=1, join='inner')
    train_data.to_csv(dest + 'xgb_train_data.csv', index=False)

    pred = pd.DataFrame(data=pred, columns=['predicted'])
    val_data = pd.concat([x_val, y_val, pred], axis=1, join='inner')
    val_data.to_csv(dest + 'xgb_val_data.csv', index=False)

    # calculate feature importance
    feature_importance(xgb, x_tv, dest)

    test_score = test_model(xgb, map, dest)
    print('test score: ', test_score)
    with open(dest + "xgb_test_score.txt", "w") as text_file:
        text_file.write('test score: {}'.format(test_score))
        text_file.close()

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


















