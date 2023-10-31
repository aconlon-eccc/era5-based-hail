import os
from datetime import datetime as dt
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, r2_score
import random
from ML.ML import get_train_val, feature_importance, test_model, evaluate, EDI

# Random Forest Classifier called by rf function below
def rf_clf_model(x_train, x_val, x_test, y_train, y_val, y_test, dest, max_depths=[None, 10, 20, 50, 75, 100, 125, 150, 175, 200], n_jobs=-1):
    # using a max_accuracy_prod as evaluation metric. initialized to zero
    max_accuracy_prod = 0

    # initialize best performing RF model
    max_rf = RandomForestClassifier()

    # initialize predictions of best performing RF model
    max_pred = []

    # search for best RF model
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

        # save scores and model parameters to dataframe
        com_sc = pd.DataFrame(
            {
             'max_depth': [d],
             'train_r2_score': [rf.score(x_train, y_train['severe'])],
             'val_r2_score': [rf.score(x_val, y_val['severe'])],
             'edi': [accuracy_prod],
             'pod': [pod],
             'pofd': [pofd],
             'true_negative': [C],
             'false_positive': [F],
             'false_negative': [M],
             'true_positive': [H]
             }
        )

        # fancy way of initializing, and adding to, the final dataframe holding details of all RF models produced in the search
        try:
            final_com_sc = pd.concat([final_com_sc, com_sc], ignore_index=True)
        except:
            final_com_sc = com_sc

    final_com_sc = final_com_sc.sort_values('edi', ascending=False)
    final_com_sc.to_csv(os.path.join(dest, 'all_scores.csv'), index=False)

    # test scores
    pred_test = max_rf.predict(X=x_test)
    pred_test_df = y_test
    pred_test_df['prediction'] = pred_test
    pred_test_df.to_csv(os.path.join(dest, 'test_pred.csv'), index=False)

    test_cm = confusion_matrix(y_test['severe'], pred_test)

    C = test_cm[0][0]
    M = test_cm[1][0]
    F = test_cm[0][1]
    H = test_cm[1][1]

    pod = H / (H + M)
    pofd = F / (F + C)

    num = np.log(pofd) - np.log(pod)
    den = np.log(pofd) + np.log(pod)
    edi = num / den

    test_sc = pd.DataFrame(
        data={
            'edi': [edi],
            'true_negative': [C],
            'false_positive': [F],
            'false_negative': [M],
            'true_positive': [H]
        })

    test_sc.to_csv(os.path.join(dest, 'test_scores.csv'), index=False)

    # save rf parameters in human readable format
    params = max_rf.get_params()
    keys = list(params.keys())
    vals = list(params.values())
    with open(os.path.join(dest, "max_rf_params.txt"), "w") as text_file:
        text_file.write('RandomForestClassifier : {')
        for i in range(len(keys)):
            text_file.write(str(keys[i]) + ': ' + str(vals[i]))
        text_file.write('}')
        text_file.close()

    return max_rf, max_pred

# Random Forest Regressor called by rf function below
def rf_reg_model(x_train, x_val, x_test, y_train, y_val, y_test, dest, max_depths=[None, 10, 20, 50, 75, 100], n_jobs=-1):
    # using a max_accuracy_prod as evaluation metric. initialized to zero
    max_accuracy_prod = 0

    # initialize best performing RF model
    max_rf = RandomForestRegressor()

    # initialize predictions of best performing RF model
    max_pred = []

    # search for best RF model
    for d in max_depths:
        rf = RandomForestRegressor(n_estimators=random.randint(100, 1000),
                                    max_features=None,
                                    max_depth=d,
                                    n_jobs=n_jobs,
                                    random_state=42)

        # had issues with certain depths
        try:
            rf.fit(x_train, y_train['hail_size'])
        except TypeError:
            print('Value error for max_depth = {}'.format(d))

        # had issues with columns matching in the training and validating training sets
        try:
            pred_val = rf.predict(X=x_val)
        except IndexError:
            print('IndexError. train cols = {}, val vols = {}'.format(len(x_train.columns), len(x_val.columns)))

        # using built-in r^2 score to find best parameters for model
        accuracy_prod = rf.score(x_val, y_val['hail_size'])
        if accuracy_prod > max_accuracy_prod:
            print('new max: ', accuracy_prod)
            max_accuracy_prod = accuracy_prod
            max_pred = pred_val
            max_rf = rf

        # save scores and model parameters to dataframe
        com_sc = pd.DataFrame(
            {
             'max_depth': [d],
             'train_r2_score': [rf.score(x_train, y_train['hail_size'])],
             'val_r2_score': [accuracy_prod]
             }
        )
        # fancy way of initializing, and adding to, the final dataframe holding details of all RF models produced in the search
        try:
            final_com_sc = pd.concat([final_com_sc, com_sc], ignore_index=True)
        except:
            final_com_sc = com_sc

    final_com_sc = final_com_sc.sort_values('val_score', ascending=False)
    final_com_sc.to_csv(os.path.join(dest, 'rf_combination_scores.csv'), index=False)

    # save rf parameters in human readable format
    params = max_rf.get_params()
    keys = list(params.keys())
    vals = list(params.values())
    with open(os.path.join(dest, "rf_reg_params.txt"), "w") as text_file:
        text_file.write('RandomForestRegressor : {')
        for i in range(len(keys)):
            text_file.write(str(keys[i]) + ': ' + str(vals[i]))
        text_file.write('}')
        text_file.close()

    # test scores
    pred_test = max_rf.predict(X=x_test)
    pred_test_df = y_test
    pred_test_df['prediction'] = pred_test
    pred_test_df['r2_score'] = [r2_score(y_test, pred_test)]
    pred_test_df.to_csv(os.path.join(dest, 'test_pred.csv'), index=False)

    return max_rf, max_pred

# Random Forest k-fold Classifier or Regressor (by specifying 'map') on all features in source_dataset
def rf_kfold(source_dataset, dest='', n_iter=5, cv=5, clean_features=True, map='classification', n_jobs=[-1]):
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
                   'n_jobs':n_jobs
                   }
    # Use the random grid to search for best hyperparameters
    # Create the base model to tune
    model_name = '.{}{}{}{}{}'.format(f'{dt.today().year:02}', f'{dt.today().month:02}', f'{dt.today().day:02}',
                                      f'{dt.today().hour:02}', f'{dt.today().minute:02}')
    if map == 'classification':
        model_name = 'rf_kfold_clf' + model_name
        dest = os.path.join(dest, model_name)
        os.mkdir(dest)
        rf = RandomForestClassifier()
        base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        label = 'severe'
    elif map == 'regression':
        model_name = 'rf_kfold_reg' + model_name
        dest = os.path.join(dest, model_name)
        os.mkdir(dest)
        rf = RandomForestRegressor()
        base_model = RandomForestRegressor(n_estimators=10, random_state=42)
        label = 'hail_size'
    else:
        print('map must be "classification" or "regression"')

    # Random search of parameters, using 10 fold cross validation,
    # search across 1000 different combinations, and use all available cores (n_jobs=-1)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=n_iter, cv=cv, verbose=4,
                                   random_state=42, n_jobs=-1)
    df = pd.read_csv(source_dataset)

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
        test_res.to_csv(os.path.join(dest, 'test_results.csv'))
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

    s = open(os.path.join(dest, 'model_performance.txt'), 'w')
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

    s = open(os.path.join(dest, 'best_params.txt'), 'w')
    s.write(str(rf_random.best_params_))
    s.close()

    feature_importance(max_rf, x_train, dest)

    return max_rf

# Random Forest k-fold Classifier or Regressor (by specifying 'map') trained on top 10 performing features by iteratively
# calling rf_kfold_fs (i.e. is a recursive function which returns a model trained on 10 most important features)
def rf_kfold_fs(source_dataset, dest='', n_iter=5, cv=5, num_features=10, clean_features=True, map='classification', n_jobs=[-1], feat_list = []):
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
            dest = os.path.join(dest, model_name)
            os.mkdir(dest)
        rf = RandomForestClassifier()
        base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        label = 'severe'
    elif map == 'regression':
        if len(feat_list) == 0:
            model_name = 'rf_10_kfold_reg' + model_name
            dest = os.path.join(dest, model_name)
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
    df = pd.read_csv(source_dataset)

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
        test_model(model=max_rf, map=map, dest= os.path.join(dest, 'feat_{}.'.format(len(x_train.columns))))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    if map == 'classification':
        H, M, F, C, pod, pofd, edi = EDI(max_rf, x_test, y_test[label])
        print('EDI = {:0.4f}'.format(edi))
        test_res_dict = {'hits': [H], 'misses': [M], 'false_alarms': [F], 'correct_negative': [C], 'pod':[pod], 'pofd':[pofd], 'edi':edi}
        test_res = pd.DataFrame(data=test_res_dict)
        test_res.to_csv(os.path.join(dest, 'test_results.feat_{}.csv'.format(len(x_train.columns))))
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
    if len(fi.index) > num_features:
        feat_list = [fi.iloc[g].name for g in range(int(np.floor(len(fi.index) / 2)), len(fi.index))]
        return rf_kfold_fs(n_iter=n_iter, cv=cv, clean_features=clean_features, map=map, n_jobs=n_jobs, feat_list=feat_list, dest=dest)
    else:
        return max_rf

def rf(source_ds, dest='', val_size=0.25, map='classification', kfold=False, n_iter=5, cv=5,
       most_important_features=False, num_features=10, clean_features=True,
       max_depths=[None, 10, 20, 50, 75, 100, 125, 150, 175, 200]):
    if kfold:
        if most_important_features:
            return rf_kfold_fs(source_dataset=source_ds, dest=dest, n_iter=n_iter, cv=cv, num_features=num_features,
                        clean_features=clean_features, map=map, n_jobs=[-1], feat_list=[])
        else:
            return rf_kfold_fs(source_dataset=source_ds, dest=dest, n_iter=n_iter, cv=cv, clean_features=clean_features,
                        map=map, n_jobs=[-1])
    else:
        # split data into training and validation data where x_tv is a dataframe composed of x_train and x_val
        x_train, x_val, y_train, y_val, x_tv, x_test, y_test = get_train_val(source_ds=source_ds, val_size=val_size)

        x_tv = x_tv.drop(['start_time', 'end_time', 'hail_size'], axis=1)

        # find optimal rf model and create unique directory name for all files saved during optimal model search
        model_name = '.{}{}{}{}{}'.format(f'{dt.today().year:02}', f'{dt.today().month:02}', f'{dt.today().day:02}', f'{dt.today().hour:02}', f'{dt.today().minute:02}')
        if map == 'classification':
            model_name = 'rf_clf' + model_name
            dest = os.path.join(dest, model_name)
            os.mkdir(dest)
            rf, pred = rf_clf_model(x_train, x_val, x_test, y_train, y_val, y_test, dest, max_depths=max_depths)

        else:
            model_name = 'rf_reg' + model_name
            dest = os.path.join(dest, model_name)
            os.mkdir(dest)
            rf, pred = rf_reg_model(x_train, x_val, x_test, y_train, y_val, y_test, dest)

        # calculate feature importance
        feature_importance(rf, x_tv, os.path.join(dest, 'feature_importance.csv'))

        # save training and validation datasets to csv files for post-analysis / repeatability
        train_data = pd.concat([x_train, y_train], axis=1, join='inner')
        train_data.to_csv(os.path.join(dest, 'training_data.csv'), index=False)

        pred_df = pd.DataFrame(data=pred, columns=['predicted'])
        val_data = pd.concat([x_val, y_val, pred_df], axis=1, join='inner')
        val_data.to_csv(os.path.join(dest, 'validation_data.csv'), index=False)

        return rf
