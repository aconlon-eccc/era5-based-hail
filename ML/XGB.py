import os
from datetime import datetime as dt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import xgboost
import random
from ML import get_train_val, feature_importance, test_model

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
        F = cm[0][1]
        M = cm[1][0]
        H = cm[1][1]

        pod = H / (H + M)
        pofd = F / (F + C)

        num = np.log(pofd) - np.log(pod)
        den = np.log(pofd) + np.log(pod)
        edi = num / den

        accuracy_prod = edi
        if accuracy_prod > max_accuracy_prod:
            max_accuracy_prod = accuracy_prod
            max_pred = pred_val
            max_xgb = xgb

        com_sc = pd.DataFrame(
            {'learning_rate': [learning_rate],
             'max_depth': [max_depth],
             'min_child_weight':[min_child_weight],
             'train_score': [xgb.score(x_train, y_train[col])],
             'val_score': [xgb.score(x_val, y_val[col])],
             'edi': [accuracy_prod],
             'true_positive': [H],
             'false_positive': [F],
             'false_negative': [M],
             'true_negative': [C]
             }
        )

        try:
            final_com_sc = pd.concat([final_com_sc, com_sc], ignore_index=True)
        except:
            final_com_sc = com_sc

    final_com_sc = final_com_sc.sort_values('edi', ascending=False)
    final_com_sc.to_csv(os.path.join(dest, 'xgb_scores.csv'), index=False)

    # save model
    max_xgb.save_model(os.path.join(dest, 'xgb_clf_model.txt'))

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
    final_com_sc.to_csv(os.path.join(dest, 'xgb_combination_scores.csv'), index=False)

    # save model
    max_xgb.save_model(os.path.join(dest, 'xgb_reg_model.txt'))

    return max_xgb, max_pred

def xgb_rand(n_iter=1000, val_size=0.25, map='classification', dest=''):
    # load data
    x_train, x_val, y_train, y_val, x_tv = get_train_val(val_size=val_size)

    x_tv = x_tv.drop(['start_time', 'end_time', 'hail_size'], axis=1)

    # set model mapping
    model_name = '.{}{}{}{}{}'.format(f'{dt.today().year:02}', f'{dt.today().month:02}', f'{dt.today().day:02}',
                                      f'{dt.today().hour:02}', f'{dt.today().minute:02}')
    if map == 'regression':
        model_name = 'xgb_reg' + model_name
        col = 'hail_size'
        dest = os.path.join(dest, model_name)
        os.mkdir(dest)
        xgb, pred = xgb_reg_rand_params(x_train, x_val, y_train, y_val, col, dest, n_iter=n_iter)
    else:
        model_name = 'xgb_clf_edi' + model_name
        col = 'severe'
        dest = os.path.join(dest, model_name)
        os.mkdir(dest)
        xgb, pred = xgb_clf_rand_params(x_train, x_val, y_train, y_val, col, dest, n_iter=n_iter)

    # save xgb parameters in human readable format
    params = xgb.get_params()
    keys = list(params.keys())
    vals = list(params.values())
    with open(os.path.join(dest, "xgb_params.txt"), "w") as text_file:
        text_file.write('XGB : {')
        for i in range(len(keys)):
            text_file.write(str(keys[i]) + ': ' + str(vals[i]))
        text_file.write('}')
        text_file.close()

    # save training and validation datasets to csv files
    train_data = pd.concat([x_train, y_train], axis=1, join='inner')
    train_data.to_csv(os.path.join(dest, 'xgb_train_data.csv'), index=False)

    pred = pd.DataFrame(data=pred, columns=['predicted'])
    val_data = pd.concat([x_val, y_val, pred], axis=1, join='inner')
    val_data.to_csv(os.path.join(dest, 'xgb_val_data.csv'), index=False)

    # calculate feature importance
    feature_importance(xgb, x_tv, dest)

    test_score = test_model(xgb, map, dest)
    print('test score: ', test_score)
    with open(os.path.join(dest, "xgb_test_score.txt"), "w") as text_file:
        text_file.write('test score: {}'.format(test_score))
        text_file.close()