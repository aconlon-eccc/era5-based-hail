import os
from datetime import datetime as dt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
import random
from ML import get_train_val, feature_importance, source_ds, test_model, evaluate, EDI

def MLP_clf_model(x_train, x_val, y_train, y_val, dest, n_iter=1000):
    max_accuracy_prod = 0
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

        H = cm[1][1]
        F = cm[0][1]
        M = cm[1][0]
        C = cm[0][0]

        pod = H / (H + M)
        pofd = F / (F + C)

        num = np.log(pofd) - np.log(pod)
        den = np.log(pofd) + np.log(pod)
        edi = num / den

        # cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
        # accuracy_prod = cm_norm[0][0] * cm_norm[1][1]
        accuracy_prod = edi
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
             'train_score': [mlp.score(x_train, y_train['severe'])],
             'val_score': [mlp.score(x_val, y_val['severe'])],
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

    final_com_sc = final_com_sc.sort_values('val_score', ascending=False)
    final_com_sc.to_csv(os.path.join(dest, 'mlp_combination_scores.csv'), index=False)

    print('Final validation accuracy: ', max_accuracy_prod)

    return max_mlp, max_pred

def MLP_reg_model(x_train, x_val, y_train, y_val, dest, n_iter=1000):
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
             'val_score': [accuracy_prod]
             }
        )

        try:
            final_com_sc = pd.concat([final_com_sc, com_sc], ignore_index=True)
        except:
            final_com_sc = com_sc

    final_com_sc = final_com_sc.sort_values('val_score', ascending=False)
    final_com_sc.to_csv(os.path.join(dest, 'mlp_combination_scores.csv'), index=False)

    print('Final validation accuracy: ', max_accuracy_prod)

    return max_mlp, max_pred

def MLP(n_iter=1000, val_size=0.25, map='classification', dest=''):
    x_train, x_val, y_train, y_val, x_tv, x_test, y_test = get_train_val(val_size=val_size, x_scaled=True)

    # set model mapping
    model_name = '.{}{}{}{}{}'.format(f'{dt.today().year:02}', f'{dt.today().month:02}', f'{dt.today().day:02}',
                                      f'{dt.today().hour:02}', f'{dt.today().minute:02}')
    # find optimal rf model
    if map == 'classification':
        model_name = 'mlp_clf' + model_name
        dest = os.path.join(dest, model_name)
        os.mkdir(dest)
        mlp, pred = MLP_clf_model(x_train, x_val, y_train, y_val, dest, n_iter=n_iter)
        y_test = y_test['severe']
    else:
        model_name = 'mlp_reg' + model_name
        dest = os.path.join(dest, model_name)
        os.mkdir(dest)
        mlp, pred = MLP_reg_model(x_train, x_val, y_train, y_val, dest, n_iter=n_iter)
        y_test = y_test['hail_size']

    # test model
    test_pred_dict = {'test_pred': mlp.predict(x_test)}
    test_score = mlp.score(x_test, y_test)

    with open(os.path.join(dest, "mlp_test_score.txt"), "w") as text_file:
        text_file.write('MLP : {}'.format(test_score))
        text_file.close()

    # print mlp parameters
    params = mlp.get_params()
    keys = list(params.keys())
    vals = list(params.values())
    print('MLP parameters: ', str(params))
    with open(os.path.join(dest, "mlp_params.txt"), "w") as text_file:
        text_file.write('MLP {}: {}'.format(map, '{'))
        for i in range(len(keys)):
            text_file.write(str(keys[i]) + ': ' + str(vals[i]))
        text_file.write('}')
        text_file.close()

    # save training and testing datasets to csv files
    x_train_df = pd.DataFrame(data=x_train)
    y_train_df = pd.DataFrame(data=y_train)
    train_data = pd.concat([x_train_df, y_train_df], axis=1, join='inner')
    train_data.to_csv(os.path.join(dest, 'train_data.csv'), index=False)

    pred = pd.DataFrame(data=pred, columns=['predicted'])
    x_val_df = pd.DataFrame(data=x_val)
    y_val_df = pd.DataFrame(data=y_val)
    val_data = pd.concat([x_val_df, y_val_df, pred], axis=1, join='inner')
    val_data.to_csv(os.path.join(dest, 'val_data.csv'), index=False)

    test_data = pd.DataFrame(data=test_pred_dict)
    test_data = pd.concat([y_test, test_data], axis=1, join='inner')
    test_data.to_csv(os.path.join(dest, 'test_data.csv'), index=False)

    return mlp