import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import os
from data.kenyan_water_dataset import KenyanWaterDataset
import pandas as pd
from utils import pickle_object, read_pickle_object

from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import math
import copy
from sklearn.model_selection import train_test_split


def perform_match_exact(row, df, *args):
    # row is the the item that we want to match
    # df is the source Pandas dataframe that we want to match it with other items
    # print('Start matching')
    sub_set = df

    for arg in args:
        sub_set = sub_set.loc[sub_set[arg] == row[arg]]
        # print(sub_set)

    return sub_set.index


def perfom_matching_v2(row, indexes, df_data):
    current_index = int(row['index'])
    prop_score_logit = row['propensity_score_logit']
    for idx in indexes[current_index, :]:
        # if (current_index != idx) and (row.treatment == 1) and (df_data.loc[idx].treatment == 0):
        counter_factual_x = np.logical_xor(row.treatment,1).astype(int)
        if (current_index != idx) and (counter_factual_x == df_data.loc[idx].treatment ):
            return int(idx)

def get_counterfactual(row, df_data):
    matched = int(row['matched_element'])
    to_return = df_data.loc[matched].targets
    return int(to_return)




def logit(p):
    logit_value = math.log(p / (1 - p))
    return logit_value

#
# def propensity_score_matching(args):
#     dataset = KenyanWaterDataset(**vars(args))
#
#     dataset_base = copy.deepcopy(dataset.train)
#
#     # dataset_proc = dataset_proc.reset_index()
#     dataset_proc = dataset_base.drop(['targets', 'uy'], axis=1, inplace=False)
#
#
#
#     T = dataset_proc.treatment
#     X = dataset_proc.loc[:, dataset_proc.columns != 'treatment']
#
#     # Design pipeline to build the treatment estimator
#
#     pipe = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,
#                                       max_depth=1, random_state=0).fit(X, T)
#
#     predictions = pipe.predict_proba(X)
#     predictions_binary = pipe.predict(X)
#
#     print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(T, predictions_binary)))
#     print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(T, predictions_binary)))
#     print('F1 score is: {:.4f}'.format(metrics.f1_score(T, predictions_binary)))
#
#     predictions_logit = np.array([logit(xi) for xi in predictions[:, 1]])
#
#     X.loc[:, 'propensity_score'] = predictions[:, 1]
#     X.loc[:, 'propensity_score_logit'] = predictions_logit
#     X.loc[:, 'targets'] = dataset_base.targets
#     X.loc[:, 'treatment'] = dataset_base.treatment
#     X.loc[:, 'uy'] = dataset_base.uy
#
#     caliper = np.std(X.propensity_score) * 0.25
#
#     print('\nCaliper (radius) is: {:.4f}\n'.format(caliper))
#
#     X = X.reset_index().drop('index',axis=1)
#
#     knn = NearestNeighbors(n_neighbors=30, p=2,n_jobs=-1)
#     knn.fit(X[['propensity_score_logit']].to_numpy())
#
#     distances, indexes = knn.kneighbors(
#         X[['propensity_score_logit']].to_numpy(), n_neighbors=30)
#     X['matched_element'] = X.reset_index().apply(perfom_matching_v2, axis=1, args=(indexes, X))
#
#     treated_with_match = ~X.matched_element.isna()
#     print(np.sum(treated_with_match==1))
#
#     X['treatment_prime'] = np.logical_xor(X['treatment'],1).astype(int)
#     X = X.reset_index()
#     X.dropna(axis=0, inplace=True)
#
#     temp = []
#     for index, row in X.iterrows():
#         temp.append(get_counterfactual(row, X))
#
#     X.insert(1, 'targets_prime', temp)
#     X.drop(['index','level_0'],axis=1,inplace=True)
#
#     train_idx, test_idx = train_test_split(list(range(len(X))), test_size=0.2)
#     to_save ={}
#     to_save['train'] = X.iloc[train_idx].astype('float')
#     to_save['test'] = X.iloc[test_idx].astype('float')
#
#     pickle_object(to_save, args.save_path)


def  propensity_score_matching(args):
    dataset = SyntheticDataset(**vars(args))

    dataset.train = pd.DataFrame.from_dict(dataset.train)
    dataset.test = pd.DataFrame.from_dict(dataset.test)

    if args.test =='test':
        print('_'+args.test)
        dataset_base = copy.deepcopy(dataset.test)
    else:
        dataset_base = copy.deepcopy(dataset.train)

    # dataset_proc = dataset_proc.reset_index()
    dataset_proc = dataset_base.drop(['Y', 'Uy','Y_prime','X_prime','Ux'], axis=1, inplace=False)



    T = dataset_proc.X
    X = dataset_proc.loc[:, (dataset_proc.columns != 'X') ]


    # Design pipeline to build the treatment estimator

    pipe = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,
                                      max_depth=1, random_state=0).fit(X, T)

    predictions = pipe.predict_proba(X)
    predictions_binary = pipe.predict(X)

    print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(T, predictions_binary)))
    print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(T, predictions_binary)))
    print('F1 score is: {:.4f}'.format(metrics.f1_score(T, predictions_binary)))

    _data = X
    _data.loc[:, 'propensity_score'] = np.array([logit(xi) for xi in predictions[:, 1]])#predictions[:, 1]
    _data.loc[:, 'Y'] = dataset_base.Y
    _data.loc[:, 'X'] = dataset_base.X
    _data.loc[:, 'Uy'] = dataset_base.Uy

    data_out = copy.deepcopy(_data)

    treated = _data.loc[_data.X == 1]
    control = _data.loc[_data.X == 0]

    Y_prime = np.zeros(len(data_out))
    X_prime = np.zeros(len(data_out))

    control_neighbors = (
        NearestNeighbors(n_neighbors=args.neighb, algorithm='ball_tree')
            .fit(control['propensity_score'].values.reshape(-1, 1))
    )
    distances, indices = control_neighbors.kneighbors(treated['propensity_score'].values.reshape(-1, 1))

    att = 0
    numtreatedunits = treated.shape[0]
    # for i in range(numtreatedunits):
    #     treated_outcome = treated.iloc[i]['Y'].item()
    #     control_outcome = control.iloc[indices[i]]['Y'].item()

    random_indices =np.random.randint(0,args.neighb,len(indices))
    Y_prime[np.array(treated.index)] = np.array([control.iloc[indices[i,j]]['Y'].item() for i,j in enumerate(random_indices)])
    X_prime[np.array(treated.index)] = 0


    # Now computing ATC
    treated_neighbors = (
        NearestNeighbors(n_neighbors=args.neighb, algorithm='ball_tree')
            .fit(treated['propensity_score'].values.reshape(-1, 1))
    )
    distances, indices = treated_neighbors.kneighbors(control['propensity_score'].values.reshape(-1, 1))
    atc = 0
    random_indices =np.random.randint(0,args.neighb,len(indices))
    Y_prime[np.array(control.index)] =np.array([treated.iloc[indices[i,j]]['Y'].item() for i,j in enumerate(random_indices)])
    X_prime[np.array(control.index)] = 1

    data_out.insert(1, 'Y_prime', Y_prime)
    data_out.insert(1, 'X_prime', X_prime)


    Xs = data_out['X'].values
    Xs_prime = data_out['X_prime'].values
    Ys = data_out['Y'].values
    Ys_prime = data_out['Y_prime'].values
    idx_given_y_1 = np.where(Ys[np.where(Xs == 1)[0]] == 1)[0]
    idx_query_y_0 = np.where(Ys_prime[np.where(Xs_prime == 0)[0]] == 0)[0]
    idx_y_1_y_prime_0 = list(set(idx_given_y_1).intersection(idx_query_y_0))
    prob_necessity_1 = len(idx_y_1_y_prime_0) / len(idx_given_y_1)
    print('N',prob_necessity_1)

    idx_given_y_0 = np.where(Ys[np.where(Xs == 0)[0]] == 0)[0]
    idx_query_y_1 = np.where(Ys_prime[np.where(Xs_prime == 1)[0]] == 1)[0]
    idx_y_0_y_prime_1 = list(set(idx_given_y_0).intersection(idx_query_y_1))
    prob_suf_1 = len(idx_y_0_y_prime_1) / len(idx_given_y_0)
    print('S', prob_suf_1)

    train_idx, test_idx = train_test_split(list(range(len(X))), test_size=0.2)

    data_out.reset_index().drop(['index'],axis=1,inplace=True)

    to_save ={}
    to_save['train'] = data_out.iloc[train_idx].astype('float')
    to_save['test'] = data_out.iloc[test_idx].astype('float')

    pickle_object(to_save, args.save_path)
    print(args.save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data',
                        default='/vol/medic01/users/av2514/Pycharm_projects/Twin_Nets_Causality/data/Datasets/dataverse_files/dta')
    parser.add_argument('--load_path', default='./Datasets/kenyan_water_proc_single_uy_normal_3.pkl')
    parser.add_argument('--load_dataset', default=False)
    parser.add_argument('--save_dataset', default=True)
    parser.add_argument('--save_path', default='./Datasets/kenyan_water_proc_single_uy_normal_with_propensity_3.pkl')
    args = parser.parse_args()

    propensity_score_matching(args)
