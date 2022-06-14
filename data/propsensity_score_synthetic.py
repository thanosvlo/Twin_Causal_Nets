import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
from data.synthetic_dataset_wt_conf import SyntheticDataset
import pandas as pd
from utils import pickle_object

from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import math
import copy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
        counter_factual_x = np.logical_xor(row.X,1).astype(int)
        if (current_index != idx) and (counter_factual_x == df_data.loc[idx].X ):
            return int(idx)

def get_counterfactual(row, df_data):
    matched = int(row['matched_element'])
    to_return = df_data.loc[matched].Y
    return int(to_return)




def logit(p):
    if p <= 1e-10:
        p = 1e-10
    if p ==1:
        p= 1-1e-10
    logit_value = math.log(p / (1 - p))
    return logit_value

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


    pipe = GradientBoostingClassifier(n_estimators=200, learning_rate=1,
                                      max_depth=2, random_state=0, ).fit(X, T)

    predictions = pipe.predict_proba(X)
    predictions_binary = pipe.predict(X)

    print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(T, predictions_binary)))
    print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(T, predictions_binary)))
    print('F1 score is: {:.4f}'.format(metrics.f1_score(T, predictions_binary)))

    _data = X
    _data.loc[:, 'propensity_score'] = predictions[:, 1]
    _data.loc[:, 'Y'] = dataset_base.Y
    _data.loc[:, 'X'] = dataset_base.X
    _data.loc[:, 'Uy'] = dataset_base.Uy

    data_out = copy.deepcopy(_data)

    treated = _data.loc[_data.X == 1]
    control = _data.loc[_data.X == 0]




    Y_prime = np.zeros(len(data_out))
    X_prime = np.zeros(len(data_out))

    control_neighbors = (
        NearestNeighbors(n_neighbors=args.neighb, algorithm='auto')
            .fit(control['propensity_score'].values.reshape(-1, 1))
    )
    distances, indices = control_neighbors.kneighbors(treated['propensity_score'].values.reshape(-1, 1))


    random_indices =np.random.randint(0,args.neighb,len(indices))
    Y_prime[np.array(treated.index)] = np.array([control.iloc[indices[i,j]]['Y'].item() for i,j in enumerate(random_indices)])

    X_prime[np.array(treated.index)] = 0

    print('im here')

    treated_neighbors = (
        NearestNeighbors(n_neighbors=args.neighb, algorithm='auto')
            .fit(treated['propensity_score'].values.reshape(-1, 1))
    )
    distances, indices = treated_neighbors.kneighbors(control['propensity_score'].values.reshape(-1, 1))

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


    data_out.reset_index().drop(['index'],axis=1,inplace=True)

    train_idx, test_idx = train_test_split(list(range(len(data_out))), test_size=0.2)
    data_out.reset_index().drop(['index'],axis=1,inplace=True)


    to_save ={}
    to_save['train'] = data_out.iloc[train_idx].astype('float')
    to_save['test'] = data_out.iloc[test_idx].astype('float')

    pickle_object(to_save, args.save_path)
    print(args.save_path)

def  propensity_score_matching_2(args):
    dataset = SyntheticDataset(**vars(args))

    dataset.train = pd.DataFrame.from_dict(dataset.train)
    dataset.test = pd.DataFrame.from_dict(dataset.test)

    if args.test =='test':
        print('_'+args.test)
        dataset_base = copy.deepcopy(dataset.test)
    else:
        dataset_base = copy.deepcopy(dataset.train)

    # dataset_proc = dataset_proc.reset_index()
    dataset_proc = dataset_base.drop(['Y','Y_prime','X_prime','Ux'], axis=1, inplace=False)




    T = dataset_proc.X
    X = dataset_proc.loc[:, (dataset_proc.columns != 'X') ]


    pipe = GradientBoostingClassifier(n_estimators=200, learning_rate=1,
                                      max_depth=2, random_state=0, ).fit(X, T)

    predictions = pipe.predict_proba(X)
    predictions_binary = pipe.predict(X)

    print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(T, predictions_binary)))
    print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(T, predictions_binary)))
    print('F1 score is: {:.4f}'.format(metrics.f1_score(T, predictions_binary)))

    _data = X
    _data.loc[:, 'propensity_score'] = predictions[:, 1]
    _data.loc[:, 'Y'] = dataset_base.Y
    _data.loc[:, 'X'] = dataset_base.X
    _data.loc[:, 'Uy'] = dataset_base.Uy

    data_out = copy.deepcopy(_data)

    treated = _data.loc[_data.X == 1]
    control = _data.loc[_data.X == 0]


    high_propensity = _data[_data["propensity_score"] > 0.5]
    low_propensity = _data[_data["propensity_score"] <= 0.5]
    counts = np.array([len(low_propensity), len(high_propensity)])
    percentages = counts / np.sum(counts)

    samples = []
    for i in range(args.n_samples):
        is_high_propensity = np.random.random() > percentages[0]
        if is_high_propensity:
            treated_patient = high_propensity[high_propensity["X"] == 1].sample().iloc[0]
            untreated_patient = high_propensity[high_propensity["X"] == 0].sample().iloc[0]
        else:
            treated_patient = low_propensity[low_propensity["X"] == 1].sample().iloc[0]
            untreated_patient = low_propensity[low_propensity["X"] == 0].sample().iloc[0]
        samples.append((treated_patient, untreated_patient))
    print('im here')
    data_out = pd.DataFrame()
    for tr,con in samples:
        is_high_propensity = np.random.random() > 0.5
        if is_high_propensity:
            tr['X_prime'] = con['X']
            tr['Y_prime'] = con['Y']
            data_out = data_out.append(tr)
        else:
            con['X_prime'] = tr['X']
            con['Y_prime'] = tr['Y']
            data_out = data_out.append(con)

    Xs = data_out['X'].values
    Xs_prime = data_out['X_prime'].values
    Ys = data_out['Y'].values
    Ys_prime = data_out['Y_prime'].values
    idx_given_y_1 = np.where(Ys[np.where(Xs == 1)[0]] == 1)[0]
    idx_query_y_0 = np.where(Ys_prime[np.where(Xs_prime == 0)[0]] == 0)[0]
    idx_y_1_y_prime_0 = list(set(idx_given_y_1).intersection(idx_query_y_0))
    prob_necessity_1 = len(idx_y_1_y_prime_0) / len(idx_given_y_1)
    print('N', prob_necessity_1)

    idx_given_y_0 = np.where(Ys[np.where(Xs == 0)[0]] == 0)[0]
    idx_query_y_1 = np.where(Ys_prime[np.where(Xs_prime == 1)[0]] == 1)[0]
    idx_y_0_y_prime_1 = list(set(idx_given_y_0).intersection(idx_query_y_1))
    prob_suf_1 = len(idx_y_0_y_prime_1) / len(idx_given_y_0)
    print('S', prob_suf_1)

    data_out.reset_index().drop(['index'], axis=1, inplace=True)

    train_idx, test_idx = train_test_split(list(range(len(data_out))), test_size=0.2)
    data_out.reset_index().drop(['index'], axis=1, inplace=True)

    to_save = {}
    to_save['train'] = data_out.iloc[train_idx].astype('float')
    to_save['test'] = data_out.iloc[test_idx].astype('float')

    pickle_object(to_save, args.save_path)
    print(args.save_path)

def propensity_score_matching_3(args):
    dataset = SyntheticDataset(**vars(args))

    dataset.train = pd.DataFrame.from_dict(dataset.train)
    dataset.test = pd.DataFrame.from_dict(dataset.test)


    dataset = pd.concat([dataset.train,dataset.test])
    dataset_proc = dataset.drop(['Y_prime','X_prime'], axis=1, inplace=False)

    z_0 = dataset_proc.loc[dataset.Z==0]
    z_1 = dataset_proc.loc[dataset.Z==1]

    data_out = pd.DataFrame()

    treated = z_0.loc[z_0.X == 1]
    control = z_0.loc[z_0.X == 0]
    percentages = len(treated) / len(z_0)

    for i, item in tqdm(enumerate(z_0.iterrows()),total=args.n_samples):
        if i == args.n_samples:
            break
        item = item[1]
        x = item.X
        if x==0:
            x_prime = 1
            to_get = treated
        else:
            x_prime = 0
            to_get = control
        random_indices = np.random.randint(0, len(to_get), args.neighb, )

        for j in random_indices:
            item_ = copy.deepcopy(item)
            item_['Y_prime'] = to_get.iloc[j].Y
            item_['X_prime'] = x_prime
            data_out = data_out.append(item_)

    print('im here')
    treated = z_1.loc[z_1.X == 1]
    control = z_1.loc[z_1.X == 0]
    percentages = len(treated) / len(z_1)


    for i, item in tqdm(enumerate(z_1.iterrows()),total=args.n_samples):
        if i == args.n_samples:
            break
        item = item[1]
        x = item.X
        if x == 0:
            x_prime = 1
            to_get = treated
        else:
            x_prime = 0
            to_get = control
        random_indices = np.random.randint(0, len(to_get), args.neighb, )

        for j in random_indices:
            item_ = copy.deepcopy(item)
            item_['Y_prime'] = to_get.iloc[j].Y
            item_['X_prime'] = x_prime
            data_out = data_out.append(item_)

    Xs = data_out['X'].values
    Xs_prime = data_out['X_prime'].values
    Ys = data_out['Y'].values
    Ys_prime = data_out['Y_prime'].values
    idx_given_y_1 = np.where(Ys[np.where(Xs == 1)[0]] == 1)[0]
    idx_query_y_0 = np.where(Ys_prime[np.where(Xs_prime == 0)[0]] == 0)[0]
    idx_y_1_y_prime_0 = list(set(idx_given_y_1).intersection(idx_query_y_0))
    prob_necessity_1 = len(idx_y_1_y_prime_0) / len(idx_given_y_1)
    print('N', prob_necessity_1)

    idx_given_y_0 = np.where(Ys[np.where(Xs == 0)[0]] == 0)[0]
    idx_query_y_1 = np.where(Ys_prime[np.where(Xs_prime == 1)[0]] == 1)[0]
    idx_y_0_y_prime_1 = list(set(idx_given_y_0).intersection(idx_query_y_1))
    prob_suf_1 = len(idx_y_0_y_prime_1) / len(idx_given_y_0)
    print('S', prob_suf_1)

    data_out.reset_index().drop(['index'], axis=1, inplace=True)

    train_idx, test_idx = train_test_split(list(range(len(data_out))), test_size=0.2)
    data_out.reset_index().drop(['index'], axis=1, inplace=True)

    to_save = {}
    to_save['train'] = data_out.iloc[train_idx].astype('float')
    to_save['test'] = data_out.iloc[test_idx].astype('float')

    pickle_object(to_save, args.save_path)
    print(args.save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path',
                        default='./Datasets/synthetic_dataset_200000_samples_X_{}_Uy_{}_Z_{}_with_counterfactual_and_z_a_link_2_final.pkl')
    parser.add_argument('--u_distribution', default='normal')
    parser.add_argument('--z_distribution', default='uniform')
    parser.add_argument('--x_distribution', default='bernouli')
    parser.add_argument('--p_1', type=float, default=0.05)
    parser.add_argument('--p_2', type=float, default=0.7)
    parser.add_argument('--p_3', type=float, default=0.3)

    parser.add_argument('--neighb', type=int, default=200)
    parser.add_argument('--test', type=str, default='')
    parser.add_argument('--load_dataset', default=True)
    parser.add_argument('--save_dataset', default=False)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--save_path', default='./Datasets/synthetic_dataset_{}_samples_X_{}_Uy_{}_Z_{}_with_counterfactual_and_z_a_link_with_propensity_1_final.pkl')
    # parser.add_argument('--save_path', default='./Datasets/dev.pkl')
    args = parser.parse_args()

    z_dist = '{}_{}'.format(args.z_distribution, args.p_2,
                            args.p_2) if args.z_distribution == 'bernouli' else '{}'.format(
        args.z_distribution)
    x_dist = '{}_{}'.format(args.x_distribution, args.p_1,
                            args.p_1) if args.x_distribution == 'bernouli' else '{}'.format(
        args.x_distribution)


    args.path_to_data = args.load_path = args.load_path.format(x_dist, args.u_distribution, z_dist)

    print(args.load_path)
    # args.save_path = args.save_path.format(x_dist, args.u_distribution, z_dist)
    args.save_path = args.save_path.format(args.n_samples,x_dist, args.u_distribution, z_dist,args.test)
    propensity_score_matching_3(args)
