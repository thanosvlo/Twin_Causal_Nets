import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
from data.twin_dataset import TwinsDataset
import pandas as pd
from utils import pickle_object

from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import math
import copy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler


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

def propensity_score_matching(args):
    dataset = TwinsDataset(**vars(args))

    dataset.train = pd.DataFrame.from_dict(dataset.train)
    dataset.test = pd.DataFrame.from_dict(dataset.test)


    dataset = pd.concat([dataset.train,dataset.test])

    dataset_base = copy.deepcopy(dataset)

    Xs = dataset['X'].values
    Xs_prime = dataset['X_prime'].values
    Ys = dataset['Y'].values
    Ys_prime = dataset['Y_prime'].values
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

    idx_given_y_0 = np.where(Ys == 0)[0]
    idx_query_y_1 = np.where(Ys_prime == 1)[0]
    idx_y_0_y_prime_1 = set(idx_given_y_0).intersection(idx_query_y_1)

    prob_nec_and_suficiency = len(idx_y_0_y_prime_1) / len(Ys)
    print('NS ', prob_nec_and_suficiency)


    # dataset_proc = dataset_proc.reset_index()
    dataset_proc = dataset_base.drop(['Y', 'Uy','Y_prime','X_prime'], axis=1, inplace=False)



    Tr = dataset_proc.X
    X = dataset_proc.loc[:, (dataset_proc.columns != 'X') ]


    # Design pipeline to build the treatment estimator

    # pipe = GradientBoostingClassifier(n_estimators=5, learning_rate=6.0,
    #                                   max_depth=1, random_state=0,).fit(X, X)

    pipe = GradientBoostingClassifier(n_estimators=50, learning_rate=1,
                                       random_state=0, ).fit(X,Tr)

    predictions = pipe.predict_proba(X)
    predictions_binary = pipe.predict(X)

    print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(Tr, predictions_binary)))
    print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(Tr, predictions_binary)))
    print('F1 score is: {:.4f}'.format(metrics.f1_score(Tr, predictions_binary)))

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


    random_indices =np.random.randint(0,args.neighb,len(indices))
    Y_prime[np.array(treated.index)] = np.array([control.iloc[indices[i,j]]['Y'].item() for i,j in enumerate(random_indices)])
    X_prime[np.array(treated.index)] = 0


    # Now computing AXC
    treated_neighbors = (
        NearestNeighbors(n_neighbors=args.neighb, algorithm='ball_tree')
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
    print('N', prob_necessity_1)

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



def  propensity_score_matching_Jobs(args):
    dataset = TwinsDataset(**vars(args))

    dataset.train = pd.DataFrame.from_dict(dataset.train)
    dataset.test = pd.DataFrame.from_dict(dataset.test)
    dataset_base = dataset = pd.concat([dataset.train,dataset.test])

    if args.balance:

        y = dataset['Y']

        dataset_proc = dataset.drop(['Y'], axis=1, inplace=False)

        smote = SMOTE()

        x_sm, y_sm = smote.fit_resample(dataset_proc, y)
        x_sm.insert(0, 'Y', y_sm)

        x_sm['Y'].loc[x_sm['Y'] >= 0.5] = 1
        x_sm['Y'].loc[x_sm['Y'] < 0.5] = 0
        x_sm['X'].loc[x_sm['X'] >= 0.5] = 1
        x_sm['X'].loc[x_sm['X'] < 0.5] = 0
        x_sm['X_prime'].loc[x_sm['X_prime'] >= 0.5] = 1
        x_sm['X_prime'].loc[x_sm['X_prime'] < 0.5] = 0
        dataset = x_sm




    # dataset_proc = dataset_proc.reset_index()
    dataset_proc = dataset_base.drop(['Y', 'Uy','X_prime'], axis=1, inplace=False)

    T = dataset_proc.X
    X = dataset_proc.loc[:, (dataset_proc.columns != 'X') ]


    pipe = GradientBoostingClassifier(n_estimators=50, learning_rate=1,
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
    dataset = TwinsDataset(**vars(args))

    dataset.train = pd.DataFrame.from_dict(dataset.train)
    dataset.test = pd.DataFrame.from_dict(dataset.test)
    dataset_base = dataset = pd.concat([dataset.train,dataset.test])

    if args.balance:

        y = dataset['Y']

        dataset_proc = dataset.drop(['Y'], axis=1, inplace=False)

        smote = SMOTE()

        x_sm, y_sm = smote.fit_resample(dataset_proc, y)
        x_sm.insert(0, 'Y', y_sm)
        scaler = MinMaxScaler()
        x_sm['Y_prime'] = scaler.fit_transform(x_sm['Y_prime'].values[..., np.newaxis])
        x_sm['Y_prime'].loc[x_sm['Y_prime'] >= 0.5] = 1
        x_sm['Y_prime'].loc[x_sm['Y_prime'] < 0.5] = 0
        y_2 = x_sm['Y_prime']
        x_sm.drop(['Y_prime'], axis=1, inplace=True)
        x_sm_2, y_sm_2 = smote.fit_resample(x_sm, y_2)
        x_sm_2.insert(0, 'Y_prime', y_sm_2)
        x_sm_2['Y'].loc[x_sm_2['Y'] >= 0.5] = 1
        x_sm_2['Y'].loc[x_sm_2['Y'] < 0.5] = 0
        x_sm_2['X'].loc[x_sm_2['X'] >= 0.5] = 1
        x_sm_2['X'].loc[x_sm_2['X'] < 0.5] = 0
        x_sm_2['X_prime'].loc[x_sm_2['X_prime'] >= 0.5] = 1
        x_sm_2['X_prime'].loc[x_sm_2['X_prime'] < 0.5] = 0
        dataset = x_sm_2

    Xs = dataset['X'].values
    Xs_prime = dataset['X_prime'].values
    Ys = dataset['Y'].values
    Ys_prime = dataset['Y_prime'].values
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

    idx_given_y_0 = np.where(Ys == 0)[0]
    idx_query_y_1 = np.where(Ys_prime == 1)[0]
    idx_y_0_y_prime_1 = set(idx_given_y_0).intersection(idx_query_y_1)

    prob_nec_and_suficiency = len(idx_y_0_y_prime_1) / len(Ys)
    print('NS ', prob_nec_and_suficiency)



    # dataset_proc = dataset_proc.reset_index()
    dataset_proc = dataset_base.drop(['Y', 'Uy','Y_prime','X_prime'], axis=1, inplace=False)




    T = dataset_proc.X
    X = dataset_proc.loc[:, (dataset_proc.columns != 'X') ]


    pipe = GradientBoostingClassifier(n_estimators=50, learning_rate=1,
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

def propensity_score_matching_3(args):
    dataset = TwinsDataset(**vars(args))

    dataset.train = pd.DataFrame.from_dict(dataset.train)
    dataset.test = pd.DataFrame.from_dict(dataset.test)

    if args.test == 'test':
        print('_' + args.test)
        dataset_base = copy.deepcopy(dataset.test)
    else:
        dataset_base = copy.deepcopy(dataset.train)

    # dataset_proc = dataset_proc.reset_index()
    dataset_proc = dataset_base.drop(['Y', 'Uy', 'Y_prime', 'X_prime'], axis=1, inplace=False)

    T = dataset_proc.X
    X = dataset_proc.loc[:, (dataset_proc.columns != 'X')]

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
    for i in tqdm(range(args.n_samples)):
        is_high_propensity = np.random.random() > percentages[0]
        if is_high_propensity:
            treated_patient = high_propensity[high_propensity["X"] == 1].sample(args.neighb)
            untreated_patient = high_propensity[high_propensity["X"] == 0].sample(args.neighb)
        else:
            treated_patient = low_propensity[low_propensity["X"] == 1].sample(args.neighb)
            untreated_patient = low_propensity[low_propensity["X"] == 0].sample(args.neighb)
        samples.append((treated_patient, untreated_patient))
    print('im here')
    data_out = pd.DataFrame()
    for tr,con in tqdm(samples):
        is_high_propensity = np.random.random() > 0.5
        if is_high_propensity:
            for j_con in con:
                tr_temp  = copy.deepcopy(tr)
                tr_temp['X_prime'] = j_con['X']
                tr_temp['Y_prime'] = j_con['Y']
                data_out = data_out.append(tr_temp)
        else:
            for j_tr in tr:
                tr_con = copy.deepcopy(tr)
                tr_con['X_prime'] = j_tr['X']
                tr_con['Y_prime'] = j_tr['Y']
                data_out = data_out.append(tr_temp)
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path',
                        # default='./Datasets/twins_uy_normal_twins_as_counterfactuals_interpolate_all.pkl')
                        # default='./Datasets/ganite_twins_uy_normal_twins.pkl')
                        default='./Datasets/jobs_uy_normal.pkl')

    parser.add_argument('--neighb', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--test', type=str, default='')
    parser.add_argument('--load_dataset', default=True)
    parser.add_argument('--save_dataset', default=False)
    parser.add_argument('--balance', default=False)

    # parser.add_argument('--save_path', default='./Datasets/twins_uy_normal_twins_with_propensity_interpolated_2_balanced_all.pkl')
    # parser.add_argument('--save_path', default='./Datasets/ganite_twins_uy_normal_twins_with_propensity.pkl')
    parser.add_argument('--save_path', default='./Datasets/jobs_uy_normal_twins_with_propensity.pkl')
    # parser.add_argument('--save_path', default='./Datasets/dev.pkl')
    args = parser.parse_args()




    args.path_to_data = args.load_path #= args.load_path.format(x_dist, args.u_distribution, z_dist)

    print(args.load_path)
    # args.save_path = args.save_path.format(x_dist, args.u_distribution, z_dist)
    # args.save_path = args.save_path.format(x_dist, args.u_distribution, z_dist,args.test)
    # propensity_score_matching_2(args)
    propensity_score_matching_Jobs(args)
