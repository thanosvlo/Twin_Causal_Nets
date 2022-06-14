import random

import numpy as np
import os
import glob
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import itertools
from prettytable import PrettyTable
from utils import pickle_object

sys.path.append("..")
sys.path.append("../..")
import argparse
from copy import deepcopy
from sklearn.model_selection import train_test_split
from scipy import stats
import random
# from german_metadata import categories_to_numerical, CategoricalFeatures, decre_monoto

decre_monoto = ['AGE']

def percentage(x,i):
    return np.sum(x==i)/len(x)

def get_subportion_confounders(df, to_keep):
    if 'all' not in to_keep:
        return df[to_keep]
    else:
        return df

def make_g( x, sex, age, uy, cons):
    g = x + sex + 0.2*(cons-1)+ 0.5*x * sex*age + uy

    g = 1/(1+np.exp(-np.array(g,float)))
    # g = x + (uy>0.5)

    # g = np.zeros_like(x)
    #
    # g[uy==0] = (x[uy==0] + sex[uy==0])/2
    # g[uy==1] = np.around((x[uy==1] + cons[uy==1])/2)
    # g[uy==2] =  (x[uy==2] + age[uy==2])/2
    # g[uy==3] =  np.around((x[uy==3] + cons[uy==3])/2)
    # g =  np.abs(g)
    # g = g-g.min()
    # g = g/g.max()
    # m = age * (1 - sex) * cons
    # g = (0.5 * x + (m + uy))
    # g[g > 3] = 3
    return g


class ISTDataset:
    def __init__(self, path_to_data, save_path=None, load_path=None, save_name=None, save_dataset=False,
                 load_dataset=False, u_distribution='normal',ux_distribution='uniform', p=0.5,  mu=0.5, sigma=1/9, low=0, high=7, bins=4,
                 train_test_split_fr=0.8, propensity_score=False, treatment='Credit-History', outcome='Synthetic',
                 **kwargs):

        self.save_path = save_path
        self.save_name = save_name
        self.load_path = load_path
        self.load_path = load_path
        self.u_distribution = u_distribution
        self.ux_distribution = ux_distribution
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high

        self.bins = bins

        self.treatment = treatment
        self.outcome = outcome

        if save_dataset:
            assert save_path is not None
        if load_dataset:
            assert path_to_data is not None

        self.path_to_data = path_to_data
        if not load_dataset:
            self.df = self.get_data()
            if propensity_score:
                self.n_samples = kwargs.get('n_samples', 1000)
                self.neighb = kwargs.get('neighb', 1000)
                self.df = self.do_propensity_matching()

        else:
            self.train = pd.read_csv(self.path_to_data.format('train'))
            self.test = pd.read_csv(self.path_to_data.format('test'))

        if save_dataset:
            self.df.drop_duplicates(inplace=True)
            train_idx, test_idx = train_test_split(list(range(len(self.df))), test_size=1 - train_test_split_fr)
            self.train = self.df.iloc[train_idx].astype('float')
            self.test = self.df.iloc[test_idx].astype('float')

            self.save_dataset(self.train, save_path,
                              save_name.format(self.treatment.replace('-', '_'), self.outcome,
                                               'train'))
            self.save_dataset(self.test, save_path,
                              save_name.format(self.treatment.replace('-', '_'), self.outcome,
                                               'test'))

    def save_dataset(self, item, save_path, save_name):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        item.to_csv(os.path.join(save_path, save_name))

    def get_uy_samples(self, N_samples):
        if self.u_distribution == 'normal':
            # temp = np.random.normal(self.mu, self.sigma, N_samples)
            temp = np.random.normal(self.mu, self.sigma, N_samples)
            # temp = np.digitize(temp, np.arange(temp.min(),temp.max(),(temp.max()- temp.min())/8)) - 1
            # temp = stats.binned_statistic(temp, temp, 'sum',bins= 7).binnumber -1
            # # temp[temp > 1] = 0
            # # temp[temp == 1] = -1
            # # temp[temp == 0] = 1
            # # temp[temp == -1] = 0
            # print('P(Uy=0) = {}'.format(np.sum(temp == 0) / len(temp)))
            # print('P(Uy=1) = {}'.format(np.sum(temp == 1) / len(temp)))
            # print('P(Uy=2) = {}'.format(np.sum(temp == 2) / len(temp)))
            # print('P(Uy=3) = {}'.format(np.sum(temp ==3) / len(temp)))
            # print('P(Uy=4) = {}'.format(np.sum(temp ==4) / len(temp)))
            # print('P(Uy=5) = {}'.format(np.sum(temp ==5) / len(temp)))
            # print('P(Uy=6) = {}'.format(np.sum(temp ==6) / len(temp)))

        elif self.u_distribution == 'uniform':
            temp = np.random.randint(self.low, self.high, N_samples)
        return temp

    def get_ux_samples(self, N_samples):
        if self.ux_distribution == 'normal':
            temp = np.random.normal(self.mu, self.sigma, N_samples)

        elif self.ux_distribution == 'uniform':
            temp = np.random.randint(self.low, self.high, N_samples)
        return temp

    def get_data(self):

        df_data = pd.read_csv(os.path.join(self.path_to_data, 'IST_corrected.csv'))
        # df_data = pd.read_csv(os.path.join(self.path_to_data,'german.data-numeric'),header=None,)
        df_data_expl = pd.read_csv(os.path.join(self.path_to_data, 'IST_variables.csv'), delimiter=';',
                                   error_bad_lines=False)

        data_encode = df_data.copy()
        if 'Y' in args.outcome:

            if 'heparin' not in self.treatment.lower():
                data_encode.loc[data_encode[self.treatment] == 'Y', self.treatment] = 1
                data_encode.loc[data_encode[self.treatment] == 'y', self.treatment] = 1
                data_encode.loc[data_encode[self.treatment] == 'N', self.treatment] = 0
                data_encode.loc[data_encode[self.treatment] == 'n', self.treatment] = 0
                data_encode.loc[data_encode[self.treatment] == 'U', self.treatment] = 0
                data_encode[self.treatment].fillna(0, inplace=True)
            else:
                indx_low = data_encode[data_encode['DLH14'] == 'Y'].index
                indx_medium = data_encode[data_encode['DMH14'] == 'Y'].index
                indx_high = data_encode[data_encode['DHH14'] == 'Y'].index
                data_encode[self.treatment] = 0
                data_encode[self.treatment].iloc[indx_low] = 1
                data_encode[self.treatment].iloc[indx_medium] = 2
                # data_encode[self.treatment].iloc[indx_high] = 3
                data_encode.drop(indx_high)

            data_encode = data_encode[['RCONSC', 'AGE', 'SEX', self.treatment]]

            data_encode.loc[data_encode['AGE'] <= 73, 'AGE'] = 0
            data_encode.loc[data_encode['AGE'] > 73, 'AGE'] = 1
            data_encode.loc[data_encode['RCONSC'] == 'D', 'RCONSC'] = 1
            data_encode.loc[data_encode['RCONSC'] == 'F', 'RCONSC'] = 2
            data_encode.loc[data_encode['RCONSC'] == 'U', 'RCONSC'] = 0
            data_encode.loc[data_encode['SEX'] == 'M', 'SEX'] = 0
            data_encode.loc[data_encode['SEX'] == 'F', 'SEX'] = 1
            data_encode.insert(0, 'Uy', self.get_uy_samples(len(data_encode)))

            data_encode.insert(0, 'Ux', self.get_ux_samples(len(data_encode)))

            #
            # data_encode = data_encode.drop_duplicates(inplace=False)
            #
            # data_encodeprimed = pd.DataFrame()
            # for jj, (_, row) in tqdm(enumerate(data_encode.iterrows()), total=len(data_encode)):
            #     cands = [i for i in np.unique(data_encode[self.treatment].values) if i != row[self.treatment]]
            #     data_encodeprimed = data_encodeprimed.append(row)
            #     for cand in cands:
            #         row['{}'.format(self.treatment)] = cand
            #         data_encodeprimed = data_encodeprimed.append(row)
            # data_encode = data_encodeprimed.drop_duplicates(inplace=False)

            if self.u_distribution =='normal':
                data_encode = self.selection_bias(data_encode)

            pos_ = data_encode[(data_encode[self.treatment] == 0)]
            neg_ = data_encode[(data_encode[self.treatment] != 0)]
            ids = len(pos_)
            choices_ = np.random.choice(ids, len(neg_))
            res_pos_features = pos_.iloc[choices_]
            data_encode = pd.concat([res_pos_features,neg_], axis=0).reset_index().drop(['index'],axis=1,inplace=False)


            if np.unique(data_encode[self.treatment].values).max()>1:
                data_encodeprimed = pd.DataFrame()
                for jj,(_, row) in tqdm(enumerate(data_encode.iterrows()),total=len(data_encode)):
                    cands = [i for i in np.unique(data_encode[self.treatment].values) if i != row[self.treatment]]
                    for cand in cands:
                        row['{}_prime'.format(self.treatment)] = cand
                        data_encodeprimed = data_encodeprimed.append(row)
                data_encode = data_encodeprimed.drop_duplicates(inplace=False)
                x_prime = data_encode['{}_prime'.format(self.treatment)]
            else:
                x_prime = np.logical_xor(data_encode[self.treatment].values, 1).astype(int)
            data_encode['{}_prime'.format(self.treatment)] = x_prime

                #
            uy = data_encode.Ux.values
            sex = data_encode['SEX'].values
            cons = data_encode['RCONSC'].values
            age = data_encode['AGE'].values

            g, g_prime = self.get_g_ours( data_encode, uy, cons, sex)

            # g = make_g(data_encode['{}'.format(self.treatment)].values, sex, age, uy, cons)
            # g_prime = make_g(data_encode['{}_prime'.format(self.treatment)].values, sex, age, uy, cons)
            # min_ = g.min()
            # max_ = g.max()
            # g = np.digitize(g, np.arange(min_, max_, (max_ - min_) / args.bins), right=False) - 1
            # min_ = g_prime.min()
            # max_ = g_prime.max()
            # g_prime = np.digitize(g_prime, np.arange(min_, max_, (max_ - min_) / args.bins), right=False) - 1
            #
            #
            data_encode['{}'.format(self.outcome)] = g
            data_encode['{}_prime'.format(self.outcome)] = g_prime
            # #


            self.calc_probs(data_encode)


            self.check_causal_ordering_wrap(data_encode)
        return data_encode

    def get_g_ours(self,data_encode,uy,cons,sex):
        g = np.zeros_like(data_encode[self.treatment].values)
        g[np.where(uy == 0)[0]] = data_encode[self.treatment].iloc[np.where(uy == 0)[0]] + cons[np.where(uy == 0)[0]]
        g[np.where(uy == 2)[0]] = data_encode[self.treatment].iloc[np.where(uy == 2)[0]] * cons[np.where(uy == 2)[0]]
        g[g > 2] = 2

        g[np.where(uy == 3)[0]] = 2
        g[np.where(uy == 4)[0]] = 1
        g[np.where(uy == 5)[0]] = np.heaviside(
            data_encode['{}'.format(self.treatment)].iloc[np.where(uy == 5)[0]].values * sex[np.where(uy == 5)] - 1, 0)
        g[np.where(uy == 6)[0]] = np.heaviside(data_encode['{}'.format(self.treatment)].iloc[np.where(uy == 6)[0]] - 1,
                                               0) * 2
        g[g < 0] = 0
        # g[data_encode[data_encode.Uy==1].index] = 2

        g_prime = np.zeros_like(data_encode['{}_prime'.format(self.treatment)].values)
        g_prime[np.where(uy == 0)[0]] = data_encode['{}_prime'.format(self.treatment)].iloc[np.where(uy == 0)[0]] + \
                                        cons[np.where(uy == 0)[0]]
        g_prime[np.where(uy == 2)[0]] = data_encode['{}_prime'.format(self.treatment)].iloc[np.where(uy == 2)[0]] * \
                                        cons[np.where(uy == 2)[0]]
        g_prime[g_prime > 2] = 2

        g_prime[np.where(uy == 3)[0]] = 2
        g_prime[np.where(uy == 4)[0]] = 1
        g_prime[np.where(uy == 5)[0]] = np.heaviside(
            data_encode['{}_prime'.format(self.treatment)].iloc[np.where(uy == 5)[0]] * sex[np.where(uy == 5)] - 1, 0)
        g_prime[np.where(uy == 6)[0]] = np.heaviside(
            data_encode['{}_prime'.format(self.treatment)].iloc[np.where(uy == 6)[0]] - 1, 0) * 2
        g_prime[g_prime < 0] = 0

        return g.astype(int), g_prime.astype(int)


    def check_causal_ordering_wrap(self, data_encode):
        for treat, treat_counter in itertools.permutations(np.unique(data_encode[self.outcome]), 2):
            treat = int(treat)
            treat_counter = int(treat_counter)
            x_1 = PrettyTable(['Case_1_cond_{}_count_{}'.format(treat,treat_counter), *[str(i) for i in np.unique(data_encode[self.outcome])]])
            x_2 = PrettyTable(['Case_2_cond_{}_count_{}'.format(treat,treat_counter), *[str(i) for i in np.unique(data_encode[self.outcome])]])
            res_1 = np.zeros((data_encode[self.outcome].max() + 1, data_encode[self.outcome].max() + 1))
            res_2 = np.zeros((data_encode[self.outcome].max() + 1, data_encode[self.outcome].max() + 1))
            for outcome, outcome_counter in itertools.permutations(np.unique(data_encode[self.outcome]), 2):
                outcome = int(outcome)
                outcome_counter = int(outcome_counter)
                _, (case_1, case_2) = self.check_causal_ordering(data_encode, treat, treat_counter, outcome, outcome_counter)
                res_1[outcome, outcome_counter] = case_1
                res_2[outcome, outcome_counter] = case_2
            for i in np.unique(data_encode[self.outcome]):
                x_1.add_row([str(i), *list(res_1[i, :])])
                x_2.add_row([str(i), *list(res_2[i, :])])
            print(x_1.get_string())
            print(x_2.get_string())

    def check_causal_ordering(self, data_encode, outcome , outcome_counter , treat, treat_counter):
        y_1 = np.where(data_encode[self.outcome] == outcome)[0]
        x_0 = np.where(data_encode[self.treatment] == treat)[0]
        y_1_x_0 = set(y_1).intersection(x_0)
        y_prime_0 = np.where(data_encode['{}_prime'.format(self.outcome)] == outcome_counter)[0]
        x_prime_1 = np.where(data_encode['{}_prime'.format(self.treatment)] == treat_counter)[0]
        y_prime_0_x_prime_1 = set(y_prime_0).intersection(x_prime_1)
        shouldnt_exist = set(y_1_x_0).intersection(y_prime_0_x_prime_1)
        print(len(shouldnt_exist) / len(y_1_x_0))
        case_1 = len(shouldnt_exist) / len(y_1_x_0)

        y_0 = np.where(data_encode[self.outcome] == outcome_counter)[0]
        x_1 = np.where(data_encode[self.treatment] == treat_counter)[0]
        y_0_x_1 = set(y_0).intersection(x_1)
        y_prime_1 = np.where(data_encode['{}_prime'.format(self.outcome)] == outcome)[0]
        x_prime_0 = np.where(data_encode['{}_prime'.format(self.treatment)] == treat_counter)[0]
        y_prime_1_x_prime_0 = set(y_prime_1).intersection(x_prime_0)
        shouldnt_exist_2 = set(y_0_x_1).intersection(y_prime_1_x_prime_0)
        print(len(shouldnt_exist_2) / len(y_0_x_1))
        case_2 = len(shouldnt_exist_2) / len(y_0_x_1)

        if treat < treat_counter:
            return not (case_2 <= 0.01 and case_1 <= 0.01), (case_1, case_2)
        if treat > treat_counter:
            return (case_1 <= 0.01 and case_2 <= 0.01), (case_1, case_2)

    def selection_bias(self, data_encode):
        dataset_base = deepcopy(data_encode)
        dataset_proc = dataset_base.drop(['Uy'], axis=1, inplace=False)
        T = dataset_proc[self.treatment]
        X = dataset_proc.loc[:, (dataset_proc.columns != self.treatment)]

        pipe = GradientBoostingClassifier(n_estimators=250, learning_rate=1,
                                          max_depth=2, random_state=0, ).fit(
            np.reshape(data_encode['Uy'].values, (-1, 1)), T)
        # max_depth=2, random_state=0, ).fit(X, T)

        # predictions_binary = pipe.predict(X)
        predictions_binary = pipe.predict(np.reshape(data_encode['Uy'].values, (-1, 1)))
        print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(T, predictions_binary)))
        print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(T, predictions_binary)))
        print('F1 score is: {:.4f}'.format(metrics.f1_score(T, predictions_binary, average='macro')))

        idxs = np.where(predictions_binary == T)
        data_encode = data_encode.iloc[idxs]
        return data_encode

    def calc_probs(self, data_encode):

        for treat, treat_counter in itertools.permutations(np.unique(data_encode[self.treatment]), 2):
            treat = int(treat)
            treat_counter = int(treat_counter)
            x_pn = PrettyTable(
                ['P_N_{}_{}'.format(treat, treat_counter), *[str(i) for i in np.unique(data_encode[self.outcome])]])
            x_ps = PrettyTable(
                ['P_S_{}_{}'.format(treat, treat_counter), *[str(i) for i in np.unique(data_encode[self.outcome])]])
            x_pns = PrettyTable(
                ['P_NS_{}_{}'.format(treat, treat_counter), *[str(i) for i in np.unique(data_encode[self.outcome])]])

            res_1 = np.zeros((data_encode[self.outcome].max() + 1, data_encode[self.outcome].max() + 1))
            res_2 = np.zeros((data_encode[self.outcome].max() + 1, data_encode[self.outcome].max() + 1))
            res_3 = np.zeros((data_encode[self.outcome].max() + 1, data_encode[self.outcome].max() + 1))

            for outcome, outcome_counter in itertools.permutations(np.unique(data_encode[self.outcome]), 2):
                outcome = int(outcome)
                outcome_counter = int(outcome_counter)
                pn = self.p_n(data_encode[self.treatment], data_encode['{}_prime'.format(self.treatment)],
                              data_encode[self.outcome],
                              data_encode['{}_prime'.format(self.outcome)], treat, treat_counter, outcome,
                              outcome_counter)
                res_1[outcome, outcome_counter] = pn
                ps = self.p_s(data_encode[self.treatment], data_encode['{}_prime'.format(self.treatment)],
                              data_encode[self.outcome],
                              data_encode['{}_prime'.format(self.outcome)], treat, treat_counter, outcome,
                              outcome_counter)
                res_2[outcome, outcome_counter] = ps
                pns = self.p_ns(data_encode[self.treatment], data_encode['{}_prime'.format(self.treatment)],
                                data_encode[self.outcome],
                                data_encode['{}_prime'.format(self.outcome)], treat, treat_counter, outcome,
                                outcome_counter)
                res_3[outcome, outcome_counter] = pns
            for i in np.unique(data_encode[self.outcome]):
                x_pn.add_row([str(i), *list(res_1[i, :])])
                x_ps.add_row([str(i), *list(res_2[i, :])])
                x_pns.add_row([str(i), *list(res_3[i, :])])
            print(x_pn.get_string())
            print(x_ps.get_string())
            print(x_pns.get_string())

            to_out = {'PN': res_1,
                      'PS': res_2,
                      'PNS': res_3}
            pickle_object(to_out, os.path.join(self.save_path,
                                               self.save_name.replace('csv', 'pkl').format(
                                                   self.treatment.replace('-', '_'), self.outcome,
                                                   'PN_tr{}_trcounter{}'.format(treat, treat_counter))))

    def p_n(self, x, x_prime, y, y_prime, treatment, treatment_counter, outcome, outcome_counter):
        x_treat = np.where(x == treatment)[0]
        y_out = np.where(y == outcome)[0]
        y_out_x_treat = set(y_out).intersection(x_treat)

        x_prime_treat_count = np.where(x_prime == treatment_counter)[0]
        y_prime_out_count = np.where(y_prime == outcome_counter)[0]
        y_prime_out_counter_x_prime_treat_counter = set(y_prime_out_count).intersection(x_prime_treat_count)

        idx_y_1_y_prime_0 = set(y_out_x_treat).intersection(y_prime_out_counter_x_prime_treat_counter)

        prob_necessity = len(idx_y_1_y_prime_0) / len(y_out_x_treat)
        return prob_necessity

    def p_s(self, x, x_prime, y, y_prime, treatment, treatment_counter, outcome, outcome_counter):
        x_treat_counter = np.where(x == treatment_counter)[0]
        y_out_counter = np.where(y == outcome_counter)[0]
        y_out_counter_x_treat_counter = set(y_out_counter).intersection(x_treat_counter)

        x_prime_treat = np.where(x_prime == treatment)[0]
        y_prime_out = np.where(y_prime == outcome)[0]
        y_prime_out_x_prime_treat = set(y_prime_out).intersection(x_prime_treat)

        idx_y_1_y_prime_0 = set(y_out_counter_x_treat_counter).intersection(y_prime_out_x_prime_treat)

        prob_suf = len(idx_y_1_y_prime_0) / len(y_out_counter_x_treat_counter)
        return prob_suf

    def p_ns(self, x, x_prime, y, y_prime, treatment, treatment_counter, outcome, outcome_counter):
        x_treat_counter = np.where(x == treatment_counter)[0]
        y_out_counter = np.where(y == outcome_counter)[0]
        y_out_counter_x_treat_counter = set(y_out_counter).intersection(x_treat_counter)

        x_prime_treat = np.where(x_prime == treatment)[0]
        y_prime_out = np.where(y_prime == outcome)[0]
        y_prime_out_x_prime_treat = set(y_prime_out).intersection(x_prime_treat)

        idx_y_1_y_prime_0 = set(y_out_counter_x_treat_counter).intersection(y_prime_out_x_prime_treat)
        prob_nec_suf = len(idx_y_1_y_prime_0) / len(y)

        return prob_nec_suf

    def do_propensity_matching(self):
        dataset_base = deepcopy(self.df)
        dataset_proc = dataset_base.drop([self.outcome], axis=1, inplace=False)
        T = dataset_proc[self.treatment]
        X = dataset_proc.loc[:, (dataset_proc.columns != self.treatment)]

        pipe = GradientBoostingClassifier(n_estimators=100, learning_rate=1,
                                          max_depth=2, random_state=0, ).fit(X, T)

        predictions = pipe.predict_proba(X)
        predictions_binary = pipe.predict(X)

        print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(T, predictions_binary)))
        print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(T, predictions_binary)))
        print('F1 score is: {:.4f}'.format(metrics.f1_score(T, predictions_binary, average='macro')))

        _data = X
        _data.loc[:, 'propensity_score'] = predictions[:, 1]
        _data.loc[:, self.outcome] = dataset_base[self.outcome]
        _data.loc[:, self.treatment] = dataset_base[self.treatment]
        _data.loc[:, 'Uy'] = dataset_base.Uy

        high_propensity = _data[_data["propensity_score"] > 0.5]
        low_propensity = _data[_data["propensity_score"] <= 0.5]
        counts = np.array([len(low_propensity), len(high_propensity)])
        percentages = counts / np.sum(counts)
        to_out = pd.DataFrame()
        for ii in np.unique(_data[self.treatment]):
            treated = _data.loc[_data[self.treatment] == ii]
            control = _data.loc[_data[self.treatment] != ii]
            control_neighbors = (
                NearestNeighbors(n_neighbors=self.neighb, algorithm='ball_tree')
                    .fit(control['propensity_score'].values.reshape(-1, 1))
            )
            distances, indices = control_neighbors.kneighbors(treated['propensity_score'].values.reshape(-1, 1))
            # random_indices = np.random.randint(0, args.neighb, len(indices))

            prime_cands = [control.iloc[j] for i, j in enumerate(indices)]

            to_out = to_out.append(self.get_valid_counters(treated, prime_cands, indices))

        return to_out

    def get_valid_counters(self, treated, prime_cands, indices):
        if self.treatment in decre_monoto:
            return self.get_valid_counters_decr(treated, prime_cands, indices)
        else:
            return self.get_valid_counters_incresing(treated, prime_cands, indices)

    def get_valid_counters_incresing(self, treated, prime_cands, indices):
        to_out = pd.DataFrame()

        for jj, (_, item_) in tqdm(enumerate(treated.iterrows()), total=len(treated)):
            cands = prime_cands[jj]
            for ll, cand in cands.iterrows():
                if cand[self.treatment] >= item_[self.treatment] and cand[self.outcome] >= item_[self.outcome]:
                    item_['{}_prime'.format(self.treatment)] = cand[self.treatment]
                    item_['{}_prime'.format(self.outcome)] = cand[self.outcome]
                    to_out = to_out.append(item_)
                elif cand[self.treatment] <= item_[self.treatment] and cand[self.outcome] <= item_[self.outcome]:
                    item_['{}_prime'.format(self.treatment)] = cand[self.treatment]
                    item_['{}_prime'.format(self.outcome)] = cand[self.outcome]
                    to_out = to_out.append(item_)
        return to_out

    def get_valid_counters_decr(self, treated, prime_cands, indices):
        to_out = pd.DataFrame()

        for jj, (_, item_) in tqdm(enumerate(treated.iterrows()), total=len(treated)):
            cands = prime_cands[jj]
            for ll, cand in cands.iterrows():
                if cand[self.treatment] >= item_[self.treatment] and cand[self.outcome] <= item_[self.outcome]:
                    item_['{}_prime'.format(self.treatment)] = cand[self.treatment]
                    item_['{}_prime'.format(self.outcome)] = cand[self.outcome]
                    to_out = to_out.append(item_)
                elif cand[self.treatment] <= item_[self.treatment] and cand[self.outcome] >= item_[self.outcome]:
                    item_['{}_prime'.format(self.treatment)] = cand[self.treatment]
                    item_['{}_prime'.format(self.outcome)] = cand[self.outcome]
                    to_out = to_out.append(item_)
        return to_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', type=str, default='../Data/DS_10283_124/')
    parser.add_argument('--load_dataset'
                        '', type=bool, default=False)
    parser.add_argument('--save_dataset', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='../Data/DS_10283_124/')
    parser.add_argument('--save_name', type=str, default='stroke_data_treatment_{}_outcome_{}_{}.csv')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--neighb', type=int, default=50)
    parser.add_argument('--bins', type=int, default=3)
    parser.add_argument('--propensity_score', type=bool, default=False)
    parser.add_argument('--treatment', type=str, default='Heparin')
    # parser.add_argument('--treatment', type=str, default='DASP14')
    parser.add_argument('--outcome', type=str, default='Y_3')

    args = parser.parse_args()

    ISTDataset(**vars(args))
