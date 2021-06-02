import sys

sys.path.append(".")
sys.path.append("..")
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from utils import pickle_object, read_pickle_object
from collections import defaultdict
import pandas as pd
import glob
import json
from data.twin_data_metadata import cov_description, cov_types
import copy
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit



def get_subportion_confounders(df,to_keep):
    if 'all' not in to_keep:
        return df[to_keep]
    else:
        return df



class TwinsDataset:
    def __init__(self, path_to_data, save_path=None, load_path=None, save_dataset=False, load_dataset=False,
                 u_distribution='normal', p= 0.5, mu=1, sigma=2 / 3, low=0, high=3,
                 train_test_split_fr=0.8, **kwargs):
        self.u_distribution = u_distribution
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high
        self.train_test_split_fr = train_test_split_fr
        if save_dataset:
            assert save_path is not None
        if load_dataset:
            assert load_dataset is not None

        self.path_to_data = path_to_data
        if not load_dataset:
            if not args.ganite:
                self.df = self.get_data()
            else:
                self.df = self.get_ganite_data()
            if save_dataset:
                pickle_object(self.df, save_path.format(u_distribution))
        else:
            self.df = read_pickle_object(load_path)

        self.train = self.df['train']
        self.test = self.df['test']

    def get_uy_samples(self, N_samples):
        if self.u_distribution == 'normal':
            temp = np.random.normal(self.mu, self.sigma, N_samples)
        elif self.u_distribution == 'uniform':
            temp = np.random.randint(self.low, self.high, N_samples)
        return temp

    def get_ganite_data(self):
        data = pd.read_csv(os.path.join(self.path_to_data, 'Twin_data.csv'))
        ori_data = np.loadtxt(os.path.join(self.path_to_data, 'Twin_data.csv'), delimiter=",", skiprows=1)

        # Define features
      
        x = ori_data[:, :30]
        no, dim = x.shape

        # Define potential outcomes
        potential_y = ori_data[:, 30:]
        # Die within 1 year = 1, otherwise = 0
        potential_y = np.array(potential_y < 9999, dtype=float)

        ## Assign treatment
        coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])
        prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))

        prob_t = prob_temp / (2 * np.mean(prob_temp))
        prob_t[prob_t > 1] = 1

        t = np.random.binomial(1, prob_t, [no, 1])
        t = t.reshape([no, ])

        uy_ = self.get_uy_samples(len(t))

        df = pd.DataFrame()
        df.insert(0,'X',t)
        df.insert(0,'Uy',uy_)
        df.insert(0,'Y',potential_y[:,0])
        df.insert(0,'Y_prime',potential_y[:,1])
        df.insert(0,'X_prime',np.logical_xor(t,1).astype(int))

        for i in range(dim):
            df.insert(0,data.columns[i].replace('\'',''),x[:,i])

        train_idx, test_idx = train_test_split(list(range(len(df))), test_size=1 - self.train_test_split_fr)

        train_ = df.iloc[train_idx]
        test_ = df.iloc[test_idx]

        to_save = {
            'train': train_,
            'test': test_
        }
        return to_save



    def get_data(self):
        covariates = pd.read_csv(os.path.join(self.path_to_data, 'twin_pairs_X_3years_samesex.csv'))
        treatment = pd.read_csv(os.path.join(self.path_to_data, 'twin_pairs_T_3years_samesex.csv'))
        outcome = pd.read_csv(os.path.join(self.path_to_data, 'twin_pairs_Y_3years_samesex.csv'))

        to_drop = [k for k in cov_description.keys() if 'risk' not in cov_description[k]]
        # to_drop = []
        covariates.drop([*to_drop, 'Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
        treatment.drop(['Unnamed: 0'], axis=1, inplace=True)
        outcome.drop(['Unnamed: 0'], axis=1, inplace=True)

        uy_ = self.get_uy_samples(len(covariates))

        first_, sec_ = train_test_split(list(range(len(treatment))), test_size=0.5)

        treat = []
        treat_prime = []
        out = []
        out_prime = []
        for idxs in first_:
            out.append(outcome.mort_0.iloc[idxs])
            out_prime.append(outcome.mort_1.iloc[idxs])
            treat.append(0)
            treat_prime.append(1)
        for idxs in sec_:
            out.append(outcome.mort_1.iloc[idxs])
            out_prime.append(outcome.mort_0.iloc[idxs])
            treat.append(1)
            treat_prime.append(0)

        df = copy.deepcopy(covariates)
        df.insert(0, 'X', treat)
        df.insert(0,'X_prime',treat_prime)

        df.insert(0,'Y_prime',out_prime)
        df.insert(0,'Y',out)

        df.insert(0,'Uy',uy_)

        if args.smote:
            y = df['Y']

            dataset_proc = df.drop(['Y'], axis=1, inplace=False)
            dataset_proc = dataset_proc.interpolate(axis=1)
            smote = SMOTE()
            x_sm, y_sm = smote.fit_resample(dataset_proc, y)
            x_sm.insert(0,'Y',y_sm)
            scaler = MinMaxScaler()
            x_sm['Y_prime'] = scaler.fit_transform(x_sm['Y_prime'].values[...,np.newaxis])
            x_sm['Y_prime'].loc[x_sm['Y_prime']>=0.5] = 1
            x_sm['Y_prime'].loc[x_sm['Y_prime']<0.5] = 0
            y_2 = x_sm['Y_prime']
            x_sm.drop(['Y_prime'], axis=1, inplace=True)
            x_sm_2, y_sm_2 = smote.fit_resample(x_sm, y_2)
            x_sm_2.insert(0,'Y_prime',y_sm_2)
            x_sm_2['Y'].loc[x_sm_2['Y'] >= 0.5] = 1
            x_sm_2['Y'].loc[x_sm_2['Y'] < 0.5] = 0
            x_sm_2['X'].loc[x_sm_2['X'] >= 0.5] = 1
            x_sm_2['X'].loc[x_sm_2['X'] < 0.5] = 0
            x_sm_2['X_prime'].loc[x_sm_2['X_prime'] >= 0.5] = 1
            x_sm_2['X_prime'].loc[x_sm_2['X_prime'] < 0.5] = 0

            df = x_sm_2

        if args.interpolate:
            df = df.interpolate(axis=1)

        else:
            df.dropna(axis=0, inplace=True)

        df = df.reset_index()

        df.drop('index', 1, inplace=True)

        train_idx, test_idx = train_test_split(list(range(len(df))), test_size=1-self.train_test_split_fr)

        train_ = df.iloc[train_idx]
        test_ = df.iloc[test_idx]

        to_save = {
            'train': train_,
            'test': test_
        }
        return to_save


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data',
                        # default='/vol/medic01/users/av2514/Pycharm_projects/Twin_Nets_Causality/data/Datasets/TWINS')
                        default='/vol/medic01/users/av2514/Pycharm_projects/Twin_Nets_Causality/data/Datasets/Twins_ganite/data')
    parser.add_argument('--load_path', default='./Datasets/twins_uy_gaussian_twins_as_counterfactuals.pkl')
    parser.add_argument('--load_dataset', default=False)
    parser.add_argument('--save_dataset', default=True)
    parser.add_argument('--smote', default=False)
    parser.add_argument('--ganite', default=True)
    parser.add_argument('--interpolate', default=True)
    # parser.add_argument('--save_path', default='./Datasets/ganite_twins_uy_{}_twins_as_counterfactuals_interpolate_all.pkl')
    parser.add_argument('--save_path', default='./Datasets/ganite_twins_uy_{}_twins.pkl')
    args = parser.parse_args()

    TwinsDataset(**vars(args))
