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
sys.path.append("..")
sys.path.append("../..")
import argparse
from copy import deepcopy
from sklearn.model_selection import train_test_split
from german_metadata import categories_to_numerical, CategoricalFeatures, decre_monoto
from tqdm import tqdm
import itertools
from prettytable import PrettyTable
from utils import pickle_object

def percentage(x,i):
    return np.sum(x==i)/len(x)


def get_subportion_confounders(df,to_keep):
    if 'all' not in to_keep:
        return df[to_keep]
    else:
        return df


class GermanCreditDataset:
    def __init__(self, path_to_data, save_path=None, load_path=None, save_name=None, save_dataset=False,
                 load_dataset=False, u_distribution='normal', p=0.5, mu=1, sigma=1, low=0, high=7, #2/3
                 train_test_split_fr=0.8, propensity_score=False, treatment='Credit-History', outcome='Status',switch_treatments=None,
                 **kwargs):


        self.save_path = save_path
        self.save_name = save_name
        self.load_path = load_path
        self.load_path = load_path
        self.u_distribution = u_distribution
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high

        self.treatment = treatment
        self.switch_treatments = switch_treatments
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
            elif 'Y' in self.outcome:
                self.df = self.make_synthetic()

        else:
            self.train = pd.read_csv(self.path_to_data.format('train'))
            self.test = pd.read_csv(self.path_to_data.format('test'))


        if save_dataset:
            self.df.drop_duplicates(inplace=True)
            train_idx, test_idx = train_test_split(list(range(len(self.df))), test_size=1 - train_test_split_fr)
            self.train = self.df.iloc[train_idx].astype('float')
            self.test = self.df.iloc[test_idx].astype('float')
            if propensity_score:
                self.save_dataset(self.train, save_path, save_name.format(self.treatment.replace('-', '_'), self.outcome, self.neighb,'train'))
                self.save_dataset(self.test, save_path, save_name.format(self.treatment.replace('-', '_'), self.outcome, self.neighb,'test'))
            elif 'Y' in self.outcome:
                self.save_dataset(self.train, save_path,
                                  save_name.format(self.treatment.replace('-', '_'), self.outcome,
                                                   'train'))
                self.save_dataset(self.test, save_path,
                                  save_name.format(self.treatment.replace('-', '_'), self.outcome, 'test'))

    def save_dataset(self, item , save_path, save_name):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        item.to_csv(os.path.join(save_path, save_name))

    def get_uy_samples(self, N_samples):
        if self.u_distribution == 'normal':
            temp = np.random.normal(self.mu, self.sigma, N_samples)
        elif self.u_distribution == 'digi_normal':
            temp = np.random.normal(self.mu, self.sigma, N_samples)
            temp = np.digitize(temp, [0,1, 2])
            print('Uy=0 {}'.format(percentage(temp,0)))
            print('Uy=1 {}'.format(percentage(temp,1)))
            print('Uy=2 {}'.format(percentage(temp,2)))
            print('Uy=3 {}'.format(percentage(temp,3)))
        elif self.u_distribution == 'uniform':
            temp = np.random.randint(self.low, self.high, N_samples)
        return temp

    def get_data(self):

        df_data = pd.read_csv(os.path.join(self.path_to_data, 'german.data'), header=None, delim_whitespace=True)
        # df_data = pd.read_csv(os.path.join(self.path_to_data,'german.data-numeric'),header=None,)
        df_data.columns = ["Existing-Account-Status", "Month-Duration", "Credit-History", "Purpose", "Credit-Amount",
                           "Saving-Acount", "Present-Employment", "Instalment-Rate", "Sex", "Guarantors", "Residence",
                           "Property", "Age", "Installment", "Housing", "Existing-Credits", "Job", "Num-People",
                           "Telephone", "Foreign-Worker", "Status"]

        data_encode = df_data.copy()

        data_encode = data_encode.replace(categories_to_numerical)
        data_encode.insert(0, 'Uy', self.get_uy_samples(len(data_encode)))
        data_encode.Status = np.abs(data_encode.Status - 2)

        if self.switch_treatments is not None:
            idx_0 = np.where(data_encode[self.treatment].values == self.switch_treatments[0])[0]
            idx_1 = np.where(data_encode[self.treatment].values == self.switch_treatments[1])[0]

            data_encode[self.treatment].iloc[idx_0] = self.switch_treatments[1]
            data_encode[self.treatment].iloc[idx_1] = self.switch_treatments[0]

        return data_encode
    def make_synthetic(self):
        data_encode = deepcopy(self.df)
        tret = data_encode[self.treatment].values
        data_encode[self.treatment].iloc[np.where(tret==3)[0]] = 2

        data_encode = data_encode.reset_index().drop(['index'],axis=1,inplace=False)

        if np.unique(data_encode[self.treatment].values).max() > 1:
            data_encodeprimed = pd.DataFrame()
            for jj, (_, row) in tqdm(enumerate(data_encode.iterrows()), total=len(data_encode)):
                cands = [i for i in np.unique(data_encode[self.treatment].values) if i != row[self.treatment]]
                for cand in cands:
                    row['{}_prime'.format(self.treatment)] = cand
                    data_encodeprimed = data_encodeprimed.append(row)
            data_encode = data_encodeprimed.drop_duplicates(inplace=False)
            x_prime = data_encode['{}_prime'.format(self.treatment)]
        else:
            x_prime = np.logical_xor(data_encode[self.treatment].values, 1).astype(int)

        dig_cred_history = np.ones_like(data_encode['Credit-History'].values)
        dig_cred_history[np.where(data_encode['Credit-History'].values==0)[0]] = 0
        dig_cred_history[np.where(data_encode['Credit-History'].values==1)[0]]  = 0
        dig_cred_history[np.where(data_encode['Credit-History'].values==2)[0]]  = 1
        dig_cred_history[np.where(data_encode['Credit-History'].values==3)[0]]  = 2
        dig_cred_history[np.where(data_encode['Credit-History'].values==4)[0]]  = 2
        data_encode.insert(0, 'dig', dig_cred_history)

        uy = data_encode.Uy.values

        # g = data_encode[self.treatment].values + 1/(1+np.exp(-uy-dig_cred_history))
        # g_prime = data_encode['{}_prime'.format(self.treatment)].values + 1/(1+np.exp(-uy-dig_cred_history))
        #
        g = np.zeros_like(data_encode[self.treatment].values)
        g[np.where(uy==0)[0]] =data_encode[self.treatment].iloc[np.where(uy==0)[0]] + dig_cred_history[np.where(uy==0)[0]]
        g[np.where(uy==2)[0]] = data_encode[self.treatment].iloc[np.where(uy==2)[0]] *  dig_cred_history[np.where(uy==2)[0]]
        g[g>2] =2

        g[np.where(uy==3)[0]] = 2
        g[np.where(uy==4)[0]] = 1
        g[np.where(uy==5)[0]] =  np.heaviside(data_encode['{}'.format(self.treatment)].iloc[np.where(uy==5)[0]]-1,0)#data_encode['{}'.format(self.treatment)].iloc[np.where(uy==5)[0]]  * dig_cred_history[np.where(uy==5)[0]]/2
        g[np.where(uy==6)[0]] =  np.heaviside(data_encode['{}'.format(self.treatment)].iloc[np.where(uy==6)[0]]-1,0)*2#data_encode['{}'.format(self.treatment)].iloc[np.where(uy==5)[0]]  * dig_cred_history[np.where(uy==5)[0]]/2
        g[g < 0] = 0
        # g[data_encode[data_encode.Uy==1].index] = 2

        g_prime = np.zeros_like(data_encode['{}_prime'.format(self.treatment)].values)
        g_prime[np.where(uy==0)[0]] = data_encode['{}_prime'.format(self.treatment)].iloc[np.where(uy==0)[0]] + dig_cred_history[np.where(uy==0)[0]]
        g_prime[np.where(uy==2)[0]] = data_encode['{}_prime'.format(self.treatment)].iloc[np.where(uy==2)[0]] *  dig_cred_history[np.where(uy==2)[0]]
        g_prime[g_prime >2] = 2

        g_prime[np.where(uy==3)[0]] = 2
        g_prime[np.where(uy==4)[0]] = 1
        g_prime[np.where(uy==5)[0]] = np.heaviside(data_encode['{}_prime'.format(self.treatment)].iloc[np.where(uy==5)[0]]-1,0)#data_encode['{}_prime'.format(self.treatment)].iloc[np.where(uy==5)[0]] * dig_cred_history[np.where(uy==5)[0]]/2
        g_prime[np.where(uy==6)[0]] = np.heaviside(data_encode['{}_prime'.format(self.treatment)].iloc[np.where(uy==6)[0]]-1,0)*2#data_encode['{}_prime'.format(self.treatment)].iloc[np.where(uy==5)[0]] * dig_cred_history[np.where(uy==5)[0]]/2
        g_prime[g_prime<0] = 0
        # g_prime[data_encode[data_encode.Uy == 1].index] = 2

        #
        # g = np.around((  data_encode['{}_prime'.format(self.treatment)].values+ np.abs(data_encode.Uy))/2)
        # g[g<0]=0
        data_encode['{}'.format(self.outcome)] = g.astype(int)
        #
        # g_prime = np.around((  data_encode[self.treatment].values+ np.abs(data_encode.Uy))/2)
        # g_prime[g_prime < 0] = 0
        data_encode['{}_prime'.format(self.outcome)] = g_prime.astype(int)

        self.calc_probs(data_encode)

        return data_encode


    def do_propensity_matching(self):
        dataset_base = deepcopy(self.df)
        dataset_proc = dataset_base.drop(['Uy', self.outcome], axis=1, inplace=False)
        T = dataset_proc[self.treatment]
        X = dataset_proc.loc[:, (dataset_proc.columns != self.treatment)]

        pipe = GradientBoostingClassifier(n_estimators=50, learning_rate=1,
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

    def get_valid_counters(self,treated, prime_cands, indices):
        if self.treatment in decre_monoto:
            return self.get_valid_counters_decr(treated, prime_cands, indices)
        else:
            return self.get_valid_counters_incresing( treated, prime_cands, indices)

    def get_valid_counters_incresing(self, treated, prime_cands, indices):
        to_out = pd.DataFrame()

        for jj,(_, item_) in tqdm(enumerate(treated.iterrows()),total=len(treated)):
            cands = prime_cands[jj]
            for ll, cand in cands.iterrows():
                # if cand[self.treatment] >= item_[self.treatment] and cand[self.outcome] >= item_[self.outcome]:
                item_['{}_prime'.format(self.treatment)] = cand[self.treatment]
                item_['{}_prime'.format(self.outcome)] = cand[self.outcome]
                to_out = to_out.append(item_)
                # elif cand[self.treatment] <= item_[self.treatment] and cand[self.outcome] <= item_[self.outcome]:
                #     item_['{}_prime'.format(self.treatment)] = cand[self.treatment]
                #     item_['{}_prime'.format(self.outcome)] = cand[self.outcome]
                #     to_out = to_out.append(item_)
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
                pn = self.p_n(data_encode[self.treatment].values, data_encode['{}_prime'.format(self.treatment)].values,
                              data_encode[self.outcome].values,
                              data_encode['{}_prime'.format(self.outcome)].values, treat, treat_counter, outcome,
                              outcome_counter,data_encode['Uy'].values, data_encode.dig.values)
                res_1[outcome, outcome_counter] = pn
                ps = self.p_s(data_encode[self.treatment].values, data_encode['{}_prime'.format(self.treatment)].values,
                              data_encode[self.outcome].values,
                              data_encode['{}_prime'.format(self.outcome)].values, treat, treat_counter, outcome,
                              outcome_counter)
                res_2[outcome, outcome_counter] = ps
                pns = self.p_ns(data_encode[self.treatment].values, data_encode['{}_prime'.format(self.treatment)].values,
                                data_encode[self.outcome].values,
                                data_encode['{}_prime'.format(self.outcome)].values, treat, treat_counter, outcome,
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

    def p_n(self, x, x_prime, y, y_prime, treatment, treatment_counter, outcome, outcome_counter,uy,z):
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', type=str, default='../Data/German Credit/')
    parser.add_argument('--load_dataset'
                        '', type=bool, default=False)
    parser.add_argument('--save_dataset', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='../Data/German Credit/')
    # parser.add_argument('--save_name', type=str, default='german_data_treatment_{}_outcome_{}_{}.csv')
    parser.add_argument('--save_name', type=str, default='german_data_treatment_{}_2_outcome_{}_neighb_{}_{}.csv')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--neighb', type=int, default=50)
    parser.add_argument('--propensity_score', type=bool, default=True)
    # parser.add_argument('--treatment', type=str, default='Credit-History')
    parser.add_argument('--treatment', type=str, default='Existing-Account-Status')
    parser.add_argument('--outcome', type=str, default='Status')
    parser.add_argument('--switch_treatments',default=[0,1])
    # parser.add_argument('--outcome', type=str, default='Y')

    args = parser.parse_args()

    GermanCreditDataset(**vars(args))
