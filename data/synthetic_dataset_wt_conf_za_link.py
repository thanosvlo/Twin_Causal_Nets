import sys

sys.path.append(".")
sys.path.append("..")
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from utils import pickle_object, read_pickle_object
from collections import defaultdict

confounder_monotonicities_1 = {
    'Z_1': 'None',
    'Z_2': 'None',
    'Z': 'None',
    'Ux': 'None'
}


def get_subportion_confounders(df, to_keep):
    if 'all' not in to_keep:
        return df[to_keep]
    else:
        return df


class SyntheticDataset:

    def __init__(self, path_to_data=None, n_samples=200000, x_distribution='normal', u_distribution='uniform',
                 z_distribution='uniform', p=0.5,
                 mu=1.5, sigma=2 / 3, low=0, high=2, split=0.8, p_1=1 / 4, p_2=3 / 4, p_3=1 / 4, save_path=None,
                 save_dataset=False,
                 load_dataset=False,
                 **kwargs):
        self.N_samples = n_samples

        self.p_1 = p_1
        self.p_2 = p_2
        self.p_3 = p_3
        self.X_distribution = x_distribution
        self.U_distribution = u_distribution
        self.Z_distribution = z_distribution
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high
        self.split = split
        self.save_path = save_path

        if not load_dataset:
            self.meta = vars(self)
            self.train, self.test, self.probs = self.create_dataset()

        else:
            read_object = read_pickle_object(path_to_data)
            # self.meta_ = read_object['metadata']
            self.train = read_object['train']
            self.test = read_object['test']

        if save_dataset:
            self.save_name = kwargs.get('save_name', 'synthetic_dataset_{}_samples_X_{}_Uy_{}.pkl')
            z_dist = '{}_{}'.format(self.Z_distribution, self.p_2,
                                    self.p_2) if self.Z_distribution == 'bernouli' else '{}'.format(
                self.Z_distribution)
            x_dist = '{}_{}'.format(self.X_distribution, self.p_1,
                                    self.p_1) if self.X_distribution == 'bernouli' else '{}'.format(
                self.X_distribution)
            self.save_name = self.save_name.format(self.N_samples, x_dist, self.U_distribution,
                                                   z_dist)
            self.save_dataset()

    def get_x_samples(self, N_samples, p=1 / 3):
        print('X Dist: {}'.format(self.X_distribution))
        if self.X_distribution == 'normal':
            temp = np.random.normal(self.mu, self.sigma, N_samples)
            temp = np.digitize(temp, [0, 2])
            temp[temp > 1] = 0
            print('P(X=0) = {}'.format(np.sum(temp == 0) / len(temp)))
            print('P(X=1) = {}'.format(np.sum(temp >= 1) / len(temp)))
            return temp
        elif self.X_distribution == 'uniform':
            return np.random.randint(0, 2, N_samples)
        elif self.X_distribution == 'bernouli':
            return  np.random.binomial(n=1, p=p, size=N_samples)


    def get_uy_samples(self, N_samples, p=1 / 3,z=None):
        print('Uy Dist: {}'.format(self.U_distribution))
        if self.U_distribution == 'normal':
            temp = np.random.normal(self.mu, self.sigma, N_samples)
            temp = np.digitize(temp, [1, 2])
            # temp[temp > 1] = 0
            temp[temp == 1] = -1
            temp[temp == 0] = 1
            temp[temp == -1] = 0
            print('P(Uy=0) = {}'.format(np.sum(temp == 0) / len(temp)))
            print('P(Uy=1) = {}'.format(np.sum(temp == 1) / len(temp)))
            print('P(Uy=2) = {}'.format(np.sum(temp >= 2) / len(temp)))
            return temp
        elif self.U_distribution == 'uniform':
            return np.random.randint(0, 3, N_samples)
        elif self.U_distribution=='bernouli':
            temp = np.random.binomial(n=1, p=p, size=N_samples)
            isx = np.where(temp == 1)[0]
            idx_1 = isx[0:int(len(isx) / 2)]
            idx_2 = isx[int(len(isx) / 2):]
            temp[idx_1] = 1
            temp[idx_2] = 2
            print('P(Uy=0) = {}'.format(np.sum(temp == 0) / len(temp)))
            print('P(Uy=1) = {}'.format(np.sum(temp == 1) / len(temp)))
            print('P(Uy=2) = {}'.format(np.sum(temp >= 2) / len(temp)))
            return temp
        elif self.U_distribution == 'p_test':
            temp = np.random.binomial(n=1, p=1 / 3, size=N_samples)
            temp = temp * 2
            temp_2 = np.random.binomial(n=1, p=p, size=np.sum(temp == 0))
            to_out = np.concatenate([temp_2, temp[temp == 2]])
            gt = self.calc_probs(z,to_out)
            return to_out, gt

    def get_z_samples(self, N_samples, p=1 / 3):
        print('Z Dist: {}'.format(self.Z_distribution))

        if self.Z_distribution == 'normal':
            temp = np.random.normal(self.mu, self.sigma, N_samples)
            temp = np.digitize(temp, [0, 2])
            temp[temp > 1] = 0
            print('P(Z=0) = {}'.format(np.sum(temp == 0) / len(temp)))
            print('P(Z=1) = {}'.format(np.sum(temp >= 1) / len(temp)))
            return temp
        elif self.Z_distribution == 'uniform':
            return np.random.randint(0, 2, N_samples)
        elif self.Z_distribution == 'bernouli':
            return np.random.binomial(n=1, p=p, size=N_samples)

    def create_dataset(self):
        Zs = self.get_z_samples(self.N_samples, self.p_2)
        z_0 = np.where(Zs == 0)[0]
        z_1 = np.where(Zs == 1)[0]

        Ux_1 = self.get_x_samples(self.N_samples, self.p_1)
        ux_0 = np.where(Ux_1 == 0)[0]
        ux_1 = np.where(Ux_1 == 1)[0]

        # Xs =Zs #Ux_1 #
        Xs = np.zeros(self.N_samples)
        x_1 = np.array(list(set(ux_0).intersection(z_1)))
        Xs[x_1] = 1
        x_1 = np.array(list(set(ux_1).intersection(z_0)))
        Xs[x_1] = 1
        # Xs = Zs * Ux_1

        Xs_prime = np.logical_xor(Xs, np.ones_like(Xs)).astype(int)

        Uy = self.get_uy_samples(self.N_samples,self.p_3)
        uy_1 = np.where(Uy == 1)[0]
        uy_0 = np.where(Uy == 0)[0]
        uy_2 = np.where(Uy == 2)[0]

        probs = self.calc_probs(Zs, Uy, Ux_1)

        Ys = np.zeros(self.N_samples)

        Ys[uy_0] = Xs[uy_0] * Zs[uy_0]
        Ys[uy_2] = 1

        Ys_prime = np.zeros(self.N_samples)
        Ys_prime[uy_0] = Xs_prime[uy_0] * Zs[uy_0]
        Ys_prime[uy_2] = 1

        idx_given_y_1 = np.where(Ys[np.where(Xs == 1)[0]] == 1)[0]
        idx_query_y_0 = np.where(Ys_prime[np.where(Xs_prime == 0)[0]] == 0)[0]

        idx_y_1_y_prime_0 = list(set(idx_given_y_1).intersection(idx_query_y_0))

        prob_necessity_1 = len(idx_y_1_y_prime_0) / len(idx_given_y_1)

        idx_given_y_0 = np.where(Ys[np.where(Xs == 0)[0]] == 0)[0]
        idx_query_y_1 = np.where(Ys_prime[np.where(Xs_prime == 1)[0]] == 1)[0]

        idx_y_0_y_prime_1 = list(set(idx_given_y_0).intersection(idx_query_y_1))

        prob_suf_1 = len(idx_y_0_y_prime_1) / len(idx_given_y_0)
        print('N {} \n S {}'.format(prob_necessity_1, prob_suf_1))

        p_y_1_x_0 = len(np.where(Ys[np.where(Xs == 0)[0]] == 1)[0]) / len(Ys)
        p_y_1 = len(np.where(Ys == 1)[0]) / len(Ys)
        p_y_1_x_1 = len(np.where(Ys[np.where(Xs == 1)[0]] == 1)[0]) / len(Ys)
        print('{} >= {} >= {}'.format(p_y_1_x_1, p_y_1, p_y_1_x_0))

        idx_split = int(self.N_samples * self.split)

        X_train = (Xs[:idx_split], Uy[:idx_split], Zs[:idx_split], Ux_1[:idx_split], Xs_prime[:idx_split])
        X_test = (Xs[idx_split:], Uy[idx_split:], Zs[idx_split:], Ux_1[idx_split:], Xs_prime[idx_split:])

        Y_train = (Ys[:idx_split], Ys_prime[:idx_split])
        Y_test = (Ys[idx_split:], Ys_prime[idx_split:])

        return (X_train, Y_train), (X_test, Y_test), probs

    def save_dataset(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        out_path = os.path.join(self.save_path, self.save_name)
        print(self.save_name)
        to_save = defaultdict()
        to_save['metadata'] = vars(self)
        to_save['probs'] = {
            'Necessity': self.probs[0],
            'Sufficiency': self.probs[1],
            'Necessity & Sufficiency': self.probs[2]
        }

        to_save['train'] = {'X': self.train[0][0],
                            'Uy': self.train[0][1],
                            'Z': self.train[0][2],
                            'Ux': self.train[0][3],
                            'X_prime': self.train[0][4],
                            'Y': self.train[1][0],
                            'Y_prime': self.train[1][1]}

        to_save['test'] = {'X': self.test[0][0],
                           'Uy': self.test[0][1],
                           'Z': self.test[0][2],
                           'Ux': self.test[0][3],
                           'X_prime': self.test[0][4],
                           'Y': self.test[1][0],
                           'Y_prime': self.test[1][1]}

        del to_save['metadata']['train']
        del to_save['metadata']['test']
        pickle_object(to_save, out_path)

    def calc_probs(self, z, u):
        p_z_0 = np.sum(z == 0) / len(z)
        p_z_1 = np.sum(z == 1) / len(z)
        p_u_0 = np.sum(u == 0) / len(u)
        p_u_1 = np.sum(u == 1) / len(u)
        p_u_2 = np.sum(u == 2) / len(u)
        # p_ux_0 = np.sum(ux == 0) / len(ux)
        # p_ux_1 = np.sum(ux == 1) / len(ux)

        p_n = (p_u_0 * p_z_1) / (p_u_2 + p_u_0 * p_z_1)  # * p_ux_0)
        p_s = (p_u_0 * p_z_1) / (p_u_1 + p_u_0)  # *p_ux_0 )
        p_n_s = (p_u_0 * p_z_1)

        print('Prob Necessity: {}'.format(p_n))
        print('Prob Sufficiency: {}'.format(p_s))
        print('Prob Necessity & Sufficiency: {}'.format(p_n_s))

        return (p_n, p_s, p_n_s)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='./Datasets/')
    parser.add_argument('--save_name',
                        default='synthetic_dataset_{}_samples_X_{}_Uy_{}_Z_{}_with_counterfactual_and_z_a_link_2_final.pkl')
    parser.add_argument('--save_dataset', default=True)
    parser.add_argument('--load_dataset', default=False)
    parser.add_argument('--u_distribution', default='bernouli')
    parser.add_argument('--z_distribution', default='uniform')
    parser.add_argument('--x_distribution', default='bernouli')
    parser.add_argument('--p_1', type=float, default=0.05)
    parser.add_argument('--p_2', type=float, default=0.7)
    parser.add_argument('--p_3', type=float, default=0.3)
    parser.add_argument('--path_to_data',
                        default='./Datasets/synthetic_dataset_200000_samples_X_bernouli_Uy_uniform_Z_0_1_with_counterfactual.pkl')
    args = parser.parse_args()

    dataset = SyntheticDataset(**vars(args))
