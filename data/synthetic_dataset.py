import sys

sys.path.append(".")
sys.path.append("..")
import numpy as np
import os
from utils import pickle_object, read_pickle_object
from collections import defaultdict


class SyntheticDataset:

    def __init__(self, path_to_data=None, n_samples=100000, x_distribution='bernouli', u_distribution='normal', p=0.5,
                 mu=1, sigma=2/3, low=0, high=3, split=0.8, save_path=None, save_dataset=False, load_dataset=False,
                 **kwargs):
        self.N_samples = n_samples

        self.X_distribution = x_distribution
        self.U_distribution = u_distribution
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high
        self.split = split
        self.save_path = save_path

        if not load_dataset:
            self.meta = vars(self)
            self.train, self.test = self.create_dataset()

        else:
            read_object = read_pickle_object(path_to_data)
            self.meta_ = read_object['metadata']
            self.train = read_object['train']
            self.test = read_object['test']

        if save_dataset:
            self.save_name = kwargs.get('save_name', 'synthetic_dataset_{}_samples_X_{}_Uy_{}.pkl')
            self.save_name = self.save_name.format(self.N_samples, self.X_distribution, self.U_distribution)
            self.save_dataset()

    def get_x_samples(self):
        if self.X_distribution == 'bernouli':
            return np.random.binomial(n=1, p=self.p, size=self.N_samples)
        else:
            raise NotImplementedError

    def get_uy_samples(self,p=0.1):
        if self.U_distribution == 'normal':
            temp = np.random.normal(self.mu, self.sigma, self.N_samples)
            temp = np.digitize(temp,[1,2])
            print('P(Uy=0) = {}'.format(np.sum(temp==0)/len(temp)))
            print('P(Uy=1) = {}'.format(np.sum(temp==1)/len(temp)))
            print('P(Uy=2) = {}'.format(np.sum(temp==2)/len(temp)))
            return temp
        elif self.U_distribution == 'uniform':
            return np.random.randint(self.low, self.high, self.N_samples)
        elif self.U_distribution =='p_test':
            temp = np.random.binomial(n=1, p=1/3, size=self.N_samples)
            temp = temp * 2
            temp_2 = np.random.binomial(n=1, p=p, size=np.sum(temp==0))
            to_out = np.concatenate([temp_2,temp[temp==2]])
            gt = self.calc_probs(to_out)
            return to_out,  gt

    def calc_probs(self,uy):
        uy_0 = np.sum(uy == 0) / len(uy)
        uy_1 = np.sum(uy == 1) / len(uy)
        uy_2 = np.sum(uy == 2) / len(uy)
        print('P(Uy=0) = {}'.format(uy_0))
        print('P(Uy=1) = {}'.format(uy_1))
        print('P(Uy=2) = {}'.format(uy_2))

        p_n = uy_0 / (uy_0+uy_2)
        p_s = uy_0 / (uy_0+uy_1)
        p_ns = uy_0
        print ('Prob Necessity: {}'.format(p_n))
        print ('Prob Sufficiency: {}'.format(p_s))
        print ('Prob Necessity & Sufficiency: {}'.format(p_ns))
        return (p_n,p_s,p_ns)

    def create_dataset(self):

        Xs = self.get_x_samples()
        Xs_prime = np.logical_xor(Xs, np.ones_like(Xs)).astype(int)

        Uy = self.get_uy_samples()

        Ys = np.zeros(self.N_samples)

        Ys[np.where(Uy == 0)[0]] = Xs[np.where(Uy == 0)[0]]

        Ys[np.where(Uy == 2)[0]] = 1

        Ys_prime = np.zeros(self.N_samples)
        Ys_prime[np.where(Uy == 0)[0]] = Xs_prime[np.where(Uy == 0)[0]]
        Ys_prime[np.where(Uy == 2)[0]] = 1

        idx_split = int(self.N_samples * self.split)

        X_train = (Xs[:idx_split], Uy[:idx_split], Xs_prime[:idx_split])
        X_test = (Xs[idx_split:], Uy[idx_split:], Xs_prime[idx_split:])

        Y_train = (Ys[:idx_split], Ys_prime[:idx_split])
        Y_test = (Ys[idx_split:], Ys_prime[idx_split:])

        return (X_train, Y_train), (X_test, Y_test)

    def save_dataset(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        out_path = os.path.join(self.save_path, self.save_name)
        to_save = defaultdict()
        to_save['metadata'] = vars(self)

        to_save['train'] = {'X': self.train[0][0],
                            'Uy': self.train[0][1],
                            'X_prime': self.train[0][2],
                            'Y': self.train[1][0],
                            'Y_prime': self.train[1][1]}

        to_save['test'] = {'X': self.test[0][0],
                           'Uy': self.test[0][1],
                           'X_prime': self.test[0][2],
                           'Y': self.test[1][0],
                           'Y_prime': self.test[1][1]}

        del to_save['metadata']['train']
        del to_save['metadata']['test']
        pickle_object(to_save, out_path)
        print(self.save_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='./Datasets/')
    parser.add_argument('--save_name', default='synthetic_dataset_{}_samples_X_{}_Uy_{}_with_counterfactual_2.pkl')
    parser.add_argument('--save_dataset', default=True)
    parser.add_argument('--u_distribution', default='normal')
    parser.add_argument('--load_dataset', default=False)
    parser.add_argument('--path_to_data',
                        default='./Datasets/synthetic_dataset_10000_samples_X_bernouli_Uy_uniform.pkl')
    args = parser.parse_args()

    dataset = SyntheticDataset(**vars(args))
