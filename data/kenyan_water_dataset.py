import sys

sys.path.append(".")
sys.path.append("..")
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import pickle_object, read_pickle_object
import pandas as pd

confounder_monotonicities_4 = {'c13_c_child_gender': 'none',  # gender
                               'base_age': 'decreasing',  # age at baseline
                               'momeduc_orig': 'increasing',  # Mother's years of education
                               'splnecmpn_base': 'increasing',
                               # Baseline water quality, ln(spring water E. coli MPN) # High quality water: MPN <=1, High or moderate quality: MPN < 126, water is poor quality: MPN = 126-1000
                               'e1_iron_roof_base': 'decreasing',  # Home has iron roof indicator
                               'hygiene_know_base': 'decreasing',
                               # Mother's hygiene knowledge at baseline. average of demeaned sum of number of correct responses given to the open-ended question “to your knowledge, what can be done to prevent diarrhea?
                               'latrine_density_base': 'decreasing',  # Baseline latrine density
                               'numkids_base': 'increasing'  # Number of children under 12 living at home
                               }

confounder_monotonicities_3 = {'c13_c_child_gender': 'none',  # gender
                               'base_age': 'increasing',  # age at baseline
                               'momeduc_orig': 'increasing',  # Mother's years of education
                               'splnecmpn_base': 'increasing',
                               # Baseline water quality, ln(spring water E. coli MPN) # High quality water: MPN <=1, High or moderate quality: MPN < 126, water is poor quality: MPN = 126-1000
                               'e1_iron_roof_base': 'increasing',  # Home has iron roof indicator
                               'hygiene_know_base': 'increasing',
                               # Mother's hygiene knowledge at baseline. average of demeaned sum of number of correct responses given to the open-ended question “to your knowledge, what can be done to prevent diarrhea?
                               'latrine_density_base': 'increasing',  # Baseline latrine density
                               'numkids_base': 'increasing'  # Number of children under 12 living at home
                               }

confounder_monotonicities_2 = {'c13_c_child_gender': 'none',  # gender
                               'base_age': 'none',  # age at baseline
                               'momeduc_orig': 'none',  # Mother's years of education
                               'splnecmpn_base': 'increasing',
                               # Baseline water quality, ln(spring water E. coli MPN) # High quality water: MPN <=1, High or moderate quality: MPN < 126, water is poor quality: MPN = 126-1000
                               'e1_iron_roof_base': 'none',  # Home has iron roof indicator
                               'hygiene_know_base': 'none',
                               # Mother's hygiene knowledge at baseline. average of demeaned sum of number of correct responses given to the open-ended question “to your knowledge, what can be done to prevent diarrhea?
                               'latrine_density_base': 'none',  # Baseline latrine density
                               'numkids_base': 'increasing'  # Number of children under 12 living at home
                               }

confounder_monotonicities_1 = {'c13_c_child_gender': 'none',  # gender
                               'base_age': 'none',  # age at baseline
                               'momeduc_orig': 'none',  # Mother's years of education
                               'splnecmpn_base': 'none',
                               # Baseline water quality, ln(spring water E. coli MPN) # High quality water: MPN <=1, High or moderate quality: MPN < 126, water is poor quality: MPN = 126-1000
                               'e1_iron_roof_base': 'none',  # Home has iron roof indicator
                               'hygiene_know_base': 'none',
                               # Mother's hygiene knowledge at baseline. average of demeaned sum of number of correct responses given to the open-ended question “to your knowledge, what can be done to prevent diarrhea?
                               'latrine_density_base': 'none',  # Baseline latrine density
                               'numkids_base': 'none'  # Number of children under 12 living at home
                               }

median_child_paper = {'c13_c_child_gender':1,
                'base_age': 6,
                'momeduc_orig': 6,
                'splnecmpn_base':4,
                'e1_iron_roof_base':1,
                'hygiene_know_base':3,
                'latrine_density_base':0.4,
                'numkids_base':5
                }
median_child = {'c13_c_child_gender':1,
                'base_age': 1.66666,
                'momeduc_orig': 7,
                'splnecmpn_base':3.42,
                'e1_iron_roof_base':1,
                'hygiene_know_base':4,
                'latrine_density_base':0.446429,
                'numkids_base':3
                }

def get_subportion_confounders(df,to_keep):
    if 'all' not in to_keep:
        return df[to_keep]
    else:
        return df[median_child.keys()]


def undo_demean(x, c=0):
    x_temp = x - (np.nanmin(x[x < 0]) + np.nanmax(x[x < 0]))
    x = x_temp - np.nanmin(x_temp) + c
    return x


class KenyanWaterDataset:

    def __init__(self, path_to_data, save_path=None,load_path=None,save_dataset=False,load_dataset=False,
                 u_distribution='normal', p=0.5, mu=1, sigma=2/3, low=0, high=3,
                 train_test_split_fr=0.8,**kwargs):
        self.u_distribution = u_distribution
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high
        if save_dataset:
            assert save_path is not None
        if load_dataset:
            assert load_dataset is not None

        self.path_to_data = path_to_data
        if not load_dataset:
            self.df = self.get_data()
            if save_dataset:
                pickle_object(self.df,save_path.format(u_distribution))
        else:
            self.df = read_pickle_object(load_path)
        if 'propensity' not in path_to_data:
            train_idx,test_idx = train_test_split(list(range(len(self.df))),test_size=1-train_test_split_fr)

            self.train = self.df.iloc[train_idx].astype('float')
            self.test = self.df.iloc[test_idx].astype('float')
        else:
            self.train = self.df['train']
            self.test = self.df['test']

    def get_uy_samples(self,N_samples):
        if self.u_distribution == 'normal':
            temp = np.random.normal(self.mu, self.sigma, N_samples)
        elif self.u_distribution == 'uniform':
            temp = np.random.randint(self.low, self.high, N_samples)
        return temp

    def get_data(self):
        """
        Data Processing from https://github.com/mariacuellar/probabilityofcausation/blob/master/Application
        :return: pandas dataframe with targets and features
        """
        df = pd.read_stata(os.path.join(self.path_to_data, "reg_data_children_Aug2010.dta"))
        # no multiusers
        df = df.loc[(df['multiusers_l_base'] == 0) | (df['multiusers_l_base'].isna())]
        # no height outliers
        df = df.loc[(df['height_outlier_severe'] == 0) | (df['height_outlier_severe'].isna())]
        # height problems
        df = df.loc[(df['problem_weight'] == 0) | (df['problem_weight'].isna())]
        # BMI problem
        df = df.loc[(df['problem_bmi'] == 0) | (df['problem_bmi'].isna())]
        # Flag age, not sure what that is
        df = df.loc[(df['flag_age'] == 0) | (df['flag_age'].isna())]
        # Problem age
        df = df.loc[(df['problem_age'] == 0) | (df['problem_age'].isna())]
        # Remove more than 3 years
        df = df.loc[(df['base_age'] <= 3) | (df['base_age'].isna())]
        assert len(df) == 22620

        df['evertreat'] = df['evertreat'].map(lambda x: 0 if x == 'TREAT' else 1)
        df['evertreat'] = df['evertreat'].fillna(1)

        # ##### Make Features #####
        # Treatment
        a = df['evertreat'].astype(int)
        # Diarrhea in past week. Diarrhea defined as three or more “looser than normal” stools within
        # 24 hours at any time in the past week
        y = df['c14_d_child_diarrhea']

        features_to_include = ['c13_c_child_gender',  # gender
                               'base_age',  # age at baseline
                               'momeduc_orig',  # Mother's years of education
                               'splnecmpn_base',
                               # Baseline water quality, ln(spring water E. coli MPN) # High quality water: MPN <=1, High or moderate quality: MPN < 126, water is poor quality: MPN = 126-1000
                               'e1_iron_roof_base',  # Home has iron roof indicator
                               'hygiene_know_base',
                               # Mother's hygiene knowledge at baseline. average of demeaned sum of number of correct responses given to the open-ended question “to your knowledge, what can be done to prevent diarrhea?
                               'latrine_density_base',  # Baseline latrine density
                               'numkids_base',  # Number of children under 12 living at home
                               ]
        df_x = df[features_to_include]

        for column_to_prc in ['splnecmpn_base', 'hygiene_know_base', 'latrine_density_base', 'numkids_base']:
            c = 1 if column_to_prc == 'hygiene_know_base' else 0
            df_x[column_to_prc] = undo_demean(df_x[column_to_prc].values, c)

        df_x.insert(0, 'targets', y.values)

        df_x.insert(1, 'treatment', a.values)
        df_x.insert(2, 'uy', self.get_uy_samples(len(df_x)))



        df_x.dropna(axis=0, inplace=True)
        df_x = df_x.reset_index()

        df_x.drop('index', 1, inplace=True)
        df_x =  df_x.astype({'targets': 'int32',
                     'treatment': 'int32'})
        return df_x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data',
                        default='/vol/medic01/users/av2514/Pycharm_projects/Twin_Nets_Causality/data/Datasets/dataverse_files/dta')
    parser.add_argument('--load_path',default='./Datasets/kenyan_water_proc_single.pkl')
    parser.add_argument('--load_dataset',default=False)
    parser.add_argument('--save_dataset',default=True)
    parser.add_argument('--save_path',default='./Datasets/kenyan_water_proc_single_uy_{}.pkl')
    args = parser.parse_args()

    KenyanWaterDataset(**vars(args))
