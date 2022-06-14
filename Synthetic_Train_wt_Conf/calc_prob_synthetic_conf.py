import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score
import tensorflow as tf

from data.synthetic_dataset_wt_conf_za_link import SyntheticDataset, confounder_monotonicities_1,get_subportion_confounders
import pandas as pd
from models_synthetic import Twin_Net_with_Z_A, dice_loss,Twin_Net,class_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import copy
from utils import pickle_object, read_pickle_object


def get_test_confs(dataset, args, N=None, mode='test'):
    if mode == 'test':
        if args.multiple_confounders:
            conf_to_input = [dataset.test[i].values.astype(np.float32) for i in dataset.test.columns]
        else:
            conf_to_input = [dataset.test.values.astype(np.float32)]

    elif mode=='sample':
        conf_to_input = [dataset.get_z_samples(N,args.p_2)]
    elif mode == 'dataset_median':
        if 'all' in args.confounders:
            args.confounders = dataset.train.columns
        conf_to_input = [np.tile(dataset.test[i].median(), (N)) for i in args.confounders]
        if not args.multiple_confounders:
            conf_to_input = np.array(conf_to_input).T

    return conf_to_input


def prob_nec(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args):
    preds = model.predict([treatment_factual, treatment_counter, uy_to_input,
                           conf_to_input],
                          args.batch_size, 1)
    pred_factual = preds[0]
    pred_counter = preds[1]

    scaler = MinMaxScaler()
    y = scaler.fit_transform(pred_factual)
    y_prime = scaler.fit_transform(pred_counter)

    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    y_prime[y_prime >= 0.5] = 1
    y_prime[y_prime < 0.5] = 0

    idx_given_y_1 = np.where(y == outcomes_factual)[0]
    idx_query_y_0 = np.where(y_prime == outcomes_counter)[0]

    idx_y_1_y_prime_0 = list(set(idx_given_y_1).intersection(idx_query_y_0))

    prob_necessity_1 = len(idx_y_1_y_prime_0) / len(idx_given_y_1)
    return prob_necessity_1


def prob_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args):
    preds = model.predict([treatment_factual, treatment_counter, uy_to_input,
                           conf_to_input],
                          args.batch_size, 1)
    pred_factual = preds[0]
    pred_counter = preds[1]

    scaler = MinMaxScaler()
    y = scaler.fit_transform(pred_factual)
    y_prime = scaler.fit_transform(pred_counter)

    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    y_prime[y_prime >= 0.5] = 1
    y_prime[y_prime < 0.5] = 0

    idx_given_y_0 = np.where(y == outcomes_factual)[0]
    idx_query_y_1 = np.where(y_prime == outcomes_counter)[0]

    idx_y_0_y_prime_1 = set(idx_given_y_0).intersection(idx_query_y_1)

    prob_suficiency = len(idx_y_0_y_prime_1) / len(idx_given_y_0)
    return prob_suficiency


def prob_nec_and_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args):
    preds = model.predict([treatment_factual, treatment_counter, uy_to_input,
                           conf_to_input],
                          args.batch_size, 1)
    pred_factual = preds[0]
    pred_counter = preds[1]

    scaler = MinMaxScaler()
    y = scaler.fit_transform(pred_factual)
    y_prime = scaler.fit_transform(pred_counter)

    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    y_prime[y_prime >= 0.5] = 1
    y_prime[y_prime < 0.5] = 0

    idx_given_y_0 = np.where(y == outcomes_factual)[0]
    idx_query_y_1 = np.where(y_prime == outcomes_counter)[0]
    idx_y_0_y_prime_1 = set(idx_given_y_0).intersection(idx_query_y_1)

    prob_nec_and_suficiency = len(idx_y_0_y_prime_1)/len(y)

    return prob_nec_and_suficiency


def ate(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args):
    preds = model.predict([treatment_factual, treatment_counter, uy_to_input,
                           conf_to_input],
                          args.batch_size, 1)
    pred_factual = preds[0]
    pred_counter = preds[1]

    scaler = MinMaxScaler()
    y = scaler.fit_transform(pred_factual)
    y_prime = scaler.fit_transform(pred_counter)

    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    y_prime[y_prime >= 0.5] = 1
    y_prime[y_prime < 0.5] = 0

    idx_given_y_0 = np.sum(y == 1)/ len(y)
    idx_query_y_1 = np.sum(y_prime == 1)/ len(y_prime)

    ate_ = idx_query_y_1 - idx_given_y_0
    return ate_



def calc_ate(dataset):

    a = dataset.index[(dataset['Y'] == 1) & (dataset['T'] == 0)].values
    ab = dataset.index[(dataset['Y_prime'] == 1) & (dataset['T_prime'] == 0)].values
    lighter = np.hstack((a, ab))

    a = dataset.index[(dataset['Y'] == 1) & (dataset['T'] == 1)].values
    ab = dataset.index[(dataset['Y_prime'] == 1) & (dataset['T_prime'] == 1)].values
    heavier = np.hstack((a, ab))

    prob_lighter = len(lighter)/len(dataset)
    prob_heavier = len(heavier)/len(dataset)

    ate = prob_heavier - prob_lighter

    return ate


def run_inference(args):
    dataset = SyntheticDataset(**vars(args))
    dataset.train = pd.DataFrame.from_dict(dataset.train)
    dataset.test = pd.DataFrame.from_dict(dataset.test)

    try:
        dataset.train.drop(['propensity_score', 'matched_element', 'propensity_score_logit'], axis=1, inplace=True)
        dataset.test.drop(['propensity_score', 'matched_element', 'propensity_score_logit'], axis=1, inplace=True)
    except:
        print('Not propensity')

    target = dataset.train.pop('Y')
    target_prime = dataset.train.pop('Y_prime')
    treatment = dataset.train.pop('X')
    uy = dataset.train.pop('Uy')

    dataset.test = dataset.test.reset_index().drop(['index'], axis=1)

    target_test = dataset.test.pop('Y')
    target_prime_test = dataset.test.pop('Y_prime')
    treatment_test = dataset.test.pop('X')
    treatment_prime_test = dataset.test.pop('X_prime')
    uy_test = dataset.test.pop('Uy')

    # Get confounders
    dataset.train = get_subportion_confounders(dataset.train, args.confounders)
    dataset.test = get_subportion_confounders(dataset.test, args.confounders)

    args.len_conf = len(dataset.train.columns)

    if args.multiple_confounders:
        args.z_monotonicity = []

        for i, col in enumerate(dataset.train.columns):
            args.z_monotonicity.append(args.z_monotonicity_base[col])
            args.lattice_sizes.append(args.z_calib_units)
        input_len = args.len_conf + 2
    else:
        input_len = 3

        args.z_monotonicity = [args.z_monotonicity]
        args.lattice_sizes.append(2)
    if 'za' in args.runPath:
        model = Twin_Net_with_Z_A(treatment, uy, dataset.train, args)
    else:
        model = Twin_Net(treatment, uy, dataset.train, args)

    # Set up loss
    if 'mse' in args.loss:
        loss_func = tf.keras.losses.mean_squared_error
    elif 'mae' in args.loss:
        loss_func = tf.keras.losses.mean_absolute_error
    elif 'bce' in args.loss:
        loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    elif 'dice' in args.loss:
        loss_func = dice_loss
    elif 'weighted_loss':
        negs = np.sum(target == 0)
        negs += np.sum(target_prime == 0)
        pos = np.sum(target == 1)
        pos += np.sum(target_prime == 1)

        weight_for_0 = (1 / negs) * (len(dataset.train) * 2) / 2.0
        weight_for_1 = (1 / pos) * (len(dataset.train) * 2) / 2.0
        loss_func = class_loss(np.array([weight_for_0, weight_for_1]))

    if args.weighted_loss:
        model.compile(
            loss=[loss_func, loss_func],
            loss_weights=[args.weight_1, args.weight_2],
            )
    else:
        model.compile(
            loss=loss_func)
    print('-------------------------Experiment: {} ---------------------'.format(args.inference_name))
    model.build((1, input_len))
    model.load_weights(args.runPath + '/best')
    conf_to_input = [dataset.test.values.astype(np.float32)]
    if args.multiple_confounders:
        conf_to_input = [dataset.test[i].values.astype(np.float32) for i in dataset.test.columns]

    test_loss = model.evaluate(
        [treatment_test.values.astype(np.float32), treatment_prime_test.values.astype(np.float32),
         uy_test.values.astype(np.float32),
         conf_to_input],
        [target_test[..., np.newaxis], target_prime_test[..., np.newaxis]])
    print('Test Loss : {}'.format(test_loss))

    preds = model.predict([treatment_test.values.astype(np.float32), treatment_prime_test.values.astype(np.float32),
                           uy_test.values.astype(np.float32), conf_to_input],
                          args.batch_size, 1)
    title = ['Factual', 'Counterfactual']
    preds = preds[0:2]

    for i, pred in enumerate(preds):

        scaler = MinMaxScaler()
        pred = scaler.fit_transform(pred)

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        ac = accuracy_score(pred, target_test)
        print('{} Acc: {}'.format(title[i], ac))
        f1 = f1_score(target_test.values, pred)
        print('{} F1 : {}'.format(title[i], f1))


    if not args.train:
        prob_necessities,prob_sufficiencies, prob_both , ate_, gt = calc_probs(args, treatment_test, dataset, model)
        return prob_necessities,prob_sufficiencies, prob_both, ate_, gt
    else:
        calc_probs(args, treatment_test, dataset, model)


def calc_probs(args, treatment_test, dataset, model):
    N = len(treatment_test)

    treatment_factual = np.ones(N)
    treatment_counter = np.zeros(N)

    outcomes_factual = 1
    outcomes_counter = 0

    gt = None
    if args.u_distribution == 'p_test':
        conf_to_input = get_test_confs(dataset, args, mode='test')

        uy_samples, gt = dataset.get_uy_samples(N,args.p,conf_to_input[0])
    else:
        uy_samples = dataset.get_uy_samples(N)

    uy_to_input = uy_samples

    conf_to_input = get_test_confs(dataset, args, N ,mode='test')

    prob_necessity_1 = prob_nec(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input,
                                outcomes_factual, outcomes_counter, args)

    print('Test Probability of Necessity : {}'.format(prob_necessity_1))



    # ##################### Prob of Sufficiency #####################

    N = len(treatment_test)

    treatment_factual = np.zeros(N)
    treatment_counter = np.ones(N)

    outcomes_factual = 0
    outcomes_counter = 1

    conf_to_input = get_test_confs(dataset, args, mode='test')

    prob_suf_1 = prob_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args)

    print('Test Probability of Sufficiency : {}'.format(prob_suf_1))


    # ############ PROB of Nec and Suf ##################
    N = len(treatment_test)

    treatment_factual = np.zeros(N)
    treatment_counter = np.ones(N)

    outcomes_factual = 0
    outcomes_counter = 1

    conf_to_input = get_test_confs(dataset, args, mode='test')

    prob_nec_suf_1 = prob_nec_and_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
                          outcomes_counter, args)

    print('Test Probability of Necessity and Sufficiency : {}'.format(prob_nec_suf_1))


    conf_to_input = get_test_confs(dataset, args, mode='test')

    N = len(conf_to_input[0])
    if args.u_distribution == 'p_test':
        conf_to_input = get_test_confs(dataset, args, mode='test')

        uy_samples, _ = dataset.get_uy_samples(N, args.p, conf_to_input[0])
    else:
        uy_samples = dataset.get_uy_samples(N)

    uy_to_input = uy_samples
    treatment_factual = np.zeros(N)
    treatment_counter = np.ones(N)

    outcomes_factual = 0
    outcomes_counter = 1


    ate_ = ate(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input,
                                      outcomes_factual,
                                      outcomes_counter, args)
    print('Test ATE : {}'.format(ate_))

    if not args.train:
        return (prob_necessity_1), (prob_suf_1), (prob_nec_suf_1), ate_, gt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--inference_name',
                        default='twin_net_arch_za_lattice_none_uy_monotonicity_none_z_monoton_opt_1_z_layer_multiple_calib_units_3_3_z_3_lr_0_001_loss_mse_Synthetics_with_2_normal_uniform_bernouli_0_05_confounders_1_confs')
    parser.add_argument('--prob_type', default='paper')
    # Logging
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--log_root', type=str, default='./experiments/Synthetic_wt_conf/')

    # Dataset Hparams
    parser.add_argument('--save_path', default='./data/Datasets/')
    parser.add_argument('--save_name', default='kenyan_water_proc.pkl')
    parser.add_argument('--save_dataset', default=False)
    parser.add_argument('--load_dataset', default=True)
    parser.add_argument('--dataset_mode', type=str, default='synthetic')
    parser.add_argument('--path_to_data',

                        default = '../data/Datasets/synthetic_dataset_200000_samples_X_{}_Uy_{}_Z_{}_with_counterfactual_and_z_a_link_2_final.pkl')

    parser.add_argument('--confounders',default=['Z'])

    parser.add_argument('--u_distribution', default='normal')
    parser.add_argument('--z_distribution', default='uniform')
    parser.add_argument('--x_distribution', default='bernouli')
    parser.add_argument('--p_1', type=float, default=0.05)
    parser.add_argument('--p_2', type=float, default=0.7)
    parser.add_argument('--p_3', type=float, default=0.2)

    # Model Hparams
    parser.add_argument('--lattice_sizes', default=[3, 3])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lattice_units', type=int, default=1)  # 1 or 2
    parser.add_argument('--hidden_dims', type=int, default=3)
    parser.add_argument('--calib_units', type=int, default=3)
    parser.add_argument('--z_calib_units', type=int, default=3)
    parser.add_argument('--layer', default='lattice')
    parser.add_argument('--uy_layer', default='none')
    parser.add_argument('--z_layer', default='none')
    parser.add_argument('--uy_monotonicity', default='none')
    parser.add_argument('--z_monotonicity', default='none')
    parser.add_argument('--z_monot_opt', default=1)
    parser.add_argument('--concats', type=bool, default=False)

    parser.add_argument('--end_activation', default='none')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--weighted_loss',default=False)
    parser.add_argument('--weight_1',type=float,default=1)
    parser.add_argument('--weight_2',type=float,default=1)
    parser.add_argument('--multiple_confounders', default=True, help='split confounders')

    # General
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()

    # GPU setup
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set Randomness
    # Set Randomness
    if args.seed == 0: args.seed = int(np.random.randint(0, 2 ** 32 - 1, (1,)))
    print('seed', args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set logdirs

    # z_dist = '{}_{}_{}'.format(args.z_distribution, args.p_1,
    #                            args.p_2) if args.z_distribution == 'bernouli' else '{}'.format(
    #     args.z_distribution)

    z_dist = '{}_{}'.format(args.z_distribution, args.p_2,
                            args.p_2) if args.z_distribution == 'bernouli' else '{}'.format(
        args.z_distribution)
    x_dist = '{}_{}'.format(args.x_distribution, args.p_1,
                            args.p_1) if args.x_distribution == 'bernouli' else '{}'.format(
        args.x_distribution)

    args.path_to_data = args.load_path = args.path_to_data.format(x_dist, args.u_distribution, z_dist)

    args.runPath = os.path.join(args.log_root, args.inference_name)

    if args.multiple_confounders:
        # args.z_calib_units = 2
        # args.z_calib_units = args.lattice_sizes[0]

        z_monotonicity = 'opt_{}'.format(args.z_monot_opt)
        args.z_monotonicity_base = eval('confounder_monotonicities_{}'.format(args.z_monot_opt))
    else:
        z_monotonicity = args.z_monotonicity
        args.z_calib_units = len(args.confounders)+1 if 'all' not in args.confounders else 2

    if not args.train:
        p_test = False
        if not p_test:
            nec_test_prob = []
            nec_dataset_prob = []
            suf_test_prob = []
            suf_dataset_prob = []
            nec_and_suf_test_prob = []
            nec_and_suf_dataset_prob = []
            ates =[]
            for i in range(1, 20):
                prob_necessities,prob_sufficiencies, prob_both,ate_ = run_inference(copy.deepcopy(args))
                nec_test_prob.append(prob_necessities)

                suf_test_prob.append(prob_sufficiencies)

                nec_and_suf_test_prob.append(prob_both)
                ates.append(ate_)


            print('\n \nTest average Prob of Necessity {}, std: {}'.format(np.array(nec_test_prob).mean(), np.array(nec_test_prob).std()))

            print('Test average Prob of Sufficiency {}, std: {}'.format(np.array(suf_test_prob).mean(),
                                                                      np.array(suf_test_prob).std()))

            print('Test average Prob of Necessity and Sufficiency {}, std: {}'.format(np.array(nec_and_suf_test_prob).mean(),
                                                                        np.array(nec_and_suf_test_prob).std()))

            print('Test ATE mean {}, std: {} \n \n'.format(
                np.array(ates).mean(),
                np.array(ates).std()))
        else:
            args.u_distribution = 'p_test'

            probs = np.zeros((20, 10, 3))

            gts = np.zeros((20, 10, 3))
            for i in range(0, 20):
                for j, p in enumerate(np.linspace(0.05, 0.95, 10)):
                    args.p = p
                    n, s, ns,ate_, gt = run_inference(copy.deepcopy(args))
                    probs[i, j, 0] = n
                    probs[i, j, 1] = s
                    probs[i, j, 2] = ns
                    gts[i, j, 0] = gt[0]
                    gts[i, j, 1] = gt[1]
                    gts[i, j, 2] = gt[2]

            pn = np.array(probs[:, :, 0])
            ps = np.array(probs[:, :, 1])
            pns = np.array(probs[:, :, 2])
            print('\n Prob of Necessity {} ; std: {}'.format(pn.mean(), pn.std()))
            print('Prob of Sufficiency {} ; std: {}'.format(ps.mean(), ps.std()))
            print('Prob of Necessity & Sufficiency  {} ; std: {}'.format(pns.mean(), pns.std()))

            to_save = {
                'pn': pn,
                'ps': ps,
                'pns': pns,
                'gt_n': gts[:, :, 0],
                'gt_s': gts[:, :, 0],
                'gt_ns': gts[:, :, 0]
            }
            pickle_object(to_save, './experiments/p_test_conf.npz')
    else:
        run_inference(args)


