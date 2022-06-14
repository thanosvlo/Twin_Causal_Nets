import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score
import tensorflow as tf
from data.kenyan_water_dataset import KenyanWaterDataset, get_subportion_confounders
from data.kenyan_water_dataset import median_child_paper, median_child
from models_kenyan import Twin_Net_Kenyan, Twin_Net_Kenyan_with_Z_A
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import copy


def get_test_confs(dataset, args, treatment_factual=None, mode='test'):
    if mode == 'test':
        if args.multiple_confounders:
            conf_to_input = [dataset.test[i].values.astype(np.float32) for i in dataset.test.columns]
        else:
            conf_to_input = [dataset.test.values.astype(np.float32)]

    elif mode == 'paper_median':
        if 'all' in args.confounders:
            args.confounders = median_child_paper.keys()
        conf_to_input = [np.tile(median_child_paper[i], (len(treatment_factual))) for i in args.confounders]
        if not args.multiple_confounders:
            conf_to_input = np.array(conf_to_input).T

    elif mode == 'dataset_median':
        if 'all' in args.confounders:
            args.confounders = median_child.keys()
        conf_to_input = [np.tile(median_child[i], (len(treatment_factual))) for i in args.confounders]
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

    idx_y_1_y_prime_0 = set(idx_given_y_1).intersection(idx_query_y_0)

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

    # idx_given_y_0 = np.sum(y == outcomes_factual) / len(y)
    # idx_query_y_1 = np.sum(y_prime == outcomes_counter)/ len(y_prime)

    idx_given_y_0 = np.where(y == outcomes_factual)[0]
    idx_query_y_1 = np.where(y_prime == outcomes_counter)[0]
    idx_y_0_y_prime_1 = set(idx_given_y_0).intersection(idx_query_y_1)

    prob_nec_and_suficiency = len(idx_y_0_y_prime_1)/len(y)

    # prob_nec_and_suficiency = np.abs(idx_given_y_0 - idx_query_y_1)
    return prob_nec_and_suficiency


def run_inference(args):
    dataset = KenyanWaterDataset(**vars(args))

    target = dataset.train.pop('targets')
    target_prime = dataset.train.pop('targets_prime')
    treatment = dataset.train.pop('treatment')
    treatment_prime = dataset.train.pop('treatment_prime')
    uy = dataset.train.pop('uy')

    target_test = dataset.test.pop('targets')
    target_prime_test = dataset.test.pop('targets_prime')
    treatment_test = dataset.test.pop('treatment')
    treatment_prime_test = dataset.test.pop('treatment_prime')
    uy_test = dataset.test.pop('uy')

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
        args.lattice_sizes.append(args.len_conf)
    if 'az' in args.runPath and args.layer != 'linear':
        model = Twin_Net_Kenyan_with_Z_A(treatment, uy, dataset.train, args)
    else:
        model = Twin_Net_Kenyan(treatment, uy, dataset.train, args)

    # Set up loss
    if 'mse' in args.loss:
        loss_func = tf.keras.losses.mean_squared_error
    elif 'mae' in args.loss:
        loss_func = tf.keras.losses.mean_absolute_error
    elif 'bce' in args.loss:
        loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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

        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        ac = accuracy_score(pred, target_test)
        print('{} Acc: {}'.format(title[i], ac))
        f1 = f1_score(target_test, pred)
        print('{} F1 : {}'.format(title[i], f1))
    if not args.train:
        prob_necessities,prob_sufficiencies, prob_both  = calc_probs(args, treatment_test, dataset, model)
        return prob_necessities,prob_sufficiencies, prob_both
    else:
        calc_probs(args, treatment_test, dataset, model)


def calc_probs(args, treatment_test, dataset, model):
    N = len(treatment_test)
    # N = 10000
    uy_samples = dataset.get_uy_samples(N)

    treatment_factual = np.ones(N)
    treatment_counter = np.zeros(N)

    outcomes_factual = 1
    outcomes_counter = 0

    uy_to_input = uy_samples

    conf_to_input = get_test_confs(dataset, args, mode='test')

    prob_necessity_1 = prob_nec(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input,
                                outcomes_factual, outcomes_counter, args)

    print('Test Probability of Necessity : {}'.format(prob_necessity_1))

    conf_to_input = get_test_confs(dataset, args, treatment_factual=treatment_factual, mode='paper_median')

    prob_necessity_2 = prob_nec(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input,
                                outcomes_factual, outcomes_counter, args)

    print('Median Child Paper Test Probability of Necessity : {}'.format(prob_necessity_2))

    conf_to_input = get_test_confs(dataset, args, treatment_factual=treatment_factual, mode='dataset_median')


    prob_necessity_3 = prob_nec(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input,
                                outcomes_factual, outcomes_counter, args)
    print('Median Child Dataset Test Probability of Necessity : {}'.format(prob_necessity_3))

    # ##################### Prob of Sufficiency #####################

    N = len(treatment_test)
    # N = 10000
    # uy_samples = dataset.get_uy_samples(N)
    # uy_to_input = uy_samples

    treatment_factual = np.zeros(N)
    treatment_counter = np.ones(N)

    outcomes_factual = 0
    outcomes_counter = 1

    conf_to_input = get_test_confs(dataset, args, mode='test')

    prob_suf_1 = prob_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args)

    print('Test Probability of Sufficiency : {}'.format(prob_suf_1))

    # N = 10000
    # uy_samples = dataset.get_uy_samples(N)
    # uy_to_input = uy_samples

    treatment_factual = np.zeros(N)
    treatment_counter = np.ones(N)

    conf_to_input = get_test_confs(dataset, args, treatment_factual=treatment_factual, mode='paper_median')

    prob_suf_2 = prob_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
                          outcomes_counter, args)

    print('Median Child Paper Test Probability of Sufficiency : {}'.format(prob_suf_2))

    conf_to_input = get_test_confs(dataset, args, treatment_factual=treatment_factual, mode='dataset_median')

    prob_suf_3 = prob_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
                          outcomes_counter, args)
    print('Median Child Dataset Test Probability of Sufficiency : {}'.format(prob_suf_3))

    # ############ PROB of Nec and Suf ##################
    N = len(treatment_test)
    # N = 10000
    # uy_samples = dataset.get_uy_samples(N)
    # uy_to_input = uy_samples

    treatment_factual = np.zeros(N)
    treatment_counter = np.ones(N)

    outcomes_factual = 0
    outcomes_counter = 1

    conf_to_input = get_test_confs(dataset, args, mode='test')

    prob_nec_suf_1 = prob_nec_and_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
                          outcomes_counter, args)

    print('Test Probability of Necessity and Sufficiency : {}'.format(prob_nec_suf_1))

    N = 10000
    uy_samples = dataset.get_uy_samples(N)
    uy_to_input = uy_samples

    treatment_factual = np.zeros(N)
    treatment_counter = np.ones(N)

    conf_to_input = get_test_confs(dataset, args, treatment_factual=treatment_factual, mode='paper_median')

    prob_nec_suf_2 = prob_nec_and_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
                          outcomes_counter, args)

    print('Median Child Paper Test Probability of Necessity and Sufficiency : {}'.format(prob_nec_suf_2))

    conf_to_input = get_test_confs(dataset, args, treatment_factual=treatment_factual, mode='dataset_median')

    prob_nec_suf_3 = prob_nec_and_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
                          outcomes_counter, args)
    print('Median Child Dataset Test Probability of Necessity and Sufficiency : {}'.format(prob_nec_suf_3))

    if not args.train:
        return (prob_necessity_1, prob_necessity_2, prob_necessity_3), (prob_suf_1,prob_suf_2, prob_suf_3), (prob_nec_suf_1,prob_nec_suf_2, prob_nec_suf_3)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--inference_name',
                        default='twin_net_arch_lattice_azlink_none_uy_none_uy_monotonicity_none_z_monoton_opt_2_z_layer_multiple_none_calib_units_4_4_z_4_lr_0_001_loss_mse_weighted_kenyan_confounders_3_confs')
    parser.add_argument('--prob_type', default='paper')
    # Logging
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--log_root', type=str, default='./experiments/KenyanWater/Twin')

    # Dataset Hparams
    parser.add_argument('--save_path', default='./data/Datasets/')
    parser.add_argument('--save_name', default='kenyan_water_proc.pkl')
    parser.add_argument('--save_dataset', default=False)
    parser.add_argument('--load_dataset', default=True)
    parser.add_argument('--dataset_mode', type=str, default='synthetic')
    parser.add_argument('--path_to_data',
                        default='../data/Datasets/kenyan_water_proc_single_uy_normal_with_propensity.pkl')
    # default='./data/Datasets/synthetic_dataset_200000_samples_X_bernouli_Uy_normal_with_counterfactual.pkl')
    # parser.add_argument('--confounders', default=['base_age','splnecmpn_base','latrine_density_base','numkids_base'])
    # parser.add_argument('--confounders', default=['base_age','splnecmpn_base','e1_iron_roof_base','latrine_density_base','numkids_base'])
    # parser.add_argument('--confounders', default='all')
    parser.add_argument('--confounders', default=['base_age', 'splnecmpn_base', 'latrine_density_base'])
    # parser.add_argument('--confounders', default=['splnecmpn_base','latrine_density_base'])

    parser.add_argument('--u_distribution', default='normal')
    # Model Hparams
    parser.add_argument('--lattice_sizes', default=[4, 4])
    parser.add_argument('--lattice_monotonicities', default=['increasing', 'increasing'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lattice_units', type=int, default=1)  # 1 or 2
    parser.add_argument('--hidden_dims', type=int, default=4)
    parser.add_argument('--calib_units', type=int, default=4)
    parser.add_argument('--z_calib_units', type=int, default=4)
    parser.add_argument('--layer', default='lattice')
    parser.add_argument('--uy_layer', default='none')
    parser.add_argument('--z_layer', default='none')
    parser.add_argument('--uy_monotonicity', default='none')
    parser.add_argument('--z_monotonicity', default='none')
    parser.add_argument('--z_monot_opt', default=2)
    parser.add_argument('--concats', type=bool, default=False)

    parser.add_argument('--end_activation', default='none')
    parser.add_argument('--loss', default='mse')
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

    args.path_to_data = args.load_path = args.path_to_data.format(args.u_distribution)

    args.runPath = os.path.join(args.log_root, args.inference_name)

    if args.multiple_confounders:
        # args.z_calib_units = 2
        # args.z_calib_units = args.lattice_sizes[0]

        z_monotonicity = 'opt_{}'.format(args.z_monot_opt)
        args.z_monotonicity_base = eval('confounder_monotonicities_{}'.format(args.z_monot_opt))
    else:
        z_monotonicity = args.z_monotonicity
        args.z_calib_units = len(args.confounders) if 'all' not in args.confounders else 8

    if not args.train:

        nec_test_prob = []
        nec_median_paper_prob = []
        nec_dataset_prob = []
        suf_test_prob = []
        suf_median_paper_prob = []
        suf_dataset_prob = []
        nec_and_suf_test_prob = []
        nec_and_suf_median_paper_prob = []
        nec_and_suf_dataset_prob = []
        for i in range(1, 20):
            prob_necessities,prob_sufficiencies, prob_both = run_inference(copy.deepcopy(args))
            nec_test_prob.append(prob_necessities[0])
            nec_median_paper_prob.append(prob_necessities[1])
            nec_dataset_prob.append(prob_necessities[2])

            suf_test_prob.append(prob_sufficiencies[0])
            suf_median_paper_prob.append(prob_sufficiencies[1])
            suf_dataset_prob.append(prob_sufficiencies[2])

            nec_and_suf_test_prob.append(prob_both[0])
            nec_and_suf_median_paper_prob.append(prob_both[1])
            nec_and_suf_dataset_prob.append(prob_both[2])


        print('\n \nTest average Prob of Necessity {}, std: {}'.format(np.array(nec_test_prob).mean(), np.array(nec_test_prob).std()))
        print('Paper Median average Prob of Necessity {}, std: {}'.format(np.array(nec_median_paper_prob).mean(),
                                                                    np.array(nec_median_paper_prob).std()))
        print('Dataset median average Prob of Necessity {}, std: {}'.format(np.array(nec_dataset_prob).mean(),
                                                                      np.array(nec_dataset_prob).std()))

        print('Test average Prob of Sufficiency {}, std: {}'.format(np.array(suf_test_prob).mean(),
                                                                  np.array(suf_test_prob).std()))
        print('Paper Median average Prob of Sufficiency {}, std: {}'.format(np.array(suf_median_paper_prob).mean(),
                                                                          np.array(suf_median_paper_prob).std()))
        print('Dataset median average Prob of Sufficiency {}, std: {}'.format(np.array(suf_dataset_prob).mean(),
                                                                            np.array(suf_dataset_prob).std()))

        print('Test average Prob of Necessity and Sufficiency {}, std: {}'.format(np.array(nec_and_suf_test_prob).mean(),
                                                                    np.array(nec_and_suf_test_prob).std()))
        print('Paper Median average Prob of Necessity and Sufficiency {}, std: {}'.format(np.array(nec_and_suf_median_paper_prob).mean(),
                                                                            np.array(nec_and_suf_median_paper_prob).std()))
        print('Dataset median average Prob of Necessity and Sufficiency {}, std: {} \n \n'.format(np.array(nec_and_suf_dataset_prob).mean(),
                                                                              np.array(nec_and_suf_dataset_prob).std()))
    else:
        run_inference(args)

    # raise NotImplementedError
