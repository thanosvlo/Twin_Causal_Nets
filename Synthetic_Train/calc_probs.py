import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score
import tensorflow as tf
from data.synthetic_dataset import SyntheticDataset
import pandas as pd
from models import TwinNet, custom_loss
from sklearn.preprocessing import MinMaxScaler


def run_inference(args):
    dataset = SyntheticDataset(**vars(args))
    train_set = pd.DataFrame.from_dict(dataset.train)
    test_set = pd.DataFrame.from_dict(dataset.test)

    target_test = test_set.pop('Y')
    target_prime_test = test_set.pop('Y_prime')

    model = TwinNet(train_set, args)

    # Set up loss
    if args.loss == 'mse':
        loss_func = tf.keras.losses.mean_squared_error
    elif args.loss == 'mae':
        loss_func = tf.keras.losses.mean_absolute_error
    elif args.loss == 'bce':
        loss_func = tf.keras.losses.binary_crossentropy
    elif 'custom' in args.loss:
        loss_func = custom_loss

    model.compile(
        loss=loss_func)

    model.build((1, 3))
    model.load_weights(args.runPath + '/best')

    if 'single' in args.runPath:
        titles = ['Factual', 'Counterfactual']
        for i in range(0, 3, 2):
            if i == 0:
                y_test = target_test
                title = titles[0]
            else:
                title = titles[1]
                y_test = target_prime_test
            test_loss = model.evaluate(
                [test_set.values.astype(np.float32)[:, i], test_set.values.astype(np.float32)[:, 1]],
                [y_test[..., np.newaxis]])
            print('{} test Loss : {}'.format(title, test_loss))

            pred = model.predict([test_set.values.astype(np.float32)[:, i], test_set.values.astype(np.float32)[:, 1]],
                                 args.batch_size, 1)
            pred = (pred - np.min(pred)) / np.ptp(pred)
            pred[pred > 0.5] = 1
            pred[pred < 0.5] = 0
            ac = accuracy_score(pred, target_test)
            print('{} Acc : {}'.format(title, ac))
    else:
        test_loss = model.evaluate([test_set.values.astype(np.float32)[:, 0], test_set.values.astype(np.float32)[:, 1],
                                    test_set.values.astype(np.float32)[:, 2]],
                                   [target_test.values[..., np.newaxis], target_prime_test.values[..., np.newaxis]])
        print('Test Loss : {}'.format(test_loss))
        preds = model.predict([test_set.values.astype(np.float32)[:, 0], test_set.values.astype(np.float32)[:, 1],
                               test_set.values.astype(np.float32)[:, 2]],
                              args.batch_size, 1)
        title = ['Factual Acc', 'Counterfactual Acc']
        for i, pred in enumerate(preds):
            pred = (pred - np.min(pred)) / np.ptp(pred)
            pred[pred > 0.5] = 1
            pred[pred < 0.5] = 0
            ac = accuracy_score(pred, target_test)
            print('{} : {}'.format(title[i], ac))
    gt = None
    if args.u_distribution=='p_test':
        uy_samples, gt = dataset.get_uy_samples(args.p)
    else:
        uy_samples = dataset.get_uy_samples()

    if 'both' in args.prob_type:
        assert len(args.cond_xs) == len(args.query_ys)

        # Xs = []
        # for cond_x in args.cond_Xs:
        Xs = (np.zeros((args.n_samples, 1)), np.ones((args.n_samples, 1)))
        Ys = []
        for query_y in args.query_ys:
            Ys.append(np.ones((args.n_samples, 1)) * query_y)


        preds = model.predict([Xs[0].astype(np.float32), uy_samples.astype(np.float32), Xs[1].astype(np.float32)],
                              args.batch_size, 1)

        scaler = MinMaxScaler()
        y = scaler.fit_transform(preds[0])
        y_prime = scaler.fit_transform(preds[1])

        y[y >= 0.5] = 1
        y[y < 0.5] = 0

        y_prime[y_prime >= 0.5] = 1
        y_prime[y_prime < 0.5] = 0

        p_x_0 = np.sum(y == 0) / len(y)
        p_x_1 = np.sum(y_prime == 0) / len(y_prime)
        pns = p_x_0 - p_x_1
        print('Probability of Necessity and Sufficiency : {}'.format(pns))
    if 'necessity' in args.prob_type:
        given_x = np.ones((args.n_samples, 1))
        query_x = np.zeros((args.n_samples, 1))

        given_y = 1
        query_y = 0

        preds = model.predict(
            [given_x.astype(np.float32), uy_samples.astype(np.float32), query_x.astype(np.float32)],
            args.batch_size, 1)

        scaler = MinMaxScaler()
        y = scaler.fit_transform(preds[0])
        y_prime = scaler.fit_transform(preds[1])

        y[y >= 0.5] = 1
        y[y < 0.5] = 0

        y_prime[y_prime >= 0.5] = 1
        y_prime[y_prime < 0.5] = 0

        idx_given_y_1 = np.where(y == given_y)[0]
        idx_query_y_0 = np.where(y_prime == query_y)[0]

        idx_y_1_y_prime_0 = set(idx_given_y_1).intersection(idx_query_y_0)

        prob_necessity = len(idx_y_1_y_prime_0) / len(idx_given_y_1)

        print('Probability of Necessity : {}'.format(prob_necessity))

    if 'sufficiency' in args.prob_type:
        given_x = np.zeros((args.n_samples, 1))
        query_x = np.ones((args.n_samples, 1))

        given_y = 0
        query_y = 1


        preds = model.predict(
            [given_x.astype(np.float32), uy_samples.astype(np.float32), query_x.astype(np.float32)],
            args.batch_size, 1)

        scaler = MinMaxScaler()
        y = scaler.fit_transform(preds[0])
        y_prime = scaler.fit_transform(preds[1])

        y[y >= 0.5] = 1
        y[y < 0.5] = 0

        y_prime[y_prime >= 0.5] = 1
        y_prime[y_prime < 0.5] = 0

        idx_given_y_0 = np.where(y == given_y)[0]
        idx_query_y_1 = np.where(y_prime == query_y)[0]

        idx_y_0_y_prime_1 = set(idx_given_y_0).intersection(idx_query_y_1)

        prob_sufficiency = len(idx_y_0_y_prime_1) / len(idx_given_y_0)

        print('Probability of Sufficiency : {}'.format(prob_sufficiency))

    return prob_necessity, prob_sufficiency, pns , gt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_name',
                        default='twin_net_arch_lattice_uy_none_uy_monotonicity_none_end_act_none_calib_units_2_lr_0.001_loss_mse_uy_uniform')
    # default='twin_net_arch_lattice_uy_none_uy_monotonicity_none_end_act_none_calib_units_1_lr_0.001_loss_mse_uy_normal')

    # parser.add_argument('--prob_type', default='sufficiency', help='both, necessity, sufficiency')
    parser.add_argument('--prob_type', default=['both', 'necessity', 'sufficiency'],
                        help='both, necessity, sufficiency')
    parser.add_argument('--n_samples', default=10000)
    parser.add_argument('--query_ys', default=[0, 0])
    parser.add_argument('--cond_xs', default=[0, 1])

    # Logging
    parser.add_argument('--restore', type=bool, default=False)
    # parser.add_argument('--log_root', type=str, default='./Synthetic_Train/experiments')
    parser.add_argument('--log_root', type=str, default='../experiments')
    parser.add_argument('--name', type=str,
                        default='twin_net_arch_{}_uy_{}_uy_monotonicity_{}_end_act_{}_calib_units_{}_lr_{}_loss_{}_uy_{}')
    # Dataset Hparams
    parser.add_argument('--save_path', default='./data/Datasets/')
    parser.add_argument('--save_name', default='synthetic_dataset_{}_samples_X_{}_Uy_{}_TEST.pkl')
    parser.add_argument('--save_dataset', default=False)
    parser.add_argument('--load_dataset', default=True)
    parser.add_argument('--dataset_mode', type=str, default='synthetic')
    parser.add_argument('--path_to_data',
                        default='../data/Datasets/synthetic_dataset_200000_samples_X_bernouli_Uy_{}_with_counterfactual.pkl')
    parser.add_argument('--u_distribution', default='uniform', help='normal, uniform')

    # Model Hparams
    parser.add_argument('--lattice_sizes', default=[2, 3, 2, 2, 3])
    parser.add_argument('--lattice_monotonicities', default=['increasing', 'increasing'])
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--hidden_dims', default=1)
    parser.add_argument('--calib_units', default=2)
    parser.add_argument('--layer', default='lattice')
    parser.add_argument('--uy_layer', default='none')
    parser.add_argument('--uy_monotonicity', default='none')
    parser.add_argument('--end_activation', default='none')
    parser.add_argument('--loss', default='mse')

    # General
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()

    # GPU setup
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set Randomness
    if args.seed == 0: args.seed = int(np.random.randint(0, 2 ** 32 - 1, (1,)))
    print('seed', args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set logdirs
    args.name = args.name.format(args.layer, args.uy_layer, args.uy_monotonicity, args.end_activation, args.calib_units,
                                 args.lr, args.loss, args.u_distribution)
    if args.u_distribution == 'p_test':
        args.path_to_data = args.path_to_data.format('normal')
    else:
        args.path_to_data = args.path_to_data.format(args.u_distribution)

    if 'single' in args.inference_name:
        args.log_root = os.path.join(args.log_root, 'SingleNet')

    # if args.inference_name is not None:
    args.runPath = os.path.join(args.log_root, args.inference_name)
    # else:
    #     args.runPath = os.path.join(args.log_root, args.name)
    probs = np.zeros((20, 1, 3))

    gts = np.zeros((20, 10, 3))
    for i in range(0, 20):
        # for j, p in enumerate(np.linspace(0.05, 0.95, 10)):
        for j, p in [0]:
            args.p = p
            n, s, ns, gt = run_inference(args)
            probs[i, j, 0] = n
            probs[i, j, 1] = s
            probs[i, j, 2] = ns
            # gts[i, j, 0] = gt[0]
            # gts[i, j, 1] = gt[1]
            # gts[i, j, 2] = gt[2]

    pn = np.array(  probs[:, :, 0])
    ps = np.array(  probs[:, :, 1])
    pns = np.array(  probs[:, :, 2])
    print('\n Prob of Necessity {} ; std: {}'.format(pn.mean(), pn.std()))
    print('Prob of Sufficiency {} ; std: {}'.format(ps.mean(), ps.std()))
    print('Prob of Necessity & Sufficiency  {} ; std: {}'.format(pns.mean(), pns.std()))

    # to_save ={
    #     'pn': pn,
    #     'ps':ps,
    #     'pns':pns,
    #     'gt_n':gts[:,:,0],
    #     'gt_s':gts[:,:,0],
    #     'gt_ns':gts[:,:,0]
    # }
    # pickle_object(to_save,'./experiments/p_test.npz')
    # raise NotImplementedError
