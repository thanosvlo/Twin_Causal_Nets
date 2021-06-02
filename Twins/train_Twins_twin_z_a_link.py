import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score
import tensorflow as tf
from data.twin_dataset import TwinsDataset, get_subportion_confounders
from data.twin_data_metadata import confounder_monotonicities_1, confounder_monotonicities_2
import pandas as pd
from models_twins import Twin_Net_with_Z_A, dice_loss,Twin_Net
from sklearn.preprocessing import MinMaxScaler
from utils import pickle_object, read_pickle_object
import sklearn
from calc_prob_twin_Twins import calc_probs
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE,ADASYN

def run_train(args):

    dataset = TwinsDataset(**vars(args))
    if 'prop' in args.runPath:
        try:
            dataset.train.drop(['propensity_score'], axis=1, inplace=True)
            dataset.test.drop(['propensity_score'], axis=1, inplace=True)
        except:
            print('Not Propensity matched')


    if args.oversample:
        y = dataset.train['Y']

        dataset_proc = dataset.train.drop(['Y'], axis=1, inplace=False)

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
        dataset.train = x_sm_2

        y = dataset.test['Y']

        dataset_proc = dataset.test.drop(['Y'], axis=1, inplace=False)



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
        dataset.test = x_sm_2

    elif args.undersample:
        neg_ = dataset.train[(dataset.train['Y'] == 0) | (dataset.train['Y_prime'] == 0)]
        pos_ = dataset.train[(dataset.train['Y'] == 1) | (dataset.train['Y_prime'] == 1)]
        ids = len(neg_)
        choices = np.random.choice(ids, len(pos_))
        res_neg_features = neg_.iloc[choices]
        dataset.train = pd.concat([res_neg_features, pos_], axis=0)

    target = dataset.train.pop('Y')
    target_prime = dataset.train.pop('Y_prime')
    treatment = dataset.train.pop('X')
    treatment_prime = dataset.train.pop('X_prime')
    uy = dataset.train.pop('Uy')

    dataset.test = dataset.test.reset_index().drop(['index'],axis=1)

    target_test = dataset.test.pop('Y')
    target_prime_test = dataset.test.pop('Y_prime')
    treatment_test = dataset.test.pop('X')
    treatment_prime_test = dataset.test.pop('X_prime')
    uy_test = dataset.test.pop('Uy')

    # Get confounders

    dataset.train = get_subportion_confounders(dataset.train, args.confounders)
    dataset.test = get_subportion_confounders(dataset.test, args.confounders)

    args.len_conf = len(dataset.train.columns)

    # using multiple inputs or all confounders together ?
    if args.multiple_confounders:
        args.z_monotonicity = []
        input_len = args.len_conf + 2

        args.z_monotonicity_latice = []
        for i, col in enumerate(dataset.train.columns):
            args.z_monotonicity.append(args.z_monotonicity_base[col])

            args.lattice_sizes.append(args.z_calib_units)

    else:
        input_len = 3
        args.z_monotonicity = [args.z_monotonicity]
        args.lattice_sizes.append(args.len_conf)

    # Define Model

    if 'azlink' not in args.runPath:
        model = Twin_Net(treatment, uy, dataset.train, args)
    else:
        model = Twin_Net_with_Z_A(treatment, uy, dataset.train, args)

    # Set up loss
    if 'mse' in args.loss:
        loss_func = tf.keras.losses.mean_squared_error
    elif 'mae' in args.loss:
        loss_func = tf.keras.losses.mean_absolute_error
    elif 'bce' in args.loss:
        loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    elif 'dice' in args.loss:
        loss_func = dice_loss
    elif 'hinge' in args.loss:
        loss_func = tf.keras.losses.hinge


    # Save Hparams
    pickle_object(vars(args), os.path.join(args.runPath, 'hparams.pkl'))
    if args.lr_schedule:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            args.lr,
            decay_steps=len(dataset.train)*10,
            decay_rate=0.96,
            staircase=True)

        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
    else:
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=args.lr)

    if args.weighted_loss:
        model.compile(
            loss=[loss_func,loss_func],
            loss_weights = [args.weight_1, args.weight_2],
            optimizer=optimizer)
    else:
        model.compile(
            loss=loss_func,
            optimizer=optimizer)

    cp_callback_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.runPath + '/best',
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss')
    cp_callback_latest = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.runPath + '/latest',
        verbose=0,
        save_weights_only=True,
        save_best_only=False,
        save_freq=100 * args.batch_size)

    if args.train:
        print('-------------------------Experiment: {} ---------------------'.format(args.name))
        conf_to_input = [dataset.train.values.astype(np.float32)]
        conf_to_input_test = [dataset.test.values.astype(np.float32)]

        if args.multiple_confounders:
            conf_to_input = [dataset.train[i].values.astype(np.float32) for i in dataset.train.columns]
            conf_to_input_test = [dataset.test[i].values.astype(np.float32) for i in dataset.test.columns]

        model.fit(
            [treatment.values.astype(np.float32), treatment_prime.values.astype(np.float32),
             uy.values.astype(np.float32), conf_to_input],
            [target[..., np.newaxis],target_prime[..., np.newaxis]],
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=0.2,
            shuffle=True,

            callbacks=[cp_callback_best, cp_callback_latest],
            verbose=1)
    print('-------------------------Experiment: {} ---------------------'.format(args.name))


    model.load_weights(args.runPath + '/best')

    conf_to_input = [dataset.test.values.astype(np.float32)]
    if args.multiple_confounders:
        conf_to_input = [dataset.test[i].values.astype(np.float32) for i in dataset.test.columns]
    test_loss = model.evaluate([treatment_test.values.astype(np.float32),treatment_prime_test.values.astype(np.float32),
                                uy_test.values.astype(np.float32),
                                conf_to_input],
                               [target_test[..., np.newaxis],target_prime_test[..., np.newaxis]])
    print('Test Loss : {}'.format(test_loss))
    preds = model.predict([treatment_test.values.astype(np.float32),treatment_prime_test.values.astype(np.float32),
                           uy_test.values.astype(np.float32), conf_to_input],
                          args.batch_size, 1)

    title = ['Factual', 'Counterfactual']

    for i, pred in enumerate(preds):
        if title[i] == 'Counterfactual':
            gt = target_prime_test
        else:
            gt = target_test
        scaler = MinMaxScaler()
        pred = scaler.fit_transform(pred)


        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        ac = accuracy_score(pred, gt)
        print('{} Acc: {}'.format(title[i], ac))
        f1 = f1_score(gt.values, pred)
        print('{} F1 : {}'.format(title[i], f1))
        auc_r = roc_auc_score(gt.values, pred)
        print('{} AUC-ROC : {}'.format(title[i], auc_r))



    calc_probs(args, treatment_test, dataset, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--inference_name',
                        default='')
    # Logging
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--log_root', type=str, default='./experiments/Twins/')
    parser.add_argument('--name', type=str,
                        default='twin_net_arch_{}_{}_uy_{}_uy_monotonicity_{}_z_monoton_{}_z_layer_{}_calib_units_{}_z_{}_lr_{}_loss_{}_Twins_{}_confounders_{}_ganite_2000')

    # Dataset Hparams
    parser.add_argument('--save_path', default='./data/Datasets/')
    parser.add_argument('--save_name', default='')
    parser.add_argument('--save_dataset', default=False)
    parser.add_argument('--load_dataset', default=True)
    parser.add_argument('--dataset_mode', type=str, default='synthetic')
    parser.add_argument('--path_to_data',

                        default='../data/Datasets/ganite_twins_uy_normal_twins.pkl')

    parser.add_argument('--confounders', default=['all'])
    parser.add_argument('--u_distribution', default='normal')
    parser.add_argument('--oversample',type=bool, default=False)
    parser.add_argument('--undersample',type=bool, default=False)
    # Model Hparams
    parser.add_argument('--lattice_sizes', default=[3,3])
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lattice_units', type=int, default=1)  # 1 or 2
    parser.add_argument('--hidden_dims', type=int, default=3)
    parser.add_argument('--calib_units', type=int, default=3)
    parser.add_argument('--z_calib_units', type=int, default=3)
    parser.add_argument('--layer', default='lattice')
    parser.add_argument('--uy_layer', default='none')
    parser.add_argument('--z_layer', default='none')
    parser.add_argument('--uy_monotonicity', default='none')
    parser.add_argument('--z_monotonicity', default='none')
    parser.add_argument('--z_monotonicity_latice', default='same', help='4 or same')
    parser.add_argument('--concats', type=bool,default=False)

    parser.add_argument('--z_monot_opt', type=int, default=1)
    parser.add_argument('--end_activation', default='none')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--lr_schedule', default=False)
    parser.add_argument('--lr', type=float, default=1e-3)


    parser.add_argument('--weighted_loss',default=False)
    parser.add_argument('--weight_1',type=float,default=0.75)
    parser.add_argument('--weight_2',type=float,default=1.75)
    parser.add_argument('--multiple_confounders', default=False, help='split confounders')

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

    lr_log = str(args.lr).replace('.', '_')
    lr_log = '{}_scheduled'.format(lr_log) if args.lr_schedule else lr_log

    confounders = '_'.join(args.confounders)
    multiple = 'multiple' if args.multiple_confounders else 'single'
    z_layer = '{}_{}'.format(multiple, args.z_layer)

    if args.multiple_confounders:

        z_monotonicity = 'opt_{}'.format(args.z_monot_opt)
        args.z_monotonicity_base = eval('confounder_monotonicities_{}'.format(args.z_monot_opt))

    else:
        z_monotonicity = args.z_monotonicity
        args.z_calib_units = len(args.confounders) if 'all' not in args.confounders else 30#20#52# 20#19

    if args.oversample:
        oversample = '_oversampled'
    elif args.undersample:
        oversample = '_undersampled'
    else:
        oversample = ''
    if args.weighted_loss:
        loss = '{}_weighted_{}_{}'.format(args.loss,args.weight_1,args.weight_2)
    else:
        loss = args.loss
    conf_to_print = '{}_confs'.format(len(args.confounders)) if 'all' not in args.confounders else 'all'
    calib_units = '{}_{}'.format(args.calib_units,args.hidden_dims)

    layer = '{}_{}'.format(args.layer,'concat') if args.concats else args.layer

    args.name = args.name.format(layer, args.end_activation, args.uy_layer, args.uy_monotonicity, z_monotonicity,
                                 z_layer,
                                 calib_units, args.z_calib_units, lr_log, loss, oversample,
                                 conf_to_print)
    args.path_to_data = args.load_path = args.path_to_data.format(args.u_distribution)

    if args.train:
        if not os.path.exists(args.log_root):
            os.makedirs(args.log_root)
        if args.restore:
            oldrunId = args.name
            oldrunpath = os.path.join(args.log_root, oldrunId)
            runId = args.name + '_cont'
        args.runPath = os.path.join(args.log_root, args.name)
        if not os.path.exists(args.runPath):
            os.makedirs(args.runPath)
    elif args.inference_name is not None:
        args.runPath = os.path.join(args.log_root, args.inference_name)
    else:
        args.runPath = os.path.join(args.log_root, args.name)

    run_train(args)


