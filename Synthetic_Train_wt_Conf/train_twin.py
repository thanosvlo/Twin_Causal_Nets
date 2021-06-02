import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score
import tensorflow as tf
# from data.synthetic_dataset_wt_confo_za_link_ciaran import SyntheticDataset, confounder_monotonicities_1
from data.synthetic_dataset_wt_conf_za_link import SyntheticDataset, confounder_monotonicities_1,get_subportion_confounders
import pandas as pd
from models_synthetic import Twin_Net, dice_loss, class_loss, DiceBCELoss,Twin_Net_with_Z_A
from sklearn.preprocessing import MinMaxScaler
from utils import pickle_object, read_pickle_object
import sklearn
from calc_prob_synthetic_conf import calc_probs
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

def run_train(args):

    dataset = SyntheticDataset(**vars(args))
    dataset.train = pd.DataFrame.from_dict(dataset.train)
    dataset.test = pd.DataFrame.from_dict(dataset.test)



    Xs = dataset.train['X'].values
    Xs_prime = dataset.train['X_prime'].values
    Ys = dataset.train['Y'].values
    Ys_prime = dataset.train['Y_prime'].values
    idx_given_y_1 = np.where(Ys[np.where(Xs == 1)[0]] == 1)[0]
    idx_query_y_0 = np.where(Ys_prime[np.where(Xs_prime == 0)[0]] == 0)[0]
    idx_y_1_y_prime_0 = list(set(idx_given_y_1).intersection(idx_query_y_0))
    prob_necessity_1 = len(idx_y_1_y_prime_0) / len(idx_given_y_1)
    print('N',prob_necessity_1)
    idx_given_y_0 = np.where(Ys[np.where(Xs == 0)[0]] == 0)[0]
    idx_query_y_1 = np.where(Ys_prime[np.where(Xs_prime == 1)[0]] == 1)[0]
    idx_y_0_y_prime_1 = list(set(idx_given_y_0).intersection(idx_query_y_1))
    prob_suf_1 = len(idx_y_0_y_prime_1) / len(idx_given_y_0)
    print('S', prob_suf_1)

    if args.oversample:
        y = dataset.train['Y']
        dataset_proc = dataset.train.drop(['Y'], axis=1, inplace=False)
        smote = SMOTE()

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
        dataset.train = x_sm_2


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
        if len(dataset.train.columns) ==1:
            temp = 2 
        else:
            temp = len(dataset.train.columns)
        args.lattice_sizes.append(temp)

    # Define Model

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
        loss_func = DiceBCELoss
    elif 'weighted_loss':
        negs = np.sum(target == 0)
        negs += np.sum(target_prime== 0)
        pos = np.sum(target == 1)
        pos += np.sum(target_prime == 1)

        weight_for_0 = (1 / negs) * (len(dataset.train) * 2) / 2.0
        weight_for_1 = (1 / pos) * (len(dataset.train) * 2) / 2.0
        loss_func = class_loss(np.array([weight_for_0,weight_for_1]))


    # Save Hparams
    pickle_object(vars(args), os.path.join(args.runPath, 'hparams.pkl'))
    if args.lr_schedule:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            args.lr,
            decay_steps=len(dataset.train)*50,
            decay_rate=0.98,
            staircase=True)

        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    if args.weighted_loss:
        model.compile(
            loss=[loss_func,loss_func],
            loss_weights = [args.weight_1, args.weight_2],
            optimizer=optimizer)
    else:
        model.compile(
            loss=loss_func,
            optimizer=optimizer)

    if args.oldrunpath is not None:
        layer_dict = {}
        for i,l in enumerate(model.layers):
            layer_dict[l.name] = i

        model.build((1, input_len))
        model.load_weights(args.oldrunpath + '/best')
        layers_to_freeze = ['A','a_calib','concatenate','linear', 'Y']
        for la in layers_to_freeze:
            if la in layer_dict:
                model.layers[layer_dict[la]].trainable=False


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
        print('-------------------------Experiment: {} ---------------------'.format(args.runId))
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
            # validation_split=0.2,
            shuffle=True,
            validation_data= ([treatment_test.values.astype(np.float32), treatment_prime_test.values.astype(np.float32),
             uy_test.values.astype(np.float32), conf_to_input_test],[target_test[..., np.newaxis],target_prime_test[..., np.newaxis]]),

            callbacks=[cp_callback_best, cp_callback_latest],
            verbose=1)
    print('-------------------------Experiment: {} ---------------------'.format(args.runId))


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
        scaler = MinMaxScaler()
        pred = scaler.fit_transform(pred)

        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        ac = accuracy_score(pred, target_test)
        print('{} Acc: {}'.format(title[i], ac))
        f1 = f1_score(target_test, pred)
        print('{} F1 : {}'.format(title[i],f1))

    calc_probs(args, treatment_test, dataset, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--inference_name',
                        default='')
    # Logging
    parser.add_argument('--restore', type=bool, default=False)
    # parser.add_argument('--restore_path', default='twin_net_arch_za_lattice_none_uy_monotonicity_none_z_monoton_opt_1_z_layer_multiple_calib_units_3_3_z_5_lr_0_0001_loss_mse_weighted_1_0_Synthetics_ciaran_with_3_normal_uniform_bernouli_0_8_confounders_1_confs')
    parser.add_argument('--log_root', type=str, default='./experiments/Synthetic_wt_conf/')
    parser.add_argument('--name', type=str,
                        default='twin_net_arch_za_{}_{}{}_uy_monotonicity_{}_z_monoton_{}_z_layer_{}_calib_units_{}_z_{}_lr_{}_loss_{}_Synthetics_ciaran_with_2_prop_{}_confounders_{}')
    # parser.add_argument('--name', type=str, default='dev')
    # Dataset Hparams
    parser.add_argument('--save_path', default='./data/Datasets/')
    parser.add_argument('--save_name', default='')
    parser.add_argument('--save_dataset', default=False)
    parser.add_argument('--load_dataset', default=True)


    parser.add_argument('--dataset_mode', type=str, default='synthetic')
    parser.add_argument('--path_to_data',

                        default='../data/Datasets/synthetic_dataset_1000_samples_X_{}_Uy_{}_Z_{}_with_counterfactual_and_z_a_link_with_propensity_1_final.pkl')

    parser.add_argument('--confounders',default=['Z'] )

    parser.add_argument('--u_distribution', default='bernouli')
    parser.add_argument('--z_distribution', default='uniform')
    parser.add_argument('--x_distribution', default='bernouli')
    parser.add_argument('--p_1', type=float, default=0.05)
    parser.add_argument('--p_2', type=float, default=0.7)
    parser.add_argument('--p_3', type=float, default=0.3)

    parser.add_argument('--oversample',type=bool, default=False)
    parser.add_argument('--undersample',type=bool, default=False)
    # Model Hparams
    parser.add_argument('--lattice_sizes', default=[3,3])
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lattice_units', type=int, default=1)  # 1 or 2
    parser.add_argument('--hidden_dims', type=int, default=3)
    parser.add_argument('--calib_units', type=int, default=3)
    parser.add_argument('--z_calib_units', type=int, default=3)
    parser.add_argument('--layer', default='linear')
    parser.add_argument('--uy_layer', default='none')
    parser.add_argument('--z_layer', default='none')
    parser.add_argument('--uy_monotonicity', default='none')
    parser.add_argument('--z_monotonicity', default='none')
    parser.add_argument('--z_monotonicity_latice', default='same', help='4 or same')
    parser.add_argument('--concats', type=bool,default=False)

    parser.add_argument('--z_monot_opt', type=int, default=1)
    parser.add_argument('--end_activation', default='none')
    parser.add_argument('--loss', default='mae')
    parser.add_argument('--lr_schedule', default=False)
    parser.add_argument('--lr', type=float, default=1e-3)


    parser.add_argument('--weighted_loss',default=False)
    parser.add_argument('--weight_1',type=float,default=1)
    parser.add_argument('--weight_2',type=float,default=2)
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
    if args.seed == 0: args.seed = int(np.random.randint(0, 2 ** 32 - 1, (1,)))
    print('seed', args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set logdirs

    lr_log = str(args.lr).replace('.', '_')
    lr_log = '{}_scheduled'.format(lr_log) if args.lr_schedule else lr_log

    confounders = '_'.join(args.confounders)
    multiple = 'multiple' if args.multiple_confounders else 'single'
    z_layer = '{}_{}'.format(multiple, args.z_layer) if 'none' not in args.z_layer else '{}'.format(multiple)

    if args.multiple_confounders:

        z_monotonicity = 'opt_{}'.format(args.z_monot_opt)
        args.z_monotonicity_base = eval('confounder_monotonicities_{}'.format(args.z_monot_opt))

    else:
        z_monotonicity = args.z_monotonicity
        args.z_calib_units = len(args.confounders)+1 if 'all' not in args.confounders else 2

    if args.oversample:
        oversample = '_oversampled'
    elif args.undersample:
        oversample = '_undersampled'
    else:
        oversample = ''

    p = '_{}'.format(str(args.p_1).replace('.', '_')) if args.x_distribution == 'bernouli' else ''
    oversample = oversample+'{}_{}_{}{}'.format(args.u_distribution,args.z_distribution,args.x_distribution,p)

    if args.weighted_loss:
        weig_1 = str(args.weight_1).replace('.', '_')
        weig_2 =  str(args.weight_2).replace('.', '_')
        loss = '{}_weighted_{}_{}'.format(args.loss,weig_1,weig_2)
    else:
        loss = args.loss
    conf_to_print = '{}_confs'.format(len(args.confounders)) if 'all' not in args.confounders else 'all'
    calib_units = '{}_{}'.format(args.calib_units,args.hidden_dims)

    layer = '{}_{}'.format(args.layer,'concat') if args.concats else args.layer

    if  'none' not in args.uy_layer:
        uy_la = '_uy_{}'.format(args.uy_layer)
    else:
        uy_la =''

    args.name = args.name.format(layer, args.end_activation,uy_la, args.uy_monotonicity, z_monotonicity,
                                 z_layer,
                                 calib_units, args.z_calib_units, lr_log, loss, oversample,
                                 conf_to_print)

    z_dist = '{}_{}'.format(args.z_distribution, args.p_2,
                            args.p_2) if args.z_distribution == 'bernouli' else '{}'.format(
        args.z_distribution)
    x_dist = '{}_{}'.format(args.x_distribution, args.p_1,
                            args.p_1) if args.x_distribution == 'bernouli' else '{}'.format(
        args.x_distribution)
    
    args.path_to_data = args.load_path = args.path_to_data.format(x_dist,args.u_distribution,z_dist)
    args.oldrunpath = None
    if args.train:
        if not os.path.exists(args.log_root):
            os.makedirs(args.log_root)
        if args.restore:
            oldrunId = args.restore_path
            args.oldrunpath = os.path.join(args.log_root, oldrunId)
            args.runId = args.name + '_cont'
        else:
            args.runId = args.name
        args.runPath = os.path.join(args.log_root, args.runId)
        if not os.path.exists(args.runPath):
            os.makedirs(args.runPath)
    elif args.inference_name is not None:
        args.runPath = os.path.join(args.log_root, args.inference_name)
    else:
        args.runPath = os.path.join(args.log_root, args.name)

    run_train(args)


