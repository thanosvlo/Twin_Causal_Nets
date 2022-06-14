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


def run_train(args):
    dataset = SyntheticDataset(**vars(args))
    train_set = pd.DataFrame.from_dict(dataset.train)
    test_set = pd.DataFrame.from_dict(dataset.test)

    target = train_set.pop('Y')
    target_prime = train_set.pop('Y_prime')

    target_test = test_set.pop('Y')
    target_prime_test = test_set.pop('Y_prime')

    model = TwinNet(train_set,args)
    # Set up loss
    if 'mse'in args.loss:
        loss_func = tf.keras.losses.mean_squared_error
    elif 'mae'in args.loss:
        loss_func = tf.keras.losses.mean_absolute_error
    elif 'bce'in args.loss:
        loss_func = tf.keras.losses.binary_crossentropy

    if 'custom' in args.loss:
        loss_func = {'Y':loss_func,'Y_1':custom_loss}

    optimizer = tf.keras.optimizers.Adagrad(learning_rate=args.lr)

    model.compile(
        loss=loss_func,
        optimizer=optimizer)

    cp_callback_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.runPath+'/best',
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss')
    cp_callback_latest = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.runPath + '/latest',
        verbose=0,
        save_weights_only=True,
        save_best_only=False,
        save_freq=100*args.batch_size)

    if args.train:
        print('-------------------------Experiment: {} ---------------------'.format(args.name))
        model.fit(
            [train_set.values.astype(np.float32)[:,0],train_set.values.astype(np.float32)[:,1],train_set.values.astype(np.float32)[:,2]],
            [target[...,np.newaxis],target_prime[...,np.newaxis]],
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=0.2,
            shuffle=False,
            callbacks=[cp_callback_best,cp_callback_latest],
            verbose=1)
    print('-------------------------Experiment: {} ---------------------'.format(args.name))

    # if not args.train:
    model.build((1,3))
    model.load_weights(args.runPath+'/best')
    test_loss =  model.evaluate( [test_set.values.astype(np.float32)[:,0],test_set.values.astype(np.float32)[:,1],test_set.values.astype(np.float32)[:,2]],
        [target_test[...,np.newaxis],target_prime_test[...,np.newaxis]])
    print('Test Loss : {}'.format(test_loss))
    preds =  model.predict( [test_set.values.astype(np.float32)[:,0],test_set.values.astype(np.float32)[:,1],test_set.values.astype(np.float32)[:,2]],
                            args.batch_size,1)
    title=['Factual Acc', 'Counterfactual Acc']
    for i, pred in enumerate(preds):
        pred = (pred - np.min(pred))/np.ptp(pred)
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        ac = accuracy_score(pred,target_test)
        print('{} : {}'.format(title[i],ac))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--inference_name', default='twin_net_arch_lattice_uy_linear_uy_monotonicity_increasing_lr_0.001_loss_bce')
    # Logging
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--log_root', type=str, default='./experiments')
    parser.add_argument('--name', type=str, default='twin_net_arch_{}_uy_{}_uy_monotonicity_{}_end_act_{}_calib_units_{}_lr_{}_loss_{}_uy_{}_2')
    # parser.add_argument('--name', type=str, default='dev')
    # Dataset Hparams
    parser.add_argument('--save_path', default='./data/Datasets/')
    parser.add_argument('--save_name', default='synthetic_dataset_{}_samples_X_{}_Uy_{}.pkl')
    parser.add_argument('--save_dataset', default=False)
    parser.add_argument('--load_dataset', default=True)
    parser.add_argument('--dataset_mode', type=str, default='synthetic')
    parser.add_argument('--path_to_data',
                        default='../data/Datasets/synthetic_dataset_200000_samples_X_bernouli_Uy_{}_with_counterfactual.pkl')
    parser.add_argument('--u_distribution', default='uniform',help='normal, uniform')
    # Model Hparams
    parser.add_argument('--lattice_sizes', default=[2, 3, 2, 2, 3])
    parser.add_argument('--lattice_monotonicities', default=['increasing', 'increasing'])
    parser.add_argument('--lr',type =float, default=1e-3)
    parser.add_argument('--epochs', type =int,default=200)
    parser.add_argument('--batch_size',type =int, default=64)
    parser.add_argument('--hidden_dims',type =int, default=1)
    parser.add_argument('--calib_units', type =int,default=2)
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
    args.name = args.name.format(args.layer,args.uy_layer,args.uy_monotonicity,args.end_activation,args.calib_units,args.lr,args.loss,args.u_distribution)
    args.path_to_data = args.path_to_data.format(args.u_distribution)

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
