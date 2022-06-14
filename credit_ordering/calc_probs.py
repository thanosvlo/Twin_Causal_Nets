import random
import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score
import tensorflow as tf
from dataloader import GermanCreditDataset, get_subportion_confounders
import itertools
from german_metadata import decre_monoto
import pandas as pd
from model_twins import Twin_Net_with_Z_A, dice_loss,Twin_Net,multi_class_twin_za

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score
import copy
from imblearn.over_sampling import SMOTE
import random
from utils import pickle_object,read_pickle_object
from collections import defaultdict
from tabulate import tabulate

def get_test_confs(dataset, args, treatment_factual=None, mode='test'):
    if mode == 'test':
        if args.multiple_confounders:
            conf_to_input = [dataset.test[i].values.astype(np.float32) for i in dataset.test.columns]
        else:
            conf_to_input = [dataset.test.values.astype(np.float32)]


    elif mode == 'dataset_median':
        if 'all' in args.confounders:
            args.confounders = dataset.train.columns
        conf_to_input = [np.tile(dataset.test[i].median(), (len(treatment_factual))) for i in args.confounders]
        if not args.multiple_confounders:
            conf_to_input = np.array(conf_to_input).T

    return conf_to_input


def prob_nec(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args, preds=None):
    if preds is None:
        preds = model.predict([treatment_factual, treatment_counter, uy_to_input,
                               conf_to_input],
                              args.batch_size, 1)
        pred_factual = preds[0]
        pred_counter = preds[1]



        if 'Y' not in args.outcome or args.architecture == 'multi_azlink':
            scaler = MinMaxScaler((args.target_min, args.target_max))
            y = scaler.fit_transform(pred_factual)
            y_prime = scaler.fit_transform(pred_counter)
            y = np.digitize(y, np.arange(y.min(), y.max(), 1 / args.bins), right=False) - 1
            y_prime = np.digitize(y_prime, np.arange(y_prime.min(), y_prime.max(), 1 / args.bins), right=False) - 1
        else:
            y_prime = np.digitize(pred_counter,
                                  [args.target_min, (args.target_max - args.target_min) / 4,(args.target_max - args.target_min)*3 / 4]) - 1
            y = np.digitize(pred_factual,
                            [args.target_min, (args.target_max - args.target_min) / 4,(args.target_max - args.target_min)*3 / 4]) - 1
    else:
        y = preds[0]
        y_prime=preds[1]


    idx_given_y_1 = np.where(y == outcomes_factual)[0]
    idx_query_y_0 = np.where(y_prime == outcomes_counter)[0]

    idx_y_1_y_prime_0 = set(idx_given_y_1).intersection(idx_query_y_0)

    prob_necessity_1 = len(idx_y_1_y_prime_0) / len(idx_given_y_1)
    return prob_necessity_1


def prob_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args,preds=None):
    if preds is  None:
        preds = model.predict([treatment_factual, treatment_counter, uy_to_input,
                               conf_to_input],
                              args.batch_size, 1)
        pred_factual = preds[0]
        pred_counter = preds[1]

        if 'Y' not in args.outcome or args.architecture == 'multi_azlink':
            scaler = MinMaxScaler((args.target_min, args.target_max))
            y = scaler.fit_transform(pred_factual)
            y_prime = scaler.fit_transform(pred_counter)
            y = np.digitize(y, np.arange(y.min(), y.max(), 1 / args.bins), right=False) - 1
            y_prime = np.digitize(y_prime, np.arange(y_prime.min(), y_prime.max(), 1 / args.bins), right=False) - 1
        else:
            y_prime = np.digitize(pred_counter,
                                  [args.target_min, (args.target_max - args.target_min) / 4,
                                   (args.target_max - args.target_min) * 3 / 4]) - 1
            y = np.digitize(pred_factual,
                            [args.target_min, (args.target_max - args.target_min) / 4,
                             (args.target_max - args.target_min) * 3 / 4]) - 1
    else:
        y = preds[0]
        y_prime=preds[1]

    idx_given_y_0 = np.where(y == outcomes_factual)[0]
    idx_query_y_1 = np.where(y_prime == outcomes_counter)[0]

    idx_y_0_y_prime_1 = set(idx_given_y_0).intersection(idx_query_y_1)

    prob_suficiency = len(idx_y_0_y_prime_1) / len(idx_given_y_0)
    return prob_suficiency


def prob_nec_and_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args,preds=None):
    if preds is  None:
        preds = model.predict([treatment_factual, treatment_counter, uy_to_input,
                               conf_to_input],
                              args.batch_size, 1)
        pred_factual = preds[0]
        pred_counter = preds[1]

        if 'Y' not in args.outcome or args.architecture == 'multi_azlink':
            scaler = MinMaxScaler((args.target_min, args.target_max))
            y = scaler.fit_transform(pred_factual)
            y_prime = scaler.fit_transform(pred_counter)
            y = np.digitize(y, np.arange(y.min(), y.max(), 1 / args.bins), right=False) - 1
            y_prime = np.digitize(y_prime, np.arange(y_prime.min(), y_prime.max(), 1 / args.bins), right=False) - 1
        else:
            y_prime = np.digitize(pred_counter,
                                  [args.target_min, (args.target_max - args.target_min) / 4,
                                   (args.target_max - args.target_min) * 3 / 4]) - 1
            y = np.digitize(pred_factual,
                            [args.target_min, (args.target_max - args.target_min) / 4,
                             (args.target_max - args.target_min) * 3 / 4]) - 1
    else:
        y = preds[0]
        y_prime=preds[1]

    idx_given_y_0 = np.where(y == outcomes_factual)[0]
    idx_query_y_1 = np.where(y_prime == outcomes_counter)[0]
    idx_y_0_y_prime_1 = set(idx_given_y_0).intersection(idx_query_y_1)

    prob_nec_and_suficiency = len(idx_y_0_y_prime_1)/len(y)

    return prob_nec_and_suficiency


def ate(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args, preds=None):
    if preds is  None:
        preds = model.predict([treatment_factual, treatment_counter, uy_to_input,
                               conf_to_input],
                              args.batch_size, 1)
        pred_factual = preds[0]
        pred_counter = preds[1]

        if 'Y' not in args.outcome or args.architecture == 'multi_azlink':
            scaler = MinMaxScaler((args.target_min, args.target_max))
            y = scaler.fit_transform(pred_factual)
            y_prime = scaler.fit_transform(pred_counter)
            y = np.digitize(y, np.arange(y.min(), y.max(), 1 / args.bins), right=False) - 1
            y_prime = np.digitize(y_prime, np.arange(y_prime.min(), y_prime.max(), 1 / args.bins), right=False) - 1
        else:
            y_prime = np.digitize(pred_counter,
                                  [args.target_min, (args.target_max - args.target_min) / 4,
                                   (args.target_max - args.target_min) * 3 / 4]) - 1
            y = np.digitize(pred_factual,
                            [args.target_min, (args.target_max - args.target_min) / 4,
                             (args.target_max - args.target_min) * 3 / 4]) - 1
    else:
        y = preds[0]
        y_prime=preds[1]

    idx_given_y_0 = np.sum(y == 1)/ len(y)
    idx_query_y_1 = np.sum(y_prime == 1)/ len(y_prime)

    ate_ = idx_query_y_1 - idx_given_y_0
    return ate_



def calc_ate(dataset,args):

    a = dataset.index[(dataset[args.outcome] == 1) & (dataset[args.treatment] == args.treat_contrast)].values
    # ab = dataset.index[(dataset['{}_prime'.format(args.outcome)] == 1) & (dataset['{}_prime'.format(args.treatment)] == args.treat_contrast)].values
    # lighter = np.hstack((a, ab))
    lighter = a

    a = dataset.index[(dataset[args.outcome] == 1) & (dataset[args.treatment] == args.treat_cond)].values
    # ab = dataset.index[(dataset['{}_prime'.format(args.outcome)] == 1) & (dataset['{}_prime'.format(args.treatment)] == args.treat_cond)].values
    # heavier = np.hstack((a, ab))
    heavier = a

    prob_lighter = len(lighter)/len(dataset)
    prob_heavier = len(heavier)/len(dataset)

    ate = prob_heavier - prob_lighter

    return ate

def calc_acc(gt_fact,gt_counter,preds,args):

    pred_fact = preds[0]
    pred_counter = preds[1]

    if 'Y' not in args.outcome or args.architecture == 'multi_azlink':
        scaler = MinMaxScaler((args.target_min, args.target_max))
        y = scaler.fit_transform(pred_fact)
        y_prime = scaler.fit_transform(pred_counter)
        # pred_fact = np.digitize(y, np.arange(y.min(), y.max(), 1 / args.bins), right=False) - 1
        # pred_counter = np.digitize(y_prime, np.arange(y_prime.min(), y_prime.max(), 1 / args.bins), right=False) - 1
        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        y_prime[y_prime >= 0.5] = 1
        y_prime[y_prime < 0.5] = 0
        pred_counter = y_prime
        pred_fact = y
    else:
        pred_counter = np.digitize(pred_counter,
                              [args.target_min, (args.target_max - args.target_min) / 4,
                               (args.target_max - args.target_min) * 3 / 4]) - 1
        pred_fact = np.digitize(pred_fact,
                        [args.target_min, (args.target_max - args.target_min) / 4,
                         (args.target_max - args.target_min) * 3 / 4]) - 1
    try:
        gt_fact = gt_fact.values
    except:
        a=1
    idx_given_cor = np.where(pred_fact[:,0]==gt_fact)[0]

    ac = accuracy_score(gt_counter[idx_given_cor],pred_counter[idx_given_cor,:])
    print('Counterfactual Acc : {}'.format(ac))
    if len(np.unique(gt_counter)) > 2:
        f1 = f1_score(gt_counter[idx_given_cor],pred_counter[idx_given_cor],average='macro')
    else:
        f1 = f1_score(gt_counter[idx_given_cor], pred_counter[idx_given_cor], )

    print('Counterfactual F1 : {}'.format(f1))
    # au = roc_auc_score(gt_counter.iloc[idx_given_cor], pred_counter[idx_given_cor],multi_class='ovo')
    # print('Counterfactual AUC : {}'.format(au))
    pred_counter = pred_counter[idx_given_cor,:]
    pred_fact = pred_fact[idx_given_cor,:]

    # print('prob fact 0 : {}'.format(np.sum(pred_fact[0]==0)/len(pred_fact[0])))
    # print('prob counter 0 : {}'.format(np.sum(pred_counter[1]==0)/len(pred_counter[1])))
    # nec = prob_nec(None,1,0,None,None,1,0,args,[pred_fact,pred_counter])
    # print('Nec: {}'.format(nec))
    # suf = prob_suf(None, 0, 1, None, None, 0, 1, args, [pred_fact,pred_counter])
    # print('Suf: {}'.format(suf))
    # ns = prob_nec_and_suf(None, 0, 1, None, None, 0, 1, args, [pred_fact,pred_counter])
    # print('Nec  & Suf: {}'.format(ns))
    # at = ate(None,0,1,None,None,0,1,args,[pred_fact,pred_counter])
    # print('ATE: {}'.format(at))
    #
    # return (nec,nec), (suf,suf), (ns,ns), (at,at)


def run_inference(args):
    dataset = GermanCreditDataset(**vars(args))
    # if 'prop' in args.runPath:
    try:
        dataset.train.drop(['Unnamed: 0'], axis=1, inplace=True)
        dataset.test.drop(['Unnamed: 0'], axis=1, inplace=True)
        dataset.train.drop(['propensity_score'], axis=1, inplace=True)
        dataset.test.drop(['propensity_score'], axis=1, inplace=True)
    except:
        print('Not Propensity matched')
    # if args.oversample:
    #     y = dataset.test['Y']
    #
    #     dataset_proc = dataset.test.drop(['Y'], axis=1, inplace=False)
    #
    #     smote = SMOTE()
    #
    #     x_sm, y_sm = smote.fit_resample(dataset_proc, y)
    #     x_sm.insert(0, 'Y', y_sm)
    #     scaler = MinMaxScaler()
    #     x_sm['Y_prime'] = scaler.fit_transform(x_sm['Y_prime'].values[..., np.newaxis])
    #     x_sm['Y_prime'].loc[x_sm['Y_prime'] >= 0.5] = 1
    #     x_sm['Y_prime'].loc[x_sm['Y_prime'] < 0.5] = 0
    #     y_2 = x_sm['Y_prime']
    #     x_sm.drop(['Y_prime'], axis=1, inplace=True)
    #     x_sm_2, y_sm_2 = smote.fit_resample(x_sm, y_2)
    #     x_sm_2.insert(0, 'Y_prime', y_sm_2)
    #     x_sm_2['Y'].loc[x_sm_2['Y'] >= 0.5] = 1
    #     x_sm_2['Y'].loc[x_sm_2['Y'] < 0.5] = 0
    #     x_sm_2['X'].loc[x_sm_2['X'] >= 0.5] = 1
    #     x_sm_2['X'].loc[x_sm_2['X'] < 0.5] = 0
    #     x_sm_2['X_prime'].loc[x_sm_2['X_prime'] >= 0.5] = 1
    #     x_sm_2['X_prime'].loc[x_sm_2['X_prime'] < 0.5] = 0
    #     dataset.test = x_sm_2

    target = dataset.train.pop(args.outcome)
    args.target_max = target.values.max()
    args.target_min = target.values.min()
    target_prime = dataset.train.pop('{}_prime'.format(args.outcome))
    treatment = dataset.train.pop(args.treatment)
    treatment_prime = dataset.train.pop('{}_prime'.format(args.treatment))
    uy = dataset.train.pop('Uy')

    dataset.test = dataset.test.reset_index().drop(['index'],axis=1)
    print('ATE : {}'.format(calc_ate(dataset.test,args)))


    target_test = dataset.test.pop('{}'.format(args.outcome))
    target_prime_test = dataset.test.pop('{}_prime'.format(args.outcome))
    treatment_test = dataset.test.pop('{}'.format(args.treatment))
    treatment_prime_test = dataset.test.pop('{}_prime'.format(args.treatment))
    uy_test = dataset.test.pop('Uy')

    if 'multi_azlink' == args.architecture:
        nb_classes = args.target_max + 1
        target = np.eye(nb_classes.astype(int))[target.astype(int)]
        target_prime = np.eye(nb_classes.astype(int))[target_prime.astype(int)]
        target_test = np.eye(nb_classes.astype(int))[target_test.astype(int)]
        target_prime_test = np.eye(nb_classes.astype(int))[target_prime_test.astype(int)]

    # Get confounders

    dataset.train = get_subportion_confounders(dataset.train, args.confounders)
    dataset.test = get_subportion_confounders(dataset.test, args.confounders)

    args.len_conf = len(dataset.train.columns)

    if args.multiple_confounders:
        args.z_monotonicity = []
        input_len = args.len_conf + 2

        args.z_monotonicity_latice = []
        for i, col in enumerate(dataset.train.columns):

            if col in decre_monoto and args.z_monotonicity_base != 0:
                args.z_monotonicity.append(0)
            else:
                args.z_monotonicity.append(args.z_monotonicity_base)

            args.lattice_sizes.append(args.z_calib_units)

    else:
        args.z_monotonicity = [args.z_monotonicity]
        args.lattice_sizes.append(args.len_conf)

    # if 'az' in args.runPath:
    if 'twin' == args.architecture:
        model = Twin_Net(treatment, uy, dataset.train, args)
    elif 'multi_azlink' == args.architecture:
        model = multi_class_twin_za(treatment, uy, dataset.train, args)
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
    # model = tf.keras.models.load_model(args.runPath + '/best_model.tf')
    conf_to_input = [dataset.test.values.astype(np.float32)]

    if args.architecture != 'multi_azlink':

        target_test = [target_test[..., np.newaxis]]
        target_prime_test = [target_prime_test[..., np.newaxis]]
    else:

        target_test = [target_test[:, i][..., np.newaxis] for i in range(target_test.shape[-1])]
        target_prime_test = [target_prime_test[:, i][..., np.newaxis] for i in range(target_prime_test.shape[-1])]

    if args.multiple_confounders:
        conf_to_input = [dataset.test[i].values.astype(np.float32) for i in dataset.test.columns]

    test_loss = model.evaluate(
        [treatment_test.values.astype(np.float32), treatment_prime_test.values.astype(np.float32),
         uy_test.values.astype(np.float32),
         conf_to_input],
        [*target_test, *target_prime_test],)
    print('Test Loss : {}'.format(test_loss))

    preds = model.predict([treatment_test.values.astype(np.float32), treatment_prime_test.values.astype(np.float32),
                           uy_test.values.astype(np.float32), conf_to_input],
                          args.batch_size, 1)
    title = ['Factual', 'Counterfactual']
    if args.architecture != 'multi_azlink':
        preds = preds[0:2]
        target_prime_test= target_prime_test[0]
        target_test= target_test[0]
    else:
        preds = [np.hstack(preds[0:3]),np.hstack(preds[3:])]
        target_prime_test = np.hstack(target_prime_test)
        target_test = np.hstack(target_test)
    preds_to_infer = []
    for i, pred in enumerate(preds):
        if title[i]=='Counterfactual':
            gt = target_prime_test
        else:
            gt = target_test
        try:
            gt = gt.values
        except:
            a=1
            # print('not a series')
        if 'Y' not in args.outcome or args.architecture == 'multi_azlink':
            scaler = MinMaxScaler()
            pred = scaler.fit_transform(pred)
            # pred = np.digitize(pred, np.arange(gt.min(), gt.max(), 1 / args.bins), right=False) - 1
            pred[pred>=0.5]=1
            pred[pred<0.5]=0
        else:
            pred = np.digitize(pred ,  [args.target_min, (args.target_max - args.target_min) / 4,
                         (args.target_max - args.target_min) * 3 / 4]) - 1

        preds_to_infer.append(pred)
        ac = accuracy_score(pred, gt)
        print('{} Acc: {}'.format(title[i], ac))
        if len(np.unique(gt)) >2 or  args.architecture == 'multi_azlink':
            f1 = f1_score(gt, pred,average='macro')
        else:
            f1 = f1_score(gt, pred, )
        print('{} F1 : {}'.format(title[i], f1))
        # auc_r = roc_auc_score(gt.values, pred,multi_class='ovo')
        # print('{} AUC-ROC : {}'.format(title[i], auc_r))

    # prob_necessities, prob_sufficiencies, prob_both, ate_= calc_acc(target_test,target_prime_test,preds)
    calc_acc(target_test,target_prime_test,preds,args=args)


    if not args.train:
        prob_necessities,prob_sufficiencies, prob_both , ate_ = calc_probs(args, treatment_test, dataset, model)
        return prob_necessities,prob_sufficiencies, prob_both, ate_
    else:
        calc_probs(args, treatment_test, dataset, model)


def calc_probs(args, treatment_test, dataset, model):


    N = len(treatment_test)

    uy_samples = dataset.get_uy_samples(N)

    treatment_factual = np.ones(N) * args.treat_cond

    treatment_counter = np.ones(N) * args.treat_contrast #np.array(random.choices([kk for kk in np.unique(treatment_test.values) if kk > args.treat_cond],k=N))
    # if len(treatment_counter)==0:
    #     treatment_counter = np.array(
    #         random.choices([kk for kk in np.unique(treatment_test.values) if kk < args.treat_cond], k=N))

    outcomes_factual = args.outcome_cond
    outcomes_counter = args.outcome_counter

    uy_to_input = uy_samples

    conf_to_input = get_test_confs(dataset, args, mode='test')

    prob_necessity_1 = prob_nec(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input,
                                outcomes_factual, outcomes_counter, args)

    print('Test Probability of Necessity : {}'.format(prob_necessity_1))

    # ##################### Prob of Sufficiency #####################

    N = len(treatment_test)
    # N = 10000
    # uy_samples = dataset.get_uy_samples(N)
    # uy_to_input = uy_samples

    treatment_factual = np.ones(N) * args.treat_contrast #np.array(random.choices([kk for kk in np.unique(treatment_test.values) if kk > args.treat_cond],k=N))
    treatment_counter = np.ones(N) * args.treat_cond
    # if len(treatment_factual)==0:
    #     treatment_factual = np.array(
    #         random.choices([kk for kk in np.unique(treatment_test.values) if kk < args.treat_cond], k=N))

    outcomes_factual = args.outcome_counter
    outcomes_counter = args.outcome_cond

    conf_to_input = get_test_confs(dataset, args, mode='test')

    prob_suf_1 = prob_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
             outcomes_counter, args)

    print('Test Probability of Sufficiency : {}'.format(prob_suf_1))


    # ############ PROB of Nec and Suf ##################
    N = len(treatment_test)
    # N = 10000
    # uy_samples = dataset.get_uy_samples(N)
    # uy_to_input = uy_samples

    treatment_factual =  np.ones(N) * args.treat_contrast # np.array(
        # random.choices([kk for kk in np.unique(treatment_test.values) if kk > args.treat_cond], k=N))
    treatment_counter = np.ones(N) * args.treat_cond
    # if len(treatment_factual) == 0:
    #     treatment_factual = np.array(
    #         random.choices([kk for kk in np.unique(treatment_test.values) if kk < args.treat_cond], k=N))

    outcomes_factual = args.outcome_counter
    outcomes_counter = args.outcome_cond

    conf_to_input = get_test_confs(dataset, args, mode='test')

    prob_nec_suf_1 = prob_nec_and_suf(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input, outcomes_factual,
                          outcomes_counter, args)

    print('Test Probability of Necessity and Sufficiency : {}'.format(prob_nec_suf_1))


    conf_to_input = get_test_confs(dataset, args, mode='test')

    N = len(conf_to_input[0])
    uy_samples = dataset.get_uy_samples(N)
    uy_to_input = uy_samples

    treatment_factual =  np.ones(N) * args.treat_contrast #np.array(
        # random.choices([kk for kk in np.unique(treatment_test.values) if kk > args.treat_cond], k=N))
    treatment_counter = np.ones(N) * args.treat_cond
    # if len(treatment_factual) == 0:
    #     treatment_factual = np.array(
    #         random.choices([kk for kk in np.unique(treatment_test.values) if kk < args.treat_cond], k=N))

    outcomes_factual = args.outcome_counter
    outcomes_counter = args.outcome_cond


    ate_ = ate(model, treatment_factual, treatment_counter, uy_to_input, conf_to_input,
                                      outcomes_factual,
                                      outcomes_counter, args)
    print('Test ATE : {}'.format(ate_))

    if not args.train:
        return (prob_necessity_1), (prob_suf_1), (prob_nec_suf_1), ate_


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--inference_name',
                        # default='twin_net_arch_lattice_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_3_4_z_4_lr_0_001_loss_mse_German__confounders_11_confs')
                        # default='twin_net_arch_lattice_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_3_4_z_4_lr_0_001_loss_mse_German__confounders_11_confs')
                        # default='twin_net_arch_azlink_lattice_categoric_treat_calib_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_3_4_z_4_lr_0_0001_loss_mse_German__confounders_5_confs')
                        # default='twin_net_arch_azlink_lattice_categoric_treat_calib_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_3_4_z_4_lr_0_001_loss_mse_German__confounders_11_confs')
                        # default='twin_net_arch_azlink_lattice_treat_monot_none_none_uy_none_uy_monot_none_z_monot_opt_1_z_layer_multiple_none_calib_units_4_4_z_4_lr_0_001_loss_mse_German__confounders_11_confs')
                        # default='twin_net_arch_twin_lattice_treat_monot_increasing_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_4_7_z_3_lr_0_001_loss_mse_German__confounders_1_confs_synthetic')
                        # default='twin_net_arch_twin_lattice_treat_monot_increasing_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_3_4_z_3_lr_0_001_loss_mse_German__confounders_2_confs_synthetic')

                        # default='twin_net_arch_twin_linear_treat_monot_none_none_uy_none_uy_monot_none_z_monot_opt_1_z_layer_multiple_none_calib_units_3_4_z_3_lr_0_001_loss_mse_German__confounders_2_confs_synthetic') ## USED IN PAPER
                        # default='twin_net_arch_azlink_lattice_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_4_4_z_4_lr_0_001_loss_mse_German__confounders_11_confs') ## USED IN PAPER
                        # default='twin_net_arch_azlink_lattice_treat_monot_none_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_4_4_z_4_lr_0_001_loss_mse_German__confounders_11_confs_credit_history_2')
                        # default='twin_net_arch_multi_azlink_lattice_treat_monot_increasing_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_3_4_z_3_lr_0_001_loss_mse_German__confounders_2_confs')## USED IN PAPER
                        # default='twin_net_arch_azlink_linear_concat_treat_monot_none_none_uy_none_uy_monot_none_z_monot_opt_1_z_layer_multiple_none_calib_units_3_3_z_3_lr_0_001_loss_mse_German__confounders_11_confs_existing_risk')## USED IN PAPER
                        # default='twin_net_arch_azlink_linear_concat_treat_monot_increasing_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_3_3_z_3_lr_0_001_loss_mse_German__confounders_11_confs_existing_risk')## USED IN PAPER
                        default='twin_net_arch_azlink_lattice_treat_monot_increasing_none_uy_none_uy_monot_none_z_monot_opt_2_z_layer_multiple_none_calib_units_3_4_z_3_lr_0_001_loss_mse_German__confounders_11_confs_switched_treatments')## USED IN PAPER
    # Logging
    parser.add_argument('--restore', type=bool, default=False)
    # parser.add_argument('--log_root', type=str, default='./experiments/German_Credit/')
    # parser.add_argument('--log_root', type=str, default='./experiments/German_Credit_treat_{}_outcome_{}/')
    # parser.add_argument('--log_root', type=str, default='./experiments/German_Credit_treat_{}_outcome_{}_neighb_{}/')
    # parser.add_argument('--log_root', type=str, default='./experiments/German_Credit_treat_{}_outcome_{}_2_neighb_{}/')
    parser.add_argument('--log_root', type=str, default='./experiments/German_Credit_treat_{}_2_outcome_{}_neighb_{}/')
    parser.add_argument('--name', type=str,
                        default='twin_net_arch_{}_{}_uy_{}_uy_monot_{}_z_monot_{}_z_layer_{}_calib_units_{}_z_{}_lr_{}_loss_{}_German_{}_confounders_{}')


    # Dataset Hparams
    # parser.add_argument('--path_to_data', type=str, default='../Data/German Credit/german_data_treatment_{}_outcome_{}_{}.csv')
    # parser.add_argument('--path_to_data', type=str, default='../Data/German Credit/german_data_treatment_{}_outcome_{}_neighb_{}_{}.csv')
    # parser.add_argument('--path_to_data', type=str, default='../Data/German Credit/german_data_treatment_{}_outcome_{}_2_neighb_{}_{}.csv')
    parser.add_argument('--path_to_data', type=str, default='../Data/German Credit/german_data_treatment_{}_2_outcome_{}_neighb_{}_{}.csv')
    parser.add_argument('--load_dataset', type=bool, default=True)
    parser.add_argument('--save_dataset', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default='../Data/German Credit/')
    # parser.add_argument('--save_name', type=str, default='german_data_treatment_{}_outcome_{}_neighb_{}_{}.csv')
    parser.add_argument('--save_name', type=str, default='german_data_treatment_{}_outcome_{}_{}.csv')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--neighb', type=int, default=50)
    parser.add_argument('--bins', type=int, default=4)
    parser.add_argument('--propensity_score', type=bool, default=True)
    # parser.add_argument('--treatment', type=str, default='Credit-History')
    parser.add_argument('--treatment', type=str, default='Existing-Account-Status')
    # parser.add_argument('--outcome', type=str, default='Y')
    parser.add_argument('--outcome', type=str, default='Status')


    # parser.add_argument('--confounders', default=['all'])
    # parser.add_argument('--confounders',
    # default=["Existing-Account-Status", "Month-Duration", "Purpose", "Credit-Amount",
    #                        "Saving-Acount", "Present-Employment", "Instalment-Rate", "Sex", "Guarantors", "Job", "Num-People",]) ## Used in paper

    # parser.add_argument('--confounders', default=['dig',"Credit-History"]) ## Used in paper

    # parser.add_argument('--confounders',
    #                     default=["Existing-Account-Status", "Month-Duration", "Purpose", "Credit-Amount",
    #                              "Saving-Acount"])
    #
    parser.add_argument('--confounders',
                        default=["Credit-History", "Month-Duration", "Purpose", "Credit-Amount",
                                 "Saving-Acount", "Present-Employment", "Instalment-Rate", "Sex", "Guarantors", "Job", ## Used in paper
                                 "Num-People"])

    parser.add_argument('--multiple_confounders', default=True, help='split confounders')



    parser.add_argument('--u_distribution', default='uniform')
    parser.add_argument('--oversample',type=bool, default=False)
    parser.add_argument('--undersample',type=bool, default=False)
    # Model Hparams
    parser.add_argument('--architecture', default='azlink')
    # parser.add_argument('--architecture', default='multi_azlink')
    # parser.add_argument('--architecture', default='twin')
    parser.add_argument('--cat_treat', default=False)
    parser.add_argument('--cat_buckets', default=5)
    parser.add_argument('--treat_monot', default=[(0, 1), (0, 3), (0, 4),
                                                  (1, 3), (1, 4),
                                                  (2, 0), (2, 1), (2, 3), (2, 4),
                                                  (3, 4)])
    # parser.add_argument('--cat_buckets', default=4)
    # parser.add_argument('--treat_monot', default=[(0, 1), (0, 2), (0, 3),
    #                                               (1, 3), (1, 2),
    #                                               (2, 3),
    #                                               ])

    parser.add_argument('--lattice_sizes', default=[3,4])
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lattice_units', type=int, default=1)
    parser.add_argument('--treat_calib_units', type=int, default=3)
    parser.add_argument('--uy_hidden_dims', type=int, default=4)
    parser.add_argument('--z_calib_units', type=int, default=3)
    parser.add_argument('--layer', default='lattice')
    # parser.add_argument('--layer', default='linear')
    parser.add_argument('--uy_layer', default='none')
    parser.add_argument('--z_layer', default='none')
    parser.add_argument('--treat_monotonicity', default='increasing')
    # parser.add_argument('--treat_monotonicity', default='none')

    parser.add_argument('--uy_monotonicity', default='none')
    parser.add_argument('--z_monotonicity', default='none')
    parser.add_argument('--z_monotonicity_latice', default='same', help='4 or same')
    # parser.add_argument('--concats', type=bool,default=True)
    parser.add_argument('--concats', type=bool,default=False)

    parser.add_argument('--z_monot_opt', type=int, default=2)
    # parser.add_argument('--z_monot_opt', type=int, default=1)
    parser.add_argument('--end_activation', default='none')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--lr_schedule', default=False)
    parser.add_argument('--lr', type=float, default=1e-3)


    parser.add_argument('--weighted_loss',default=False)
    parser.add_argument('--weight_1',type=float,default=0.75)
    parser.add_argument('--weight_2',type=float,default=1.75)

    # General
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')
    parser.add_argument('--gpu', type=str, default='1')
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

    if 'Y' in args.outcome:

        args.log_root = args.log_root.format(args.treatment.replace('-', '_'), args.outcome,)
        args.path_to_data = args.path_to_data.format(args.treatment.replace('-', '_'), args.outcome, '{}')

    else:
        args.log_root = args.log_root.format(args.treatment.replace('-', '_'), args.outcome, args.neighb,)
        args.path_to_data = args.path_to_data.format(args.treatment.replace('-', '_'), args.outcome, args.neighb, '{}')

    args.runPath = os.path.join(args.log_root, args.inference_name)
    confounders = '_'.join(args.confounders)
    multiple = 'multiple' if args.multiple_confounders else 'single'
    z_layer = '{}_{}'.format(multiple, args.z_layer)

    if args.multiple_confounders:

        z_monotonicity = 'opt_{}'.format(args.z_monot_opt)
        if args.z_monot_opt == 1:
            args.z_monotonicity_base = 0
        else:
            args.z_monotonicity_base = 1

    else:
        z_monotonicity = args.z_monotonicity
        args.z_calib_units = len(args.confounders) if 'all' not in args.confounders else 19

    # run_inference(args)

# def inference_wrapper(args):
    args.train = True
    if not args.train:
        dataset = GermanCreditDataset(**vars(args))
        # for treat in np.unique(dataset.test[args.treatment]):
        cumulative_results_wrap = defaultdict(dict)

        for treat, treat_counter in itertools.permutations(np.unique(dataset.test[args.treatment]), 2):
            cumulative_results = defaultdict(dict)

            args.treat_cond = treat
            args.treat_contrast = treat_counter
            for outcome_cond, outcome_counter in itertools.permutations(np.unique(dataset.test[args.outcome]), 2):
                args.outcome_cond = outcome_cond
                args.outcome_counter = outcome_counter
                nec_test_prob = []
                nec_dataset_prob = []
                suf_test_prob = []
                suf_dataset_prob = []
                nec_and_suf_test_prob = []
                nec_and_suf_dataset_prob = []
                ates =[]
                for i in range(1, 5):
                    print(args.outcome_cond,args.outcome_counter)
                    prob_necessities,prob_sufficiencies, prob_both,ate_ = run_inference(copy.deepcopy(args))
                    nec_test_prob.append(prob_necessities)
                    suf_test_prob.append(prob_sufficiencies)

                    nec_and_suf_test_prob.append(prob_both)
                    ates.append(ate_)

                cumulative_results['{}'.format(args.outcome_cond)]['{}'.format(args.outcome_counter)] = defaultdict()
                cumulative_results['{}'.format(args.outcome_cond)]['{}'.format(args.outcome_counter)]['PN'] = nec_test_prob
                cumulative_results['{}'.format(args.outcome_cond)]['{}'.format(args.outcome_counter)]['PS'] = suf_test_prob
                cumulative_results['{}'.format(args.outcome_cond)]['{}'.format(args.outcome_counter)]['PNS'] = nec_and_suf_test_prob
                cumulative_results['{}'.format(args.outcome_cond)]['{}'.format(args.outcome_counter)]['ATE'] = ates

                print('\n \n Cond {}, TR {}; Contrast {}, TR {} average Prob of Necessity {}, std: {}'.format(args.outcome_cond, args.treat_cond, args.outcome_counter, args.treat_contrast , np.array(nec_test_prob).mean(), np.array(nec_test_prob).std()))


                print('Cond {}, TR {}; Contrast {}, TR {}; Test average Prob of Sufficiency {}, std: {}'.format( args.outcome_cond,args.treat_cond, args.outcome_counter, args.treat_contrast , np.array(suf_test_prob).mean(),
                                                                          np.array(suf_test_prob).std()))


                print('Cond {}, TR {}; Contrast {}, TR {}; Test average Prob of Necessity and Sufficiency {}, std: {}'.format( args.outcome_cond,args.treat_cond, args.outcome_counter,args.treat_contrast ,  np.array(nec_and_suf_test_prob).mean(),
                                                                            np.array(nec_and_suf_test_prob).std()))

                print('Cond {}, TR {}; Contrast {}, TR {}; Test ATE mean {}, std: {} \n \n'.format( args.outcome_cond,args.treat_cond, args.outcome_counter,args.treat_contrast ,
                    np.array(ates).mean(),
                    np.array(ates).std()))

                cumulative_results[str(args.treat_cond)]['cumulative'] = [( np.array(nec_test_prob).mean(), np.array(nec_test_prob).std()),
                                                       (np.array(suf_test_prob).mean(),
                                                                          np.array(suf_test_prob).std()),
                                                       (np.array(nec_and_suf_test_prob).mean(),
                                                                            np.array(nec_and_suf_test_prob).std()),
                                                       (np.array(ates).mean(),
                                                        np.array(ates).std())  ]
            cumulative_results_wrap['{}'.format(args.treat_cond)]['{}'.format(args.treat_contrast)] = cumulative_results
        print (cumulative_results_wrap)
        pickle_object(cumulative_results_wrap,os.path.join(args.runPath,'cumulative_results.pkl'))

    # raise NotImplementedError

    args.analysis = True
    if args.analysis:
        from prettytable import PrettyTable
        import tabulate as T

        del (T.LATEX_ESCAPE_RULES[u'$'])
        del (T.LATEX_ESCAPE_RULES[u'\\'])

        cumulative_results = read_pickle_object(os.path.join(args.runPath, 'cumulative_results.pkl'))
        dataset = GermanCreditDataset(**vars(args))

        for treat, treat_counter in itertools.permutations(np.unique(dataset.test[args.treatment]), 2):
            if len(np.unique(dataset.test[args.outcome])) > 2:

                try:
                    gt= True
                    gt_ = read_pickle_object(args.path_to_data.replace('csv','pkl').format('PN_tr{}_trcounter{}'.format(int(treat),int(treat_counter))))
                    x_pn_gt = PrettyTable(
                        ['GT_PN_{:.0f}_{:.0f}'.format(treat, treat_counter),
                         *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                    x_ps_gt = PrettyTable(
                        ['GT_PS_{:.0f}_{:.0f}'.format(treat, treat_counter),
                         *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                    x_pns_gt = PrettyTable(
                        ['GT_PNS_{:.0f}_{:.0f}'.format(treat, treat_counter),
                         *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                    x_ate_gt = PrettyTable(
                        ['GT_ATE_{:.0f}_{:.0f}'.format(treat, treat_counter),
                         *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                    res_1_gt = np.zeros((int(dataset.test[args.outcome].max() + 2), int(dataset.test[args.outcome].max() + 2)),
                                        dtype=object)
                    res_2_gt = np.zeros((int(dataset.test[args.outcome].max() + 2), int(dataset.test[args.outcome].max() + 2)),
                                        dtype=object)
                    res_3_gt = np.zeros((int(dataset.test[args.outcome].max() + 2), int(dataset.test[args.outcome].max() + 2)),
                                        dtype=object)

                    res_1_gt[0, :] = ['PN_GT_{}_{}'.format(treat, treat_counter),
                                      *[str(i) for i in np.unique(dataset.test[args.outcome])]]
                    res_2_gt[0, :] = ['PS_GT_{}_{}'.format(treat, treat_counter),
                                      *[str(i) for i in np.unique(dataset.test[args.outcome])]]
                    res_3_gt[0, :] = ['PNS_GT_{}_{}'.format(treat, treat_counter),
                                      *[str(i) for i in np.unique(dataset.test[args.outcome])]]

                    res_1_gt[1:, 1:] = gt_['PN']
                    res_2_gt[1:, 1:] = gt_['PS']
                    res_3_gt[1:, 1:] = gt_['PNS']
                except:
                    gt = False
                    print('No Ground truth available')

                x_pn = PrettyTable(['PN_{}_{}'.format(treat,treat_counter), *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                x_ps = PrettyTable(['PS_{}_{}'.format(treat,treat_counter), *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                x_pns = PrettyTable(['PNS_{}_{}'.format(treat,treat_counter), *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                x_ate = PrettyTable(['ATE_{}_{}'.format(treat,treat_counter), *[str(i) for i in np.unique(dataset.test[args.outcome])]])


                res_1 = np.zeros((int(dataset.test[args.outcome].max() + 2), int(dataset.test[args.outcome].max() + 2)),dtype=object)
                res_2 = np.zeros((int(dataset.test[args.outcome].max() + 2), int(dataset.test[args.outcome].max() + 2)),dtype=object)
                res_3 = np.zeros((int(dataset.test[args.outcome].max() + 2), int(dataset.test[args.outcome].max() + 2)),dtype=object)
                res_4 = np.zeros((int(dataset.test[args.outcome].max() + 2), int(dataset.test[args.outcome].max() + 2)),dtype=object)

                res_1[0,:] = ['PN_{:.0f}_{:.0f}'.format(treat,treat_counter),*[str(i) for i in np.unique(dataset.test[args.outcome])]]
                res_2[0,:] = ['PS_{:.0f}_{:.0f}'.format(treat,treat_counter),*[str(i) for i in np.unique(dataset.test[args.outcome])]]
                res_3[0,:] = ['PNS_{:.0f}_{:.0f}'.format(treat,treat_counter),*[str(i) for i in np.unique(dataset.test[args.outcome])]]
                res_4[0,:] = ['ATE_{:.0f}_{:.0f}'.format(treat,treat_counter),*[str(i) for i in np.unique(dataset.test[args.outcome])]]





                for outcome_cond, outcome_counter in itertools.permutations(np.unique(dataset.test[args.outcome]), 2):

                        res_1[int(outcome_cond)+1,int(outcome_counter)+1] = "${:.4f} \pm {:.4f} $".format(np.array(cumulative_results[str(treat)][str(treat_counter)][str(outcome_cond)][str(outcome_counter)]['PN']).mean(),np.array(cumulative_results[str(treat)][str(treat_counter)][str(outcome_cond)][str(outcome_counter)]['PN']).std())
                        res_2[int(outcome_cond)+1,int(outcome_counter)+1] = "${:.4f} \pm {:.4f} $".format(np.array(cumulative_results[str(treat)][str(treat_counter)][str(outcome_cond)][str(outcome_counter)]['PS']).mean(),np.array(cumulative_results[str(treat)][str(treat_counter)][str(outcome_cond)][str(outcome_counter)]['PS']).std())
                        res_3[int(outcome_cond)+1,int(outcome_counter)+1] = "${:.4f} \pm {:.4f} $".format(np.array(cumulative_results[str(treat)][str(treat_counter)][str(outcome_cond)][str(outcome_counter)]['PNS']).mean(),np.array(cumulative_results[str(treat)][str(treat_counter)][str(outcome_cond)][str(outcome_counter)]['PNS']).std())
                        res_4[int(outcome_cond)+1,int(outcome_counter)+1] = "${:.4f} \pm {:.4f} $".format(np.array(cumulative_results[str(treat)][str(treat_counter)][str(outcome_cond)][str(outcome_counter)]['ATE']).mean(),np.array(cumulative_results[str(treat)][str(treat_counter)][str(outcome_cond)][str(outcome_counter)]['ATE']).std())



                for i in np.unique(dataset.test[args.outcome]):
                    i = int(i)
                    res_1[i+1,0] = i
                    res_2[i+1,0] = i
                    res_3[i+1,0] = i
                    res_4[i+1,0] = i
                    if gt:
                        res_1_gt[ i + 1,0] = i
                        res_2_gt[ i + 1,0] = i
                        res_3_gt[ i + 1,0] = i

            else:

                try:
                    gt = True
                    gt_ = read_pickle_object(args.path_to_data.replace('csv', 'pkl').format(
                        'PN_tr{}_trcounter{}'.format(int(treat), int(treat_counter))))
                    x_pn_gt = PrettyTable(
                        ['GT_PN'.format(treat, treat_counter),
                         *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                    x_ps_gt = PrettyTable(
                        ['GT_PS'.format(treat, treat_counter),
                         *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                    x_pns_gt = PrettyTable(
                        ['GT_PNS'.format(treat, treat_counter),
                         *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                    x_ate_gt = PrettyTable(
                        ['GT_ATE'.format(treat, treat_counter),
                         *[str(i) for i in np.unique(dataset.test[args.outcome])]])
                    res_1_gt = np.zeros(
                        (int(dataset.test[args.outcome].max() + 2), int(dataset.test[args.outcome].max() + 2)),
                        dtype=object)
                    res_2_gt = np.zeros(
                        (int(dataset.test[args.outcome].max() + 2), int(dataset.test[args.outcome].max() + 2)),
                        dtype=object)
                    res_3_gt = np.zeros(
                        (int(dataset.test[args.outcome].max() + 2), int(dataset.test[args.outcome].max() + 2)),
                        dtype=object)

                    res_1_gt[0, :] = ['PN_GT'.format(treat, treat_counter),
                                      *[str(i) for i in np.unique(dataset.test[args.outcome])]]
                    res_2_gt[0, :] = ['PS_GT'.format(treat, treat_counter),
                                      *[str(i) for i in np.unique(dataset.test[args.outcome])]]
                    res_3_gt[0, :] = ['PNS_GT'.format(treat, treat_counter),
                                      *[str(i) for i in np.unique(dataset.test[args.outcome])]]

                    res_1_gt[1:, 1:] = gt_['PN']
                    res_2_gt[1:, 1:] = gt_['PS']
                    res_3_gt[1:, 1:] = gt_['PNS']
                except:
                    gt = False
                    print('No Ground truth available')

                x_pn = PrettyTable(['PN'.format(treat, treat_counter),
                                    *[str(i) for i in np.unique(dataset.test[args.treatment])]])
                x_ps = PrettyTable(['PS'.format(treat, treat_counter),
                                    *[str(i) for i in np.unique(dataset.test[args.treatment])]])
                x_pns = PrettyTable(['PNS'.format(treat, treat_counter),
                                     *[str(i) for i in np.unique(dataset.test[args.treatment])]])
                x_ate = PrettyTable(['ATE'.format(treat, treat_counter),
                                     *[str(i) for i in np.unique(dataset.test[args.treatment])]])

                res_1 = np.zeros(
                    (int(dataset.test[args.treatment].max() + 2), int(dataset.test[args.treatment].max() + 2)),
                    dtype=object)
                res_2 = np.zeros(
                    (int(dataset.test[args.treatment].max() + 2), int(dataset.test[args.treatment].max() + 2)),
                    dtype=object)
                res_3 = np.zeros(
                    (int(dataset.test[args.treatment].max() + 2), int(dataset.test[args.treatment].max() + 2)),
                    dtype=object)
                res_4 = np.zeros(
                    (int(dataset.test[args.treatment].max() + 2), int(dataset.test[args.treatment].max() + 2)),
                    dtype=object)

                res_1[0, :] = ['PN'.format(treat, treat_counter),
                               *[str(i) for i in np.unique(dataset.test[args.treatment])]]
                res_2[0, :] = ['PS'.format(treat, treat_counter),
                               *[str(i) for i in np.unique(dataset.test[args.treatment])]]
                res_3[0, :] = ['PNS'.format(treat, treat_counter),
                               *[str(i) for i in np.unique(dataset.test[args.treatment])]]
                res_4[0, :] = ['ATE'.format(treat, treat_counter),
                               *[str(i) for i in np.unique(dataset.test[args.treatment])]]

                for treat, treat_counter in itertools.permutations(np.unique(dataset.test[args.treatment]), 2):

                    res_1[int(treat) + 1, int(treat_counter) + 1] = "${:.4f} \pm {:.4f} $".format(np.array(
                        cumulative_results[str(treat)][str(treat_counter)][str(0.0)][str(1.0)][
                            'PN']).mean(), np.array(
                        cumulative_results[str(treat)][str(treat_counter)][str(0.0)][str(1.0)][
                            'PN']).std())
                    res_2[int(treat) + 1, int(treat_counter) + 1] = "${:.4f} \pm {:.4f} $".format(np.array(
                        cumulative_results[str(treat)][str(treat_counter)][str(0.0)][str(1.0)][
                            'PS']).mean(), np.array(
                        cumulative_results[str(treat)][str(treat_counter)][str(0.0)][str(1.0)][
                            'PS']).std())
                    res_3[int(treat) + 1, int(treat_counter) + 1] = "${:.4f} \pm {:.4f} $".format(np.array(
                        cumulative_results[str(treat)][str(treat_counter)][str(0.0)][str(1.0)][
                            'PNS']).mean(), np.array(
                        cumulative_results[str(treat)][str(treat_counter)][str(0.0)][str(1.0)][
                            'PNS']).std())
                    res_4[int(treat) + 1, int(treat_counter) + 1] = "${:.4f} \pm {:.4f} $".format(np.array(
                        cumulative_results[str(treat)][str(treat_counter)][str(0.0)][str(1.0)][
                            'ATE']).mean(), np.array(
                        cumulative_results[str(treat)][str(treat_counter)][str(0.0)][str(1.0)][
                            'ATE']).std())

                for i in np.unique(dataset.test[args.treatment]):
                    i = int(i)
                    res_1[i + 1, 0] = i
                    res_2[i + 1, 0] = i
                    res_3[i + 1, 0] = i
                    res_4[i + 1, 0] = i
                    if gt:
                        res_1_gt[i + 1, 0] = i
                        res_2_gt[i + 1, 0] = i
                        res_3_gt[i + 1, 0] = i


                # i = int(i)
                # x_pn.add_row([str(i), *list(res_1[i, :])])
                # x_ps.add_row([str(i), *list(res_2[i, :])])
                # x_pns.add_row([str(i), *list(res_3[i, :])])
                # x_ate.add_row([str(i), *list(res_4[i, :])])
                #
                # x_pn_gt.add_row([str(i), *list(res_1_gt[i, :])])
                # x_ps_gt.add_row([str(i), *list(res_2_gt[i, :])])
                # x_pns_gt.add_row([str(i), *list(res_3_gt[i, :])])


            print('\\begin{table}')
            print('\\centering')
            print(tabulate(res_1[1:],headers=res_1[0,:], tablefmt="latex_booktabs", floatfmt=".4f"))
            print('\\caption{Probability of Necessity -- columns and rows are outcomes -- given treatment '+str(treat)+' counter treatment ' +str(treat_counter)+'}')
            print('\\end{table}')
            print(' ')
            if gt:
                print('\\begin{table}')
                print('\\centering')
                print(tabulate(res_1_gt[1:],headers=res_1_gt[0,:], tablefmt="latex_booktabs", floatfmt=".4f"))
                print(
                    '\\caption{Probability of Necessity -- Ground Truth-- columns and rows are outcomes -- given treatment '+str(treat)+' counter treatment '+str(treat_counter)+'}')
                print('\\end{table}')
            print(' ')
            print('\\begin{table}')
            print('\\centering')
            print(tabulate(res_2[1:],headers=res_2[0,:], tablefmt="latex_booktabs", floatfmt=".4f"))
            print(
                '\\caption{Probability of Sufficiency -- columns and rows are outcomes -- given treatment '+str(treat)+' counter treatment ' +str(treat_counter)+'}')
            print('\\end{table}')

            print(' ')
            if gt:
                print('\\begin{table}')
                print('\\centering')
                print(tabulate(res_2_gt[1:],headers=res_2_gt[0,:], tablefmt="latex_booktabs", floatfmt=".4f"))
                print(
                    '\\caption{Probability of Sufficiency -- Ground Truth-- columns and rows are outcomes-- given treatment '+str(treat)+' counter treatment '+str(treat_counter)+'}')
                print('\\end{table}')
                print(' ')
            print('\\begin{table}')
            print('\\centering')
            print(tabulate(res_3[1:],headers=res_3[0,:], tablefmt="latex_booktabs", floatfmt=".4f"))
            print(
                '\\caption{Probability of Necessity \& Sufficiency -- columns and rows are outcomes -- given treatment '+str(treat)+' counter treatment '+str(treat_counter)+'}')
            print('\\end{table}')
            print(' ')
            if gt:
                print('\\begin{table}')
                print('\\centering')
                print(tabulate(res_3_gt[1:],headers=res_3_gt[0,:], tablefmt="latex_booktabs", floatfmt=".4f"))
                print(
                    '\\caption{Probability of Necessity \& Sufficiency -- Ground Truth -- columns and rows are outcomes -- given treatment '+str(treat)+' counter treatment '+str(treat_counter)+'}')
                print('\\end{table}')
            print(' ')
            print('\\begin{table}')
            print('\\centering')
            print(tabulate(res_4[1:],headers=res_4[0,:], tablefmt="latex_booktabs", floatfmt=".4f"))
            print(
                '\\caption{ ATE -- columns and rows are outcomes --- given treatment '+str(treat)+' counter treatment '+str(treat_counter)+'}')
            print('\\end{table}')
            # print(x_pn.get_string())
            # if gt:
            #     print(x_pn_gt.get_string())
            # print(x_ps.get_string())
            # if gt:
            #     print(x_ps_gt.get_string())
            # print(x_pns.get_string())
            # if gt:
            #     print(x_pns_gt.get_string())
            # print(x_ate.get_string())


        # print('PN Treat {}; Contrast {}: mean {}, std: {} '.format(a,k, np.array(cumulative_results[str(a)][k]['PN']).mean(), np.array(cumulative_results[str(a)][k]['PN']).std()))
        # print('PS Treat {}; Contrast {}: mean {}, std: {} '.format(a,k, np.array(cumulative_results[str(a)][k]['PS']).mean(), np.array(cumulative_results[str(a)][k]['PS']).std()))
        # print('PNS Treat {}; Contrast {}: mean {}, std: {} '.format(a,k, np.array(cumulative_results[str(a)][k]['PNS']).mean(), np.array(cumulative_results[str(a)][k]['PNS']).std()))
        # print('ATE Treat {}; Contrast {}: mean {}, std: {} '.format(a,k, np.array(cumulative_results[str(a)][k]['ATE']).mean(), np.array(cumulative_results[str(a)][k]['ATE']).std()))
