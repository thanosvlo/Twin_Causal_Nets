import matplotlib.pyplot as plt
import numpy as np

from utils import read_pickle_object

data =read_pickle_object('./experiments/p_test.npz')

a=1
probs_titles = ['Necessity', 'Sufficiency', 'Necessity & Sufficiency']
variable=['n','s','ns']
for i, title in enumerate(probs_titles):
    preds = data['p{}'.format(variable[i])]
    gt = data['gt_{}'.format(variable[i])]
    # preds = preds.mean(0)
    gt = gt.mean(0)
    a = np.linspace(0.05, 0.95, 10) #* 10 +0.5

    # dif = preds-gt

    plt.figure()
    fig1, ax1 = plt.subplots()
    ax1.set_title('Probability of {} '.format(title))
    plt.xlabel('P(Uy=p)')
    plt.ylabel('P({})'.format(variable[i]))
    # ax1.boxplot(preds[:,:5],positions=a)
    plt.plot(a,preds.mean(0),label='Pred')
    plt.fill_between(a, (preds.mean(0)-preds.std(0)),(preds.mean(0)+preds.std(0)), color='blue', alpha=0.4)
    plt.plot(a,gt, linestyle='--', label='GT')
    # plt.plot(a, dif.mean(0), label='Error')
    # plt.fill_between(a, (dif.mean(0)-dif.std(0)),(dif.mean(0)+dif.std(0)), color='blue', alpha=0.4)
    plt.legend()
    # plt.show()
    plt.savefig('../imgdump/p_{}_varying_p.pdf'.format(title),format='pdf')


data =read_pickle_object('../p_test_conf.npz')

a=1
probs_titles = ['Necessity', 'Sufficiency', 'Necessity & Sufficiency']
variable=['n','s','ns']
for i, title in enumerate(probs_titles):
    preds = data['p{}'.format(variable[i])]
    gt = data['gt_{}'.format(variable[i])]
    # preds = preds.mean(0)
    gt = gt.mean(0)
    a = np.linspace(0.05, 0.95, 10) #* 10 +0.5

    dif = preds-gt

    plt.figure()
    fig1, ax1 = plt.subplots()
    ax1.set_title('Probability of {} '.format(title))
    plt.xlabel('P(Uy=p)')
    plt.ylabel('P({})'.format(variable[i]))
    # ax1.boxplot(preds[:,:5],positions=a)
    plt.plot(a,preds.mean(0),label='Pred')
    plt.fill_between(a, (preds.mean(0)-preds.std(0)),(preds.mean(0)+preds.std(0)), color='blue', alpha=0.4)
    plt.plot(a,gt, linestyle='--', label='GT')
    # plt.plot(a, dif.mean(0), label='Error')
    # plt.fill_between(a, (dif.mean(0)-dif.std(0)),(dif.mean(0)+dif.std(0)), color='blue', alpha=0.4)
    plt.legend()
    # plt.show()
    plt.savefig('../imgdump/p_{}_varying_p_with_conf.pdf'.format(title),format='pdf')
