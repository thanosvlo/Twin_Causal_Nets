import numpy as np
import argparse
from dataloader import GermanCreditDataset
import pandas as pd
import itertools
from collections import defaultdict
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="",x_label='None',y_label='', **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")


    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
    #         text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
    #         texts.append(text)

    return texts
def calc_ate(dataset,args):
    #
    # p_y_x_row = dataset.index[(dataset[args.outcome] == args.outcome_cond) & (dataset[args.treatment] == args.treat_contrast)].values
    # # ab = dataset.index[(dataset['{}_prime'.format(args.outcome)] == 1) & (dataset['{}_prime'.format(args.treatment)] == args.treat_contrast)].values
    # # lighter = np.hstack((a, ab))
    # # lighter =
    #
    # p_y_x_col = dataset.index[(dataset[args.outcome] == args.outcome_cond) & (dataset[args.treatment] == args.treat_cond)].values
    # # ab = dataset.index[(dataset['{}_prime'.format(args.outcome)] == 1) & (dataset['{}_prime'.format(args.treatment)] == args.treat_cond)].values
    # # heavier = np.hstack((a, ab))
    # # heavier = a
    #
    # # prob_lighter = len(lighter)/len(dataset)
    # # prob_heavier = len(heavier)/len(dataset)
    #
    # # ate = prob_heavier - prob_lighter
    # p_y_x_row = len(p_y_x_row)/len(dataset)
    # p_y_x_col = len(p_y_x_col)/len(dataset)
    # ate = p_y_x_col - p_y_x_row
    if '{}_prime'.format(args.treatment) not in dataset.columns:

        idx_p_y_x_0 = np.where(dataset[args.treatment].values == args.treat_cond)[0]
        idx_p_y_x_1 = np.where(dataset[args.treatment].values == args.treat_contrast)[0]

        y_0 = dataset[args.outcome].iloc[idx_p_y_x_0].mean()
        y_1 = dataset[args.outcome].iloc[idx_p_y_x_1].mean()

        ate = y_1 - y_0
    else:
        idx_p_y_x_0 = np.where(dataset[args.treatment].values == args.treat_cond)[0]
        idx_p_y_x_0_prime = np.where(dataset['{}_prime'.format(args.treatment)].values == args.treat_cond)[0]
        idx_p_y_x_1 = np.where(dataset[args.treatment].values == args.treat_contrast)[0]
        idx_p_y_x_1_prime = np.where(dataset['{}_prime'.format(args.treatment)].values == args.treat_contrast)[0]

        y_0 = dataset[args.outcome].iloc[idx_p_y_x_0]
        y_0_prime = dataset['{}_prime'.format(args.outcome)].iloc[idx_p_y_x_0_prime]
        y_1 = dataset[args.outcome].iloc[idx_p_y_x_1]
        y_1_prime = dataset['{}_prime'.format(args.outcome)].iloc[idx_p_y_x_1_prime]

        y_0 = np.hstack((y_0,y_0_prime)).mean()
        y_1 = np.hstack((y_1,y_1_prime)).mean()

        ate = y_1 - y_0





    # xg = XGBTRegressor(random_state=42)
    # dataset_ = copy.deepcopy(dataset)
    # y = dataset_.pop(args.outcome)
    # y_ = dataset_.pop('{}_prime'.format(args.outcome))
    # tr =  dataset_.pop(args.treatment)
    # tr_ = dataset_.pop('{}_prime'.format(args.treatment))
    #
    # te, lb, ub = xg.estimate_ate(dataset, tr, y)
    # ate_2 = te
    return ate



if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data', type=str,
                        # default='../Data/German Credit/german_data_treatment_{}_outcome_{}_{}.csv')
                        # default='../Data/German Credit/german_data_treatment_{}_outcome_{}_neighb_{}_{}.csv')
                        default = '../Data/DS_10283_124/stroke_data_treatment_{}_outcome_{}_{}.csv')
    parser.add_argument('--load_dataset', type=bool, default=True)
    parser.add_argument('--save_dataset', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default='../Data/German Credit/')
    # parser.add_argument('--save_name', type=str, default='german_data_treatment_{}_outcome_{}_{}.csv')
    parser.add_argument('--save_name', type=str, default='german_data_treatment_{}_outcome_{}_neighb_{}_{}.csv')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--neighb', type=int, default=50)
    parser.add_argument('--propensity_score', type=bool, default=True)
    # parser.add_argument('--treatment', type=str, default='Existing-Account-Status')
    parser.add_argument('--treatment', type=str, default='Heparin')
    # parser.add_argument('--treatment', type=str, default='DASP14')
    # parser.add_argument('--treatment', type=str, default='Credit-History')
    # parser.add_argument('--outcome', type=str, default='Status')
    parser.add_argument('--outcome', type=str, default='Y_3')
    # parser.add_argument('--outcome', type=str, default='Y')
    # parser.add_argument('--outcome', type=str, default='Y_2')


    args = parser.parse_args()
    args.path_to_data = args.path_to_data.format(args.treatment.replace('-', '_'), args.outcome,'{}')
    # args.path_to_data = args.path_to_data.format(args.treatment.replace('-', '_'), args.outcome, args.neighb, '{}')


    dataset_base = GermanCreditDataset(**vars(args))

    dataset = pd.concat((dataset_base.train,dataset_base.test))

    # args.treatment = 'Credit-History'

    treatment_values = np.unique(dataset[args.treatment])

    to_out = defaultdict(dict)
    #
    for outcome_cond in np.unique(dataset[args.outcome]):
        args.outcome_cond = outcome_cond
        res_1 = np.zeros((int(dataset[args.treatment].max() + 1), int(dataset[args.treatment].max() + 1)))

        x_ate = PrettyTable(
            ['ATE', *[str(i) for i in np.unique(dataset[args.treatment])]])
        for treat, treat_counter in itertools.permutations(treatment_values, 2):
            args.treat_cond = treat
            args.treat_contrast = treat_counter

            ate_ = calc_ate(dataset, args)
            res_1[int(treat),int(treat_counter)] = ate_



        for i in np.unique(dataset[args.treatment]):
            i = int(i)
            x_ate.add_row([str(i), *list(res_1[i, :])])

        print(x_ate.get_string())
        outname = args.path_to_data.format('').replace('.csv','ate.pkl'.format(outcome_cond))
        # pickle_object(res_1,outname)

    fig, ax = plt.subplots()

    im, cbar = heatmap(res_1, np.unique(dataset[args.outcome]).astype(int),
                       np.unique(dataset[args.treatment]).astype(int), ax=ax,
                       cbarlabel="ATE", cmap='plasma', x_label='Treatments', y_label='Treatments')
    texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    # fig.tight_layout()
    plt.title('{} - Synthetic Outcome'.format(args.treatment))
    plt.savefig(args.path_to_data.format('').replace('.csv', 'ate.pdf'), format='pdf')



    res_1 = np.zeros((int(dataset[args.treatment].max() + 1), int(dataset[args.outcome].max() + 1)))
    x_ate = PrettyTable(
            ['P(Y|do(X))', *[str(i) for i in np.unique(dataset[args.outcome])]])
    for treat in np.unique(dataset[args.treatment]):
        args.treat_cond = treat
        for outcome_cond in np.unique(dataset[args.outcome]):
            args.outcome_cond = outcome_cond

            do_ = dataset.index[(dataset[args.outcome] == args.outcome_cond) & (dataset[args.treatment] == args.treat_cond)].values
            not_do_ = dataset.index[ (dataset[args.treatment] == args.treat_cond)].values

            res_1[int(treat), int(outcome_cond)] = (len(do_))/(len(not_do_))

    for i in np.unique(dataset[args.treatment]):
        i = int(i)
        x_ate.add_row([str(i), *list(res_1[i, :])])

    print(x_ate.get_string())
    outname = args.path_to_data.format('').replace('.csv', 'p_y_do_x.pkl')
    import plotly.figure_factory as ff

    # fig = go.Figure(data=go.Heatmap(
    #             z=res_1,
    #             x=np.unique(dataset[args.outcome]),
    #             y=np.unique(dataset[args.treatment]),
    #             # hoverlabel=dict(x="Outcome", y="Treatment", color="P(Y|do(X))"),
    #             hoverongaps=False,
    #             colorscale='RdBu_r'))
    #
    # fig.update_layout(
    #     title="P(Y|do(X))",
    #
    #     width=2000, height=400,
    #     legend=dict(x=-.1, y=.5))
    # if os.path.exists(args.path_to_data.format('').replace('.csv', 'test.html')):
    #     os.remove(args.path_to_data.format('').replace('.csv', 'test.html'))
    # with open( args.path_to_data.format('').replace('.csv', 'test.html'), 'a') as f:
    #     f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


    fig, ax = plt.subplots()

    im, cbar = heatmap(res_1, np.unique(dataset[args.outcome]).astype(int), np.unique(dataset[args.treatment]).astype(int), ax=ax,
                        cbarlabel="P(Y|do(X))",cmap='plasma', x_label='Treatments',y_label='Outcomes')
    texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    # fig.tight_layout()
    plt.title('{} - Syntehetic Outcome'.format(args.treatment))
    plt.savefig( args.path_to_data.format('').replace('.csv', 'py_do_x.pdf'),format='pdf')
# pickle_object(res_1, outname)
    # a=1
