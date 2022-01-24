import seaborn as sns
import pandas as pd

def plot_embed(databases):
    features = []
    values = []
    for feat, d in databases.items():
        values.extend(d.values())
        features.extend([feat]*len(d))
    df = pd.DataFrame({'feature': features, 'value': values})
    print('Preparing plots')
    g = sns.FacetGrid(df, col='feature', col_wrap=3,
                      sharex=False, sharey=False)
    g.map(sns.distplot, 'value')
    plt.show()




def plot_bar(labels_plot, func, ylabel, title, yerr_func=None):
    """
    Return bar plot & list of values plotted
    given function to apply for each feature
    """
    x = np.arange(len(labels_plot))
    y = [func(label) for label in labels_plot]
    fig, ax = plt.subplots()
    plt.grid(color='grey', which='both',
             linestyle='-', axis='y', linewidth=0.5)
    if yerr_func:  # error bars
        yerr = [yerr_func(label) for label in labels_plot]
        ax.bar(x, y, yerr=yerr, align='center', alpha=0.5)
    else:
        ax.bar(x, y, align='center', alpha=0.5)
    ax.set(ylabel=ylabel, title=title)
    plt.xticks(x, labels_plot, rotation='vertical')
    return fig, y




def plot_all(result, labels_plot, wordlst_l, save=False, save_path=None):
    """
    Bar plots on NAs in data
    """
    numSentences = len(np.unique(result['Sentence no.']))
    numUniqueWords = len(np.unique(result['Word cleaned']))

    figA, yA = plot_bar(labels_plot, lambda feature: countNA(result[feature]),
                        ylabel=f'No. NA values out of {len(result)} words',
                        title='A. No. NA values for each feature')
    figB, yB = plot_bar(labels_plot, lambda feature: len(uniqueNA(result, feature)),
                        ylabel=f'No. unique NA values out of {len(result)} words ({numUniqueWords} unique words)',
                        title='B. No. unique words with NA values for each feature')
    figC, yC = plot_bar(labels_plot, lambda feature: np.mean(avgNA(result, feature)),
                        ylabel='Mean proportion of NA values within each sentence',
                        # \n Errorbars denote std across sentences'
                        title='C. Mean proportion of words with NA values within each sentence',
                        yerr_func=lambda feature: np.std(avgNA(result, feature)))

    def findProp(v, cutoff=0.5):
        return sum(v >= cutoff)

    figD, yD = plot_bar(labels_plot, lambda feature: findProp(avgNA(result, feature)),
                        ylabel=f'No. sentences out of {numSentences} sentences',
                        title='D. No. sentences with mean proportion of NA values .50 or higher')
    if save:
        for fig, name in zip([figA, figB, figC, figD], ['A', 'B', 'C', 'D']):
            fig.savefig(save_path + name + '.png', bbox_inches='tight')
    return figA, figB, figC, figD


def annotateDecomp(X_decomp, annotation, adjustment_method='skip', x_min=-0.02, x_max=0.02, y_min=-0.01, y_max=0.02, skip_no=2, dims=[0, 1], save=False):
    fig, ax = plt.subplots(figsize=(35, 25))
    # plot all points
    plt.plot(X_decomp[:, dims[0]], X_decomp[:, dims[1]], 'bo')

    plotted_texts = []
    added_indices = [1]

    if adjustment_method == 'simple':
        texts = [plt.text(X_decomp[i, dims[0]], X_decomp[i, dims[1]], annotation[i],
                          ha='center', va='center') for i in range(len(X_decomp[:, 0]))]
        adjust_text(texts)

    if adjustment_method == 'none':
        texts = [plt.text(X_decomp[i, dims[0]] * (1 + 0.01), X_decomp[i, dims[1]] * (1 + 0.02),
                          annotation[i], ha='center', va='center') for i in range(len(X_decomp[:, 0]))]

    if adjustment_method == 'skip':
        for i in range(len(X_decomp[:, 0])):
            if x_min < X_decomp[i, 0] < x_max:
                if y_min < X_decomp[i, 1] < y_max:
                    if i % skip_no == 0:
                        texts = plt.text(
                            X_decomp[i, dims[0]], X_decomp[i, dims[1]], annotation[i], ha='center', va='center')
                        plotted_texts.append(texts)
            else:
                texts = plt.text(
                    X_decomp[i, dims[0]], X_decomp[i, dims[1]], annotation[i], ha='center', va='center')
                plotted_texts.append(texts)
        adjust_text(plotted_texts)

    if adjustment_method == 'distance':  # OBS HEAVY
        x_dists = []
        for k in added_indices:
            x_dist = np.abs(X_decomp[i, 0] - X_decomp[k, 0])
            x_dists.append(x_dist)
            if min(x_dists) > frac:
                y_dists = []
                for g in added_indices:
                    y_dist = np.abs(X_decomp[i, 1] - X_decomp[g, 1])
                    y_dists.append(y_dist)
                    if min(y_dists) > frac:
                        # plot
                        texts = plt.text(
                            X_decomp[i, dims[0]], X_decomp[i, dims[1]], annotation[i], ha='center', va='center')
                        plotted_texts.append(texts)
                        added_indices.append(i)
        adjust_text(plotted_texts)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=240)


def plot_usr_input_against_benchmark_dist_plots(bench_df, usr_df):
    '''
    This function plots the lexical feature data. So it accepts a df of the form that gets saved at the very end of the pipeline.
    bench_df: A df that Greta et al determined would be a benchmark (i.e. a corpus that we passed through the pipeline)
    usr_df: This is the df that the user would like to compare against the benchmark
    '''
    plt.figure(figsize=(17, 10))
    plt.rcParams.update({'font.size': 5})
    plot_number = 1
    binz = 13
    for col in usr_df.columns:
        if 'Sentence no' in col:
            #plot_number = plot_number + 1
            continue
        ax = plt.subplot(4, 5, plot_number)
        bench_df[col].hist(bins=binz, density=True,
                           alpha=0.25, ax=ax, color="skyblue")
        bench_df[col].plot.kde(color="skyblue")
        usr_df[col].hist(bins=binz, density=True,
                         alpha=0.25, ax=ax, color="orange")
        usr_df[col].plot.kde(color="orange", title=col)

        # plot legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['benchmark', 'usr_input'])

        # Go to the next plot for the next loop
        plot_number = plot_number + 1

    plt.show()
