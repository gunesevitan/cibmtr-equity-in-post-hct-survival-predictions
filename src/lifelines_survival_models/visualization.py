import numpy as np
import matplotlib.pyplot as plt


def visualize_survival_function(model, title, path=None):

    """
    Visualize survival function of the given model

    Parameters
    ----------
    model: lifelines.BaseFitter
        Lifelines model fit on dataset

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(32, 12))
    model.plot_survival_function(
        ax=ax,
        label='efs',
        alpha=0.5,
        show_censors=True,
        censor_styles={
            'ms': 5,
            'marker': 's'
        }
    )
    ax.set_xlabel('Timeline', size=20, labelpad=15)
    ax.set_ylabel('Survival Probability', size=20, labelpad=15)
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='best', prop={'size': 18})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_hazard(model, title, path=None):

    """
    Visualize hazard of the given model

    Parameters
    ----------
    model: lifelines.BaseFitter
        Lifelines model fit on dataset

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(32, 12))
    model.plot_hazard(ax=ax)
    ax.set_xlabel('Timeline', size=20, labelpad=15)
    ax.set_ylabel('Hazard', size=20, labelpad=15)
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='best', prop={'size': 18})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_cumulative_hazard(model, title, path=None):

    """
    Visualize cumulative hazard of the given model

    Parameters
    ----------
    model: lifelines.BaseFitter
        Lifelines model fit on dataset

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(32, 12))
    model.plot_cumulative_hazard(ax=ax)
    ax.set_xlabel('Timeline', size=20, labelpad=15)
    ax.set_ylabel('Hazard', size=20, labelpad=15)
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='best', prop={'size': 18})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_cumulative_density(model, title, path=None):

    """
    Visualize cumulative density of the given model

    Parameters
    ----------
    model: lifelines.BaseFitter
        Lifelines model fit on dataset

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(32, 12))
    model.plot_cumulative_density(ax=ax)
    ax.set_xlabel('Timeline', size=20, labelpad=15)
    ax.set_ylabel('Density', size=20, labelpad=15)
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='best', prop={'size': 18})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_target(df, target, path=None):

    """
    Visualize cumulative density of the given Aalen-Johansen fitter

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with efs_time, efs and target columns

    target: str
        Name of the target column

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, axes = plt.subplots(figsize=(24, 20), nrows=2, dpi=100)
    axes[0].hist(df.loc[df['efs'] == 0, 'efs_time'], bins=32, alpha=0.6, label='EFS 0')
    axes[0].hist(df.loc[df['efs'] == 1, 'efs_time'], bins=32, alpha=0.6, label='EFS 1')
    axes[1].hist(df.loc[df['efs'] == 0, target], bins=32, alpha=0.6, label='EFS 0')
    axes[1].hist(df.loc[df['efs'] == 1, target], bins=32, alpha=0.6, label='EFS 1')
    for ax in axes:
        ax.tick_params(axis='x', labelsize=15, pad=10)
        ax.tick_params(axis='y', labelsize=15, pad=10)
    axes[0].legend(loc='best', prop={'size': 18})
    axes[1].legend(loc='best', prop={'size': 18})
    axes[0].set_title('efs_time Histogram', size=20, pad=15)
    axes[1].set_title(f'{target} Histogram', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_feature_importance(df_feature_importance, title, path=None):

    """
    Visualize feature importance/coefficients of models

    Parameters
    ----------
    df_feature_importance: pandas.DataFrame of shape (n_features, n_splits)
        Dataframe of feature importance

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 60), dpi=100)
    ax.barh(
        range(len(df_feature_importance)),
        df_feature_importance['mean'],
        xerr=df_feature_importance['std'],
        ecolor='black',
        capsize=10,
        align='center',
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks(range(len(df_feature_importance)))
    ax.set_yticklabels([f'{k} ({v:.2f})' for k, v in df_feature_importance['mean'].to_dict().items()])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    plt.gca().invert_yaxis()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


def visualize_scores(scores, title, path=None):

    """
    Visualize scores of models

    Parameters
    ----------
    scores: pandas.DataFrame of shape (n_splits + 1, n_metrics)
        Dataframe with one or multiple scores and metrics of folds and oof scores

    title: str
        Title of the plot

    path: str, pathlib.Path or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    metric_columns = [column for column in scores.columns.tolist() if column != 'fold']
    fold_scores = scores.loc[scores['fold'] != 'OOF', metric_columns].agg(['mean', 'std']).T.fillna(0)
    oof_scores = scores.loc[scores['fold'] == 'OOF', metric_columns].reset_index(drop=True).T.rename(columns={0: 'score'})

    fig, ax = plt.subplots(figsize=(32, 24))
    ax.barh(
        y=np.arange(fold_scores.shape[0]) - 0.2,
        width=fold_scores['mean'],
        height=0.4,
        xerr=fold_scores['std'],
        align='center',
        ecolor='black',
        capsize=10,
        label='Fold Scores'
    )
    ax.barh(
        y=np.arange(oof_scores.shape[0]) + 0.2,
        width=oof_scores['score'],
        height=0.4,
        align='center',
        capsize=10,
        label='OOF Scores'
    )
    ax.set_yticks(np.arange(fold_scores.shape[0]))
    ax.set_yticklabels([
        f'{metric}\nOOF: {oof:.4f}\nMean: {mean:.4f} (±{std:.4f})' for metric, mean, std, oof in zip(
            fold_scores.index,
            fold_scores['mean'].values,
            fold_scores['std'].values,
            oof_scores['score'].values
        )
    ])
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='best', prop={'size': 18})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
