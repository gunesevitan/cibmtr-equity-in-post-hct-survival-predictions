import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_correlations(df, columns, title, path=None):

    """
    Visualize correlations of given columns in given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given columns

    columns: list
        List of names of columns

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(25, 25), dpi=100)
    ax = sns.heatmap(
        df[columns].corr(),
        annot=True,
        square=True,
        cmap='coolwarm',
        annot_kws={'size': 16},
        fmt='.3f'
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    ax.tick_params(axis='x', labelsize=10, rotation=90)
    ax.tick_params(axis='y', labelsize=10, rotation=0)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
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


def visualize_roc_curves(roc_curves, title, path=None):

    """
    Visualize ROC curves of the model(s)

    Parameters
    ----------
    roc_curves: array-like of shape (n_splits, 3)
        List of ROC curves (tuple of false positive rates, true positive rates and thresholds)

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    true_positive_rates_interpolated = []
    aucs = []
    mean_false_positive_rate = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(16, 16))

    # Plot random guess curve
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2.5, color='r', alpha=0.75)

    # Plot individual ROC curves of multiple models
    for fprs, tprs, _ in roc_curves:
        true_positive_rates_interpolated.append(np.interp(mean_false_positive_rate, fprs, tprs))
        true_positive_rates_interpolated[-1][0] = 0.0
        roc_auc = auc(fprs, tprs)
        aucs.append(roc_auc)
        ax.plot(fprs, tprs, lw=1, alpha=0.1)

    # Plot mean ROC curve of N models
    mean_true_positive_rate = np.mean(true_positive_rates_interpolated, axis=0)
    mean_true_positive_rate[-1] = 1.0
    mean_auc = auc(mean_false_positive_rate, mean_true_positive_rate)
    std_auc = np.std(aucs)
    ax.plot(mean_false_positive_rate, mean_true_positive_rate, color='b', label=f'Mean ROC Curve (AUC: {mean_auc:.4f} ±{std_auc:.4f})', lw=2.5, alpha=0.9)
    best_threshold_idx = np.argmax(mean_true_positive_rate - mean_false_positive_rate)
    ax.scatter(
        [mean_false_positive_rate[best_threshold_idx]], [mean_true_positive_rate[best_threshold_idx]],
        marker='o',
        color='r',
        s=100,
        label=f'Best Threshold\nSensitivity: {mean_true_positive_rate[best_threshold_idx]:.4f}\nSpecificity {mean_false_positive_rate[best_threshold_idx]:.4f}'
    )

    # Plot confidence interval of ROC curves
    std_tpr = np.std(true_positive_rates_interpolated, axis=0)
    tprs_upper = np.minimum(mean_true_positive_rate + std_tpr, 1)
    tprs_lower = np.maximum(mean_true_positive_rate - std_tpr, 0)
    ax.fill_between(mean_false_positive_rate, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='±1 sigma')

    ax.set_xlabel('False Positive Rate', size=15, labelpad=12)
    ax.set_ylabel('True Positive Rate', size=15, labelpad=12)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='lower right', prop={'size': 14})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_pr_curves(pr_curves, title, path=None):

    """
    Visualize PR curves of the model(s)

    Parameters
    ----------
    pr_curves: array-like of shape (n_splits, 3)
        List of PR curves (tuple of precision, recall and thresholds)

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    precisions_interpolated = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(16, 16))

    # Plot individual PR curves of multiple models
    for precisions, recalls, _ in pr_curves:
        precisions_interpolated.append(np.interp(mean_recall, 1 - recalls, precisions)[::-1])
        precisions_interpolated[-1][0] = 0.0
        precisions_interpolated[-1][0] = 1 - precisions_interpolated[-1][0]
        pr_auc = auc(recalls, precisions)
        aucs.append(pr_auc)
        ax.plot(recalls, precisions, lw=1, alpha=0.1)

    # Plot mean PR curve of N models
    mean_precision = np.mean(precisions_interpolated, axis=0)
    mean_precision[-1] = 0
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)
    ax.plot(mean_recall, mean_precision, color='b', label=f'Mean PR Curve (AUC: {mean_auc:.4f} ±{std_auc:.4f})', lw=2.5, alpha=0.9)

    f1_scores = 2 * mean_recall * mean_precision / (mean_recall + mean_precision)
    best_threshold_idx = np.argmax(f1_scores)
    ax.scatter(
        [mean_recall[best_threshold_idx]], [mean_precision[best_threshold_idx]],
        marker='o',
        color='r',
        s=100,
        label=f'Best Threshold\nRecall: {mean_recall[best_threshold_idx]:.4f}\nPrecision {mean_precision[best_threshold_idx]:.4f}'
    )

    # Plot confidence interval of PR curves
    std_tpr = np.std(precisions_interpolated, axis=0)
    tprs_upper = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower = np.maximum(mean_precision - std_tpr, 0)
    ax.fill_between(mean_recall, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='±1 sigma')

    ax.set_xlabel('Recall', size=15, labelpad=12)
    ax.set_ylabel('Precision', size=15, labelpad=12)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='lower right', prop={'size': 14})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
