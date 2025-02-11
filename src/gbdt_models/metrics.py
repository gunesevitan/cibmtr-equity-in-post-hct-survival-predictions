import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
)
from lifelines.utils import concordance_index


def ranking_score(df, group_column, time_column, event_column, prediction_column):

    """
    Calculate global and group-wise concordance index scores

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples, 4)
        Dataframe with group, time, event and prediction columns

    group_column: str
        Name of the group column

    time_column: str
        Name of the continuous time column

    event_column: str
        Name of the binary event column

    prediction_column: str
        Name of the prediction column

    Returns
    -------
    scores: dict of {metric: score}
        Dictionary of metrics and scores
    """

    scores = {
        'micro_concordance_index': float(concordance_index(
            event_times=df[time_column],
            predicted_scores=-df[prediction_column],
            event_observed=df[event_column]
        ))
    }

    group_scores = []
    for group, df_group in df.groupby(group_column):
        group_score = concordance_index(
            event_times=df_group[time_column],
            predicted_scores=-df_group[prediction_column],
            event_observed=df_group[event_column]
        )
        group_scores.append(group_score)
        scores[f'{"_".join(group.lower().split())}_concordance_index'] = float(group_score)

    macro_concordance_index = np.mean(group_scores)
    std_concordance_index = np.std(group_scores)
    stratified_concordance_index = macro_concordance_index - std_concordance_index

    scores['macro_concordance_index'] = float(macro_concordance_index)
    scores['std_concordance_index'] = float(std_concordance_index)
    scores['stratified_concordance_index'] = float(stratified_concordance_index)

    return scores


def classification_score(df, group_column, event_column, prediction_column, threshold=0.5):

    """
    Calculate global and group-wise classification scores

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples, 3)
        Dataframe with group, event and prediction columns

    group_column: str
        Name of the group column

    event_column: str
        Name of the binary event column

    prediction_column: str
        Name of the prediction column

    threshold: float
        Threshold for converting probabilities into labels

    Returns
    -------
    scores: dict of {metric: score}
        Dictionary of metrics and scores
    """

    probabilities = df[prediction_column].values
    labels = np.zeros_like(probabilities, dtype=np.uint8)
    labels[probabilities >= threshold] = 1

    scores = {
        'accuracy': accuracy_score(df[event_column], labels),
        'roc_auc': roc_auc_score(df[event_column], df[prediction_column]),
        'f1': f1_score(df[event_column], labels)
    }

    for group, df_group in df.groupby(group_column):
        group_probabilities = df_group[prediction_column].values
        group_labels = np.zeros_like(group_probabilities, dtype=np.uint8)
        group_labels[group_probabilities >= threshold] = 1
        group_scores = {
            'accuracy': accuracy_score(df_group[event_column], group_labels),
            'roc_auc': roc_auc_score(df_group[event_column], df_group[prediction_column]),
            'f1': f1_score(df_group[event_column], group_labels)
        }
        group_scores = {f'{"_".join(group.lower().split())}_{k}': v for k, v in group_scores.items()}
        scores.update(group_scores)

    return scores


def regression_score(df, group_column, time_column, prediction_column):

    """
    Calculate global and group-wise regression scores

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_samples, 3)
        Dataframe with group, time and prediction columns

    group_column: str
        Name of the group column

    time_column: str
        Name of the continuous time column

    prediction_column: str
        Name of the prediction column

    Returns
    -------
    scores: dict of {metric: score}
        Dictionary of metrics and scores
    """

    scores = {
        'mean_squared_error': mean_squared_error(df[time_column], df[prediction_column]),
        'mean_absolute_error': mean_absolute_error(df[time_column], df[prediction_column]),
        'mean_absolute_percentage_error': mean_absolute_percentage_error(df[time_column] + 1, df[prediction_column] + 1),
    }

    for group, df_group in df.groupby(group_column):
        group_scores = {
            'mean_squared_error': mean_squared_error(df_group[time_column], df_group[prediction_column]),
            'mean_absolute_error': mean_absolute_error(df_group[time_column], df_group[prediction_column]),
            'mean_absolute_percentage_error': mean_absolute_percentage_error(df_group[time_column] + 1, df_group[prediction_column] + 1),
        }
        group_scores = {f'{"_".join(group.lower().split())}_{k}': v for k, v in group_scores.items()}
        scores.update(group_scores)

    return scores
