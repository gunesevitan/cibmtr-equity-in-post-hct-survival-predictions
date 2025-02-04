import numpy as np
from lifelines.utils import concordance_index


def score(df, group_column, time_column, event_column, prediction_column):

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
