from lifelines.utils import concordance_index


def score(y_true_time, y_true_event, y_pred):

    """
    Calculate metric scores on targets and predictions

    Parameters
    ----------
    y_true_time: numpy.ndarray of shape (n_samples)
        Continuous event times

    y_true_event: numpy.ndarray of shape (n_samples)
        Binary event

    y_pred: numpy.ndarray of shape (n_samples)
        Score predictions

    Returns
    -------
    scores: dict
        Dictionary of metric scores
    """

    scores = {
        'concordance_index': float(concordance_index(
            event_times=y_true_time,
            predicted_scores=-y_pred,
            event_observed=y_true_event
        ))
    }

    return scores
