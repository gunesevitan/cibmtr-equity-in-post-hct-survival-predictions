import sys
import json
import pandas as pd

sys.path.append('..')
import settings
import metrics
import visualization


def load_oof_predictions(model_directory):

    """
    Load oof predictions of a trained model from the given model directory

    Parameters
    ----------
    model_directory: str or pathlib.Path
        Model directory relative to root/models

    Returns
    -------
    df_oof_predictions: pandas.DataFrame
        Dataframe with oof predictions
    """

    df_oof_predictions = pd.read_csv(settings.MODELS / model_directory / 'oof_predictions.csv').rename(columns={
        'prediction': f'{str(model_directory)}_prediction'
    })

    return df_oof_predictions


if __name__ == '__main__':

    model_directory = settings.MODELS / 'ensemble'
    model_directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(settings.DATA / 'datasets' / 'dataset.parquet')
    settings.logger.info(f'Raw Dataset Shape {df.shape}')

    df_logistic_regression_efs = load_oof_predictions('logistic_regression_efs')
    df_knn_classifier_efs = load_oof_predictions('knn_classifier_efs')
    df = pd.concat((
        df,
        df_logistic_regression_efs,
        df_knn_classifier_efs
    ), axis=1)

    prediction_columns = ['logistic_regression_efs_prediction', 'knn_classifier_efs_prediction']
    for column in prediction_columns:
        oof_mask = df[column].notna()
        oof_scores = metrics.classification_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            event_column='efs',
            prediction_column=column
        )
        settings.logger.info(f'{column} OOF Scores: {json.dumps(oof_scores, indent=2)}')

    df['efs_prediction'] = df['logistic_regression_efs_prediction'] * 0.975 + \
                           df['knn_classifier_efs_prediction'] * 0.025
    df['efs_prediction_error'] = df['efs'] - df['efs_prediction']

    oof_mask = df['efs_prediction'].notna()
    oof_scores = metrics.classification_score(
        df=df.loc[oof_mask],
        group_column='race_group',
        event_column='efs',
        prediction_column='efs_prediction'
    )
    settings.logger.info(f'EFS Classifier Blend OOF Scores: {json.dumps(oof_scores, indent=2)}')

    oof_curves = metrics.classification_curves(
        df=df.loc[oof_mask],
        event_column='efs',
        prediction_column='efs_prediction',
    )

    visualization.visualize_roc_curves(
        roc_curves=[oof_curves['roc']],
        title=f'EFS Classifier Blend Validation ROC Curves',
        path=model_directory / 'efs_classifier_blend_roc_curves.png'
    )
    settings.logger.info(f'Saved efs_classifier_blend_roc_curves.png to {model_directory}')

    visualization.visualize_pr_curves(
        pr_curves=[oof_curves['pr']],
        title=f'EFS Classifier Blend Validation PR Curves',
        path=model_directory / 'efs_classifier_blend_pr_curves.png'
    )
    settings.logger.info(f'Saved efs_classifier_blend_roc_curves.png to {model_directory}')

    prediction_columns = ['efs_prediction', 'efs_prediction_error']
    df.loc[:, prediction_columns].to_csv(settings.MODELS / 'ensemble' / 'efs_predictions.csv', index=False)
    settings.logger.info(f'Saved efs_predictions.csv to {model_directory}')
