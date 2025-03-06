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

    df_hgbm_efs = load_oof_predictions('hist_gradient_boosting_classifier_efs')
    df_lgb_efs = load_oof_predictions('lightgbm_efs_binary_classifier')
    df_xgb_efs = load_oof_predictions('xgboost_efs_binary_classifier')
    df_cb_efs = load_oof_predictions('catboost_efs_binary_classifier')
    df_mlp_sparse_efs = load_oof_predictions('mlp_sparse_efs_binary_classifier')
    df_mlp_embeddings_efs = load_oof_predictions('mlp_embeddings_efs_binary_classifier')
    df = pd.concat((
        df,
        df_hgbm_efs,
        df_lgb_efs,
        df_xgb_efs,
        df_cb_efs,
        df_mlp_sparse_efs,
        df_mlp_embeddings_efs
    ), axis=1)

    prediction_columns = [
        'hist_gradient_boosting_classifier_efs_prediction',
        'lightgbm_efs_binary_classifier_prediction',
        'xgboost_efs_binary_classifier_prediction',
        'catboost_efs_binary_classifier_prediction',
        'mlp_sparse_efs_binary_classifier_prediction',
        'mlp_embeddings_efs_binary_classifier_prediction'
    ]

    visualization.visualize_correlations(
        df=df,
        columns=prediction_columns,
        title='Classifier Predictions Correlations',
        path=model_directory / 'classifier_predictions_correlations.png'
    )

    for column in prediction_columns:
        oof_mask = df[column].notna()
        oof_scores = metrics.classification_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            event_column='efs',
            prediction_column=column
        )
        settings.logger.info(f'{column} OOF Scores: {json.dumps(oof_scores, indent=2)}')

    df['efs_prediction'] = df['hist_gradient_boosting_classifier_efs_prediction'] * 0.45 + \
                           df['lightgbm_efs_binary_classifier_prediction'] * 0.05 + \
                           df['xgboost_efs_binary_classifier_prediction'] * 0.05 + \
                           df['catboost_efs_binary_classifier_prediction'] * 0.2 + \
                           df['mlp_sparse_efs_binary_classifier_prediction'] * 0.125 + \
                           df['mlp_embeddings_efs_binary_classifier_prediction'] * 0.125

    df.loc[df['race_group'] == 'More than one race', 'efs_prediction'] *= 1.
    df.loc[df['race_group'] == 'Asian', 'efs_prediction'] *= 1.
    df.loc[df['race_group'] == 'White', 'efs_prediction'] *= 1.
    df.loc[df['race_group'] == 'American Indian or Alaska Native', 'efs_prediction'] *= 1.
    df.loc[df['race_group'] == 'Native Hawaiian or other Pacific Islander', 'efs_prediction'] *= 1.
    df.loc[df['race_group'] == 'Black or African-American', 'efs_prediction'] *= 1.
    #df.loc[df['efs_prediction'] >= 0.97, 'efs_prediction'] = 1
    #df.loc[df['efs_prediction'] <= 0.03, 'efs_prediction'] = 0

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

    prediction_columns = ['efs_prediction'] + prediction_columns
    df.loc[:, prediction_columns].to_csv(settings.MODELS / 'ensemble' / 'efs_predictions.csv', index=False)
    settings.logger.info(f'Saved efs_predictions.csv to {model_directory}')
