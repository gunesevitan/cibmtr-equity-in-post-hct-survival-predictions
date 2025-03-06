import sys
import json
import numpy as np
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

    df['efs_prediction'] = pd.read_csv(settings.MODELS / 'ensemble' / 'efs_predictions.csv')['efs_prediction']

    df_hgbm_log_efs_time = load_oof_predictions('hist_gradient_boosting_regressor_log_efs_time')
    df_hgbm_log_km_proba = load_oof_predictions('hist_gradient_boosting_regressor_log_km_proba')
    df_hgbm_na_cum_hazard = load_oof_predictions('hist_gradient_boosting_regressor_na_cum_hazard')
    df_xgb_log_efs_time = load_oof_predictions('xgboost_log_efs_time_regressor')
    df_xgb_log_km_proba = load_oof_predictions('xgboost_log_km_proba_regressor')

    df = pd.concat((
        df,
        df_hgbm_log_efs_time,
        df_hgbm_log_km_proba,
        df_hgbm_na_cum_hazard,
        df_xgb_log_efs_time,
        df_xgb_log_km_proba
    ), axis=1)

    prediction_columns = [
        'hist_gradient_boosting_regressor_log_efs_time_prediction',
        'hist_gradient_boosting_regressor_log_km_proba_prediction',
        'hist_gradient_boosting_regressor_na_cum_hazard_prediction',
        'xgboost_log_efs_time_regressor_prediction',
        'xgboost_log_km_proba_regressor_prediction'
    ]

    visualization.visualize_correlations(
        df=df,
        columns=prediction_columns,
        title='Ranking Predictions Correlations',
        path=model_directory / 'ranking_predictions_correlations.png'
    )

    for column in prediction_columns:
        oof_mask = df[column].notna()
        oof_scores = metrics.ranking_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column='efs_time',
            event_column='efs',
            prediction_column=column
        )
        settings.logger.info(f'{column} OOF Scores: {json.dumps(oof_scores, indent=2)}')

    df['prediction'] = df['hist_gradient_boosting_regressor_log_efs_time_prediction'] * 0.29 + \
                       df['hist_gradient_boosting_regressor_log_km_proba_prediction'] * 0.29 + \
                       df['hist_gradient_boosting_regressor_na_cum_hazard_prediction'] * 0.29 + \
                       df['xgboost_log_efs_time_regressor_prediction'] * 0.06 + \
                       df['xgboost_log_km_proba_regressor_prediction'] * 0.07

    oof_mask = df['prediction'].notna()
    oof_scores = metrics.ranking_score(
        df=df.loc[oof_mask],
        group_column='race_group',
        time_column='efs_time',
        event_column='efs',
        prediction_column='prediction'
    )
    settings.logger.info(f'Blend OOF Scores: {json.dumps(oof_scores, indent=2)}')

    step = 0.01
    for threshold in np.arange(0, 0.33, step):
        df.loc[(df['efs_prediction'] >= threshold) & (
                    df['efs_prediction'] < threshold + step), 'prediction'] *= threshold + step

    oof_scores = metrics.ranking_score(
        df=df.loc[oof_mask],
        group_column='race_group',
        time_column='efs_time',
        event_column='efs',
        prediction_column='prediction'
    )
    settings.logger.info(f'Postprocessed Blend OOF Scores: {json.dumps(oof_scores, indent=2)}')

    df.loc[:, 'prediction'].to_csv(settings.MODELS / 'ensemble' / 'ranking_predictions.csv', index=False)
    settings.logger.info(f'Saved ranking_predictions.csv to {model_directory}')
