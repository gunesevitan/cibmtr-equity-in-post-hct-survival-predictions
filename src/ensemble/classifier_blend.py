import sys
import json
import pandas as pd

sys.path.append('..')
import settings
import metrics


if __name__ == '__main__':

    df = pd.read_parquet(settings.DATA / 'datasets' / 'dataset.parquet')

    df_lgb_efs_classifier = pd.read_csv(settings.MODELS / 'lightgbm_efs_classifier' / 'oof_predictions.csv').rename(columns={'prediction': 'lgb_efs_prediction'})
    df_cb_efs_classifier = pd.read_csv(settings.MODELS / 'catboost_efs_classifier' / 'oof_predictions.csv').rename(columns={'prediction': 'cb_efs_prediction'})
    df = pd.concat((
        df,
        df_lgb_efs_classifier,
        df_cb_efs_classifier
    ), axis=1)

    prediction_columns = ['lgb_efs_prediction', 'cb_efs_prediction']


    for column in prediction_columns:

        oof_mask = df[column].notna()
        oof_scores = metrics.classification_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            event_column='efs',
            prediction_column=column
        )
        settings.logger.info(f'{column} OOF Scores: {json.dumps(oof_scores, indent=2)}')

    df['efs_prediction'] = df['lgb_efs_prediction'] * 0.4 + \
                           df['cb_efs_prediction'] * 0.6

    oof_mask = df['efs_prediction'].notna()
    oof_scores = metrics.classification_score(
        df=df.loc[oof_mask],
        group_column='race_group',
        event_column='efs',
        prediction_column='efs_prediction'
    )
    settings.logger.info(f'Classifier Blend OOF Scores: {json.dumps(oof_scores, indent=2)}')

    df.loc[:, 'efs_prediction'].to_csv(settings.MODELS / 'ensemble' / 'efs_prediction.csv', index=False)
    settings.logger.info(f'Saved efs_prediction.csv to {settings.DATA}')
