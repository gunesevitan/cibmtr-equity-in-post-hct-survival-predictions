import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')
import settings
import metrics


def visualize_target(df, path=None):

    """
    Visualize predictions

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
    axes[0].hist(df.loc[df['efs'] == 0, 'efs_prediction'], bins=32, alpha=0.6, label='EFS 0 Predictions')
    axes[0].hist(df.loc[df['efs'] == 1, 'efs_prediction'], bins=32, alpha=0.6, label='EFS 1 Predictions')
    axes[1].hist(df.loc[df['efs'] == 0, 'efs'], bins=32, alpha=0.6, label='EFS 0')
    axes[1].hist(df.loc[df['efs'] == 1, 'efs'], bins=32, alpha=0.6, label='EFS 1')
    for ax in axes:
        ax.tick_params(axis='x', labelsize=15, pad=10)
        ax.tick_params(axis='y', labelsize=15, pad=10)
    axes[0].legend(loc='best', prop={'size': 18})
    axes[1].legend(loc='best', prop={'size': 18})
    axes[0].set_title('efs_time Histogram', size=20, pad=15)
    axes[1].set_title(f'Target Histogram', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


if __name__ == '__main__':

    df = pd.read_parquet(settings.DATA / 'datasets' / 'dataset.parquet')

    df_lgb_efs_classifier = pd.read_csv(settings.MODELS / 'lightgbm_efs_binary_classifier' / 'oof_predictions.csv').rename(columns={'prediction': 'lgb_efs_prediction'})
    df_cb_efs_classifier = pd.read_csv(settings.MODELS / 'catboost_efs_binary_classifier' / 'oof_predictions.csv').rename(columns={'prediction': 'cb_efs_prediction'})
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

    df['efs_prediction'] = df['lgb_efs_prediction'] * 0.5 + \
                           df['cb_efs_prediction'] * 0.5
    df['efs_prediction_error'] = df['efs'] - df['efs_prediction']

    visualize_target(
        df=df,
        path=settings.MODELS / 'ensemble' / 'efs_prediction.png'
    )

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


