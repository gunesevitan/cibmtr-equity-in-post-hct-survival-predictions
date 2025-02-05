import sys
import argparse
from pathlib import Path
import pickle
import yaml
import json
import numpy as np
import pandas as pd
import sksurv.ensemble
from sksurv.util import Surv

sys.path.append('..')
import settings
import tree_based_model_preprocessing
import metrics
import visualization


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df = pd.read_parquet(settings.DATA / 'datasets' / config['dataset']['name'])
    df = pd.concat((
        df,
        pd.read_csv(settings.DATA / 'folds.csv')
    ), axis=1, ignore_index=False)
    settings.logger.info(f'Raw Dataset Shape {df.shape}')

    categorical_columns = config['dataset']['categorical_columns']

    df = tree_based_model_preprocessing.preprocess(
        df=df,
        categorical_columns=categorical_columns,
        transformer_directory=settings.DATA / 'tree_based_model_transformers',
        load_transformers=False
    )

    folds = config['training']['folds']
    features = config['training']['features']
    time_column = config['training']['time']
    event_column = config['training']['event']

    settings.logger.info(
        f'''
        Running sksurv linear model trainer {config['model_class']}
        Dataset Shape: {df.shape}
        Folds: {folds}
        Time: {time_column}
        Event: {event_column}
        Features: {json.dumps(features, indent=2)}
        '''
    )

    scores = []

    for fold in folds:

        training_mask = df[f'fold{fold}'] == 0
        validation_mask = df[f'fold{fold}'] == 1

        df.loc[validation_mask, 'prediction'] = 0.

        settings.logger.info(
            f'''
            Fold: {fold} 
            Training: ({np.sum(training_mask)})
            Validation: ({np.sum(validation_mask)})
            '''
        )

        model = getattr(sksurv.ensemble, config['model_class'])(**config['model_parameters'])
        model.fit(
            X=df.loc[training_mask, features],
            y=Surv.from_dataframe(event='efs', time='efs_time', data=df.loc[training_mask]),
        )
        model_file_name = f'model_fold_{fold}.pickle'
        with open(model_directory / model_file_name, mode='wb') as f:
            pickle.dump(model, f)
        settings.logger.info(f'{model_file_name} is saved to {model_directory}')

        df.loc[validation_mask, 'prediction'] = model.predict(df.loc[validation_mask, features])
        validation_scores = metrics.score(
            df=df.loc[validation_mask],
            group_column='race_group',
            time_column='efs_time',
            event_column='efs',
            prediction_column='prediction'
        )
        settings.logger.info(f'Fold: {fold} - Validation Scores: {json.dumps(validation_scores, indent=2)}')
        scores.append(validation_scores)

    scores = pd.DataFrame(scores)
    settings.logger.info(
        f'''
        Mean Validation Scores
        ----------------------
        {json.dumps(scores.mean(axis=0).to_dict(), indent=2)}

        Standard Deviations
        -------------------
        ±{json.dumps(scores.std(axis=0).to_dict(), indent=2)}

        Fold Scores
        -----------
        Micro Concordance Index: {scores['micro_concordance_index'].values.tolist()}
        Macro Concordance Index: {scores['macro_concordance_index'].values.tolist()}
        Std Concordance Index: {scores['std_concordance_index'].values.tolist()}
        Stratified Concordance Index: {scores['stratified_concordance_index'].values.tolist()}
        '''
    )

    oof_mask = df['prediction'].notna()
    oof_scores = metrics.score(
        df=df.loc[oof_mask],
        group_column='race_group',
        time_column='efs_time',
        event_column='efs',
        prediction_column='prediction'
    )
    settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

    scores = pd.concat((
        scores,
        pd.DataFrame([oof_scores])
    )).reset_index(drop=True)
    scores['fold'] = folds + ['OOF']
    scores = scores[scores.columns.tolist()[::-1]]
    scores.to_csv(model_directory / 'scores.csv', index=False)
    settings.logger.info(f'scores.csv is saved to {model_directory}')

    visualization.visualize_scores(
        scores=scores,
        title=f'lifelines Regression Model Scores of {len(folds)} Fold(s)',
        path=model_directory / 'scores.png'
    )
    settings.logger.info(f'Saved scores.png to {model_directory}')

    df.loc[:, 'prediction'].to_csv(model_directory / 'oof_predictions.csv', index=False)
    settings.logger.info(f'oof_predictions.csv is saved to {model_directory}')
