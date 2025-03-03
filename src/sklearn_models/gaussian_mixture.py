import sys
import argparse
from pathlib import Path
import pickle
import yaml
import json
import numpy as np
import pandas as pd
import sklearn.mixture

sys.path.append('..')
import settings
import preprocessing
import metrics
import visualization


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df = pd.read_parquet(settings.DATA / 'datasets' / config['dataset']['name'])
    df = pd.concat((
        df,
        pd.read_csv(settings.DATA / 'folds.csv')
    ), axis=1, ignore_index=False)
    settings.logger.info(f'Raw Dataset Shape {df.shape}')

    df = preprocessing.preprocess(
        df=df,
        categorical_columns=config['dataset']['categorical_columns'],
        continuous_columns=config['dataset']['continuous_columns'],
        transformer_directory=settings.DATA / 'linear_model_transformers',
        load_transformers=False,
        efs_predictions_path=config['dataset']['efs_predictions_path'],
        kaplan_meier_targets_path=config['dataset']['kaplan_meier_targets_path'],
        nelson_aalen_targets_path=config['dataset']['nelson_aalen_targets_path'],
        efs_weight=config['training']['efs_weight']
    )

    task = config['training']['task']
    folds = config['training']['folds']
    target = config['training']['target']
    features = config['training']['features']

    settings.logger.info(
        f'''
        Running Trainer for {config['model_class']}
        Dataset Shape: {df.shape}
        Folds: {folds}
        Features: {json.dumps(features, indent=2)}
        Target: {target}
        '''
    )

    df_coefficients = pd.DataFrame(
        data=np.zeros((len(features), len(folds))),
        index=features,
        columns=folds
    )

    model = getattr(sklearn.mixture, config['model_class'])(**config['model_parameters'])
    model.fit(
        X=df.loc[:, features],
        y=df.loc[:, target]
    )

    model_file_name = f'model.pickle'
    with open(model_directory / model_file_name, mode='wb') as f:
        pickle.dump(model, f)
    settings.logger.info(f'{model_file_name} is saved to {model_directory}')

    validation_predictions = model.predict_proba(df.loc[:, features])[:, 0]
    df.loc[:, 'prediction'] = validation_predictions

    oof_mask = df['prediction'].notna()
    if task == 'ranking':
        oof_scores = metrics.ranking_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column='efs_time',
            event_column='efs',
            prediction_column='prediction'
        )
    elif task == 'classification':
        oof_scores = metrics.classification_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            event_column='efs',
            prediction_column='prediction',
            weight_column=None
        )
    elif task == 'regression':
        oof_scores = metrics.regression_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column=target,
            prediction_column='prediction'
        )
    else:
        raise ValueError(f'Invalid task type {task}')

    settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

    df.loc[:, 'prediction'].to_csv(model_directory / 'oof_predictions.csv', index=False)
    settings.logger.info(f'oof_predictions.csv is saved to {model_directory}')
