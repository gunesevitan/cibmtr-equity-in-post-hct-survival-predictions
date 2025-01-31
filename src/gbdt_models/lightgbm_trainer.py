import sys
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.append('..')
import settings
import preprocessing
import metrics
import visualization


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
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

    df = preprocessing.preprocess(
        df=df,
        kaplan_meier_estimator_directory=settings.MODELS / 'kaplan_meier_estimator',
        categorical_columns=categorical_columns,
        transformer_directory=settings.DATA / 'transformers',
        load_transformers=False
    )

    target = config['training']['target']
    features = config['training']['features']
    categorical_features = config['training']['categorical_features']
    folds = config['training']['folds']
    seeds = config['training']['seeds']
    rank_transform = config['training']['rank_transform']

    settings.logger.info(
        f'''
        Running LightGBM trainer in {args.mode} mode
        Processed Dataset Shape: {df.shape}
        Folds: {folds}
        Features: {json.dumps(features, indent=2)}
        Categorical Features: {json.dumps(categorical_features, indent=2)}
        Target: {target}
        '''
    )

    if args.mode == 'training':

        df_feature_importance_gain = pd.DataFrame(
            data=np.zeros((len(features), len(folds))),
            index=features,
            columns=folds
        )
        df_feature_importance_split = pd.DataFrame(
            data=np.zeros((len(features), len(folds))),
            index=features,
            columns=folds
        )
        scores = []

        for fold in folds:

            training_mask = df[f'fold{fold}'] == 0
            validation_mask = df[f'fold{fold}'] == 1

            df.loc[validation_mask, 'prediction'] = 0.

            settings.logger.info(
                f'''
                Fold: {fold} 
                Training: ({training_mask.sum()}) - Target Mean: {df.loc[training_mask, target].mean():.4f}
                Validation: ({validation_mask.sum()}) - Target Mean: {df.loc[validation_mask, target].mean():.4f}
                '''
            )

            for seed in seeds:

                training_dataset = lgb.Dataset(
                    df.loc[training_mask, features],
                    label=df.loc[training_mask, target],
                    categorical_feature=categorical_features
                )
                validation_dataset = lgb.Dataset(
                    df.loc[validation_mask, features],
                    label=df.loc[validation_mask, target],
                    categorical_feature=categorical_features
                )

                config['model_parameters']['seed'] = seed
                config['model_parameters']['feature_fraction_seed'] = seed
                config['model_parameters']['bagging_seed'] = seed
                config['model_parameters']['drop_seed'] = seed
                config['model_parameters']['data_random_seed'] = seed

                model = lgb.train(
                    params=config['model_parameters'],
                    train_set=training_dataset,
                    valid_sets=[training_dataset, validation_dataset],
                    num_boost_round=config['fit_parameters']['boosting_rounds'],
                    callbacks=[
                        lgb.log_evaluation(config['fit_parameters']['log_evaluation'])
                    ]
                )
                model_file_name = f'model_fold_{fold}_seed_{seed}.lgb'
                model.save_model(model_directory / model_file_name, num_iteration=None, start_iteration=0)
                settings.logger.info(f'{model_file_name} is saved to {model_directory}')

                df_feature_importance_gain[fold] += pd.Series((model.feature_importance(importance_type='gain') / len(seeds)), index=features)
                df_feature_importance_split[fold] += pd.Series((model.feature_importance(importance_type='split') / len(seeds)), index=features)

                validation_predictions = model.predict(df.loc[validation_mask, features], num_iteration=config['fit_parameters']['boosting_rounds'])
                if rank_transform:
                    validation_predictions = pd.Series(validation_predictions).rank(pct=True).values
                df.loc[validation_mask, 'prediction'] += validation_predictions / len(seeds)

            validation_scores = metrics.score(
                y_true_time=df.loc[validation_mask, 'efs_time'],
                y_true_event=df.loc[validation_mask, 'efs'],
                y_pred=df.loc[validation_mask, 'prediction'],
            )
            validation_race_group_concordance_indices = []
            for race_group, df_group in df.loc[validation_mask].groupby('race_group'):
                group_validation_scores = metrics.score(
                    y_true_time=df_group.loc[validation_mask, 'efs_time'],
                    y_true_event=df_group.loc[validation_mask, 'efs'],
                    y_pred=df_group.loc[validation_mask, 'prediction'],
                )
                validation_race_group_concordance_indices.append(group_validation_scores['concordance_index'])
                group_validation_scores = {
                    f'{"_".join(race_group.lower().split())}_{metric}': score
                    for metric, score in group_validation_scores.items()
                }
                validation_scores.update(group_validation_scores)
            validation_mean_concordance_index = np.mean(validation_race_group_concordance_indices)
            validation_std_concordance_index = np.std(validation_race_group_concordance_indices)
            validation_stratified_concordance_index = float(validation_mean_concordance_index - validation_std_concordance_index)
            validation_scores['mean_concordance_index'] = validation_mean_concordance_index
            validation_scores['std_concordance_index'] = validation_std_concordance_index
            validation_scores['stratified_concordance_index'] = validation_stratified_concordance_index

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
            Concordance Index {scores['concordance_index'].values.tolist()}
            Mean Concordance Index {scores['mean_concordance_index'].values.tolist()}
            Std Concordance Index {scores['std_concordance_index'].values.tolist()}
            Stratified Concordance Index {scores['stratified_concordance_index'].values.tolist()}
            '''
        )

        oof_mask = df['prediction'].notna()
        oof_scores = metrics.score(
            y_true_time=df.loc[oof_mask, 'efs_time'],
            y_true_event=df.loc[oof_mask, 'efs'],
            y_pred=-df.loc[oof_mask, 'prediction'],
        )
        oof_race_group_concordance_indices = []
        for race_group, df_group in df.loc[oof_mask].groupby('race_group'):
            group_oof_scores = metrics.score(
                y_true_time=df_group.loc[oof_mask, 'efs_time'],
                y_true_event=df_group.loc[oof_mask, 'efs'],
                y_pred=df_group.loc[oof_mask, 'prediction'],
            )
            oof_race_group_concordance_indices.append(group_oof_scores['concordance_index'])
            group_oof_scores = {
                f'{"_".join(race_group.lower().split())}_{metric}': score
                for metric, score in group_oof_scores.items()
            }
            oof_scores.update(group_oof_scores)
        oof_mean_concordance_index = np.mean(oof_race_group_concordance_indices)
        oof_std_concordance_index = np.std(oof_race_group_concordance_indices)
        oof_stratified_concordance_index = float(oof_mean_concordance_index - oof_std_concordance_index)
        oof_scores['mean_concordance_index'] = oof_mean_concordance_index
        oof_scores['std_concordance_index'] = oof_std_concordance_index
        oof_scores['stratified_concordance_index'] = oof_stratified_concordance_index
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
            title=f'LightGBM Model Scores of {len(folds)} Fold(s)',
            path=model_directory / 'scores.png'
        )
        settings.logger.info(f'Saved scores.png to {model_directory}')

        for importance_type, df_feature_importance in zip(['gain', 'split'], [df_feature_importance_gain, df_feature_importance_split]):
            df_feature_importance['mean'] = df_feature_importance[config['training']['folds']].mean(axis=1)
            df_feature_importance['std'] = df_feature_importance[config['training']['folds']].std(axis=1).fillna(0)
            df_feature_importance.sort_values(by='mean', ascending=False, inplace=True)
            visualization.visualize_feature_importance(
                df_feature_importance=df_feature_importance,
                title=f'LightGBM Feature Importance ({importance_type.capitalize()})',
                path=model_directory / f'feature_importance_{importance_type}.png'
            )
            settings.logger.info(f'Saved feature_importance_{importance_type}.png to {model_directory}')

        df.loc[:, 'prediction'].to_csv(model_directory / 'oof_predictions.csv', index=False)
        settings.logger.info(f'Saved oof_predictions.csv to {model_directory}')
