import sys
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
import optuna

sys.path.append('..')
import settings
import preprocessing
import metrics


def objective(trial):

    parameters = {
        'loss': trial.suggest_categorical('loss', ['log_loss']),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, step=0.005),
        'max_iter': trial.suggest_int('max_iter', 100, 500, step=10),
        'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]),
        'max_depth': trial.suggest_int('max_depth', 1, 8),
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]),
        'l2_regularization': trial.suggest_categorical('l2_regularization', [0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10]),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0, step=0.05),
        'max_bins': trial.suggest_categorical('max_bins', [127, 255]),
        'interaction_cst': trial.suggest_categorical('interaction_cst', ['pairwise', 'no_interactions', None]),
        'warm_start': False,
        'early_stopping': False,
        'scoring': 'loss',
        'validation_fraction': None,
        'random_state': None,
        'categorical_features': [
            'dri_score_encoded', 'psych_disturb_encoded', 'cyto_score_encoded', 'diabetes_encoded', 'hla_match_c_high_encoded',
            'hla_high_res_8_encoded', 'tbi_status_encoded', 'arrhythmia_encoded', 'hla_low_res_6_encoded', 'graft_type_encoded',
            'vent_hist_encoded', 'renal_issue_encoded', 'pulm_severe_encoded', 'prim_disease_hct_encoded', 'hla_high_res_6_encoded',
            'cmv_status_encoded', 'hla_high_res_10_encoded', 'hla_match_dqb1_high_encoded', 'tce_imm_match_encoded', 'hla_nmdp_6_encoded',
            'hla_match_c_low_encoded', 'rituximab_encoded', 'hla_match_drb1_low_encoded', 'hla_match_dqb1_low_encoded', 'prod_type_encoded',
            'cyto_score_detail_encoded', 'conditioning_intensity_encoded', 'ethnicity_encoded', 'year_hct_encoded', 'obesity_encoded',
            'mrd_hct_encoded', 'in_vivo_tcd_encoded', 'tce_match_encoded', 'hla_match_a_high_encoded', 'hepatic_severe_encoded',
            'prior_tumor_encoded', 'hla_match_b_low_encoded', 'peptic_ulcer_encoded', 'hla_match_a_low_encoded', 'gvhd_proph_encoded',
            'rheum_issue_encoded', 'sex_match_encoded', 'hla_match_b_high_encoded', 'race_group_encoded', 'comorbidity_score_encoded',
            'karnofsky_score_encoded', 'hepatic_mild_encoded', 'tce_div_match_encoded', 'donor_related_encoded', 'melphalan_dose_encoded',
            'hla_low_res_8_encoded', 'cardiac_encoded', 'hla_match_drb1_high_encoded', 'pulm_moderate_encoded', 'hla_low_res_10_encoded',
        ]
    }

    df['prediction'] = 0.

    for fold in folds:

        training_mask = df[f'fold{fold}'] == 0
        validation_mask = df[f'fold{fold}'] == 1

        if config['training']['two_stage']:
            training_mask = training_mask & (df['efs'] == 1)

        for seed in seeds:

            parameters['random_state'] = seed

            model = eval(config['model_class'])(**parameters)
            model.fit(
                X=df.loc[training_mask, features],
                y=df.loc[training_mask, target],
                sample_weight=df.loc[training_mask, 'weight'] if config['training']['sample_weight'] else None
            )

            if task == 'classification':
                validation_predictions = model.predict_proba(df.loc[validation_mask, features])[:, 1]
            else:
                validation_predictions = model.predict(df.loc[validation_mask, features])

            if config['training']['two_stage']:
                if config['training']['target'] in ['log_efs_time']:
                    validation_predictions = df.loc[validation_mask, 'efs_prediction'] / np.exp(validation_predictions)
                elif config['training']['target'] in ['efs_time']:
                    df.loc[validation_mask, 'reg_1_prediction'] = validation_predictions
                    validation_predictions = df.loc[validation_mask, 'efs_prediction'] / validation_predictions
                elif config['training']['target'] in ['log_km_survival_probability', 'na_cumulative_hazard', 'gg_cumulative_hazard']:
                    validation_predictions = df.loc[validation_mask, 'efs_prediction'] * np.exp(validation_predictions)

            if config['training']['rank_transform']:
                validation_predictions = pd.Series(validation_predictions).rank(pct=True).values

            df.loc[validation_mask, 'prediction'] = validation_predictions

    oof_mask = df['prediction'].notna()
    if task == 'ranking':
        oof_scores = metrics.ranking_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column='efs_time',
            event_column='efs',
            prediction_column='prediction'
        )
        score = oof_scores['stratified_concordance_index']
    elif task == 'classification':
        oof_scores = metrics.classification_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            event_column='efs',
            prediction_column='prediction'
        )
        score = oof_scores['log_loss']
    elif task == 'regression':
        oof_scores = metrics.regression_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column=target,
            prediction_column='prediction'
        )
        score = oof_scores['mean_squared_error']
    else:
        raise ValueError(f'Invalid task type {task}')

    return score


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
    seeds = config['training']['seeds']

    try:
        storage = f'sqlite:///{model_directory}/study.db'
        study = optuna.create_study(
            study_name=f'{model_directory.name}_study',
            storage=storage,
            load_if_exists=True,
            direction='minimize'
        )
        study.optimize(objective, n_trials=300)
    except KeyboardInterrupt:
        settings.logger.info('Interrupted')
    finally:
        df_study = study.trials_dataframe().dropna()
        df_study = df_study.sort_values(by='value', ascending=False).drop_duplicates(subset='value', keep='first').reset_index(drop=True)
        df_study.to_csv(model_directory / 'study.csv', index=False)
        best_parameters = study.best_params
        with open(model_directory / 'best_parameters.json', mode='w') as f:
            json.dump(best_parameters, f, indent=2, ensure_ascii=False)
        settings.logger.info(f'Saved best_parameters.json to {model_directory}')
