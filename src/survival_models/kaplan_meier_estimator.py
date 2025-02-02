import sys
from pathlib import Path
import pickle
import yaml
import pandas as pd
from lifelines import KaplanMeierFitter

sys.path.append('..')
import settings
import visualization


if __name__ == '__main__':

    model_directory = Path(settings.MODELS / 'kaplan_meier_estimator')
    model_directory.mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df = pd.read_parquet(settings.DATA / 'datasets' / config['dataset']['name'])
    df = pd.concat((
        df,
        pd.read_csv(settings.DATA / 'folds.csv')
    ), axis=1, ignore_index=False)
    settings.logger.info(f'Raw Dataset Shape {df.shape}')

    race_groups = df['race_group'].unique()

    for fold in range(1, 6):

        training_mask = df[f'fold{fold}'] == 0
        validation_mask = df[f'fold{fold}'] == 1

        df.loc[validation_mask, 'kmf_oof_survival_probability'] = 0.

        settings.logger.info(
            f'''
            Fold: {fold} 
            Training: ({training_mask.sum()})
            Validation: ({validation_mask.sum()})
            '''
        )

        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[training_mask, 'efs_time'], df.loc[training_mask, 'efs'])
        df.loc[validation_mask, 'kmf_oof_survival_probability'] = kmf.survival_function_at_times(df.loc[validation_mask, 'efs_time']).values

        for race_group in race_groups:

            race_group_mask = df['race_group'] == race_group

            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[training_mask & race_group_mask, 'efs_time'], df.loc[training_mask & race_group_mask, 'efs'])
            df.loc[validation_mask & race_group_mask, 'kmf_oof_race_group_survival_probability'] = kmf.survival_function_at_times(df.loc[validation_mask & race_group_mask, 'efs_time']).values

    kmf = KaplanMeierFitter()
    kmf.fit(df['efs_time'], df['efs'])

    with open(model_directory / 'kaplan_meier_estimator.pickle', mode='wb') as f:
        pickle.dump(kmf, f)
    settings.logger.info(f'kaplan_meier_estimator.pickle is saved to {model_directory}')

    visualization.visualize_survival_probabilities(
        kmf=kmf,
        title='Kaplan-Meier Estimator Survival Probabilities',
        path=model_directory / 'survival_probabilities.png'
    )
    settings.logger.info(f'survival_probabilities.png is saved to {model_directory}')
    df['kmf_survival_probability'] = kmf.survival_function_at_times(df['efs_time']).values

    for race_group, df_group in df.groupby('race_group'):

        kmf = KaplanMeierFitter()
        kmf.fit(df_group['efs_time'], df_group['efs'])

        file_name = f'kaplan_meier_estimator_{"_".join(str(race_group).lower().split())}.pickle'
        with open(model_directory / file_name, mode='wb') as f:
            pickle.dump(kmf, f)
        settings.logger.info(f'{file_name} is saved to {model_directory}')

        df_group['kmf_race_group_survival_probability'] = kmf.survival_function_at_times(df_group['efs_time']).values
        group_idx = df_group.index
        df.loc[group_idx, 'kmf_race_group_survival_probability'] = df_group['kmf_race_group_survival_probability']

    df.loc[:, [
        'kmf_survival_probability', 'kmf_race_group_survival_probability',
        'kmf_oof_survival_probability', 'kmf_oof_race_group_survival_probability',
    ]].to_csv(model_directory / 'survival_probabilities.csv', index=False)
    settings.logger.info(f'survival_probabilities.csv is saved to {model_directory}')
