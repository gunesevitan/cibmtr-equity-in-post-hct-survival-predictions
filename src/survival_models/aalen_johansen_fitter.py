import sys
from pathlib import Path
import pickle
import yaml
import pandas as pd
from lifelines import AalenJohansenFitter

sys.path.append('..')
import settings
import visualization


if __name__ == '__main__':

    model_directory = Path(settings.MODELS / 'aalen_johansen_fitter')
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

        df.loc[validation_mask, 'ajf_oof_survival_probability'] = 0.

        settings.logger.info(
            f'''
            Fold: {fold} 
            Training: ({training_mask.sum()})
            Validation: ({validation_mask.sum()})
            '''
        )

        ajf = AalenJohansenFitter()
        ajf.fit(df.loc[training_mask, 'efs_time'], df.loc[training_mask, 'efs'], event_of_interest=1)
        df.loc[validation_mask, 'ajf_oof_survival_probability'] = 1 - ajf.predict(df.loc[validation_mask, 'efs_time']).values

        for race_group in race_groups:

            race_group_mask = df['race_group'] == race_group

            ajf = AalenJohansenFitter()
            ajf.fit(df.loc[training_mask & race_group_mask, 'efs_time'], df.loc[training_mask & race_group_mask, 'efs'], event_of_interest=1)
            df.loc[validation_mask & race_group_mask, 'ajf_oof_race_group_survival_probability'] = 1 - ajf.predict(df.loc[validation_mask & race_group_mask, 'efs_time']).values

    ajf = AalenJohansenFitter()
    ajf.fit(df['efs_time'], df['efs'], event_of_interest=1)
    df['ajf_survival_probability'] = 1 - ajf.predict(df['efs_time']).values

    with open(model_directory / 'aalen_johansen_fitter.pickle', mode='wb') as f:
        pickle.dump(ajf, f)
    settings.logger.info(f'aalen_johansen_fitter.pickle is saved to {model_directory}')

    visualization.visualize_cumulative_density(
        ajf=ajf,
        title='Aalen-Johansen Fitter - Survival Probabilities',
        path=model_directory / 'survival_probabilities.png'
    )
    settings.logger.info(f'survival_probabilities.png is saved to {model_directory}')

    for race_group, df_group in df.groupby('race_group'):

        ajf = AalenJohansenFitter()
        ajf.fit(df_group['efs_time'], df_group['efs'], event_of_interest=1)
        df_group['ajf_race_group_survival_probability'] = 1 - ajf.predict(df_group['efs_time']).values
        group_idx = df_group.index
        df.loc[group_idx, 'ajf_race_group_survival_probability'] = df_group['ajf_race_group_survival_probability']

        file_name = f'aalen_johansen_fitter_{"_".join(str(race_group).lower().split())}.pickle'
        with open(model_directory / file_name, mode='wb') as f:
            pickle.dump(ajf, f)
        settings.logger.info(f'{file_name} is saved to {model_directory}')

    df.loc[:, [
        'ajf_survival_probability', 'ajf_race_group_survival_probability',
        'ajf_oof_survival_probability', 'ajf_oof_race_group_survival_probability',
    ]].to_csv(model_directory / 'survival_probabilities.csv', index=False)
    settings.logger.info(f'survival_probabilities.csv is saved to {model_directory}')
