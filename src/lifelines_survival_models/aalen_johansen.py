import sys
from pathlib import Path
import yaml
import pandas as pd
from lifelines import AalenJohansenFitter

sys.path.append('..')
import settings
import visualization


if __name__ == '__main__':

    model_directory = Path(settings.MODELS / 'aalen_johansen')
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

        df.loc[validation_mask, 'oof_cumulative_density'] = 0.

        settings.logger.info(
            f'''
            Fold: {fold} 
            Training: ({training_mask.sum()})
            Validation: ({validation_mask.sum()})
            '''
        )

        ajf = AalenJohansenFitter()
        ajf.fit(df.loc[training_mask, 'efs_time'], df.loc[training_mask, 'efs'], event_of_interest=0)
        df.loc[validation_mask, 'oof_cumulative_density'] = ajf.predict(df.loc[validation_mask, 'efs_time']).values

        for race_group in race_groups:
            race_group_mask = df['race_group'] == race_group
            ajf = AalenJohansenFitter()
            ajf.fit(df.loc[training_mask & race_group_mask, 'efs_time'], df.loc[training_mask & race_group_mask, 'efs'], event_of_interest=0)
            df.loc[validation_mask & race_group_mask, 'oof_race_group_cumulative_density'] = ajf.predict(df.loc[validation_mask & race_group_mask, 'efs_time']).values

    ajf = AalenJohansenFitter()
    ajf.fit(df['efs_time'], df['efs'], event_of_interest=0)
    df['cumulative_density'] = ajf.predict(df['efs_time']).values

    visualization.visualize_cumulative_density(
        ajf=ajf,
        title='Aalen-Johansen Fitter - Cumulative Density',
        path=model_directory / 'cumulative_density.png'
    )
    settings.logger.info(f'cumulative_density.png is saved to {model_directory}')

    for race_group, df_group in df.groupby('race_group'):
        ajf = AalenJohansenFitter()
        ajf.fit(df_group['efs_time'], df_group['efs'], event_of_interest=0)
        df_group['race_group_cumulative_density'] = ajf.predict(df_group['efs_time']).values
        group_idx = df_group.index
        df.loc[group_idx, 'race_group_cumulative_density'] = df_group['race_group_cumulative_density']

    target_columns = [
        'cumulative_density', 'race_group_cumulative_density',
        'oof_cumulative_density', 'oof_race_group_cumulative_density',
    ]
    df.loc[:, target_columns].to_csv(model_directory / 'targets.csv', index=False)
    settings.logger.info(f'targets.csv is saved to {model_directory}')
