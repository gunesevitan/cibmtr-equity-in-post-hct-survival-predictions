import sys
from pathlib import Path
import yaml
import pandas as pd
from lifelines import NelsonAalenFitter
sys.path.append('..')
import settings
import visualization


if __name__ == '__main__':

    model_directory = Path(settings.MODELS / 'nelson_aalen')
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

        df.loc[validation_mask, 'oof_cumulative_hazard'] = 0.

        settings.logger.info(
            f'''
            Fold: {fold} 
            Training: ({training_mask.sum()})
            Validation: ({validation_mask.sum()})
            '''
        )

        naf = NelsonAalenFitter()
        naf.fit(df.loc[training_mask, 'efs_time'], df.loc[training_mask, 'efs'])
        df.loc[validation_mask, 'oof_cumulative_hazard'] = naf.cumulative_hazard_at_times(df.loc[validation_mask, 'efs_time']).values

        for race_group in race_groups:
            race_group_mask = df['race_group'] == race_group
            naf = NelsonAalenFitter()
            naf.fit(df.loc[training_mask & race_group_mask, 'efs_time'], df.loc[training_mask & race_group_mask, 'efs'])
            df.loc[validation_mask & race_group_mask, 'oof_race_group_cumulative_hazard'] = naf.cumulative_hazard_at_times(df.loc[validation_mask & race_group_mask, 'efs_time']).values

    naf = NelsonAalenFitter()
    naf.fit(df['efs_time'], df['efs'])
    df['cumulative_hazard'] = naf.cumulative_hazard_at_times(df['efs_time']).values
    
    visualization.visualize_cumulative_hazard(
        naf=naf,
        title='Nelson-Aalen Fitter Cumulative Hazard',
        path=model_directory / 'cumulative_hazard.png'
    )
    settings.logger.info(f'cumulative_hazard.png is saved to {model_directory}')

    for race_group, df_group in df.groupby('race_group'):
        naf = NelsonAalenFitter()
        naf.fit(df_group['efs_time'], df_group['efs'])
        df_group['race_group_cumulative_hazard'] = naf.cumulative_hazard_at_times(df_group['efs_time']).values
        group_idx = df_group.index
        df.loc[group_idx, 'race_group_cumulative_hazard'] = df_group['race_group_cumulative_hazard']

    target_columns = [
        'cumulative_hazard', 'race_group_cumulative_hazard',
        'oof_cumulative_hazard', 'oof_race_group_cumulative_hazard',
    ]
    df.loc[:, target_columns].to_csv(model_directory / 'targets.csv', index=False)
    settings.logger.info(f'targets.csv is saved to {model_directory}')
