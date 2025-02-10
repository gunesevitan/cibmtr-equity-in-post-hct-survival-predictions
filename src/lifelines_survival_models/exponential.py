import sys
from pathlib import Path
import yaml
import pandas as pd
from lifelines import ExponentialFitter

sys.path.append('..')
import settings
import visualization


if __name__ == '__main__':

    model_directory = Path(settings.MODELS / 'exponential')
    model_directory.mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df = pd.read_parquet(settings.DATA / 'datasets' / config['dataset']['name'])
    df = pd.concat((
        df,
        pd.read_csv(settings.DATA / 'folds.csv')
    ), axis=1, ignore_index=False)
    settings.logger.info(f'Raw Dataset Shape {df.shape}')

    race_groups = df['race_group'].unique()

    model = ExponentialFitter()
    model.fit(df['efs_time'], df['efs'])
    df['survival_probability'] = model.survival_function_at_times(df['efs_time'].values).values
    df['cumulative_hazard'] = model.cumulative_hazard_at_times(df['efs_time'].values).values

    visualization.visualize_survival_function(
        model=model,
        title='Exponential Fitter - Survival Function',
        path=model_directory / 'survival_function.png'
    )
    settings.logger.info(f'survival_function.png is saved to {model_directory}')

    visualization.visualize_cumulative_hazard(
        model=model,
        title='Exponential Fitter Cumulative Hazard',
        path=model_directory / 'cumulative_hazard.png'
    )
    settings.logger.info(f'cumulative_hazard.png is saved to {model_directory}')

    for race_group, df_group in df.groupby('race_group'):
        model = ExponentialFitter()
        model.fit(df_group['efs_time'], df_group['efs'])
        df_group['race_group_survival_probability'] = model.survival_function_at_times(df_group['efs_time'].values).values
        df_group['race_group_cumulative_hazard'] = model.cumulative_hazard_at_times(df_group['efs_time'].values).values

        group_idx = df_group.index
        df.loc[group_idx, 'race_group_survival_probability'] = df_group['race_group_survival_probability']
        df.loc[group_idx, 'race_group_cumulative_hazard'] = df_group['race_group_cumulative_hazard']

    target_columns = [
        'survival_probability', 'race_group_survival_probability',
        'cumulative_hazard', 'race_group_cumulative_hazard'
    ]
    for target in target_columns:
        visualization.visualize_target(
            df=df,
            target=target,
            path=model_directory / f'{target}.png'
        )
    df.loc[:, target_columns].to_csv(model_directory / 'targets.csv', index=False)
    settings.logger.info(f'targets.csv is saved to {model_directory}')
