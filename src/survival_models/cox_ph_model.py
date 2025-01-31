import sys
from pathlib import Path
import pickle
import yaml
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

sys.path.append('..')
import settings
import preprocessing
import metrics
import visualization


if __name__ == '__main__':

    model_directory = Path(settings.MODELS / 'cox_ph_model')
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
        categorical_columns=categorical_columns,
        transformer_directory=settings.DATA / 'one_hot_encoders',
        load_transformers=False
    )
    exit()

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