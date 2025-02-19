import sys
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

sys.path.append('..')
import settings


def create_folds(df, stratify_columns, n_splits, shuffle=True, random_state=42, verbose=True):

    """
    Create columns of folds on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataset to create folds

    stratify_columns: list of shape (n_stratify_columns)
        Array stratify column names

    n_splits: int
        Number of folds (2 <= n_splits)

    shuffle: bool
        Whether to shuffle before split or not

    random_state: int
        Random seed for reproducible results

    verbose: bool
        Verbosity flag

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with created fold columns
    """

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold, (training_idx, validation_idx) in enumerate(mskf.split(X=df,  y=df[stratify_columns]), 1):
        df.loc[training_idx, f'fold{fold}'] = 0
        df.loc[validation_idx, f'fold{fold}'] = 1
        df[f'fold{fold}'] = df[f'fold{fold}'].astype(np.uint8)

    if verbose:

        settings.logger.info(f'Dataset split into {n_splits} folds')

        validation_sizes = []
        race_group_value_counts = []
        efs_value_counts = []
        efs_time_means = []

        for fold in range(1, n_splits + 1):

            df_fold = df[df[f'fold{fold}'] == 1]

            fold_validation_size = df_fold.shape[0]
            validation_sizes.append(fold_validation_size)
            fold_race_group_value_counts = df_fold['race_group'].value_counts().sort_index()
            race_group_value_counts.append(fold_race_group_value_counts)
            fold_efs_value_counts = df_fold['efs'].value_counts().sort_index()
            efs_value_counts.append(fold_efs_value_counts)
            fold_efs_time_mean = df_fold['efs_time'].mean()
            efs_time_means.append(fold_efs_time_mean)

            settings.logger.info(f'Fold {fold} - Validation Size: {fold_validation_size} - race_group Value Counts {fold_race_group_value_counts.to_dict()} - efs Value Counts {fold_efs_value_counts.to_dict()} - efs_time Mean {fold_efs_time_mean:.4f} ')

        validation_sizes_std = np.std(validation_sizes)
        race_group_value_counts = pd.DataFrame(race_group_value_counts)
        race_group_average_std = race_group_value_counts.std(axis=0).mean(axis=0)
        efs_value_counts = pd.DataFrame(efs_value_counts)
        efs_average_std = efs_value_counts.std(axis=0).mean(axis=0)
        efs_time_means_std = np.std(efs_time_means)

        settings.logger.info(f'seed: {random_state} - validation size std {validation_sizes_std:.4f} - race group std {race_group_average_std:.4f} - efs std {efs_average_std:.4f} - efs_time_mean std {efs_time_means_std:.4f}')

    return df


if __name__ == '__main__':

    df = pd.read_parquet(settings.DATA / 'datasets' / 'dataset.parquet')
    settings.logger.info(f'Dataset Shape: {df.shape}')

    time_bin = (np.log1p(df['efs_time']).clip(1, 4) // 0.5).astype(int).astype(str)
    df['time_bin'] = time_bin

    n_splits = 7
    df = create_folds(
        df=df,
        stratify_columns=['race_group', 'efs', 'time_bin'],
        n_splits=n_splits,
        shuffle=True,
        random_state=60,
        verbose=True
    )

    df_folds = df.loc[:, [f'fold{fold}' for fold in range(1, n_splits + 1)]]
    df_folds.to_csv(settings.DATA / 'folds.csv', index=False)
    settings.logger.info(f'folds.csv is saved to {settings.DATA}')
