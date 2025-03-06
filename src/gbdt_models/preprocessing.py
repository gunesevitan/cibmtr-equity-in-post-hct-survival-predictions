import numpy as np
import pandas as pd


def encode_categorical_columns(df, categorical_columns, dtype):

    """
    Encode given categorical columns

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with categorical columns

    categorical_columns: list
        Array of categorical column names

    dtype: str
        Type of the categorical column

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with encoded categorical columns
    """

    for column in categorical_columns:
        df[f'{column}_{dtype.__name__ if dtype == str else dtype}'] = df[column].astype(dtype)

    return df


def create_targets(df, efs_predictions_path, kaplan_meier_targets_path, nelson_aalen_targets_path):

    """
    Create targets on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with event and time columns

    efs_predictions_path: str or pathlib.Path
        Path of the efs predictions file

    kaplan_meier_targets_path: str or pathlib.Path
        Path of the kaplan-meier targets file

    nelson_aalen_targets_path: str or pathlib.Path
        Path of the nelson-aalen targets file

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with additional targets
    """

    df['log_efs_time'] = np.log(df['efs_time'])
    df['inverted_efs_time'] = df['efs_time'].values
    df.loc[df['efs'] == 0, 'inverted_efs_time'] *= -1

    if efs_predictions_path is not None:
        df = pd.concat((
            df,
            pd.read_csv(efs_predictions_path).rename(columns={
                'prediction': 'efs_prediction'
            })
        ), axis=1, ignore_index=False)

    if kaplan_meier_targets_path is not None:
        df_kaplan_meier = pd.read_csv(kaplan_meier_targets_path).rename(columns={
            'survival_probability': 'km_survival_probability'
        })
        df = pd.concat((
            df,
            df_kaplan_meier
        ), axis=1, ignore_index=False)

        df['log_km_survival_probability'] = np.log1p(df['km_survival_probability'])
        df['log_km_survival_probability'] -= df['log_km_survival_probability'].min()
        df['log_km_survival_probability'] /= df['log_km_survival_probability'].max()

    if nelson_aalen_targets_path is not None:
        df_nelson_aalen = pd.read_csv(nelson_aalen_targets_path).rename(columns={
            'cumulative_hazard': 'na_cumulative_hazard',
            'hazard_rate': 'na_hazard_rate'
        })
        df = pd.concat((
            df,
            df_nelson_aalen
        ), axis=1, ignore_index=False)

        df['na_cumulative_hazard'] = -np.exp(df['na_cumulative_hazard'])
        df['na_hazard_rate'] = -np.exp(df['na_hazard_rate'])

    return df


def create_sample_weights(df, efs_weight):

    """
    Create sample weights on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with event column

    efs_weight: float
        Weights of event occurred samples

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with sample weights
    """

    df['weight'] = 1
    df.loc[df['efs'] == 1, 'weight'] = efs_weight

    return df


def preprocess(
        df,
        categorical_columns, categorical_dtype,
        efs_predictions_path, kaplan_meier_targets_path, nelson_aalen_targets_path,
        efs_weight
):

    """
    Preprocess given dataframe for training LightGBM model

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with raw targets

    categorical_columns: list
        Array of categorical column names

    categorical_dtype: str
        Type of the categorical column

    efs_predictions_path: str or pathlib.Path or None
        Path of the efs predictions file

    kaplan_meier_targets_path: str or pathlib.Path or None
        Path of the kaplan-meier targets file

    nelson_aalen_targets_path: str or pathlib.Path
        Path of the nelson-aalen targets file

    efs_weight: float (efs_weight >= 1)
        Weights of event occurred samples

    Returns
    -------
    df: pandas.DataFrame
        Preprocessed dataframe
    """

    df = df.fillna(np.nan)
    df = create_targets(
        df=df,
        efs_predictions_path=efs_predictions_path,
        kaplan_meier_targets_path=kaplan_meier_targets_path,
        nelson_aalen_targets_path=nelson_aalen_targets_path
    )
    df = create_sample_weights(df=df, efs_weight=efs_weight)
    df = encode_categorical_columns(df=df, categorical_columns=categorical_columns, dtype=categorical_dtype)

    return df
