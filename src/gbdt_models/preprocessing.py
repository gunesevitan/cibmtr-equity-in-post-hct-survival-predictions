import sys
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


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
        df[column] = df[column].astype(dtype)

    return df


def create_targets(df):

    """
    Create targets on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with raw targets

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with additional targets
    """

    df['log_efs_time'] = np.log(df['efs_time'])

    df_kaplan_meier = pd.read_csv(settings.MODELS / 'kaplan_meier' / 'targets.csv')
    df = pd.concat((
        df,
        df_kaplan_meier
    ), axis=1, ignore_index=False)
    df['log_survival_probability'] = np.log(df['survival_probability'])

    return df


def preprocess(df, categorical_columns, categorical_dtype):

    """
    Preprocess given dataframe for training LightGBM model

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with raw targets

    categorical_columns: list
        Array of categorical column names

    dtype: str
        Type of the categorical column

    Returns
    -------
    df: pandas.DataFrame
        Preprocessed dataframe
    """

    df = df.fillna(np.nan)
    df = create_targets(df=df)
    df = encode_categorical_columns(df=df, categorical_columns=categorical_columns, dtype=categorical_dtype)

    return df
