from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def one_hot_encode_categorical_columns(df, categorical_columns, transformer_directory, load_transformers=False):

    """
    One-hot encode given categorical columns and concatenate them to given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with categorical columns

    categorical_columns: list
        List of categorical columns

    transformer_directory: str or pathlib.Path
        Path of the serialized transformers

    load_transformers: bool
        Whether to load transformers from the given transformer directory or not

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with encoded categorical columns
    """

    Path(transformer_directory).mkdir(parents=True, exist_ok=True)

    for column in categorical_columns:

        if load_transformers:

            with open(transformer_directory / f'{column}_encoder.pickle', mode='rb') as f:
                encoder = pickle.load(f)

            encoded = encoder.fit_transform(df[column].values.reshape(-1, 1))
            encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]), index=df.index)
            df = pd.concat((df, encoded), axis=1, ignore_index=False)

        else:

            encoder = OneHotEncoder(
                categories='auto',
                drop=None,
                sparse_output=False,
                dtype=np.uint8,
                handle_unknown='ignore',
                min_frequency=128
            )
            encoded = encoder.fit_transform(df[column].values.reshape(-1, 1))
            encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]), index=df.index)
            df = pd.concat((df, encoded), axis=1, ignore_index=False)

            with open(transformer_directory / f'{column}_one_hot_encoder.pickle', mode='wb') as f:
                pickle.dump(encoder, f)

    return df


def normalize_continuous_columns(df, continuous_columns, transformer_directory, load_transformers=False):

    """
    Normalize continuous columns and concatenate them to given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with continuous columns

    continuous_columns: list
        List of continuous columns

    transformer_directory: str or pathlib.Path
        Path of the serialized transformers

    load_transformers: bool
        Whether to load transformers from the given transformer directory or not

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with encoded continuous columns
    """

    Path(transformer_directory).mkdir(parents=True, exist_ok=True)

    if load_transformers:

        with open(transformer_directory / 'standard_scaler.pickle', mode='rb') as f:
            normalizer = pickle.load(f)

        normalized_column_names = [f'{column}_normalized' for column in continuous_columns]
        df[normalized_column_names] = normalizer.transform(df[continuous_columns].fillna(df[continuous_columns].median()).values)

    else:

        normalizer = StandardScaler()
        normalizer.fit(df[continuous_columns].values)
        normalized_column_names = [f'{column}_normalized' for column in continuous_columns]
        df[normalized_column_names] = normalizer.transform(df[continuous_columns].fillna(df[continuous_columns].median()).values)

        with open(transformer_directory / 'standard_scaler.pickle', mode='wb') as f:
            pickle.dump(normalizer, f)

    return df


def preprocess(
        df,
        categorical_columns, continuous_columns,
        transformer_directory, load_transformers
):

    """
    Preprocess given dataframe for survival models

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with raw features and targets

    categorical_columns: list
        List of categorical columns

    continuous_columns: list
        List of continuous columns

    transformer_directory: str or pathlib.Path
        Path of the serialized transformers

    load_transformers: bool
        Whether to load transformers from the given transformer directory or not

    Returns
    -------
    df: pandas.DataFrame
        Preprocessed dataframe
    """

    df = df.fillna(np.nan)
    df = one_hot_encode_categorical_columns(
        df=df,
        categorical_columns=categorical_columns,
        transformer_directory=transformer_directory,
        load_transformers=load_transformers
    )
    df = normalize_continuous_columns(
        df=df,
        continuous_columns=continuous_columns,
        transformer_directory=transformer_directory,
        load_transformers=load_transformers
    )

    return df
