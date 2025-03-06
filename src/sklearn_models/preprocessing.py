from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


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


def ordinal_encode_categorical_columns(df, categorical_columns, transformer_directory, load_transformers=False):

    """
    Ordinal encode given categorical columns and concatenate them to given dataframe

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

        column_dtype = df[column].dtype
        fill_value = 'missing' if column_dtype == object else -1

        if load_transformers:

            with open(transformer_directory / f'{column}_ordinal_encoder.pickle', mode='rb') as f:
                encoder = pickle.load(f)

            df[f'{column}_encoded'] = encoder.transform(df[column].fillna(fill_value).values.reshape(-1, 1))

        else:

            encoder = OrdinalEncoder(
                categories='auto',
                dtype=np.int32,
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=np.nan,
            )
            df[f'{column}_encoded'] = encoder.fit_transform(df[column].fillna(fill_value).values.reshape(-1, 1))

            with open(transformer_directory / f'{column}_ordinal_encoder.pickle', mode='wb') as f:
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
        df[normalized_column_names] = np.nan
        df.loc[:, normalized_column_names] = normalizer.transform(df[continuous_columns].fillna(df[continuous_columns].median()).values)

        with open(transformer_directory / 'standard_scaler.pickle', mode='wb') as f:
            pickle.dump(normalizer, f)

    return df


def create_targets(
        df,
        efs_predictions_path,
        kaplan_meier_targets_path, nelson_aalen_targets_path
):

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

    df['weight'] = 1.
    df.loc[df['efs'] == 1, 'weight'] = efs_weight

    return df


def preprocess(
        df,
        categorical_columns, continuous_columns,
        transformer_directory, load_transformers,
        efs_predictions_path, kaplan_meier_targets_path, nelson_aalen_targets_path,
        efs_weight
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
    df = one_hot_encode_categorical_columns(
        df=df,
        categorical_columns=categorical_columns,
        transformer_directory=transformer_directory,
        load_transformers=load_transformers
    )
    df = ordinal_encode_categorical_columns(
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
