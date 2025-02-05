from pathlib import Path
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def encode_categorical_columns(df, categorical_columns, transformer_directory, load_transformers):

    """
    Encode given categorical columns

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with categorical columns

    categorical_columns: list
        Array of categorical column names

    transformer_directory: pathlib.Path or str
        Directory for saving/loading transformers

    load_transformers: bool
        Whether to load precomputed transformers or not

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

            df[f'{column}_encoded'] = encoder.transform(df[column].values.reshape(-1, 1))

        else:

            encoder = OrdinalEncoder(
                categories='auto',
                dtype=np.float32,
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=np.nan,
            )
            df[f'{column}_encoded'] = encoder.fit_transform(df[column].values.reshape(-1, 1))

            with open(transformer_directory / f'{column}_encoder.pickle', mode='wb') as f:
                pickle.dump(encoder, f)

    return df


def preprocess(
    df,
    categorical_columns, transformer_directory, load_transformers,
):

    """
    Preprocess given dataframe for survival models

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with raw features and targets

    categorical_columns: list
        List of categorical columns

    transformer_directory: str or pathlib.Path
        Path of the serialized transformers

    load_transformers: bool
        Whether to load transformers from the given transformer directory or not

    Returns
    -------
    df: pandas.DataFrame
        Preprocessed dataframe
    """

    columns_to_fill = [
        'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6', 'hla_high_res_10', 'hla_match_dqb1_high',
        'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low', 'hla_match_dqb1_low', 'hla_match_a_high', 'donor_age',
        'hla_match_b_low', 'hla_match_a_low', 'hla_match_b_high', 'comorbidity_score', 'karnofsky_score',
        'hla_low_res_8', 'hla_match_drb1_high', 'hla_low_res_10'
    ]
    for column in columns_to_fill:
        df[column] = df[column].fillna(df[column].median())

    df = encode_categorical_columns(
        df=df,
        categorical_columns=categorical_columns,
        transformer_directory=transformer_directory,
        load_transformers=load_transformers
    )

    return df
