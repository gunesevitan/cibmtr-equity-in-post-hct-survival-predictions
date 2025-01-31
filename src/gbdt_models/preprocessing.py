from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def clean_categorical_columns(df):

    df['dri_score'] = df['dri_score'].str.lower()

    return df


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


def load_kaplan_meier_outputs(df, kaplan_meier_estimator_directory):

    df_survival_probabilities = pd.read_csv(kaplan_meier_estimator_directory / 'survival_probabilities.csv')
    df = pd.concat((
        df,
        df_survival_probabilities
    ), axis=1, ignore_index=False)

    return df


def preprocess(
    df,
    categorical_columns, transformer_directory, load_transformers,
    kaplan_meier_estimator_directory
):

    df = load_kaplan_meier_outputs(df=df, kaplan_meier_estimator_directory=kaplan_meier_estimator_directory)
    df = clean_categorical_columns(df=df)
    df = encode_categorical_columns(
        df=df,
        categorical_columns=categorical_columns,
        transformer_directory=transformer_directory,
        load_transformers=load_transformers
    )

    return df
