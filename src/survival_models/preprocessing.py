from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def one_hot_encode_categorical_columns(df, categorical_columns, transformer_directory, load_transformers=False):

    Path(transformer_directory).mkdir(parents=True, exist_ok=True)

    for column in categorical_columns:

        if load_transformers:

            with open(transformer_directory / f'{column}_encoder.pickle', mode='rb') as f:
                encoder = pickle.load(f)

            df_column_encoded = pd.DataFrame(
                encoder.transform(df[column].values.reshape(-1, 1)),
                columns=[f'{column}_{category}' for category in encoder.categories_[0]]
            )

            df = pd.concat((df, df_column_encoded), axis=1, ignore_index=False)

        else:

            encoder = OneHotEncoder(
                categories='auto',
                drop=None,
                sparse_output=False,
                dtype=np.uint8,
                handle_unknown='error'
            )
            encoder.fit(df[column].values.reshape(-1, 1))
            df_column_encoded = pd.DataFrame(
                encoder.transform(df[column].values.reshape(-1, 1)),
                columns=[f'{column}_{category}' for category in encoder.categories_[0]]
            )
            df = pd.concat((df, df_column_encoded), axis=1, ignore_index=False)

            with open(transformer_directory / f'{column}_one_hot_encoder.pickle', mode='wb') as f:
                pickle.dump(encoder, f)

    return df


def normalize_continuous_columns(df, continuous_columns, transformer_directory, load_transformers=False):

    Path(transformer_directory).mkdir(parents=True, exist_ok=True)

    if load_transformers:

        with open(transformer_directory / 'standard_scaler.pickle', mode='rb') as f:
            normalizer = pickle.load(f)

        normalized_column_names = [f'{column}_normalized' for column in continuous_columns]
        df[normalized_column_names] = normalizer.transform(df[continuous_columns].values)

    else:

        normalizer = StandardScaler()
        normalizer.fit(df[continuous_columns].values)
        normalized_column_names = [f'{column}_normalized' for column in continuous_columns]
        df[normalized_column_names] = normalizer.transform(df[continuous_columns].values)

        with open(transformer_directory / 'standard_scaler.pickle', mode='wb') as f:
            pickle.dump(normalizer, f)

    return df


def preprocess(
        df,
        categorical_columns, one_hot_encoder_directory, load_one_hot_encoders,
):

    df = one_hot_encode_categorical_columns(
        df=df,
        categorical_columns=categorical_columns,
        transformer_directory=one_hot_encoder_directory,
        load_transformers=load_one_hot_encoders
    )

    df = normalize_continuous_columns(
        df=df,
        continuous_columns=continuous_columns,
        transformer_directory=transformer_directory,
        load_transformers=load_transformers
    )

    return df