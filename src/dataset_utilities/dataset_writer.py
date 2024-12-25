import sys
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(settings.DATA / 'equity-post-HCT-survival-predictions' / 'train.csv')
    settings.logger.info(f'Dataset Shape: {df.shape}')

    df = df.drop(columns=['ID'])

    df.to_parquet(dataset_directory / 'dataset.parquet')
    settings.logger.info(f'Saved dataset.parquet to {dataset_directory}')
