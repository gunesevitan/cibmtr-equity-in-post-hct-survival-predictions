import sys
from pathlib import Path
import pickle
import yaml
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

sys.path.append('..')
import settings


def visualize_survival_probabilities(kmf, title, path=None):

    """
    Visualize survival probabilities of the given Kaplan-Meier estimator

    Parameters
    ----------
    kmf: lifelines.KaplanMeierFitter
        Kaplan-Meier estimator fit on dataset

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(32, 12))

    kmf.plot_survival_function(
        ax=ax,
        label='efs',
        alpha=0.5,
        show_censors=True,
        censor_styles={
            'ms': 5,
            'marker': 's'
        })

    ax.set_xlabel('Timeline', size=20, labelpad=15)
    ax.set_ylabel('Probability', size=20, labelpad=15)
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='best', prop={'size': 18})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


if __name__ == '__main__':

    model_directory = Path(settings.MODELS / 'kaplan_meier_estimator')
    model_directory.mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df = pd.read_parquet(settings.DATA / 'datasets' / config['dataset']['name'])
    settings.logger.info(f'Raw Dataset Shape {df.shape}')

    kmf = KaplanMeierFitter()
    kmf.fit(df['efs_time'], df['efs'])

    with open(model_directory / 'kaplan_meier_estimator.pickle', mode='wb') as f:
        pickle.dump(kmf, f)
    settings.logger.info(f'kaplan_meier_estimator.pickle is saved to {model_directory}')

    visualize_survival_probabilities(
        kmf=kmf,
        title='Kaplan-Meier Estimator Survival Probabilities',
        path=model_directory / 'survival_probabilities.png'
    )
    settings.logger.info(f'survival_probabilities.png is saved to {model_directory}')
    df['kmf_survival_probability'] = kmf.survival_function_at_times(df['efs_time']).values

    for race_group, df_group in df.groupby('race_group'):

        kmf = KaplanMeierFitter()
        kmf.fit(df_group['efs_time'], df_group['efs'])

        file_name = f'kaplan_meier_estimator_{"_".join(str(race_group).lower().split())}.pickle'
        with open(model_directory / file_name, mode='wb') as f:
            pickle.dump(kmf, f)
        settings.logger.info(f'{file_name} is saved to {model_directory}')

        df_group['kmf_race_group_survival_probability'] = kmf.survival_function_at_times(df_group['efs_time']).values
        group_idx = df_group.index
        df.loc[group_idx, 'kmf_race_group_survival_probability'] = df_group['kmf_race_group_survival_probability']

    df.loc[:, ['kmf_survival_probability', 'kmf_race_group_survival_probability']].to_csv(model_directory / 'survival_probabilities.csv', index=False)
    settings.logger.info(f'survival_probabilities.csv is saved to {model_directory}')
