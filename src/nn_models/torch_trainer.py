import sys
import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import torch.optim as optim

sys.path.append('..')
import settings
import preprocessing
import torch_datasets
import torch_modules
import torch_utilities
import metrics
import visualization


def train(training_loader, model, criterion, optimizer, device, scheduler=None):

    """
    Train given model on given data loader

    Parameters
    ----------
    training_loader: torch.utils.data.DataLoader
        Training set data loader

    model: torch.nn.Module
        Model to train

    criterion: torch.nn.Module
        Loss function

    optimizer: torch.optim.Optimizer
        Optimizer for updating model weights

    device: torch.device
        Location of the model and inputs

    scheduler: torch.optim.LRScheduler or None
        Learning rate scheduler

    Returns
    -------
    training_losses: dict
        Dictionary of training losses

    training_predictions: torch.Tensor of shape (n_samples, n_outputs)
        Training predictions
    """

    model.train()
    progress_bar = tqdm(training_loader)

    running_loss = 0.0
    training_predictions = []

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(dim=-1))
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                if scheduler.last_epoch < scheduler.total_steps:
                    scheduler.step()
            else:
                scheduler.step()

        running_loss += loss.detach().item() * len(inputs)
        training_predictions.append(outputs.detach().cpu())
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        progress_bar.set_description(f'lr: {lr:.8f} - training loss: {running_loss / len(training_loader.sampler):.4f}')

    training_loss = running_loss / len(training_loader.sampler)
    training_predictions = torch.cat(training_predictions, dim=0).float()

    training_losses = {
        'training_loss': training_loss
    }

    return training_losses, training_predictions


def validate(validation_loader, model, criterion, device):

    """
    Validate given model on given data loader

    Parameters
    ----------
    validation_loader: torch.utils.data.DataLoader
        Validation set data loader

    model: torch.nn.Module
        Model to validate

    criterion: torch.nn.Module
        Loss function

    device: torch.device
        Location of the model and inputs

    Returns
    -------
    validation_losses: dict
        Dictionary of validation losses

    validation_predictions: torch.Tensor of shape (n_samples, n_outputs)
        Validation predictions
    """

    model.eval()
    progress_bar = tqdm(validation_loader)

    running_loss = 0.0
    validation_predictions = []

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(dim=-1))

        running_loss += loss.detach().item() * len(inputs)
        validation_predictions.append(outputs.detach().cpu())
        progress_bar.set_description(f'validation loss: {running_loss / len(validation_loader.sampler):.4f}')

    validation_loss = running_loss / len(validation_loader.sampler)
    validation_predictions = torch.cat(validation_predictions, dim=0).float()

    validation_outputs = {
        'validation_loss': validation_loss,
    }

    return validation_outputs, validation_predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df = pd.read_parquet(settings.DATA / 'datasets' / config['dataset']['name'])
    df = pd.concat((
        df,
        pd.read_csv(settings.DATA / 'folds.csv')
    ), axis=1, ignore_index=False)
    settings.logger.info(f'Raw Dataset Shape {df.shape}')

    df = preprocessing.preprocess(
        df=df,
        categorical_columns=config['dataset']['categorical_columns'],
        continuous_columns=config['dataset']['continuous_columns'],
        transformer_directory=settings.DATA / 'transformers',
        load_transformers=False,
        efs_predictions_path=config['dataset']['efs_predictions_path'],
        efs_weight=config['training']['efs_weight']
    )

    torch.multiprocessing.set_sharing_strategy('file_system')

    task = config['training']['task']
    folds = config['training']['folds']
    target = config['training']['target']
    features = config['training']['features']

    settings.logger.info(
        f'''
        Running Trainer for {config['model']['model_class']}
        Dataset Shape: {df.shape}
        Folds: {folds}
        Features: {json.dumps(features, indent=2)}
        Target: {target}
        '''
    )

    scores = []
    if task == 'classification':
        curves = []
    else:
        curves = None

    for fold in folds:

        training_mask = df[f'fold{fold}'] == 0
        validation_mask = df[f'fold{fold}'] == 1

        if config['training']['two_stage']:
            training_mask = training_mask & (df['efs'] == 1)

        df.loc[validation_mask, 'prediction'] = 0.

        settings.logger.info(
            f'''
            Fold: {fold} 
            Training: ({np.sum(training_mask)})
            Validation: ({np.sum(validation_mask)})
            '''
        )

        training_dataset = torch_datasets.TabularInMemoryDataset(
            features=df.loc[training_mask, features].values,
            targets=df.loc[training_mask, target].values,
        )
        training_loader = DataLoader(
            training_dataset,
            batch_size=config['training']['training_batch_size'],
            sampler=RandomSampler(training_dataset, replacement=False),
            pin_memory=False,
            drop_last=False,
            num_workers=config['training']['num_workers']
        )
        validation_dataset = torch_datasets.TabularInMemoryDataset(
            features=df.loc[validation_mask, features].values,
            targets=df.loc[validation_mask, target].values,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config['training']['test_batch_size'],
            sampler=SequentialSampler(validation_dataset),
            pin_memory=False,
            drop_last=False,
            num_workers=config['training']['num_workers']
        )

        torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
        device = torch.device(config['training']['device'])
        criterion = getattr(torch_modules, config['training']['loss_function'])(**config['training']['loss_function_args'])

        model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
        model_checkpoint_path = config['model']['model_checkpoint_path']
        if model_checkpoint_path is not None:
            model_checkpoint_path = settings.MODELS / model_checkpoint_path
            model.load_state_dict(torch.load(model_checkpoint_path), strict=False)
        model.to(device)

        optimizer = getattr(torch.optim, config['training']['optimizer'])(model.parameters(), **config['training']['optimizer_args'])
        scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler'])(optimizer, **config['training']['lr_scheduler_args'])

        epochs = config['training']['epochs']
        for epoch in range(1, epochs + 1):

            training_losses, training_predictions = train(
                training_loader=training_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
            )
            validation_losses, validation_predictions = validate(
                validation_loader=validation_loader,
                model=model,
                criterion=criterion,
                device=device,
            )

            settings.logger.info(
                f'''
                Epoch {epoch}
                Training Loss: {json.dumps(training_losses, indent=2)}
                Validation Loss: {json.dumps(validation_losses, indent=2)}
                '''
            )

        model_name = f'model_fold_{fold}.pt'
        torch.save(model.state_dict(), model_directory / model_name)
        settings.logger.info(f'Saved {model_name} to {model_directory}')

        if task == 'classification':
            validation_predictions = torch.sigmoid(validation_predictions)
        validation_predictions = validation_predictions.numpy().reshape(-1)

        if config['training']['two_stage']:
            if config['training']['target'] == 'log_efs_time':
                df.loc[validation_mask, 'reg_1_prediction'] = validation_predictions
                validation_predictions = df.loc[validation_mask, 'efs_prediction'] / np.exp(validation_predictions)
            elif config['training']['target'] == 'log_km_survival_probability':
                df.loc[validation_mask, 'reg_1_prediction'] = validation_predictions
                validation_predictions = df.loc[validation_mask, 'efs_prediction'] * np.exp(validation_predictions)

        if config['training']['rank_transform']:
            validation_predictions = pd.Series(validation_predictions).rank(pct=True).values

        df.loc[validation_mask, 'prediction'] += validation_predictions

        if task == 'ranking':
            validation_scores = metrics.ranking_score(
                df=df.loc[validation_mask],
                group_column='race_group',
                time_column='efs_time',
                event_column='efs',
                prediction_column='prediction'
            )
        elif task == 'classification':
            validation_scores = metrics.classification_score(
                df=df.loc[validation_mask],
                group_column='race_group',
                event_column='efs',
                prediction_column='prediction',
                weight_column=None
            )
            validation_curves = metrics.classification_curves(
                df=df.loc[validation_mask],
                event_column='efs',
                prediction_column='prediction',
            )
            curves.append(validation_curves)
        elif task == 'regression':
            validation_scores = metrics.regression_score(
                df=df.loc[validation_mask],
                group_column='race_group',
                time_column=target,
                prediction_column='prediction'
            )
        else:
            raise ValueError(f'Invalid task type {task}')

        settings.logger.info(f'Fold: {fold} - Validation Scores: {json.dumps(validation_scores, indent=2)}')
        scores.append(validation_scores)

    scores = pd.DataFrame(scores)
    settings.logger.info(
        f'''
        Mean Validation Scores
        ----------------------
        {json.dumps(scores.mean(axis=0).to_dict(), indent=2)}

        Standard Deviations
        -------------------
        ±{json.dumps(scores.std(axis=0).to_dict(), indent=2)}
        '''
    )

    oof_mask = df['prediction'].notna()
    if task == 'ranking':
        oof_scores = metrics.ranking_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column='efs_time',
            event_column='efs',
            prediction_column='prediction'
        )
    elif task == 'classification':
        oof_scores = metrics.classification_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            event_column='efs',
            prediction_column='prediction'
        )
    elif task == 'regression':
        oof_scores = metrics.regression_score(
            df=df.loc[oof_mask],
            group_column='race_group',
            time_column=target,
            prediction_column='prediction'
        )
    else:
        raise ValueError(f'Invalid task type {task}')

    settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

    scores = pd.concat((
        scores,
        pd.DataFrame([oof_scores])
    )).reset_index(drop=True)
    scores['fold'] = folds + ['OOF']
    scores = scores[scores.columns.tolist()[::-1]]
    scores.to_csv(model_directory / 'scores.csv', index=False)
    settings.logger.info(f'scores.csv is saved to {model_directory}')

    visualization.visualize_scores(
        scores=scores,
        title=f'{config["model"]["model_class"]} Model Scores of {len(folds)} Fold(s)',
        path=model_directory / 'scores.png'
    )
    settings.logger.info(f'Saved scores.png to {model_directory}')

    if task == 'classification':
        visualization.visualize_roc_curves(
            roc_curves=[curve['roc'] for curve in curves],
            title=f'{config["model"]["model_class"]} Model Validation ROC Curves',
            path=model_directory / 'roc_curves.png'
        )
        settings.logger.info(f'Saved roc_curves.png to {model_directory}')

        visualization.visualize_pr_curves(
            pr_curves=[curve['pr'] for curve in curves],
            title=f'{config["model"]["model_class"]} Model Validation PR Curves',
            path=model_directory / 'pr_curves.png'
        )
        settings.logger.info(f'Saved pr_curves.png to {model_directory}')

    df.loc[:, 'prediction'].to_csv(model_directory / 'oof_predictions.csv', index=False)
    settings.logger.info(f'Saved oof_predictions.csv to {model_directory}')
