## CIBMTR - Equity in post-HCT Survival Predictions - 2nd Place Solution (Gunes Evitan's Part)

## Getting Started

```
git clone https://github.com/gunesevitan/cibmtr-equity-in-post-hct-survival-predictions.git
cd cibmtr-equity-in-post-hct-survival-predictions
mkdir data logs models src
```

There has to be two environments created in order to train all models and use them on Kaggle notebooks.

This environment is for LightGBM, XGBoost and MLP models.
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

This environment is for sklearn and CatBoost models.

```
virtualenv catboost_venv
source catboost_venv/bin/activate
pip install -r catboost_requirements.txt
```

After setting up the environments, root path (`ROOT`) on `settings.py` should be changed.
It's the absolute path on my local machine.

## How to Reproduce

Those scripts are for creating dataset and folds.

```
cd src/dataset_utilities
python dataset_writer.py
python validation.py
```

Then, Kaplan-Meier probabilities and Nelson-Aalen cumulative hazard values can be calculated by doing

```
cd src/lifelines_survival_models
python kaplan_meier.py
python nelson_aalen.py
```

Next step is training classification models.

```
cd src/sklearn_models
python tree_model_trainer.py hist_gradient_boosting_classifier_efs

cd src/gbdt_models
python catboost_trainer.py catboost_efs_binary_classifier
python lightgbm_trainer.py lightgbm_efs_binary_classifier
python xgboost_trainer.py xgboost_efs_binary_classifier

cd src/nn_models
python torch_trainer.py mlp_sparse_efs_binary_classifier
python torch_trainer.py mlp_embeddings_efs_binary_classifier
```

After all the classification models are trained, they can be blended by doing

```
cd src/ensemble
python classifier_ensemble.py
```

Scores on classification section can be reproduced by going through those steps.
Running this script writes the `efs_predictions.csv` file which will be used on second stage.

Next step is training the regression models.

```
cd src/sklearn_models
python tree_model_trainer.py hist_gradient_boosting_regressor_log_efs_time
python tree_model_trainer.py hist_gradient_boosting_regressor_log_km_proba
python tree_model_trainer.py hist_gradient_boosting_regressor_na_cum_hazard

cd src/gbdt_models
python xgboost_trainer.py xgboost_log_efs_time_regressor
python xgboost_trainer.py xgboost_log_km_proba_regressor
```

After all the regression models are trained, they can be blended by doing

```
cd src/ensemble
python ranking_ensemble.py
```

which concludes the reproduction part.

## Overview

I used a slightly different version of two-stage pipeline:
* Classification (Models are trained to predict efs)
* Regression (Models are trained on only efs 1 samples)

Final ranks are determined by combining the outputs from both the classifiers and regressors.
The intuition was removing the noise that's coming from efs 0 samples.

* GitHub Repository: https://github.com/gunesevitan/cibmtr-equity-in-post-hct-survival-predictions
* Kaggle Submission Notebook: https://www.kaggle.com/code/gunesevitan/cibmtr-inference
* Kaggle Introduction to Survival Analysis Notebook: https://www.kaggle.com/code/gunesevitan/survival-analysis

## 1st Stage - Classification

For the classification stage, I built an ensemble of six models:
* HistGBM
* LightGBM
* XGBoost
* CatBoost
* MLP with sparse input
* MLP with embeddings

All of the models are trained using raw features. Log loss is optimized for achieving better probabilities.
For HistGBM and MLP with embeddings, all features except `donor_age` and `age_at_hct` are ordinal encoded.
For LightGBM, XGBoost and CatBoost, all features are used as categoricals.
For MLP with sparse input, all features except `donor_age` and `age_at_hct` are one-hot encoded.

| Model      	| Log Loss   | ROC AUC    |
|----------------|------------|------------|
| **HistGBM**   	| **0.5771** | **0.7622** |
| LightGBM   	| 0.5777     | 0.7613     |
| XGBoost    	| 0.5779     | 0.7611     |
| CatBoost   	| 0.5788     | 0.7604     |
| MLP Sparse 	| 0.5792     | 0.7605     |
| MLP Embeddings | 0.5792     | 0.7601     |
| **Weighted Average** | **0.5748** | **0.7645** |

I tried multiplying race group probabilities with different constants, but it didn’t bring any improvement.
I also tried overwriting probabilities less than 0.02 to 0 and probabilities greater than 0.98 to 1.
It improved the log loss slightly, but adjusting those thresholds a little bit off was reducing the score drastically, so I ended up not using that either.
I used a weighted average of those probabilities in my two stage pipeline.

## 2nd Stage - EFS 1 Regression
On 2nd stage, I used 3 different targets:
* Log EFS time
* Min-max scaled log Kaplan-Meier probability
* Negative exp Nelson-Aalen cumulative hazard

and two different models:
* HistGBM
* XGBoost

Final ranks are obtained by combining efs probability predictions (weighted average of classification models) and efs 1 regression predictions.

Those are the formulas for each target:
* Log EFS Time: efs probability prediction / exp(log efs time prediction)
* Kaplan-Meier Probability: efs probability prediction * exp(kaplan-meier probability prediction)
* Negative Nelson-Aalen Cumulative Hazard: efs probability prediction * exp(nelson-aalen cumulative hazard prediction)

All of the models on the second stage are also trained with raw features. Hyperparameters of regressors are tuned with respect to scores calculated after the two stage approach, not based on their individual regression predictions.

On test time, rank transform is applied after predicting with each single fold and their rank average is taken in order to get final predictions.

| Model                                  	| OOF    	   | Public LB | Private LB |
|--------------------------------------------|------------|-----------|------------|
| HistGBM Log EFS Time                   	| 0.6903 	   | 0.694 	| 0.697  	|
| HistGBM Kaplan-Meier Probability       	| 0.6909 	   | 0.694 	| 0.697  	|
| **HistGBM Nelson-Aalen Cumulative Hazard** | **0.6915** | **0.694** | **0.697**  |
| XGBoost Log EFS Time                   	| 0.6867 	   | 0.693 	| 0.690  	|
| XGBoost Kaplan-Meier Probability       	| 0.6860 	   | 0.693 	| 0.690  	|
| **Weighted Average**                   	| **0.6927** | **0.694** | **0.697**  |

I tried other models such as LightGBM, CatBoost, MLP, perpetual, ngboost etc., but they didn’t work well on either cross-validation or leaderboard.

## Postprocessing

To refine the final predictions, I applied a threshold-based scaling approach to the final rank transformed prediction values like this.

```
step = 0.01
for threshold in np.arange(0, 0.33, step):
    df.loc[(df['efs_prediction'] >= threshold) & (df['efs_prediction'] < threshold + step), 'rank_prediction'] *= threshold + step 
```

It increased the weighted average score from 0.6927 to 0.6934, but we ended up not using it since the risk may not be worth it.

## Things Didn’t Work
* Feature engineering
* Using more or less folds was pretty much the same
* GMM
* sksurv models
* GBDTs with cox objective
* Sample weights
* Huber, poisson, tweedie, cross-entropy loss functions
* Different lifelines univariate model target transforms
