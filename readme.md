# Freight Rate Prediction

Machine learning solution for predicting freight `posted_rate` from historical load data.

## Approach

The project uses a **CatBoost regression model** trained on a log-transformed target.

### Features

The final model uses:

- `pickup`
- `delivery`
- `distance`
- `equipment`
- `weight`
- `month`
- `day_of_month`
- `day_of_week`

Negative weight values are treated as missing values. The raw date is converted into calendar features and removed afterward.

The target is transformed using:

```python
np.log1p(posted_rate)
```

Predictions are converted back to the original scale using:

```python
np.expm1(prediction)
```

### Validation

A temporal holdout was used to simulate future prediction:

- Training/development: September 2025 and earlier
- Holdout: October 2025
- Development samples: 43,147
- Holdout samples: 4,853

The final model was selected using the October holdout and then retrained on all labeled data before generating the final predictions.

## Installation

Python 3.10+ is recommended.

Clone the repository and install the dependencies:

```bash
git clone https://github.com/NaBiRouS/freight-rate-prediction.git
cd freight-rate-prediction

python -m pip install -r requirements.txt
```

## Generate predictions

Run the notebook to train the final model and generate both:

```text
validation_predictions.csv
```

and

```text
december_chart_inputs.csv
```

## Dependencies

Main dependencies include:

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- XGBoost
- PyTorch
- Optuna
- Matplotlib

The complete dependency list is provided in `requirements.txt`.
