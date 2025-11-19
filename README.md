# House Price Prediction — Linear Regression

## Project Overview
A simple linear regression model to predict house prices using fundamental features such as square footage, number of bedrooms, and number of bathrooms. This project demonstrates end-to-end workflow including exploratory data analysis, feature selection, model training, evaluation, and basic model persistence.

## Dataset
This project uses the Kaggle **House Prices - Advanced Regression Techniques** dataset:

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

> Note: Do **not** upload the full Kaggle dataset to GitHub. Keep data locally or provide download instructions / a small sample.

## Key Features
- Square footage / living area
- Number of bedrooms
- Number of bathrooms
- Basic feature engineering (if applied)
- Train / test split and model evaluation (MAE, RMSE, R²)

## Project Structure
HOUSE_PRICE_PREDICTION/
├── data/ # sample data, DO NOT commit full dataset
├── notebooks/ # EDA and model notebooks (e.g., Task_01_HousePricePrediction.ipynb)
├── src/
│ ├── preprocess.py # data cleaning & feature engineering
│ ├── train.py # training script
│ ├── model.py # model definition & utilities
│ └── predict.py # single-sample prediction script
├── models/ # saved models (.pkl or .joblib)
├── results/ # plots, evaluation metrics
├── requirements.txt
├── README.md
└── .gitignore



## Installation
```bash
# create and activate virtual environment (example using venv)
python -m venv ml_env
# Windows
ml_env\Scripts\activate
# macOS/Linux
source ml_env/bin/activate

# install dependencies
pip install -r requirements.txt


Usage
Run notebook

Open notebooks/Task_01_HousePricePrediction.ipynb and run the cells to explore the data, train the model and see evaluation results.

Train from command line
python src/train.py --data data/HousePrice_sample.csv --output models/linear_reg_model.pkl

Make a prediction
python src/predict.py --model models/linear_reg_model.pkl --sqft 1500 --bedrooms 3 --bathrooms 2

Evaluation

This project reports standard regression metrics:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² score

Include a small section in the notebook that shows metric values and a residual plot.

Future Improvements

Add polynomial features or interaction terms

Compare with regularized linear models (Ridge, Lasso)

Add cross-validation and hyperparameter tuning

Build a small FastAPI service for inference

---

# requirements.txt
python>=3.8
numpy
pandas
matplotlib
scikit-learn
seaborn
joblib
jupyterlab




