# House-Price-Prediction
Predicting house sale prices using EDA and Linear Regression on the Ames Housing dataset (Kaggle House Prices competition).

A machine learning project that predicts residential home sale prices using the Ames Housing dataset (Kaggle: House Prices - Advanced Regression Techniques). The project covers exploratory data analysis (EDA), feature-correlation analysis, and a Linear Regression model trained to predict SalePrice.

Repository Structure

house-price-prediction/
├── data/
│   ├── train.csv              # Training data (with SalePrice target)
│   └── test.csv                # Test data (for generating predictions)
├── notebooks/
│   └── House_Price_Prediction.ipynb   # Full EDA + model notebook
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files/folders excluded from git
└── README.md                   # Project overview and instructions

Dataset

The dataset contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, and is commonly used for regression practice.


train.csv — 1460 rows, includes the target column SalePrice
test.csv — 1459 rows, used to generate final predictions (no SalePrice)


 Project Workflow

The notebook (notebooks/House_Price_Prediction.ipynb) walks through:


Data Loading — reading train.csv and test.csv with pandas
Exploratory Data Analysis

Distribution of SalePrice
Missing value inspection
Correlation of key numeric features with SalePrice
Outlier check on GrLivArea vs SalePrice
SalePrice grouped by OverallQual (boxplot)



Feature Selection — 12 numeric features most correlated with price:
OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd, MasVnrArea, Fireplaces
Modeling — a LinearRegression model (scikit-learn) trained on an 80/20 train/validation split
Evaluation — Mean Squared Error (MSE) and R² score on the validation set, plus an Actual vs Predicted scatter plot
Prediction & Submission — predictions generated on test.csv and saved to submission_v2.csv (Kaggle submission format)


Tech Stack


Python 3
pandas, numpy
matplotlib
scikit-learn


Getting Started


Clone the repo:


bash   git clone https://github.com/<Ayushvats23>/house-price-prediction.git
   cd house-price-prediction



Launch the notebook:


bash   jupyter notebook notebooks/House_Price_Prediction.ipynb

Results

The Linear Regression model is evaluated using MSE and R² on a held-out validation split. See the notebook for the exact metric values and the Actual vs Predicted plot.

Possible Improvements


Handle missing values more thoroughly (e.g. imputation instead of fillna(0))
Include categorical features via one-hot encoding
Try regularized models (Ridge, Lasso) or tree-based models (Random Forest, XGBoost)
Log-transform SalePrice to reduce skew
Cross-validation instead of a single train/val split


License

This project is open source and available under the MIT License.
