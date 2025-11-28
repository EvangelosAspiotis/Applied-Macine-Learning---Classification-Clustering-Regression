# Applied-Macine-Learning---Classification-Clustering-Regression

This repository collects three example projects demonstrating supervised and unsupervised machine learning techniques: **classification**, **clustering (market segmentation)**, and **regression**.  
Each project is implemented in a separate Jupyter notebook, using a different dataset and target problem:

- **Classification** — predict whether a person’s annual income is > 50K.  
- **Clustering** — perform market segmentation on a dataset (unsupervised).  
- **Regression** — predict house sale prices.  


## Structure

    /
    ├── ML Project_Classification.ipynb       # Predict income > 50K (classification)
    ├── ML Project_Clustering.ipynb           # Market segmentation (clustering)
    ├── ML_Project_Regression.ipynb           # Predict house prices (regression)
    ├── ML Project Data/                      # Datasets used in the notebooks
    └── README.md
------------------------------------------------------------------------

## Classification -- Predicting Income 

**Notebook:** `ML Project_Classification.ipynb`\
**Goal:** Predict whether a person earns **more than 50,000 USD per
year**.

### What the notebook does

-   Loads dataset of demographic & employment features\
-   Performs basic EDA (distributions, correlations, missing values)\
-   Preprocesses data:
    -   Handles missing values
    -   Encodes categorical variables (One-Hot, Label Encoding, etc.)
    -   Scales numerical columns
-   Splits data into **train/test**
-   Trains classification model(s) such as:
    -   Logistic Regression\
    -   Decision Tree / Random Forest\
    -   Other sklearn classifiers
-   Evaluates performance using:
    -   Accuracy
    -   Precision, Recall
    -   Confusion Matrix
    -   ROC-AUC (if included)
-   Provides interpretation (feature importance / coefficients)

------------------------------------------------------------------------

## Clustering -- Market Segmentation

**Notebook:** `ML Project_Clustering.ipynb`\
**Goal:** Group customers into meaningful segments using unsupervised
learning.

### What the notebook does

-   Loads customer-related dataset\
-   Cleans & preprocesses:
    -   Missing values
    -   Categorical encoding (if needed)
    -   Feature scaling (important for distance-based clustering)
-   Determines number of clusters\
-   Fits **K-Means** clustering model\
-   Visualizes clusters with:
    -   PCA / t-SNE 2D projection
    -   Scatter plots
-   Performs cluster profiling:
    -   Identifies traits of each segment
 
------------------------------------------------------------------------


## Regression -- Predicting House Prices

**Notebook:** `ML_Project_Regression.ipynb`\
**Goal:** Predict house sale prices from property features.

### What the notebook does

-   Loads housing dataset\
-   Performs EDA\
-   Preprocesses input features:
    -   Handle missing values\
    -   Encode categorical fields\
    -   Scale numerical fields\
-   Splits dataset into training/testing\
-   Trains regression model(s):
    -   Linear Regression
    -   Ridge / Lasso
    -   Tree-based Regressors
-   Evaluates model using:
    -   MAE, MSE, RMSE, R²\
-   Visualizes:
    -   Predictions vs. actual values
    -   Residual plots
    -   Feature importance
 
------------------------------------------------------------------------


#  How to Run

``` bash
git clone https://github.com/EvangelosAspiotis/Applied-Macine-Learning---Classification-Clustering-Regression.git
cd Applied-Macine-Learning---Classification-Clustering-Regression
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```
