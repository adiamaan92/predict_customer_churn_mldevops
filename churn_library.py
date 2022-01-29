"""
Churn Library
-------------

This library contains functions for performing EDA, feature engineering,
and model training.

The following functions are available:
1. import_data: Imports data given a file path
2. perform_eda: Performs EDA and stores the resulting graphs in results_path
3. perform_feature_engineering: Performs feature engineering
4. train_model: Trains a model given a dataframe and a response variable

The following variables are available:
1. eda_path: Path to store the EDA graphs
2. results_path: Path to store the results

"""
import os
from typing import Any, Callable, List, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

sns.set()
rcParams.update({"figure.autolayout": True})
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Define paths
eda_path = "./images/eda"
results_path = "./images/results"
models_path = "./models/"


def import_data(pth: str) -> pd.DataFrame:
    """Import data given a file path

    Args:
        pth (str): File path as string

    Returns:
        pd.DataFrame: Imported pandas dataframe
    """
    return pd.read_csv(pth)


def plot_result(
    df: pd.DataFrame,
    variable: str,
    plotter: Callable,
    title: str,
    path: str = eda_path,
):
    """Plot the results given a dataframe, variable, a plotter function
    and a path to store

    Args:
        df (pd.DataFrame): Dataframe
        variable (str): Variable to use in plot
        plotter (Callable): Lambda plotter function
        title (str): Title of the plot
        path (str, optional): [description]. Defaults to eda_path.
    """
    plt.figure(figsize=(20, 10))
    plotter(df, variable)
    plt.title(title)
    plt.savefig(f"{path}/{variable.lower()}_distribution.png")


def perform_eda(df: pd.DataFrame):
    """Performs eda and stores the resulting graphs in results_path

    Args:
        df (pd.DataFrame): DataFrame
    """
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    plot_result(df, "Churn", lambda df, x: df[x].hist(), "Churn Distribution")
    plot_result(
        df,
        "Customer_Age",
        lambda df, x: df[x].hist(),
        "Customer Age Distribution",
    )
    plot_result(
        df,
        "Marital_Status",
        lambda df, x: df[x].value_counts("normalize").plot(kind="bar"),
        "Marital Status Distribution",
    )
    plot_result(
        df,
        "Total_Trans_Ct",
        lambda df, x: sns.distplot(df[x]),
        "Total Trans Distribution",
    )
    plot_result(
        df,
        "Correlation_Plot",
        lambda df, x: sns.heatmap(
            df.corr(), annot=False, cmap="Dark2_r", linewidths=2
        ),
        "Correlation Plot",
    )


def encoder_helper(
    df: pd.DataFrame, category_lst: List[str], response: Optional[str] = None
) -> pd.DataFrame:
    """Encodes category columns with the mean of churn ratio for that category

    Args:
        df (pd.DataFrame): DataFrame
        category_lst (List[str]): List of categorical variables
        response (Optional[str]): Optional response variable

    Returns:
        pd.DataFrame: DataFrame with encoded categorical variables
    """
    response = "Churn" if response is None else response
    for category in category_lst:
        cat_list = []
        cat_groups = df.groupby(category).mean()["Churn"]

        for val in df[category]:
            cat_list.append(cat_groups.loc[val])

        df[f"{category}_Churn"] = cat_list

    return df


def perform_feature_engineering(
    df: pd.DataFrame, response: Optional[str] = None
) -> Any:
    """Feature engineering

    Args:
        df (pd.DataFrame): DataFrame
        response (Optional[str], optional): [description]. Defaults to None.

    Returns:
        List[Sequence]:
               X_train: X training data
               X_test: X testing data
               y_train: y training data
               y_test: y testing data
    """
    response = "Churn" if response is None else response
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]
    cat_cols = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    df = encoder_helper(df, cat_cols)

    X = df.loc[:, keep_cols]
    y = df["Churn"]

    return train_test_split(X, y, test_size=0.3, random_state=42)


def plot_scores(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_dash: np.ndarray,
    y_test_dash: np.ndarray,
    model_str: str,
    output_pth: str = results_path,
):
    """Plot and store scores given y and ydash for train and test dataset

    Args:
        y_train (np.ndarray): Training data response variable
        y_test (np.ndarray): Testing data response variable
        y_train_dash (np.ndarray): Training data predicted variable
        y_test_dash (np.ndarray): Testing data predicted variable
        model_str (np.ndarray): Model name as string. Used in storing the
        result
        output_pth ([str], optional): [description]. Defaults to results_path.
    """
    plt.figure(figsize=(5, 5))
    norm_model_str = model_str.replace("_", " ").title()
    # plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        norm_model_str,
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_dash)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        norm_model_str + " Test",
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_dash)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig(f"{output_pth}/{model_str}_metrics.png")


def classification_report_image(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_preds_lr: np.ndarray,
    y_train_preds_rf: np.ndarray,
    y_test_preds_lr: np.ndarray,
    y_test_preds_rf: np.ndarray,
):
    """
    produces classification report for training and testing results and
    stores report as image in images folder
    Args:
        y_train (np.ndarray): training response values
        y_test (np.ndarray):  test response values
        y_train_preds_lr (np.ndarray): training preds from logistic regression
        y_train_preds_rf (np.ndarray): training preds from random forest
        y_test_preds_lr (np.ndarray): test preds from logistic regression
        y_test_preds_rf (np.ndarray): test preds from random forest

    Returns:
             None
    """
    plot_scores(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        "Logistic Regression",
    )
    plot_scores(
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        "Random Forest",
    )


def feature_importance_plot(
    model: Union[RandomForestClassifier, LogisticRegression],
    X_data: pd.DataFrame,
    output_pth=results_path,
):
    """
    Creates and stores the feature importances in pth
    Args:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    Returns:
             None
    """
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the plot
    plt.savefig(f"{output_pth}/feature_importance.png")


def train_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
):
    """
    Train, store model results: images + scores, and store models
    Args:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    Returns:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(
        estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1
    )
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    plot_scores(
        y_train, y_test, y_train_preds_rf, y_test_preds_rf, "random_forest"
    )
    plot_scores(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        "logistic_regression",
    )
    feature_importance_plot(cv_rfc, X_train)

    joblib.dump(cv_rfc.best_estimator_, models_path + "rfc_model.pkl")
    joblib.dump(lrc, models_path + "logistic_model.pkl")


if __name__ == "__main__":
    df = import_data(r"./data/bank_data.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)
