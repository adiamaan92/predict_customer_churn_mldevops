"""
Churn Script -> Logging and Tests
---------------------------------

This module is responsible for testing and logging of churn_script.py

The following functions are available:
    test_import()   -> Tests that given a valid path a df is returned
    test_eda()      -> Tests that eda can be performed on the read dataset.
    test_encoder_helper() -> Tests that encoding is successfule
    test_perform_feature_engineering() -> Tests successful data splitting
    test_train_models() -> Tests successful training of models

"""
import logging
import os

import numpy as np

from churn_library import (
    encoder_helper,
    import_data,
    perform_eda,
    perform_feature_engineering,
    train_models,
)

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import():
    """
    Tests that given a valid path a df is returned

    Raises:
        err:
            FileNotFoundError if file is not found
            AssertionError if df is empty
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    """
    Tests that eda can be performed on the read dataset.
    perform_eda function relies on certain columns being present and will
    fail if those columns are not present.

    Raises:
        err: Assertion error on missing coumns
    """
    df = import_data("./data/bank_data.csv")
    # df = df.drop(columns="Customer_Age")
    try:
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logging.error("Column missing")
        logging.error(err)
        raise err


def test_encoder_helper():
    """
    On successful execution of encoder_helper, it should create columns
    with an _Churn. This function ensures that those columns are created

    Raises:
        err: Assertion error on missing coumns
    """
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    cat_cols = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    df = encoder_helper(df, cat_cols)

    try:
        for col in cat_cols:
            assert col + "_Churn" in df.columns
        logging.info("Testing encoder_help: SUCCESS")
    except AssertionError as err:
        logging.error("Columns missing")
        logging.error(err)
        raise err


def test_perform_feature_engineering():
    """
    On successful data splitting the train data set sould have 70% of the data
    and test should have 30% of the data as set in the splitting function

    Raises:
        err: Assertion error on improper data splitting
    """
    df = import_data(r"./data/bank_data.csv")
    perform_eda(df)
    X_train, X_test, _, _ = perform_feature_engineering(df)

    try:
        assert np.isclose(X_train.shape[0], df.shape[0] * 0.70, atol=1)
        assert np.isclose(X_test.shape[0], df.shape[0] * 0.30, atol=1)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Data splitting error")
        logging.error(err)
        raise err


def test_train_models():
    """
    On successful training of models, the models should be exported to the
    models folder along with results being exported as images under
    images/results folder

    Raises:
        err: Assertion error on model training error
    """
    df = import_data(r"./data/bank_data.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)

    try:
        # Makes sure results are present
        assert os.path.isfile("./images/results/feature_importance.png")
        assert os.path.isfile(
            "./images/results/logistic_regression_metrics.png"
        )
        assert os.path.isfile("./images/results/random_forest_metrics.png")

        # Makes sure models are saved
        assert os.path.isfile("./models/logistic_model.pkl")
        assert os.path.isfile("./models/rfc_model.pkl")
        logging.info("Testing train_models: SUCCESS")

    except AssertionError as err:
        logging.error("Model training error")
        logging.error(err)
        raise err


if __name__ == "__main__":
    test_perform_feature_engineering()
