import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = ["proto", "service", "state"]

TARGET = "label"

DROP_COLUMNS = ["id", "attack_cat", TARGET]


def prepare_data(train_df, test_df):
    X_train = train_df.drop(columns=DROP_COLUMNS)
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=DROP_COLUMNS)
    y_test = test_df[TARGET]

    numeric_features = [
        col for col in X_train.columns
        if col not in CATEGORICAL_FEATURES
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor