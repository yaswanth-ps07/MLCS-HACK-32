import pandas as pd

TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_data()

    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)
    print("\nTraining columns:")
    print(train_df.columns.tolist())