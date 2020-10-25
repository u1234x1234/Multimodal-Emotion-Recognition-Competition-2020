from sklearn.model_selection import train_test_split
import pandas as pd


def get_split():
    df = pd.read_csv("data/train.csv")
    df = [f"data/train{}" for path, label in df.values]
    # train_test_split()
