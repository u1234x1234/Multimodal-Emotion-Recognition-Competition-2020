import sys

import pandas as pd

if __name__ == "__main__":
    df1 = pd.read_csv(sys.argv[1])
    df2 = pd.read_csv(sys.argv[2])
    dfs = df1.merge(df2, on="FileID")
    s = (dfs["Emotion_x"] == dfs["Emotion_y"]).mean()
    print(s)
    assert s == 1
