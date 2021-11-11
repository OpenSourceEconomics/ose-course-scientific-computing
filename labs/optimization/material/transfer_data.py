"""This module transfers the files from Green's textbook to data frames."""
import numpy as np
import pandas as pd


def transfer_data():
    """Transfer data from .txt-file to pickled pandas.DataFrame."""
    df = pd.read_csv(
        "TableF5-2.txt",
        sep=r"\s+",
        engine="python",
        usecols=["Year", "qtr", "realgdp", "realcons"],
    )
    df = df.astype({"Year": np.int, "qtr": np.int})
    df.set_index(["Year", "qtr"], inplace=True)

    df.to_pickle("data-consumption-function.pkl")

    df = pd.read_csv("TableF14-1.csv", index_col=["OBS"])
    df["INTERCEPT"] = 1
    df["GRADE"] = df["GRADE "]
    df.to_pickle("data-graduation-prediction.pkl")


if __name__ == "__main__":
    transfer_data()
