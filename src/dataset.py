"""
Created on April 5, 2022

dataset.py contains functions for pre-processing User Item Dataset, populate negative samples for training, generate random negative test samples for evaluation metrics.

@author: Rajasvi Vinayak Sharma (rvsharma@ucsd.edu)
"""

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pandas as pd


class MovieUserItemRatingDataset(Dataset):
    """This class is used to create a Torch.Dataset object for existing numpy array training/test data so it can be used easily by PyTorch model. 

    Args:
        users : User one-hot encoded vector data.
        items: Item one-hot encoded vector data.
        ratings: Implicit feedback (0/1) output to be predicted.
    """

    def __init__(self, users, items, ratings):
        self.users = torch.from_numpy(users)
        self.items = torch.from_numpy(items)
        self.ratings = torch.from_numpy(ratings)

    def __getitem__(self, index):
        return [self.users[index], self.items[index], self.ratings[index].float()]

    def __len__(self):
        return len(self.users)


def preprocessData(path):
    """This function is used to read raw training data and preprocess explicit feeedback into implicit feedback(0/1).

    Args:
        path (str): Path of ratings.dat

    Returns:
        _type_: _description_
    """
    df = pd.read_csv(
        path,
        sep="::",
        names=["users", "items", "rating", "timestamp"],
        engine="python",
    )

    df["rank_latest"] = df.groupby(["users"])["timestamp"].rank(
        method="first", ascending=False
    )

    df["rating"] = 1
    return df


def populateNegativeSamples(df, users_all, items_all, neg_per_pos=4):
    """This function is used to populate negative interactions for every positive interaction since the entire available data is just positive interactions in case of implicit feedback. This function populates number of negative samples as per the param neg_per_pos and returns the populated dataset.

    Args:
        df (pd.DataFrame): User-item dataframe with only positive user-item interactions.
        users_all (set): Set of unique users.
        items_all (set): Set of unique items.
        neg_per_pos (int, optional): Number of negative interactions to be populated for every positive interaction. Defaults to 4.

    Returns:
        pd.DataFrame: User-Item data with negative samples.
    """
    users_fin = df["users"].values.copy()
    items_fin = df["items"].values.copy()
    ratings_fin = df["rating"].values.copy().astype(float)

    for u in tqdm(users_all):
        items_per_user = set(df[df.users == u]["items"].unique())

        neg_item = np.random.choice(
            list(items_all - items_per_user), len(items_per_user) * neg_per_pos
        )

        users_fin = np.append(
            users_fin, np.repeat(u, len(items_per_user) * neg_per_pos)
        )
        items_fin = np.append(items_fin, neg_item)
        ratings_fin = np.append(
            ratings_fin, np.repeat(0, len(items_per_user) * neg_per_pos)
        )

    return [users_fin, items_fin, ratings_fin]


def generateNegTestSamples(test_df, items_all, neg_num=100):
    """This function is used to generate negative testing samples per user to evaluate model performance.

    Args:
        test_df (pd.DataFram): Test Data.
        items_all (set): Set of unique items.
        neg_num (int, optional ): Number of negative test samples generated per user. Defaults to 100.

    Returns:
        dict: Returns a dict with key: user, value: negative sampels per user.
    """
    neg_samples = {}
    for i in range(len(test_df)):
        user, item = test_df.iloc[i]["users"], test_df.iloc[i]["items"]
        if user not in neg_samples:
            neg_item = np.random.choice(list(items_all - set([int(item)])), neg_num)
            neg_samples[user] = neg_item
    return neg_samples
