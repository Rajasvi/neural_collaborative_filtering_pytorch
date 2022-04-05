import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pandas as pd


class MovieUserItemRatingDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.from_numpy(users)
        self.items = torch.from_numpy(items)
        self.ratings = torch.from_numpy(ratings)

    def __getitem__(self, index):
        return [self.users[index], self.items[index], self.ratings[index].float()]

    def __len__(self):
        return len(self.users)


def preprocessData(path):
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


def populateNegativeSamples(df, users_all, items_all, neg_per_pos=100):
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


def generateNegTestSamples(test_df, items_all, neg_num):
    neg_samples = {}
    for i in range(len(test_df)):
        user, item = test_df.iloc[i]["users"], test_df.iloc[i]["items"]
        if user not in neg_samples:
            neg_item = np.random.choice(list(items_all - set([int(item)])), neg_num)
            neg_samples[user] = neg_item
    return neg_samples
