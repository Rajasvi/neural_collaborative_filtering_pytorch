"""
Created on April 5, 2022

PyTorch Implementation of Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Rajasvi Vinayak Sharma (rvsharma@ucsd.edu)
"""

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import plotly.express as px
import numpy as np
from torch.utils.data import Dataset
from torch import nn
import plotly.express as px
from dataset import (
    preprocessData,
    MovieUserItemRatingDataset,
    populateNegativeSamples,
    generateNegTestSamples,
)
from utils import train, evaluate_test

device = "cuda" if torch.cuda.is_available() else "cpu"

# Sample command
# python src/GMF.py --dataset ratings.dat --epochs 2 --batch_size 256 --num_factors [8] --neg_per_pos 4 --lr 0.001 --learner adam --verbose 1 --out 1

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Generalized Matrix Factorization (GMF)."
    )
    parser.add_argument(
        "--path", nargs="?", default="../data/", help="Input data path."
    )
    parser.add_argument(
        "--dataset", nargs="?", default="ratings.dat", help="Choose a dataset."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--num_factors", nargs="?", default="[8]", help="Embedding size."
    )
    parser.add_argument(
        "--neg_per_pos",
        type=int,
        default=4,
        help="Number of negative instances to pair with a positive instance.",
    )

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--learner",
        nargs="?",
        default="adam",
        help="Specify an optimizer: adam, sgd, rmsprop, adagrad",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Show performance per epoch or not."
    )
    parser.add_argument(
        "--out", type=int, default=1, help="Whether to save the trained model."
    )
    parser.add_argument(
        "--MF", type=int, default=0, help="Whether to run Matrix Factorization or not."
    )
    return parser.parse_args()


class GMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, MF=False):
        super().__init__()
        # create user embeddings
        self.user_embeddings_GMF = torch.nn.Embedding(n_users, n_factors, sparse=False)
        # create item embeddings
        self.item_embeddings_GMF = torch.nn.Embedding(n_items, n_factors, sparse=False)
        self.h_output = torch.nn.Linear(in_features=n_factors, out_features=1)
        self.activation = torch.nn.Sigmoid()
        self.MF = MF

    def forward(self, user, item):
        # element-wise multiplication
        x = self.user_embeddings_GMF(user) * self.item_embeddings_GMF(item)

        # Param to convert GMF to MF if required.
        if self.MF:
            return self.activation(x.sum(1))

        x = self.h_output(x)
        x = self.activation(x)

        return torch.squeeze(x)


if __name__ == "__main__":
    # Parse arguments required for the model.
    args = parse_args()
    num_factors = eval(args.num_factors)
    neg_per_pos = args.neg_per_pos
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    topK = 10

    print("GMF arguments: %s" % (args))

    # Clean, pre-process and convert explicit into implicit feedback data.
    df = preprocessData(args.path + args.dataset)
    train_df, test_df = df[df["rank_latest"] > 1], df[df["rank_latest"] == 1]

    # Unique users and items.
    users_all = df["users"].unique()
    items_all = set(df["items"].unique())

    # Number of unique users and items.
    n_users = df["users"].nunique() + 1
    n_items = np.max(train_df["items"]) + 1

    # Populate negative interactions per positive instances in train data.
    train_users, train_items, train_ratings = populateNegativeSamples(
        train_df, users_all, items_all, neg_per_pos
    )

    # Generate random 100 negative interaction test samples for evaluation metrics i.e. HR@10 and NDCG@10.
    neg_test_samples = generateNegTestSamples(test_df, items_all, neg_num=100)

    training_data = MovieUserItemRatingDataset(train_users, train_items, train_ratings)

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )

    # Model hyper-params and evaluation metrics stored in all_out for different factors in model experimentations.
    all_out = []

    for f in num_factors:
        gmf = GMF(n_users=n_users, n_items=n_items, n_factors=f, MF=args.MF).to(device)
        loss_fn = nn.BCELoss()
        if learner.lower() == "adagrad":
            optimizer = torch.optim.Adagrad(gmf.parameters(), lr=learning_rate)
        elif learner.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(gmf.parameters(), lr=learning_rate)
        elif learner.lower() == "adam":
            optimizer = torch.optim.Adam(gmf.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(gmf.parameters(), lr=learning_rate)

        for t in range(epochs):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")
            loss = train(train_dataloader, gmf, loss_fn, optimizer)
            hr, ndcg, test_loss = evaluate_test(
                loss_fn, neg_test_samples, test_df, gmf, k=topK, neg_num=100
            )
            all_out.append(["GMF", t, f, loss, test_loss, hr, ndcg])
            if verbose:
                print("-------------------------------")

        if args.out:
            if args.MF:
                torch.save(gmf.state_dict(), f"../pretrain_models/{f}_MF.pt")
            else:
                torch.save(gmf.state_dict(), f"../pretrain_models/{f}_GMF.pt")
            print("Model saved successfully!")

        if verbose:
            print(f"Factor {f} | HR@10: {hr} | NDCG@10: {ndcg}")

    plot_df = pd.DataFrame(
        all_out,
        columns=[
            "model",
            "epoch",
            "factor",
            "train_loss",
            "test_loss",
            "HR@10",
            "NDCG@10",
        ],
    )
    plot_melt = pd.melt(
        plot_df,
        id_vars=["model", "epoch", "factor"],
        value_vars=["train_loss", "HR@10", "NDCG@10"],
        var_name="metric",
    )

    # Epoch-wise Loss, HR@10, NDCG@10 for each model
    fig_epoch = px.line(
        plot_melt,
        x="epoch",
        y="value",
        facet_col="metric",
        facet_row="factor",
        height=800,
    )
    fig_epoch.update_yaxes(matches=None)
    fig_epoch.update_yaxes(type="log")
    fig_epoch.update_traces(mode="markers+lines")
    fig_epoch.write_image("../plots/plot_epoch_GMF.png")

