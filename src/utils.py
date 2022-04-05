"""
Created on April 5, 2022

utils.py contains utility functions for training and evaluating model performance.

@author: Rajasvi Vinayak Sharma (rvsharma@ucsd.edu)
"""

import torch
import numpy as np
import heapq
import math


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, loss_fn, optimizer):
    """This function is used to train model for given arguments.

    Args:
        dataloader (torch.utils.data.DataLoader): Training user-item data.
        model (torch.nn.Module): PyTorch NN model. 
        loss_fn (torch.nn.): Loss function. Eg. toch.nn.BCELoss
        optimizer (torch.optim): Optimizer. Eg. torch.optim.Adam

    Returns:
        list: List of loss per 1000 epochs.
    """
    size = len(dataloader.dataset)
    model.train()
    tot_loss = []
    for batch, (X_users, X_items, y) in enumerate(dataloader):
        X_users, X_items, y = X_users.to(device), X_items.to(device), y.to(device)

        # Compute prediction error
        pred = model(X_users, X_items)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X_users)
            tot_loss.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print(f"Mean Training Loss: {np.mean(tot_loss)} ")
    return np.mean(tot_loss)


def evaluate_test(loss_fn, neg_samples, test_df, model, k=10, neg_num=100):
    """This function is used to evaluate model performance on negative test samples and calculate evaluation metrics: Eg. HR@10, NDCG@10.

    Args:
        loss_fn (torch.nn): Loss function. Eg. toch.nn.BCELoss
        neg_samples (dict): a dict with key: user, value: negative sampels per user.
        test_df (pd.DataFrame): Test data.
        model (torch.nn.Module): PyTorch model.
        k (int, optional): Top k for which evaluation metrics need to be calculated. Defaults to 10.
        neg_num (int, optional): Number of negative test samples generated per user. Defaults to 100.

    Returns:
        list: [HR@k, NDCG@k, Mean Loss]
    """
    users_test_all = test_df["users"].unique()
    ndcg, hit, total = 0, 0, len(users_test_all)
    tot_loss = []

    for i in range(len(test_df)):
        user, item = test_df.iloc[i]["users"], test_df.iloc[i]["items"]

        users_test = torch.LongTensor(np.repeat(user, neg_num + 1)).to(device)
        items_test = torch.LongTensor(np.append(item, neg_samples[user])).to(device)
        ratings_test = torch.FloatTensor(np.append(1, np.repeat(0, neg_num))).to(device)

        model.eval()
        with torch.no_grad():
            pred = model(users_test, items_test)
            loss = loss_fn(pred, ratings_test)
            tot_loss.append(loss.item())

        hp = []
        for curr_item, curr_rating in zip(items_test.cpu().numpy(), pred.cpu().numpy()):
            heapq.heappush(hp, (curr_rating, curr_item))

        topK = heapq.nlargest(k, hp, key=lambda x: x[0])
        topKitems = set([x[1] for x in topK])
        if item in topKitems:
            hit += 1

        for j, itemi in enumerate(topKitems):
            if itemi == item:
                ndcg += math.log(2) / math.log(j + 2)

    print(f"Mean Test Loss: {np.mean(tot_loss)}")
    print(f"\n Hit Ratio | HR@{k} : {hit/total}")
    print(f" NDCG@{k} : {ndcg/total}")

    return [hit / total, ndcg / total, np.mean(tot_loss)]
