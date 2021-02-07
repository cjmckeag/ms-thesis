import time

import pandas as pd
import numpy as np
import itertools


class ItemBased(object):
    """
    Class to implement an item-based recommender system. To use, initialize, run fit(), then predict_rating().
    """

    def __init__(self, df, c=10):
        """
        Constructs necessary parameters for ItemBased object.
        :param df: dataframe; must have "item", "rating", and "user" columns
        :param c: neighborhood size to use in predicting rating; default 10
        """
        self.df = df
        self.c = c

    def fit(self):
        """
        "fits" system by computing every pearson correlation between every pair of items
        :return:
        """
        # measure fitting time
        start = time.process_time()
        # create list of all pairs of items
        print("Creating pairs of items...")
        ijs = list(set(itertools.combinations(self.df.item.values, 2)))
        # create dictionary of all pearson correlations between items
        rho_dict = dict.fromkeys(ijs)
        # compute every pearson correlation
        print("Computing Pearson correlations...")
        for key in rho_dict:
            rho_dict[key] = self.pearson_correlation(key[0], key[1])
        self.rho_dict = rho_dict
        print("Done fitting in {s} seconds".format(s=time.process_time() - start))

    def predict_rating(self, i, u):
        """
        computes predicted rating for a new item i and user u.
        :param i: string, new item name
        :param u: string, user name
        :return: weighted average predicted rating
        """
        # get neighborhood of most similar items to i
        print("Finding top {c} most similar items to {item}".format(c=self.c, item=i))
        K_i = self.find_neighborhood(u, i)
        # compute predicted rating for new item i by user u
        num = 0
        denom = 0
        print("Computing prediction...")
        for k in K_i:
            # find the rating that user u gave to item k
            y_uk = self.df.loc[(self.df['item'] == k) & (self.df['user'] == u), 'rating'].values[0]
            # get correlation between i and k
            rho_ik = K_i[k]
            # add rating * weight of similarity
            num += y_uk * rho_ik
            # sum up weights
            denom += abs(rho_ik)
        # compute y-hat
        yhat_ui = num / denom
        return yhat_ui

    def pearson_correlation(self, i, j):
        """
        function to compute the Pearson Correlation between two items in a dataset
        :param i: string, item name
        :param j: string, item name
        :return: pearson correlation btwn i and j
        """
        # find ratings for each item
        df_i = self.df.loc[self.df['item'] == i]
        df_j = self.df.loc[self.df['item'] == j]
        # create set of all users who have rated both items i and j
        U_ij = pd.merge(df_i, df_j, how='inner', on='user').user.values
        # check if set is empty
        if U_ij.size == 0:
            return 0
        # initialize numerator and denominator sums of pearson correlation
        num = 0
        denom_i = 0
        denom_j = 0
        # for every user in U_ij
        for u in U_ij:
            # find rating on item i given by user u
            y_ui = self.df.loc[(self.df['item'] == i) & (self.df['user'] == u), 'rating'].values[0]
            # find item i's average rating across all users
            ybar_i = self.df.loc[self.df['item'] == i, 'rating'].mean()
            # find rating on item j given by user u
            y_uj = self.df.loc[(self.df['item'] == j) & (self.df['user'] == u), 'rating'].values[0]
            # find item j's average rating across all users
            ybar_j = self.df.loc[self.df['item'] == j, 'rating'].mean()
            # accumulate numerator and denominator
            num += (y_ui - ybar_i) * (y_uj - ybar_j)
            denom_i += (y_ui - ybar_i) ** 2
            denom_j += (y_uj - ybar_j) ** 2
        # compute pearson correlation
        # add an error buffer to prevent divide by zero
        # i.e. if |U_ij|=1 and y_ui=y_uj, then rho_ij=1
        rho_ij = (num + 10e-6) / (np.sqrt(denom_i) * np.sqrt(denom_j) + 10e-6)
        return rho_ij

    def find_neighborhood(self, u, i):
        """
        finds a size c neighborhood of the most similar items to item i
        :param u:
        :param i: item string name
        :return: dict of 10 highest item pair correlations involving i
        """
        # create dict of correlations with i - looks like {(i, j): rho}
        rho_i = {key: value for (key, value) in self.rho_dict.items() if i in key}
        # sort the correlations in descending order, and make it a dictionary
        rho_i_sorted = {k: v for k, v in sorted(rho_i.items(), key=lambda item: item[1], reverse=True)}
        # get items that user u has rated
        items_user_rated = self.df.loc[self.df['user'] == u].item.values
        # find items that user u has rated in sorted correlation dict, get top 10 (if there even are 10)
        K_i = {x: rho_i_sorted[(x, i)] for x in items_user_rated[:self.c]}  # for v in rho_i_sorted.values()}
        # return neighborhood
        return K_i
