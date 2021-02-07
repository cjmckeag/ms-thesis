import pandas as pd
import numpy as np
import itertools
import time


class UserBased:
    """
    Class to implement a user-based recommender system. To use, initialize, run fit(), then predict_rating().
    """

    def __init__(self, df, c=10):
        """
        Constructs necessary parameters for UserBased object.
        :param df: dataframe; must have "item", "rating", and "user" columns
        :param c: neighborhood size to use in predicting rating; default 10
        """
        self.df = df
        self.c = c

    def fit(self):
        """
        "fits" system by computing every pearson correlation between every pair of users
        :return:
        """
        # measure fitting time
        start = time.process_time()
        # create list of all pairs of users
        uvs = list(itertools.combinations(self.df.item.values, 2))
        # initialize dictionary of all pearson correlations between users
        rho_dict = dict.fromkeys(uvs)
        # compute every pearson correlation
        for key in rho_dict:
            rho_dict[key] = self.pearson_correlation(key[0], key[1])
        self.rho_dict = rho_dict
        print("Done fitting in {s} seconds".format(s=time.process_time() - start))

    def predict_rating(self, u, i):
        """
        computes predicted rating for a new item i and user u.
        :param u: string, user name
        :param i: string, new item name
        :return: weighted average variation predicted rating
        """
        # get neighborhood of most similar users to u, who have rated item i
        W_u = self.find_neighborhood(u, i)
        # compute predicted rating for new item i by user u
        # first get average rating by user u across all items
        ybar_u = self.df.loc[self.df['user'] == u, 'rating'].mean()
        # initialize num/denom of rating
        num = 0
        denom = 0
        # for each user in u's neighborhood
        for w in W_u:
            # get user w's average rating
            ybar_w = self.df.loc[self.df['user'] == w, 'rating'].mean()
            # get user w's rating on i
            y_wi = self.df.loc[(self.df['item'] == i) & (self.df['user'] == w), 'rating'].values[0]
            # get correlation
            rho_uw = W_u[(u, w)]
            num += (y_wi - ybar_w) * rho_uw
            denom += rho_uw
        # compute y-hat
        yhat_ui = ybar_u + num / denom
        return yhat_ui

    def pearson_correlation(self, u, v):
        """
        function to compute the Pearson Correlation between two users in a dataset
        :param u: string, user name
        :param v: string, user name
        :return: pearson correlation btwn u and v
        """
        # find ratings for each item
        df_u = self.df.loc[self.df['user'] == u]
        df_v = self.df.loc[self.df['user'] == v]
        # create set of all items that both users u and v have rated
        I_uv = pd.merge(df_u, df_v, how='inner', on='item').item.values
        # initialize numerator and denominator sums of pearson correlation
        num = 0
        denom_u = 0
        denom_v = 0
        # for every item in I_uv
        for i in I_uv:
            # find rating on item i given by user u
            y_ui = self.df.loc[(self.df['item'] == i) & (self.df['user'] == u), 'rating'].values[0]
            # find user u's average rating across all items
            ybar_u = self.df.loc[self.df['user'] == u, 'rating'].mean()
            # find rating on item i given by user v
            y_vi = self.df.loc[(self.df['item'] == i) & (self.df['user'] == u), 'rating'].values[0]
            # find user v's average rating across all items
            ybar_v = self.df.loc[self.df['user'] == v, 'rating'].mean()
            # accumulate numerator and denominator
            num += (y_ui - ybar_u) * (y_vi - ybar_v)
            denom_u += (y_ui - ybar_u) ** 2
            denom_v += (y_vi - ybar_v) ** 2
        # compute pearson correlation
        rho_uv = num / (np.sqrt(denom_u) * np.sqrt(denom_v))
        return rho_uv

    def find_neighborhood(self, u, i):
        """
        finds a size c neighborhood of the most similar users to user u. the users must also have rated item i.
        :param u: target user name string
        :param i: item name string
        :return: dict of 10 highest user pair correlations involving u
        """
        # create dict of correlations with u - looks like {(u, v): rho}
        rho_u = {key: value for (key, value) in self.rho_dict.items() if u in key}
        # sort the correlations in descending order, and make it a dictionary
        rho_u_sorted = {k: v for k, v in sorted(rho_u.items(), key=lambda item: item[1], reverse=True)}
        # get users that rated item i
        users_rated_i = self.df.loc[self.df['item'] == i].user.values
        # find users that rated i in sorted correlation dict, get top 10 (if there even are 10)
        W_u = {x: rho_u_sorted[(x, u)] for x in users_rated_i[:self.c]}  # for v in rho_i_sorted.values()}
        # return neighborhood
        return W_u
