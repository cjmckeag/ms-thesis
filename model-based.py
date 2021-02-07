import pandas as pd
import numpy as np


class NaiveBayes:
    """
    Class to implement model-based filtering using a Naive Bayes classifier.
    """

    def __init__(self, df):
        """
        Constructs necessary parameters for Naive Bayes classifier.
        :param df: dataframe; must have "item", "rating", and "user" columns
        """
        self.df = df

    def predict_rating(self, u, i):
        """
        computes predicted rating for a new item i and user u.
        :param u: string, user name
        :param i: string, new item name
        :return: weighted average variation predicted rating
        """
        priors = self.compute_prior(i)
        probs = self.compute_cond_prob(u, i)
        predicted_probs = np.dot(priors, probs)
        yhat_ui = np.argmax(predicted_probs) + 1
        return yhat_ui

    def compute_prior(self, i):
        """
        computes prior probability vector
        :param i: item name, string
        :return: prior probability vector
        """
        prior_vector = np.zeros(5)
        for rating in range(1, 6):
            rating_counts = self.df.loc[self.df['item'] == i].groupby('rating').count()
            num_ratings = rating_counts.loc[rating][0]
            num_total_ratings = rating_counts.sum()[0]
            prior_vector[rating - 1] = num_ratings / num_total_ratings
        return prior_vector

    def compute_cond_prob(self, u, i):
        """
        calculates conditional probability of user u's ratings vector given
        :param u: string, user name
        :param i: string, new item name
        :return: conditional probability vector
        """
        I_u = self.df.loc[self.df['user'] == u].item.values
        prob_vector = np.ones(5)
        for rating in range(1, 6):
            users_rated_i = self.df.loc[(self.df['item'] == i) & (self.df['rating'] == rating)]
            for j in I_u:
                users_rated_j_and_i = pd.merge(self.df.loc[self.df['item'] == j], users_rated_i, how='inner', on='user')
                prob_vector[rating - 1] *= users_rated_j_and_i.user.nunique() / users_rated_i.user.nunique()
        return prob_vector
