import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# rmse
def compute_rmse(predicted_ratings, actual_ratings):
    non_zero_indices = actual_ratings.nonzero()[0]
    rmse = sqrt(mean_squared_error(actual_ratings[non_zero_indices], predicted_ratings[non_zero_indices]))
    return rmse

# mae
def compute_mae(predicted_ratings, actual_ratings):
    non_zero_indices = actual_ratings.nonzero()[0]
    mae = mean_absolute_error(actual_ratings[non_zero_indices], predicted_ratings[non_zero_indices])
    return mae

# K precision
def precision_at_k(predictions, actual_ratings, k):
    top_k_preds = np.argsort(-predictions, axis=1)[:, :k]
    precision_scores = []

    for user_index in range(predictions.shape[0]):
        relevant_items = actual_ratings[user_index].nonzero()[0]
        top_k_items = top_k_preds[user_index]
        num_relevant_in_top_k = np.intersect1d(top_k_items, relevant_items).shape[0]
        precision_scores.append(num_relevant_in_top_k / k)

    return np.mean(precision_scores)