import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def user_based_cf(ratings, target_user, k=5):
    """
    User-based Collaborative Filtering:
    Predict ratings for the target user based on similar users.

    Args:
        ratings (pd.DataFrame): Rows are users, columns are items, values are ratings.
        target_user (str or index): User ID for whom to recommend.
        k (int): Number of nearest neighbors to consider.

    Returns:
        pd.Series: Predicted ratings for items that target_user hasn't rated, sorted in descending order.
    """
    # Fill missing values with 0 for similarity computation
    similarity_matrix = cosine_similarity(ratings.fillna(0))
    similarity_df = pd.DataFrame(similarity_matrix, index=ratings.index, columns=ratings.index)
    
    # Get similarity scores for the target user (excluding itself)
    sim_scores = similarity_df[target_user].drop(target_user)
    top_neighbors = sim_scores.sort_values(ascending=False).head(k)
    
    # For items not rated by target user, compute a weighted average from neighbors
    target_user_ratings = ratings.loc[target_user]
    unrated_items = target_user_ratings[target_user_ratings.isna()].index
    
    predictions = {}
    for item in unrated_items:
        numerator = 0.0
        denominator = 0.0
        for neighbor in top_neighbors.index:
            neighbor_rating = ratings.at[neighbor, item]
            if not pd.isna(neighbor_rating):
                numerator += top_neighbors[neighbor] * neighbor_rating
                denominator += abs(top_neighbors[neighbor])
        if denominator != 0:
            predictions[item] = numerator / denominator
        else:
            predictions[item] = np.nan
    return pd.Series(predictions).dropna().sort_values(ascending=False)

def item_based_cf(ratings, target_user, k=5):
    """
    Item-based Collaborative Filtering:
    Predict ratings for the target user based on similarity between items.

    Args:
        ratings (pd.DataFrame): Rows are users, columns are items, values are ratings.
        target_user (str or index): User ID for whom to recommend.
        k (int): Number of similar items to consider.

    Returns:
        pd.Series: Predicted ratings for items that target_user hasn't rated, sorted in descending order.
    """
    # Transpose to compute similarity between items
    item_ratings = ratings.T
    similarity_matrix = cosine_similarity(item_ratings.fillna(0))
    similarity_df = pd.DataFrame(similarity_matrix, index=item_ratings.index, columns=item_ratings.index)
    
    target_user_ratings = ratings.loc[target_user]
    unrated_items = target_user_ratings[target_user_ratings.isna()].index
    
    predictions = {}
    for item in unrated_items:
        # Consider items that target user has rated
        similar_items = similarity_df[item]
        rated_items = target_user_ratings[target_user_ratings.notna()].index
        similar_items = similar_items[rated_items].sort_values(ascending=False).head(k)
        
        numerator = 0.0
        denominator = 0.0
        for other_item, sim in similar_items.items():
            rating = target_user_ratings[other_item]
            numerator += sim * rating
            denominator += abs(sim)
        if denominator != 0:
            predictions[item] = numerator / denominator
        else:
            predictions[item] = np.nan
    return pd.Series(predictions).dropna().sort_values(ascending=False)

def svd_recommender(ratings, target_user, k=5, n_components=2):
    """
    SVD-based Recommendation:
    Fill missing ratings using Singular Value Decomposition and then recommend items for the target user.

    Args:
        ratings (pd.DataFrame): Rows are users, columns are items.
        target_user (str or index): User ID for whom to recommend.
        k (int): Number of recommendations to return.
        n_components (int): Number of singular values/components to keep.

    Returns:
        pd.Series: Predicted ratings for items that target_user hasn't rated, sorted in descending order.
    """
    # Fill missing ratings with each user's mean rating
    ratings_filled = ratings.apply(lambda row: row.fillna(row.mean()), axis=1)
    user_means = ratings.mean(axis=1)
    ratings_centered = ratings_filled.sub(user_means, axis=0)
    
    # Compute SVD decomposition
    U, sigma, Vt = np.linalg.svd(ratings_centered, full_matrices=False)
    sigma_diag = np.diag(sigma)
    
    # Keep only n_components
    U_reduced = U[:, :n_components]
    sigma_reduced = sigma_diag[:n_components, :n_components]
    Vt_reduced = Vt[:n_components, :]
    
    ratings_pred = np.dot(np.dot(U_reduced, sigma_reduced), Vt_reduced)
    ratings_pred = ratings_pred + user_means.values.reshape(-1, 1)
    ratings_pred_df = pd.DataFrame(ratings_pred, index=ratings.index, columns=ratings.columns)
    
    target_user_ratings = ratings.loc[target_user]
    unrated_items = target_user_ratings[target_user_ratings.isna()].index
    predictions = ratings_pred_df.loc[target_user, unrated_items]
    return predictions.sort_values(ascending=False).head(k)

def social_regularization(ratings, social_matrix, target_user, k=5):
    """
    Social Regularization Recommendation:
    Incorporates social influence from a trust matrix to predict ratings for the target user.

    Args:
        ratings (pd.DataFrame): Rows are users, columns are items.
        social_matrix (pd.DataFrame): Rows and columns are users, representing trust or social similarity.
        target_user (str or index): User ID for whom to recommend.
        k (int): Number of neighbors to consider.

    Returns:
        pd.Series: Predicted ratings for items that target_user hasn't rated, sorted in descending order.
    """
    # Use social_matrix values as trust scores
    trust_scores = social_matrix.loc[target_user].drop(target_user)
    top_neighbors = trust_scores.sort_values(ascending=False).head(k)
    
    target_user_ratings = ratings.loc[target_user]
    unrated_items = target_user_ratings[target_user_ratings.isna()].index
    predictions = {}
    for item in unrated_items:
        numerator = 0.0
        denominator = 0.0
        for neighbor in top_neighbors.index:
            neighbor_rating = ratings.at[neighbor, item]
            if not pd.isna(neighbor_rating):
                numerator += top_neighbors[neighbor] * neighbor_rating
                denominator += abs(top_neighbors[neighbor])
        if denominator != 0:
            predictions[item] = numerator / denominator
        else:
            predictions[item] = np.nan
    return pd.Series(predictions).dropna().sort_values(ascending=False)

# Example usage
if __name__ == '__main__':
    # Create a sample ratings DataFrame (users x courses)
    data = {
        'course1': [5, 4, np.nan, 2, 1],
        'course2': [4, np.nan, 3, 2, 1],
        'course3': [np.nan, 3, 4, 2, 5],
        'course4': [2, 5, 3, np.nan, 4],
    }
    ratings = pd.DataFrame(data, index=['user1', 'user2', 'user3', 'user4', 'user5'])
    
    target_user = 'user3'
    
    print("User-based CF Recommendations for", target_user)
    print(user_based_cf(ratings, target_user))
    
    print("\nItem-based CF Recommendations for", target_user)
    print(item_based_cf(ratings, target_user))
    
    print("\nSVD Recommendations for", target_user)
    print(svd_recommender(ratings, target_user))
    
    # For social regularization, assume a social (trust) matrix is provided.
    # Here we create a dummy trust matrix for demonstration.
    trust_matrix = pd.DataFrame(np.random.rand(5, 5), index=ratings.index, columns=ratings.index)
    np.fill_diagonal(trust_matrix.values, 1)
    
    print("\nSocial Regularization Recommendations for", target_user)
    print(social_regularization(ratings, trust_matrix, target_user))
