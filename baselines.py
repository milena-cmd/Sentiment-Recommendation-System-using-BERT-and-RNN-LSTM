# src/baselines.py

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
    class DeepCoNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(DeepCoNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cnn = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, user_reviews, item_reviews):
        user_embedded = self.embedding(user_reviews).permute(0, 2, 1)
        item_embedded = self.embedding(item_reviews).permute(0, 2, 1)
        user_features = self.relu(self.cnn(user_embedded)).max(dim=2)[0]
        item_features = self.relu(self.cnn(item_embedded)).max(dim=2)[0]
        interaction = user_features * item_features
        output = self.fc(interaction)
        return output

def deepconn_recommender(user_reviews, item_reviews, target_user, k=5):
    """
    DeepCoNN-based Recommendation
    
    Args:
        user_reviews (dict): Dictionary mapping user IDs to their review text
        item_reviews (dict): Dictionary mapping item IDs to review text
        target_user (str): User ID for whom to recommend
        k (int): Number of recommendations to return

    Returns:
        pd.Series: Predicted ratings sorted in descending order
    """
    vocab_size = 5000  # Placeholder value
    model = DeepCoNN(vocab_size, embedding_dim=128, hidden_dim=64, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    user_inputs = torch.randint(0, vocab_size, (len(user_reviews), 100))
    item_inputs = torch.randint(0, vocab_size, (len(item_reviews), 100))
    
    model.train()
    for _ in range(10):  # Placeholder training loop
        optimizer.zero_grad()
        predictions = model(user_inputs, item_inputs).squeeze()
        loss = criterion(predictions, torch.rand(len(user_reviews)))
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        user_input = torch.randint(0, vocab_size, (1, 100))
        item_inputs = torch.randint(0, vocab_size, (len(item_reviews), 100))
        predictions = model(user_input, item_inputs).squeeze()
    
    recommendations = pd.Series(predictions.numpy(), index=item_reviews.keys()).sort_values(ascending=False)
    return recommendations.head(k)

# Define NARRE Model
class NARRE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(NARRE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cnn = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, user_reviews, item_reviews):
        user_embedded = self.embedding(user_reviews).permute(0, 2, 1)
        item_embedded = self.embedding(item_reviews).permute(0, 2, 1)
        user_features = self.relu(self.cnn(user_embedded))
        item_features = self.relu(self.cnn(item_embedded))
        user_weights = torch.softmax(self.attention(user_features).squeeze(), dim=1)
        item_weights = torch.softmax(self.attention(item_features).squeeze(), dim=1)
        user_features = (user_features * user_weights.unsqueeze(2)).sum(dim=1)
        item_features = (item_features * item_weights.unsqueeze(2)).sum(dim=1)
        interaction = user_features * item_features
        output = self.fc(interaction)
        return output

def narre_recommender(user_reviews, item_reviews, target_user, k=5):
    """
    NARRE-based Recommendation
    
    Args:
        user_reviews (dict): Dictionary mapping user IDs to their review text
        item_reviews (dict): Dictionary mapping item IDs to review text
        target_user (str): User ID for whom to recommend
        k (int): Number of recommendations to return

    Returns:
        pd.Series: Predicted ratings sorted in descending order
    """
    vocab_size = 5000  # Placeholder value
    model = NARRE(vocab_size, embedding_dim=128, hidden_dim=64, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    user_inputs = torch.randint(0, vocab_size, (len(user_reviews), 100))
    item_inputs = torch.randint(0, vocab_size, (len(item_reviews), 100))
    
    model.train()
    for _ in range(10):  # Placeholder training loop
        optimizer.zero_grad()
        predictions = model(user_inputs, item_inputs).squeeze()
        loss = criterion(predictions, torch.rand(len(user_reviews)))
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        user_input = torch.randint(0, vocab_size, (1, 100))
        item_inputs = torch.randint(0, vocab_size, (len(item_reviews), 100))
        predictions = model(user_input, item_inputs).squeeze()
    
    recommendations = pd.Series(predictions.numpy(), index=item_reviews.keys()).sort_values(ascending=False)
    return recommendations.head(k)
