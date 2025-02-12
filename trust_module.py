# src/trust_module.py

import numpy as np
import math

def compute_similarity(v1, v2):
    """
    Calculate similarity between two learners using the provided formula (Equation 3.1):
    
        Sim(l_i, l_j) = (sum_{t=1}^k (v1[t] * v2[t])) / sqrt(sum_{t=1}^k (v1[t]^2)) * (1 / (1 + exp(-1/2)))
    
    Args:
        v1 (np.ndarray): Feature vector of learner 1.
        v2 (np.ndarray): Feature vector of learner 2.
    
    Returns:
        float: The calculated similarity.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.sqrt(np.sum(np.square(v1)))
    constant = 1 / (1 + math.exp(-1/2))
    
    if norm_v1 == 0:
        return 0.0
    
    similarity = (dot_product / norm_v1) * constant
    return similarity

def build_trusted_learners(learner_vectors, N):
    """
    Builds a trusted learners set (TRSet) by computing similarity between each pair of learners.
    
    Only positive similarities are considered and for each learner, the top N similar learners are retained.
    
    Args:
        learner_vectors (dict): Dictionary mapping learner ID to feature vector (np.ndarray).
        N (int): Maximum number of trusted learners to retain for each learner.
    
    Returns:
        dict: Dictionary mapping each learner to a list of tuples (other_learner_id, similarity).
    """
    trusted = {}
    for learner_id, vector in learner_vectors.items():
        similarities = []
        for other_id, other_vector in learner_vectors.items():
            if learner_id == other_id:
                continue
            sim = compute_similarity(vector, other_vector)
            if sim > 0:
                similarities.append((other_id, sim))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:N]
        trusted[learner_id] = similarities
    return trusted

def compute_confidence(r_li, r_vj, epsilon):
    """
    Computes the confidence score between two learners according to Equation (3.3).
    
    Args:
        r_li (float): Performance or rating of learner l.
        r_vj (float): Performance or rating of learner v.
        epsilon (float): Tolerance threshold.
    
    Returns:
        int: 1 if |r_li - r_vj| <= epsilon, otherwise 0.
    """
    return 1 if abs(r_li - r_vj) <= epsilon else 0

def build_confidence_matrix(learners_ratings, epsilon):
    """
    Builds the confidence matrix for all learners.
    
    Args:
        learners_ratings (dict): Dictionary mapping learner ID to rating/performance.
        epsilon (float): Threshold for computing confidence.
    
    Returns:
        tuple: (confidence_matrix (np.ndarray), learner_ids (list))
    """
    learner_ids = list(learners_ratings.keys())
    n = len(learner_ids)
    matrix = np.zeros((n, n))
    for i, learner_id in enumerate(learner_ids):
        for j, other_id in enumerate(learner_ids):
            if i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = compute_confidence(learners_ratings[learner_id], learners_ratings[other_id], epsilon)
    return matrix, learner_ids

def compute_course_utility(trusted_learners, course_counts):
    """
    Computes the utility of a course (Ut_c_N) based on the frequency of recommendations by trusted learners.
    
    This is a simplified implementation where:
      - course_counts is a dictionary mapping course IDs to their frequency or aggregated score.
      - The ratio |TRSet(N)| / |ConfSet(N)| is approximated by using the total count of trusted learners.
    
    Args:
        trusted_learners (dict): Dictionary mapping learner ID to list of trusted learners.
        course_counts (dict): Dictionary mapping course IDs to frequency or aggregated score.
    
    Returns:
        dict: Dictionary mapping course ID to utility score.
    """
    total_trusted = sum([len(lst) for lst in trusted_learners.values()])
    total_course_count = sum(course_counts.values()) if sum(course_counts.values()) > 0 else 1
    utility = {}
    for course_id, count in course_counts.items():
        utility[course_id] = (count * total_trusted) / total_course_count
    return utility

# Example usage:
if __name__ == '__main__':
    # Example feature vectors for learners
    learner_vectors = {
        'learner1': np.array([1, 2, 3]),
        'learner2': np.array([2, 1, 0]),
        'learner3': np.array([0, 1, 1])
    }
    N = 2
    trusted = build_trusted_learners(learner_vectors, N)
    print('Trusted Learners:', trusted)
    
    # Example ratings
    learners_ratings = {
        'learner1': 4.5,
        'learner2': 4.0,
        'learner3': 3.8
    }
    epsilon = 0.5
    confidence_matrix, learner_ids = build_confidence_matrix(learners_ratings, epsilon)
    print('Confidence Matrix:\n', confidence_matrix)
    
    # Example course counts
    course_counts = {'course1': 10, 'course2': 5, 'course3': 3}
    utility = compute_course_utility(trusted, course_counts)
    print('Course Utility:', utility)
