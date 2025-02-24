# src/recommend.py

def compute_reccl(performance_list, weight_list):
    """
    Computes the weighted performance score (Reccl) for a course using Equation (3.5):
    
        Reccl = (sum(P_i * W(P_i))) / (sum(P_i))
    
    Args:
        performance_list (list of floats): List of performances (P_i) for the course.
        weight_list (list of floats): List of corresponding weights (W(P_i)).
    
    Returns:
        float: The computed Reccl score.
    """
    if not performance_list or not weight_list or sum(performance_list) == 0:
        return 0.0
    numerator = sum(p * w for p, w in zip(performance_list, weight_list))
    denominator = sum(performance_list)
    reccl = numerator / denominator
    return reccl

def recommend_courses(learner_performances, course_utilities):
    """
    Generates a list of recommended courses for a learner by combining:
      - Reccl: The weighted performance score for each course.
      - Ut_c_N: The course utility computed from trusted learners.
    
    The final recommendation score is given by:
    
        Recommendation = Reccl + Ut_c_N  (Equation 3.6)
    
    Args:
        learner_performances (dict): Dictionary mapping course ID to a tuple (list_of_performances, list_of_weights)
                                      for a given learner.
        course_utilities (dict): Dictionary mapping course ID to its utility score (Ut_c_N).
    
    Returns:
        list of tuples: Sorted list of tuples (course_id, recommendation_score) in descending order.
    """
    recommendations = {}
    for course_id, (performance_list, weight_list) in learner_performances.items():
        reccl = compute_reccl(performance_list, weight_list)
        utility = course_utilities.get(course_id, 0.0)
        recommendation_score = reccl + utility
        recommendations[course_id] = recommendation_score
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations


