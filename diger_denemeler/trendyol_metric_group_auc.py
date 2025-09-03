import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import roc_auc_score

class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    This metric calculates the weighted mean group AUC for ordered and clicked items across all sessions.
    
    The metric processes a submission dataframe containing session_id and prediction columns,
    where prediction is a space-separated string of content IDs.
    
    It compares these predictions against the ground truth solution to calculate:
    1. Ordered AUC - how well predictions rank ordered items vs non-ordered items
    2. Clicked AUC - how well predictions rank clicked items vs non-clicked items
    
    For each session, AUC is calculated based on the ranking of relevant vs irrelevant items.
    The final score is the weighted mean of these two AUC values, with ordered AUC
    weighted at 70% and clicked AUC weighted at 30%.
    
    >>> import pandas as pd
    >>> row_id_column_name = "session_id"
    >>> solution = pd.DataFrame({
    ...     "session_id": ["s1", "s2"],
    ...     "ordered_items": ["c1", "c3"],
    ...     "clicked_items": ["c1 c2", "c3 c4"],
    ...     "all_items": ["c1 c2 c3 c4 c5 c6", "c3 c4 c5 c6 c7 c8"]
    ... })
    >>> submission = pd.DataFrame({
    ...     "session_id": ["s1", "s2"],
    ...     "prediction": ["c1 c2 c5 c3 c4 c6", "c3 c4 c5 c6 c7 c8"]
    ... })
    >>> round(score(solution.copy(), submission.copy(), row_id_column_name), 4)
    Ordered AUC:  1.0
    Clicked AUC:  1.0
    1.0
    '''

    # Validate input data
    if row_id_column_name not in solution.columns or row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Row ID column '{row_id_column_name}' not found in both dataframes")
    
    if "prediction" not in submission.columns:
        raise ParticipantVisibleError("Submission must contain a 'prediction' column")
    
    if "ordered_items" not in solution.columns or "clicked_items" not in solution.columns:
        raise ParticipantVisibleError("Solution must contain 'ordered_items' and 'clicked_items' columns")
    
    if "all_items" not in solution.columns:
        raise ParticipantVisibleError("Solution must contain 'all_items' column")
    
    # Process submission: expand predictions into lists
    submission_dict = {}
    for _, row in submission.iterrows():
        session_id = row[row_id_column_name]
        
        # Split predictions and strip whitespace
        predictions = [p.strip() for p in row["prediction"].split()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_predictions = []
        for pred in predictions:
            if pred not in seen:
                seen.add(pred)
                unique_predictions.append(pred)
        
        submission_dict[session_id] = unique_predictions
    
    # Calculate AUC for each session
    ordered_aucs = []
    clicked_aucs = []
    
    for _, row in solution.iterrows():
        session_id = row[row_id_column_name]
        
        # Skip if session not in submission
        if session_id not in submission_dict:
            continue
            
        predictions = submission_dict[session_id]
        
        # Get all ground truth items
        all_items = []
        if isinstance(row["all_items"], str) and row["all_items"].strip():
            all_items = [item.strip() for item in row["all_items"].split()]
        
        # Check if the set of predicted and ground truth items are exactly the same
        predicted_set = set(predictions)
        gt_set = set(all_items)
        
        if predicted_set != gt_set:
            missing_in_predictions = gt_set - predicted_set
            extra_in_predictions = predicted_set - gt_set
            
            error_msg = f"Session {session_id}: predicted and ground truth item sets must be exactly the same."
            if missing_in_predictions:
                error_msg += f" Missing from predictions: {list(missing_in_predictions)}."
            if extra_in_predictions:
                error_msg += f" Extra in predictions: {list(extra_in_predictions)}."
            
            raise ParticipantVisibleError(error_msg)
        
        # Process ordered items
        ordered_items = []
        if isinstance(row["ordered_items"], str) and row["ordered_items"].strip():
            ordered_items = [item.strip() for item in row["ordered_items"].split()]
        
        if ordered_items and len(predictions) > 0:
            # Create binary labels and scores for ordered items
            y_true_ordered = []
            y_scores_ordered = []
            
            # Add all predicted items with their ranking scores (higher rank = higher score)
            for i, pred in enumerate(predictions):
                y_true_ordered.append(1 if pred in ordered_items else 0)
                y_scores_ordered.append(len(predictions) - i)  # Higher rank gets higher score
            
            # Only calculate AUC if we have both positive and negative examples
            if len(set(y_true_ordered)) > 1:
                try:
                    ordered_auc = roc_auc_score(y_true_ordered, y_scores_ordered)
                    ordered_aucs.append(ordered_auc)
                except ValueError:
                    # In case of any AUC calculation issues, skip this session
                    pass
        
        # Process clicked items
        clicked_items = []
        if isinstance(row["clicked_items"], str) and row["clicked_items"].strip():
            clicked_items = [item.strip() for item in row["clicked_items"].split()]
        
        if clicked_items and len(predictions) > 0:
            # Create binary labels and scores for clicked items
            y_true_clicked = []
            y_scores_clicked = []
            
            # Add all predicted items with their ranking scores (higher rank = higher score)
            for i, pred in enumerate(predictions):
                y_true_clicked.append(1 if pred in clicked_items else 0)
                y_scores_clicked.append(len(predictions) - i)  # Higher rank gets higher score
            
            # Only calculate AUC if we have both positive and negative examples
            if len(set(y_true_clicked)) > 1:
                try:
                    clicked_auc = roc_auc_score(y_true_clicked, y_scores_clicked)
                    clicked_aucs.append(clicked_auc)
                except ValueError:
                    # In case of any AUC calculation issues, skip this session
                    pass
    
    # Calculate mean AUCs
    ordered_auc = np.mean(ordered_aucs) if ordered_aucs else 0.5  # Default to 0.5 (random) if no valid AUCs
    clicked_auc = np.mean(clicked_aucs) if clicked_aucs else 0.5   # Default to 0.5 (random) if no valid AUCs
    
    print("Ordered AUC: ", ordered_auc)
    print("Clicked AUC: ", clicked_auc)
    
    # Final score is the weighted mean of ordered and clicked AUC
    final_score = 0.7 * ordered_auc + 0.3 * clicked_auc
    
    return final_score