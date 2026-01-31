import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    ndcg_score
)
from scipy.stats import spearmanr

def evaluate_ranking_predictions(test_data_path, pred_column='predicted_relevance_score', label_column='relevance_label'):
    """
    Evaluate ranking predictions using multiple metrics
    """
    df = pd.read_csv(test_data_path)
    
    # Remove rows with NaN values
    df_clean = df.dropna(subset=[pred_column, label_column])
    
    predictions = df_clean[pred_column].values
    ground_truth = df_clean[label_column].values
    
    # 1. NDCG@k - Normalized Discounted Cumulative Gain
    ndcg_scores = []
    for k in [1, 3, 5, 10]:
        if len(df_clean) >= k:
            # Reshape for ndcg_score (requires 2D array)
            ndcg = ndcg_score(
                y_true=[ground_truth],
                y_score=[predictions],
                k=k
            )
            ndcg_scores.append(f"NDCG@{k}: {ndcg:.4f}")
    
    # 2. Spearman Rank Correlation - measures correlation between predicted and actual rankings
    spearman_corr, spearman_pval = spearmanr(predictions, ground_truth)
    
    # 3. Mean Absolute Error
    mae = mean_absolute_error(ground_truth, predictions)
    
    # 4. Mean Squared Error / RMSE
    mse = mean_squared_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    
    # 5. Precision metrics - if you want to treat it as binary (relevant/not relevant)
    threshold = np.median(ground_truth)  # or use 0.5
    pred_binary = (predictions >= threshold).astype(int)
    truth_binary = (ground_truth >= threshold).astype(int)
    
    tp = np.sum((pred_binary == 1) & (truth_binary == 1))
    fp = np.sum((pred_binary == 1) & (truth_binary == 0))
    fn = np.sum((pred_binary == 0) & (truth_binary == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("=" * 50)
    print("RANKING EVALUATION METRICS")
    print("=" * 50)
    print(f"\nNDCG Scores (higher is better):")
    for score in ndcg_scores:
        print(f"  {score}")
    
    print(f"\nSpearman Rank Correlation: {spearman_corr:.4f} (p-value: {spearman_pval:.4e})")
    print(f"  (Range: -1 to 1, closer to 1 means better ranking correlation)")
    
    print(f"\nRegression Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    print(f"\nBinary Classification Metrics (threshold={threshold:.4f}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    print("\n" + "=" * 50)
    print("METRIC INTERPRETATION:")
    print("=" * 50)
    print("- NDCG@k: Measures if top-k results are ranked well (0-1, higher=better)")
    print("- Spearman Correlation: -1 to 1 (closer to 1 = better ranking correlation)")
    print("- MAE/RMSE: Lower is better (absolute prediction errors)")
    print("- Precision/Recall/F1: For binary relevance classification")
    
    return {
        'ndcg_scores': ndcg_scores,
        'spearman_corr': spearman_corr,
        'mae': mae,
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    evaluate_ranking_predictions('relevance_test_data_with_predictions.csv')
