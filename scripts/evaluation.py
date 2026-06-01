# evaluation_with_visuals.py

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from movie_recommender.config import ML_RATINGS_INTEGRATED_PATH

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory for plots
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 1. Load ratings and split into train / test
# ---------------------------------------------------------

def load_ratings():
    """Load integrated MovieLens ratings."""
    df = pd.read_csv(ML_RATINGS_INTEGRATED_PATH)
    required = {"userId", "movie_id", "rating"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Ratings file must contain {required}, got {df.columns.tolist()}"
        )

    print(f"[Eval] Loaded {len(df)} ratings from {ML_RATINGS_INTEGRATED_PATH}")
    return df


def train_test_split_by_user(ratings, test_ratio=0.2, min_ratings_per_user=5, random_state=42):
    """Split ratings into train and test by user."""
    rng = np.random.default_rng(random_state)

    train_rows = []
    test_rows = []

    for user_id, group in ratings.groupby("userId"):
        if len(group) < min_ratings_per_user:
            train_rows.append(group)
            continue

        idx = np.arange(len(group))
        rng.shuffle(idx)
        split = int(len(group) * (1 - test_ratio))
        train_idx = idx[:split]
        test_idx = idx[split:]

        train_rows.append(group.iloc[train_idx])
        test_rows.append(group.iloc[test_idx])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    print(f"[Eval] Train size: {len(train_df)} ratings")
    print(f"[Eval] Test size:  {len(test_df)} ratings")

    return train_df, test_df


def visualize_data_split(train_df, test_df):
    """Visualize train/test split statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Rating distribution comparison
    ax = axes[0, 0]
    bins = np.arange(0.5, 5.6, 0.5)
    ax.hist([train_df['rating'], test_df['rating']], bins=bins, 
            label=['Train', 'Test'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Frequency')
    ax.set_title('Rating Distribution: Train vs Test')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. User activity distribution
    ax = axes[0, 1]
    train_user_counts = train_df.groupby('userId').size()
    test_user_counts = test_df.groupby('userId').size()
    ax.hist([train_user_counts, test_user_counts], bins=50, 
            label=['Train', 'Test'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Ratings per User')
    ax.set_ylabel('Number of Users')
    ax.set_title('User Activity Distribution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Movie popularity
    ax = axes[1, 0]
    train_movie_counts = train_df.groupby('movie_id').size()
    test_movie_counts = test_df.groupby('movie_id').size()
    ax.hist([train_movie_counts, test_movie_counts], bins=50, 
            label=['Train', 'Test'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Ratings per Movie')
    ax.set_ylabel('Number of Movies')
    ax.set_title('Movie Popularity Distribution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    stats_data = [
        ['Metric', 'Train', 'Test'],
        ['Total Ratings', f'{len(train_df):,}', f'{len(test_df):,}'],
        ['Unique Users', f'{train_df["userId"].nunique():,}', f'{test_df["userId"].nunique():,}'],
        ['Unique Movies', f'{train_df["movie_id"].nunique():,}', f'{test_df["movie_id"].nunique():,}'],
        ['Avg Rating', f'{train_df["rating"].mean():.3f}', f'{test_df["rating"].mean():.3f}'],
        ['Sparsity', f'{100 * (1 - len(train_df) / (train_df["userId"].nunique() * train_df["movie_id"].nunique())):.2f}%', 
         f'{100 * (1 - len(test_df) / (test_df["userId"].nunique() * test_df["movie_id"].nunique())):.2f}%']
    ]
    table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_data_split_analysis.png", dpi=300, bbox_inches='tight')
    print(f"[Visual] Saved data split analysis to {OUTPUT_DIR / '01_data_split_analysis.png'}")
    plt.close()


# ---------------------------------------------------------
# 2. Build user-item matrix from train ratings
# ---------------------------------------------------------

def build_user_item_matrix(ratings: pd.DataFrame):
    """Build a sparse user-item rating matrix."""
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movie_id"].unique()

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    idx_to_user_id = {i: uid for uid, i in user_id_to_idx.items()}

    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    idx_to_movie_id = {i: mid for mid, i in movie_id_to_idx.items()}

    rows = ratings["userId"].map(user_id_to_idx).values
    cols = ratings["movie_id"].map(movie_id_to_idx).values
    data = ratings["rating"].astype(np.float32).values

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    R = csr_matrix((data, (rows, cols)), shape=(num_users, num_movies))

    print(f"[Eval] Train rating matrix shape: {R.shape} (users x movies)")

    return R, user_id_to_idx, idx_to_user_id, movie_id_to_idx, idx_to_movie_id


# ---------------------------------------------------------
# 3. User-based CF prediction
# ---------------------------------------------------------

def predict_rating_cf(user_id, movie_id, R, user_id_to_idx, movie_id_to_idx, k_neighbors=50):
    """Predict rating using user-based CF."""
    if not hasattr(predict_rating_cf, "_global_mean"):
        all_data = R.data
        predict_rating_cf._global_mean = float(all_data.mean()) if len(all_data) > 0 else 3.5
    global_mean = predict_rating_cf._global_mean

    if user_id not in user_id_to_idx or movie_id not in movie_id_to_idx:
        return global_mean

    u_idx = user_id_to_idx[user_id]
    m_idx = movie_id_to_idx[movie_id]

    user_row = R[u_idx]
    if user_row.nnz == 0:
        return global_mean

    sims = cosine_similarity(user_row, R)[0]
    sims[u_idx] = 0.0

    neighbor_indices = np.argsort(sims)[::-1]

    num_neighbors = 0
    weighted_sum = 0.0
    sim_sum = 0.0

    for n_idx in neighbor_indices:
        if num_neighbors >= k_neighbors:
            break
        sim = sims[n_idx]
        if sim <= 0:
            continue
        rating = R[n_idx, m_idx]
        if rating == 0:
            continue

        rating_val = float(rating)
        weighted_sum += sim * rating_val
        sim_sum += sim
        num_neighbors += 1

    if num_neighbors == 0 or sim_sum == 0.0:
        user_ratings = R[u_idx].data
        if len(user_ratings) > 0:
            return float(user_ratings.mean())
        return global_mean

    return weighted_sum / sim_sum


# ---------------------------------------------------------
# 4. Evaluate rating prediction (RMSE, MAE)
# ---------------------------------------------------------

def evaluate_rating_prediction(train_df, test_df, R, user_id_to_idx, movie_id_to_idx, sample_size=3000):
    """Compute RMSE and MAE with detailed analysis."""
    if len(test_df) > sample_size:
        test_sample = test_df.sample(n=sample_size, random_state=42)
    else:
        test_sample = test_df

    y_true = []
    y_pred = []
    errors = []

    total = len(test_sample)
    print(f"[Eval] Evaluating {total} test samples...")
    
    for idx, row in enumerate(test_sample.itertuples(), 1):
        if idx % 1000 == 0:
            print(f"[Eval] Progress: {idx}/{total} ({100*idx/total:.1f}%)")
        
        uid = row.userId
        mid = row.movie_id
        true_r = float(row.rating)
        pred_r = predict_rating_cf(uid, mid, R, user_id_to_idx, movie_id_to_idx, k_neighbors=30)
        y_true.append(true_r)
        y_pred.append(pred_r)
        errors.append(abs(true_r - pred_r))

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"[Eval] Rating prediction RMSE: {rmse:.4f}")
    print(f"[Eval] Rating prediction MAE:  {mae:.4f}")

    # Visualize prediction results
    visualize_rating_predictions(y_true, y_pred, errors, rmse, mae)

    return rmse, mae, y_true, y_pred, errors


def visualize_rating_predictions(y_true, y_pred, errors, rmse, mae):
    """Create comprehensive visualizations for rating predictions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Actual vs Predicted scatter
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    ax.plot([0, 5], [0, 5], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Rating')
    ax.set_ylabel('Predicted Rating')
    ax.set_title('Actual vs Predicted Ratings')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 5.5)
    
    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(mae, color='r', linestyle='--', linewidth=2, label=f'MAE = {mae:.3f}')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Error by actual rating
    ax = axes[0, 2]
    rating_bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    rating_labels = ['1', '2', '3', '4', '5']
    binned_errors = [[] for _ in range(5)]
    for true, err in zip(y_true, errors):
        bin_idx = int(true - 0.5)
        if 0 <= bin_idx < 5:
            binned_errors[bin_idx].append(err)
    
    bp = ax.boxplot(binned_errors, labels=rating_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('Actual Rating')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Distribution by Rating Level')
    ax.grid(alpha=0.3)
    
    # 4. Residuals
    ax = axes[1, 0]
    residuals = np.array(y_true) - np.array(y_pred)
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Rating')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title('Residual Plot')
    ax.grid(alpha=0.3)
    
    # 5. Cumulative error distribution
    ax = axes[1, 1]
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, cumulative, linewidth=2)
    ax.axvline(mae, color='r', linestyle='--', linewidth=2, label=f'MAE = {mae:.3f}')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Cumulative Proportion')
    ax.set_title('Cumulative Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. Metrics summary
    ax = axes[1, 2]
    ax.axis('off')
    metrics_data = [
        ['Metric', 'Value'],
        ['RMSE', f'{rmse:.4f}'],
        ['MAE', f'{mae:.4f}'],
        ['Mean Error', f'{np.mean(residuals):.4f}'],
        ['Std Error', f'{np.std(errors):.4f}'],
        ['Max Error', f'{np.max(errors):.4f}'],
        ['% Errors < 0.5', f'{100 * np.mean(np.array(errors) < 0.5):.1f}%'],
        ['% Errors < 1.0', f'{100 * np.mean(np.array(errors) < 1.0):.1f}%'],
    ]
    table = ax.table(cellText=metrics_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_rating_prediction_analysis.png", dpi=300, bbox_inches='tight')
    print(f"[Visual] Saved rating prediction analysis to {OUTPUT_DIR / '02_rating_prediction_analysis.png'}")
    plt.close()


# ---------------------------------------------------------
# 5. Evaluate Top-K metrics (Precision@K, Recall@K, NDCG, MAP, Coverage)
# ---------------------------------------------------------

def dcg_at_k(r, k):
    """Compute Discounted Cumulative Gain at K."""
    r = np.asarray(r, dtype=np.float64)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0


def ndcg_at_k(r, k):
    """Compute Normalized Discounted Cumulative Gain at K."""
    dcg = dcg_at_k(r, k)
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(relevant_items, recommended_items, k):
    """Compute Average Precision at K."""
    if not relevant_items:
        return 0.0
    
    recommended_items = recommended_items[:k]
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    if num_hits == 0:
        return 0.0
    
    return score / min(len(relevant_items), k)


def recommend_for_user_cf(user_id, R, user_id_to_idx, movie_id_to_idx, idx_to_movie_id, top_k=10, neighbor_k=50):
    """CF-based Top-N recommendation with scores."""
    if user_id not in user_id_to_idx:
        return [], []

    u_idx = user_id_to_idx[user_id]
    user_row = R[u_idx]
    user_ratings_dense = user_row.toarray()[0]
    seen_movies = set(np.where(user_ratings_dense > 0)[0])

    sims = cosine_similarity(user_row, R)[0]
    sims[u_idx] = 0.0

    neighbor_indices = np.argsort(sims)[::-1][:neighbor_k]
    neighbor_sims = sims[neighbor_indices]

    candidate_scores = defaultdict(float)
    sim_sums = defaultdict(float)

    for n_idx, sim in zip(neighbor_indices, neighbor_sims):
        if sim <= 0:
            continue
        neighbor_row = R[n_idx].toarray()[0]
        rated_indices = np.where(neighbor_row > 0)[0]

        for m_idx in rated_indices:
            if m_idx in seen_movies:
                continue
            rating = neighbor_row[m_idx]
            candidate_scores[m_idx] += sim * rating
            sim_sums[m_idx] += sim

    scored = []
    for m_idx, score in candidate_scores.items():
        if sim_sums[m_idx] > 0:
            scored.append((m_idx, score / sim_sums[m_idx]))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    rec_movie_ids = [idx_to_movie_id[m_idx] for m_idx, _ in top]
    rec_scores = [score for _, score in top]
    return rec_movie_ids, rec_scores


def evaluate_topk(train_df, test_df, R, user_id_to_idx, movie_id_to_idx, idx_to_movie_id,
                  k=10, rating_threshold=4.0, max_users=1000):
    """Evaluate comprehensive recommendation metrics."""
    users_with_test = test_df["userId"].unique()
    if len(users_with_test) > max_users:
        rng = np.random.default_rng(42)
        users_with_test = rng.choice(users_with_test, size=max_users, replace=False)

    precisions = []
    recalls = []
    f1_scores = []
    ndcg_scores = []
    map_scores = []
    hit_rates = []
    mrr_scores = []
    user_relevant_counts = []
    user_hit_counts = []
    
    all_recommended_movies = set()
    total_movies = len(movie_id_to_idx)

    total = len(users_with_test)
    print(f"[Eval] Evaluating {total} users for K={k}...")

    for idx, uid in enumerate(users_with_test, 1):
        if idx % 200 == 0:
            print(f"[Eval] Progress: {idx}/{total} ({100*idx/total:.1f}%)")
            
        user_test = test_df[test_df["userId"] == uid]
        relevant = set(user_test[user_test["rating"] >= rating_threshold]["movie_id"].tolist())
        if len(relevant) == 0:
            continue

        recs, scores = recommend_for_user_cf(uid, R, user_id_to_idx, movie_id_to_idx, idx_to_movie_id,
                                             top_k=k, neighbor_k=30)
        if not recs:
            continue

        recs_set = set(recs)
        all_recommended_movies.update(recs)
        hits = len(recs_set & relevant)

        # Standard metrics
        precision = hits / float(k)
        recall = hits / float(len(relevant))
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        user_relevant_counts.append(len(relevant))
        user_hit_counts.append(hits)
        
        # Hit Rate (binary: did we get at least one relevant item?)
        hit_rates.append(1.0 if hits > 0 else 0.0)
        
        # NDCG (considers ranking position)
        relevance = [1 if movie in relevant else 0 for movie in recs]
        ndcg = ndcg_at_k(relevance, k)
        ndcg_scores.append(ndcg)
        
        # MAP (Mean Average Precision)
        ap = average_precision_at_k(relevant, recs, k)
        map_scores.append(ap)
        
        # MRR (Mean Reciprocal Rank)
        reciprocal_rank = 0.0
        for rank, movie in enumerate(recs, 1):
            if movie in relevant:
                reciprocal_rank = 1.0 / rank
                break
        mrr_scores.append(reciprocal_rank)

    if not precisions:
        print("[Eval] No users with evaluable test data for top-K.")
        return {}, {}

    # Calculate catalog coverage
    catalog_coverage = len(all_recommended_movies) / total_movies
    
    # Average metrics
    metrics = {
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
        'f1': float(np.mean(f1_scores)),
        'ndcg': float(np.mean(ndcg_scores)),
        'map': float(np.mean(map_scores)),
        'hit_rate': float(np.mean(hit_rates)),
        'mrr': float(np.mean(mrr_scores)),
        'coverage': catalog_coverage
    }
    
    # Detailed data for visualization
    detailed = {
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'ndcg_scores': ndcg_scores,
        'map_scores': map_scores,
        'hit_rates': hit_rates,
        'mrr_scores': mrr_scores,
        'user_relevant_counts': user_relevant_counts,
        'user_hit_counts': user_hit_counts
    }

    print(f"[Eval] Precision@{k}: {metrics['precision']:.4f}")
    print(f"[Eval] Recall@{k}:    {metrics['recall']:.4f}")
    print(f"[Eval] F1@{k}:        {metrics['f1']:.4f}")
    print(f"[Eval] NDCG@{k}:      {metrics['ndcg']:.4f}")
    print(f"[Eval] MAP@{k}:       {metrics['map']:.4f}")
    print(f"[Eval] Hit Rate@{k}:  {metrics['hit_rate']:.4f}")
    print(f"[Eval] MRR@{k}:       {metrics['mrr']:.4f}")
    print(f"[Eval] Coverage:      {metrics['coverage']:.4f}")

    # Visualize top-K results
    visualize_topk_results(detailed, k, metrics)

    return metrics, detailed


def visualize_topk_results(detailed, k, metrics):
    """Visualize top-K recommendation performance with recommendation-specific metrics."""
    precisions = detailed['precisions']
    recalls = detailed['recalls']
    f1_scores = detailed['f1_scores']
    ndcg_scores = detailed['ndcg_scores']
    map_scores = detailed['map_scores']
    hit_rates = detailed['hit_rates']
    mrr_scores = detailed['mrr_scores']
    relevant_counts = detailed['user_relevant_counts']
    hit_counts = detailed['user_hit_counts']
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # 1. Precision distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(precisions, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(metrics['precision'], color='r', linestyle='--', linewidth=2, 
               label=f'Mean = {metrics["precision"]:.3f}')
    ax.set_xlabel(f'Precision@{k}')
    ax.set_ylabel('Number of Users')
    ax.set_title(f'Precision@{k} Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Recall distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(recalls, bins=30, edgecolor='black', alpha=0.7, color='seagreen')
    ax.axvline(metrics['recall'], color='r', linestyle='--', linewidth=2, 
               label=f'Mean = {metrics["recall"]:.3f}')
    ax.set_xlabel(f'Recall@{k}')
    ax.set_ylabel('Number of Users')
    ax.set_title(f'Recall@{k} Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. NDCG distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(ndcg_scores, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(metrics['ndcg'], color='r', linestyle='--', linewidth=2, 
               label=f'Mean = {metrics["ndcg"]:.3f}')
    ax.set_xlabel(f'NDCG@{k}')
    ax.set_ylabel('Number of Users')
    ax.set_title(f'NDCG@{k} Distribution (Ranking Quality)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. MAP distribution
    ax = fig.add_subplot(gs[0, 3])
    ax.hist(map_scores, bins=30, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax.axvline(metrics['map'], color='r', linestyle='--', linewidth=2, 
               label=f'Mean = {metrics["map"]:.3f}')
    ax.set_xlabel(f'MAP@{k}')
    ax.set_ylabel('Number of Users')
    ax.set_title(f'MAP@{k} Distribution (Average Precision)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. All metrics comparison
    ax = fig.add_subplot(gs[1, 0])
    metric_names = ['Precision', 'Recall', 'F1', 'NDCG', 'MAP', 'Hit Rate', 'MRR']
    metric_values = [metrics['precision'], metrics['recall'], metrics['f1'], 
                     metrics['ndcg'], metrics['map'], metrics['hit_rate'], metrics['mrr']]
    colors = ['steelblue', 'seagreen', 'coral', 'mediumpurple', 'gold', 'tomato', 'turquoise']
    bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Score')
    ax.set_title(f'All Recommendation Metrics @{k}', fontweight='bold')
    ax.set_xlim(0, max(metric_values) * 1.15)
    ax.grid(alpha=0.3, axis='x')
    for bar, val in zip(bars, metric_values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2., f'{val:.3f}',
                ha='left', va='center', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 6. Precision vs Recall scatter
    ax = fig.add_subplot(gs[1, 1])
    scatter = ax.scatter(recalls, precisions, c=ndcg_scores, cmap='viridis', alpha=0.6, s=30)
    ax.set_xlabel(f'Recall@{k}')
    ax.set_ylabel(f'Precision@{k}')
    ax.set_title('Precision vs Recall (colored by NDCG)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('NDCG Score')
    ax.grid(alpha=0.3)
    
    # 7. MRR distribution
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(mrr_scores, bins=30, edgecolor='black', alpha=0.7, color='turquoise')
    ax.axvline(metrics['mrr'], color='r', linestyle='--', linewidth=2, 
               label=f'Mean = {metrics["mrr"]:.3f}')
    ax.set_xlabel(f'MRR@{k}')
    ax.set_ylabel('Number of Users')
    ax.set_title(f'MRR@{k} Distribution (First Relevant Item)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 8. Hit Rate pie chart
    ax = fig.add_subplot(gs[1, 3])
    hit_count = sum(1 for hr in hit_rates if hr > 0)
    no_hit_count = len(hit_rates) - hit_count
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0)
    ax.pie([hit_count, no_hit_count], labels=['Got ≥1 Hit', 'No Hits'], 
           autopct='%1.1f%%', startangle=90, colors=colors_pie, explode=explode,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title(f'Hit Rate@{k}: {metrics["hit_rate"]:.1%} of Users', fontweight='bold')
    
    # 9. Performance by number of relevant items
    ax = fig.add_subplot(gs[2, :2])
    bins = [0, 2, 5, 10, 20, max(relevant_counts) + 1]
    bin_labels = ['1-2', '3-5', '6-10', '11-20', '20+']
    binned_precision = [[] for _ in range(len(bins) - 1)]
    binned_recall = [[] for _ in range(len(bins) - 1)]
    binned_ndcg = [[] for _ in range(len(bins) - 1)]
    
    for p, r, n, count in zip(precisions, recalls, ndcg_scores, relevant_counts):
        for i in range(len(bins) - 1):
            if bins[i] <= count < bins[i + 1]:
                binned_precision[i].append(p)
                binned_recall[i].append(r)
                binned_ndcg[i].append(n)
                break
    
    x = np.arange(len(bin_labels))
    width = 0.25
    p_means = [np.mean(bp) if bp else 0 for bp in binned_precision]
    r_means = [np.mean(br) if br else 0 for br in binned_recall]
    n_means = [np.mean(bn) if bn else 0 for bn in binned_ndcg]
    
    ax.bar(x - width, p_means, width, label='Precision', alpha=0.8, color='steelblue')
    ax.bar(x, r_means, width, label='Recall', alpha=0.8, color='seagreen')
    ax.bar(x + width, n_means, width, label='NDCG', alpha=0.8, color='coral')
    ax.set_xlabel('Number of Relevant Items in Test Set', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Performance by User Test Set Size', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    # 10. Metrics summary table
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    
    metrics_data = [
        ['Metric', 'Value', 'Description'],
        [f'Precision@{k}', f'{metrics["precision"]:.4f}', 'Relevant items / Total recommended'],
        [f'Recall@{k}', f'{metrics["recall"]:.4f}', 'Relevant items found / Total relevant'],
        [f'F1@{k}', f'{metrics["f1"]:.4f}', 'Harmonic mean of P & R'],
        [f'NDCG@{k}', f'{metrics["ndcg"]:.4f}', 'Ranking quality (position matters)'],
        [f'MAP@{k}', f'{metrics["map"]:.4f}', 'Mean Average Precision'],
        [f'Hit Rate@{k}', f'{metrics["hit_rate"]:.4f}', '% users with ≥1 relevant item'],
        [f'MRR@{k}', f'{metrics["mrr"]:.4f}', 'Mean rank of 1st relevant item'],
        ['Coverage', f'{metrics["coverage"]:.4f}', '% of catalog recommended'],
        ['Users Evaluated', f'{len(precisions):,}', 'Sample size'],
        ['Avg Relevant/User', f'{np.mean(relevant_counts):.1f}', 'Test set size'],
    ]
    
    table = ax.table(cellText=metrics_data, cellLoc='left', loc='center',
                     colWidths=[0.25, 0.15, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(3):
            table[(i, j)].set_facecolor(color)
    
    plt.suptitle(f'Recommendation Performance Analysis @K={k}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(OUTPUT_DIR / "03_topk_recommendation_analysis.png", dpi=300, bbox_inches='tight')
    print(f"[Visual] Saved top-K analysis to {OUTPUT_DIR / '03_topk_recommendation_analysis.png'}")
    plt.close()


def evaluate_multiple_k_values(train_df, test_df, R, user_id_to_idx, movie_id_to_idx, idx_to_movie_id):
    """Evaluate performance across different K values."""
    k_values = [5, 10, 20, 30, 50]
    all_metrics = []
    
    print("\n[Eval] === Evaluating Multiple K Values ===")
    for k in k_values:
        print(f"\n[Eval] Evaluating K={k}...")
        metrics, _ = evaluate_topk(train_df, test_df, R, user_id_to_idx, movie_id_to_idx, 
                                   idx_to_movie_id, k=k, max_users=1000)
        all_metrics.append(metrics)
    
    # Extract metric arrays
    precisions = [m['precision'] for m in all_metrics]
    recalls = [m['recall'] for m in all_metrics]
    f1_scores = [m['f1'] for m in all_metrics]
    ndcg_scores = [m['ndcg'] for m in all_metrics]
    map_scores = [m['map'] for m in all_metrics]
    hit_rates = [m['hit_rate'] for m in all_metrics]
    mrr_scores = [m['mrr'] for m in all_metrics]
    coverages = [m['coverage'] for m in all_metrics]
    
    # Visualize K comparison with all metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Primary metrics
    ax = axes[0, 0]
    ax.plot(k_values, precisions, marker='o', linewidth=2.5, markersize=10, 
            label='Precision@K', color='steelblue')
    ax.plot(k_values, recalls, marker='s', linewidth=2.5, markersize=10, 
            label='Recall@K', color='seagreen')
    ax.plot(k_values, f1_scores, marker='^', linewidth=2.5, markersize=10, 
            label='F1@K', color='coral')
    ax.set_xlabel('K (Number of Recommendations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Primary Metrics vs K', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(k_values)
    
    # 2. Ranking metrics
    ax = axes[0, 1]
    ax.plot(k_values, ndcg_scores, marker='D', linewidth=2.5, markersize=10, 
            label='NDCG@K', color='mediumpurple')
    ax.plot(k_values, map_scores, marker='v', linewidth=2.5, markersize=10, 
            label='MAP@K', color='gold')
    ax.plot(k_values, mrr_scores, marker='p', linewidth=2.5, markersize=10, 
            label='MRR@K', color='turquoise')
    ax.set_xlabel('K (Number of Recommendations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Ranking Quality Metrics vs K', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(k_values)
    
    # 3. Hit Rate and Coverage
    ax = axes[1, 0]
    ax2 = ax.twinx()
    line1 = ax.plot(k_values, hit_rates, marker='*', linewidth=2.5, markersize=12, 
                    label='Hit Rate@K', color='tomato')
    line2 = ax2.plot(k_values, coverages, marker='h', linewidth=2.5, markersize=10, 
                     label='Coverage', color='darkgreen', linestyle='--')
    ax.set_xlabel('K (Number of Recommendations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hit Rate', fontsize=12, fontweight='bold', color='tomato')
    ax2.set_ylabel('Catalog Coverage', fontsize=12, fontweight='bold', color='darkgreen')
    ax.set_title('Hit Rate & Diversity vs K', fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='tomato')
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=11, loc='center right')
    ax.grid(alpha=0.3)
    ax.set_xticks(k_values)
    
    # 4. All metrics comparison table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [['K'] + [str(k) for k in k_values]]
    table_data.append(['Precision'] + [f'{p:.3f}' for p in precisions])
    table_data.append(['Recall'] + [f'{r:.3f}' for r in recalls])
    table_data.append(['F1'] + [f'{f:.3f}' for f in f1_scores])
    table_data.append(['NDCG'] + [f'{n:.3f}' for n in ndcg_scores])
    table_data.append(['MAP'] + [f'{m:.3f}' for m in map_scores])
    table_data.append(['Hit Rate'] + [f'{h:.3f}' for h in hit_rates])
    table_data.append(['MRR'] + [f'{m:.3f}' for m in mrr_scores])
    table_data.append(['Coverage'] + [f'{c:.3f}' for c in coverages])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.18] + [0.14]*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row and column
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#e0e0e0')
        table[(i, 0)].set_text_props(weight='bold')
        
        # Highlight best value in each row
        row_values = [float(table_data[i][j]) for j in range(1, len(table_data[i]))]
        best_idx = row_values.index(max(row_values)) + 1
        table[(i, best_idx)].set_facecolor('#90EE90')
        table[(i, best_idx)].set_text_props(weight='bold')
    
    ax.set_title('All Metrics Across K Values\n(Best values highlighted)', 
                 fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_k_value_comparison.png", dpi=300, bbox_inches='tight')
    print(f"[Visual] Saved K value comparison to {OUTPUT_DIR / '04_k_value_comparison.png'}")
    plt.close()
    
    return k_values, all_metrics


def create_final_summary(rmse, mae, k_values, all_metrics):
    """Create a comprehensive summary visualization."""
    # Extract metrics
    precisions = [m['precision'] for m in all_metrics]
    recalls = [m['recall'] for m in all_metrics]
    f1_scores = [m['f1'] for m in all_metrics]
    ndcg_scores = [m['ndcg'] for m in all_metrics]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle('Multi-Source Movie Recommendation System - Comprehensive Evaluation Summary', 
                 fontsize=17, fontweight='bold', y=0.98)
    
    # 1. Rating Prediction Metrics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_names = ['RMSE', 'MAE']
    values = [rmse, mae]
    bars = ax1.bar(metrics_names, values, color=['#e74c3c', '#3498db'], alpha=0.8, 
                   edgecolor='black', linewidth=2)
    ax1.set_ylabel('Error Value', fontsize=11, fontweight='bold')
    ax1.set_title('Rating Prediction Performance', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Best K performance (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    best_k_idx = np.argmax(f1_scores)
    best_k = k_values[best_k_idx]
    best_m = all_metrics[best_k_idx]
    best_metric_names = ['Precision', 'Recall', 'F1', 'NDCG']
    best_values = [best_m['precision'], best_m['recall'], best_m['f1'], best_m['ndcg']]
    bars2 = ax2.bar(best_metric_names, best_values, 
                    color=['#2ecc71', '#f39c12', '#9b59b6', '#e74c3c'], 
                    alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title(f'Best Performance (K={best_k})', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(best_values) * 1.2)
    ax2.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars2, best_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. All recommendation metrics at best K (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    all_rec_names = ['P', 'R', 'F1', 'NDCG', 'MAP', 'HR', 'MRR']
    all_rec_values = [best_m['precision'], best_m['recall'], best_m['f1'], 
                      best_m['ndcg'], best_m['map'], best_m['hit_rate'], best_m['mrr']]
    colors_all = ['steelblue', 'seagreen', 'coral', 'mediumpurple', 
                  'gold', 'tomato', 'turquoise']
    bars3 = ax3.barh(all_rec_names, all_rec_values, color=colors_all, 
                     alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax3.set_title(f'All Metrics @K={best_k}', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, max(all_rec_values) * 1.15)
    ax3.grid(alpha=0.3, axis='x')
    for bar, val in zip(bars3, all_rec_values):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2., f'{val:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 4. K value trends - Primary metrics (middle left, wider)
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(k_values, precisions, marker='o', linewidth=3, markersize=11, 
             label='Precision@K', color='steelblue')
    ax4.plot(k_values, recalls, marker='s', linewidth=3, markersize=11, 
             label='Recall@K', color='seagreen')
    ax4.plot(k_values, f1_scores, marker='^', linewidth=3, markersize=11, 
             label='F1@K', color='coral')
    ax4.set_xlabel('K (Number of Recommendations)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('Primary Metrics Across K Values', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11, loc='best')
    ax4.grid(alpha=0.3)
    ax4.set_xticks(k_values)
    
    # 5. K value trends - Ranking metrics (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    map_scores = [m['map'] for m in all_metrics]
    hit_rates = [m['hit_rate'] for m in all_metrics]
    ax5.plot(k_values, ndcg_scores, marker='D', linewidth=2.5, markersize=9, 
             label='NDCG', color='mediumpurple')
    ax5.plot(k_values, map_scores, marker='v', linewidth=2.5, markersize=9, 
             label='MAP', color='gold')
    ax5.plot(k_values, hit_rates, marker='*', linewidth=2.5, markersize=10, 
             label='Hit Rate', color='tomato')
    ax5.set_xlabel('K', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax5.set_title('Ranking & Hit Metrics', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9, loc='best')
    ax5.grid(alpha=0.3)
    ax5.set_xticks(k_values)
    
    # 6. Key insights (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.axis('off')
    insights = [
        "📊 Key Findings:",
        "",
        f"• Rating RMSE: {rmse:.4f}",
        f"• Best F1 at K={best_k}: {best_m['f1']:.4f}",
        f"• NDCG@{best_k}: {best_m['ndcg']:.4f} (ranking quality)",
        f"• Hit Rate: {best_m['hit_rate']:.1%} of users",
        f"• Coverage: {best_m['coverage']:.1%} of catalog",
        "",
        "📈 Trends:",
        "• Recall ↑ as K increases",
        "• Precision ↓ as K increases",
        "• NDCG considers position"
    ]
    y_pos = 0.95
    for i, insight in enumerate(insights):
        if insight.startswith("📊") or insight.startswith("📈"):
            weight = 'bold'
            size = 11
        elif insight.startswith("•"):
            weight = 'normal'
            size = 10
        else:
            weight = 'normal'
            size = 10
        ax6.text(0.05, y_pos, insight, fontsize=size, fontweight=weight, 
                verticalalignment='top', family='monospace')
        y_pos -= 0.075
    
    # 7. Metric definitions (bottom middle)
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis('off')
    definitions = [
        "📖 Metric Definitions:",
        "",
        "• Precision: Relevant/Recommended",
        "• Recall: Found/Total Relevant",
        "• F1: Harmonic mean of P & R",
        "• NDCG: Ranking quality score",
        "• MAP: Avg precision across ranks",
        "• Hit Rate: % users with ≥1 hit",
        "• MRR: Reciprocal rank of 1st hit",
        "• Coverage: % catalog diversity",
        "",
        "💡 Higher is better for all!"
    ]
    y_pos = 0.95
    for defn in definitions:
        if defn.startswith("📖") or defn.startswith("💡"):
            weight = 'bold'
            size = 11
        else:
            weight = 'normal'
            size = 10
        ax7.text(0.05, y_pos, defn, fontsize=size, fontweight=weight, 
                verticalalignment='top', family='monospace')
        y_pos -= 0.075
    
    # 8. Recommendations (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    recommendations = [
        "🎯 Recommendations:",
        "",
        "✓ K=10-20: Balanced approach",
        "✓ Use NDCG for ranking quality",
        "✓ Monitor hit rate for UX",
        "✓ Check coverage for diversity",
        "✓ F1 for overall quality",
        "",
        "🏆 Model: User-based CF",
        "   • Cosine similarity",
        "   • 30 neighbors",
        "   • Multi-source data:",
        "     IMDb, TMDb, MovieLens"
    ]
    y_pos = 0.95
    for rec in recommendations:
        if rec.startswith("🎯") or rec.startswith("🏆"):
            weight = 'bold'
            size = 11
        else:
            weight = 'normal'
            size = 10
        ax8.text(0.05, y_pos, rec, fontsize=size, fontweight=weight, 
                verticalalignment='top', family='monospace')
        y_pos -= 0.07
    
    plt.savefig(OUTPUT_DIR / "05_comprehensive_summary.png", dpi=300, bbox_inches='tight')
    print(f"[Visual] Saved comprehensive summary to {OUTPUT_DIR / '05_comprehensive_summary.png'}")
    plt.close()
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Best K performance (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    best_k_idx = np.argmax(f1_scores)
    best_k = k_values[best_k_idx]
    best_metrics = ['Precision', 'Recall', 'F1']
    best_values = [precisions[best_k_idx], recalls[best_k_idx], f1_scores[best_k_idx]]
    bars2 = ax2.bar(best_metrics, best_values, color=['#2ecc71', '#f39c12', '#9b59b6'], 
                    alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title(f'Top-K Performance (K={best_k})', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(best_values) * 1.2)
    ax2.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars2, best_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. K value trends (middle row, spanning all columns)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(k_values, precisions, marker='o', linewidth=3, markersize=10, 
             label='Precision@K', color='#2ecc71')
    ax3.plot(k_values, recalls, marker='s', linewidth=3, markersize=10, 
             label='Recall@K', color='#f39c12')
    ax3.plot(k_values, f1_scores, marker='^', linewidth=3, markersize=10, 
             label='F1@K', color='#9b59b6')
    ax3.set_xlabel('K (Number of Recommendations)', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Performance Across Different K Values', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(alpha=0.3)
    ax3.set_xticks(k_values)
    
    # 4. Key insights (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    insights = [
        "Key Insights:",
        "",
        f"• Rating Prediction RMSE: {rmse:.4f}",
        f"• Best F1 Score achieved at K={best_k}",
        f"• Recall increases with K",
        f"• Precision generally decreases with K",
        f"• F1 balances both metrics"
    ]
    y_pos = 0.9
    for insight in insights:
        weight = 'bold' if insight.startswith("Key") else 'normal'
        ax4.text(0.05, y_pos, insight, fontsize=11, fontweight=weight, 
                verticalalignment='top', family='monospace')
        y_pos -= 0.13
    
    # 5. Recommendations (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    recommendations = [
        "Recommendations:",
        "",
        "✓ Use K=10-20 for balanced P/R",
        "✓ Larger K for recall-focused",
        "✓ Smaller K for precision-focused",
        "✓ Consider user context",
        "✓ Monitor F1 for overall quality"
    ]
    y_pos = 0.9
    for rec in recommendations:
        weight = 'bold' if rec.startswith("Rec") else 'normal'
        ax5.text(0.05, y_pos, rec, fontsize=11, fontweight=weight, 
                verticalalignment='top', family='monospace')
        y_pos -= 0.13
    
    # 6. Model info (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    model_info = [
        "Model Details:",
        "",
        "• Algorithm: User-based CF",
        "• Similarity: Cosine",
        "• Neighbors: K=50",
        "• Data: Integrated sources",
        "• (IMDb, TMDb, MovieLens)"
    ]
    y_pos = 0.9
    for info in model_info:
        weight = 'bold' if info.startswith("Model") else 'normal'
        ax6.text(0.05, y_pos, info, fontsize=11, fontweight=weight, 
                verticalalignment='top', family='monospace')
        y_pos -= 0.13
    
    plt.savefig(OUTPUT_DIR / "05_comprehensive_summary.png", dpi=300, bbox_inches='tight')
    print(f"[Visual] Saved comprehensive summary to {OUTPUT_DIR / '05_comprehensive_summary.png'}")
    plt.close()


# ---------------------------------------------------------
# 6. Main
# ---------------------------------------------------------

def main():
    print("\n" + "="*70)
    print("  MULTI-SOURCE MOVIE RECOMMENDATION SYSTEM - EVALUATION")
    print("="*70 + "\n")
    
    # Load and split data
    ratings = load_ratings()
    train_df, test_df = train_test_split_by_user(ratings, test_ratio=0.2)
    
    # Visualize data split
    print("\n[Visual] Creating data split visualizations...")
    visualize_data_split(train_df, test_df)
    
    # Build user-item matrix
    R, user_id_to_idx, idx_to_user_id, movie_id_to_idx, idx_to_movie_id = build_user_item_matrix(train_df)
    
    # Evaluate rating prediction
    print("\n[Eval] === Rating Prediction Metrics ===")
    rmse, mae, y_true, y_pred, errors = evaluate_rating_prediction(
        train_df, test_df, R, user_id_to_idx, movie_id_to_idx, sample_size=3000
    )
    
    # Evaluate across multiple K values
    k_values, all_metrics = evaluate_multiple_k_values(
        train_df, test_df, R, user_id_to_idx, movie_id_to_idx, idx_to_movie_id
    )
    
    # Create comprehensive summary
    print("\n[Visual] Creating comprehensive summary...")
    create_final_summary(rmse, mae, k_values, all_metrics)
    
    print("\n" + "="*70)
    print("  EVALUATION COMPLETE!")
    print(f"  All visualizations saved to: {OUTPUT_DIR}/")
    print("="*70 + "\n")
    
    # Print final summary
    best_k_idx = np.argmax([m['f1'] for m in all_metrics])
    best_k = k_values[best_k_idx]
    best_m = all_metrics[best_k_idx]
    
    print("\n=== FINAL SUMMARY ===")
    print(f"\nRating Prediction:")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE:  {mae:.4f}")
    print(f"\nRecommendation Performance (Best at K={best_k}):")
    print(f"  - Precision@{best_k}:  {best_m['precision']:.4f}")
    print(f"  - Recall@{best_k}:     {best_m['recall']:.4f}")
    print(f"  - F1@{best_k}:         {best_m['f1']:.4f}")
    print(f"  - NDCG@{best_k}:       {best_m['ndcg']:.4f}")
    print(f"  - MAP@{best_k}:        {best_m['map']:.4f}")
    print(f"  - Hit Rate@{best_k}:   {best_m['hit_rate']:.4f}")
    print(f"  - MRR@{best_k}:        {best_m['mrr']:.4f}")
    print(f"  - Coverage:       {best_m['coverage']:.4f}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()