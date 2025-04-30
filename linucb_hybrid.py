#!/usr/bin/env python3
"""
contextual_bandit_llm_selector.py
---------------------------------
• Build rewards from cosine-similarity (0‒1)
• Train LinUCB with context = [Q-emb |  one-hot(topic)]
• Call predict_best_llm(...) to route new questions.
"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────
# 1.  Load data  (JSONL → DataFrame)
# ───────────────────────────────────────────────────────────────
SRC = sys.argv[1] if len(sys.argv) > 1 else "dataset.jsonl"
df  = pd.read_json(SRC, lines=True)

MODELS = ["gemini_ans", "phi3mini_ans", "qwen4b_ans"]

# ───────────────────────────────────────────────────────────────
# 2.  Embed answers & gold output
# ───────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

gold_emb  = embedder.encode(df["output"].tolist(), show_progress_bar=True)
model_emb = {m: embedder.encode(df[m].tolist(), show_progress_bar=True) for m in MODELS}

# ───────────────────────────────────────────────────────────────
# 3.  Reward matrix R (n_samples × 3)
# ───────────────────────────────────────────────────────────────
def scaled_cos(a, b):
    a = a.reshape(a.shape[0], -1)  # Flatten all dimensions after the first
    b = b.reshape(b.shape[0], -1)
    return (cosine_similarity(a, b) + 1.0) / 2.0   # → [0,1]

R = np.column_stack([scaled_cos(model_emb[m], gold_emb).diagonal()
                     for m in MODELS])

# ───────────────────────────────────────────────────────────────
# 4.  Build contextual feature matrix  X
#     • question-embedding (384 dims)  +  one-hot(topic) (21 dims)
# ───────────────────────────────────────────────────────────────
q_emb = embedder.encode(df["input"].tolist(), show_progress_bar=True)

topic_ohe = OneHotEncoder(sparse_output=False).fit_transform(
    df["subject_name"].fillna("Unknown").to_numpy().reshape(-1,1))

X = np.hstack([q_emb, topic_ohe]).astype(np.float32)   # shape (n, d)

# ───────────────────────────────────────────────────────────────
# 5.  Contextual bandit: LinUCB with parameter tuning
# ───────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────
# Hybrid-LinUCB  (shared + arm-specific linear models)
# ───────────────────────────────────────────────────────────────
import itertools

def kron(x, a_feat):
    """Kronecker product in 1-D (== flattened outer product)."""
    return np.kron(x, a_feat).reshape(-1, 1)

def train_hybrid_linucb(X, R, alpha=1.0, arm_attrs=None):
    """
    Hybrid LinUCB with:
      • shared regressor  β  over   z_t,a = kron(x_t, a_attr)
      • residual regressors θ_a over x_t  (per-arm)

    Parameters
    ----------
    X    : (n, d)  context vectors  x_t   (same for all arms)
    R    : (n, K)  reward matrix    r_t,a
    alpha: UCB exploration weight
    arm_attrs : list[np.ndarray] , each of shape (m,)
                attribute vector for arm a.
                If None → use one-hot identity (classic cold-start demo).

    Returns
    -------
    dict with:
        • select_arm(x)   callable
        • β_hat, θ_hat[a] estimates
        • stats …  (same keys as before for plotting)
    """
    n, d = X.shape
    K     = R.shape[1]

    # ---------- arm attribute vectors ----------
    if arm_attrs is None:                               # default: one-hot
        arm_attrs = np.eye(K, dtype=np.float32)
    m = arm_attrs[0].shape[0]                           # attr dimension
    k = d * m                                           # |z|

    # ---------- global & per-arm matrices ----------
    A0   = np.eye(k, dtype=np.float32)
    b0   = np.zeros((k, 1), dtype=np.float32)

    A    = [np.eye(d, dtype=np.float32) for _ in range(K)]
    B    = [np.zeros((d, k), dtype=np.float32) for _ in range(K)]
    b    = [np.zeros((d, 1), dtype=np.float32) for _ in range(K)]

    # ---------- statistics ----------
    rewards_history       = []
    arm_selections        = defaultdict(int)
    topic_arm_selections  = defaultdict(lambda: defaultdict(int))
    topic_rewards         = defaultdict(list)

    # ---------- helper: pick arm ----------
    def select_arm(x_vec):
        """x_vec : (d,1) column vector"""
        beta_hat  = np.linalg.inv(A0) @ b0
        p_vals = []

        for a in range(K):
            Ainv = np.linalg.inv(A[a])
            theta_hat = Ainv @ (b[a] - B[a] @ beta_hat)

            z_vec = kron(x_vec, arm_attrs[a])           # (k,1)

            # UCB mean
            mu = float(theta_hat.T @ x_vec + beta_hat.T @ z_vec)

            # UCB variance  (Li et al., eq. 16)
            s2  =  (z_vec.T @ np.linalg.inv(A0) @ z_vec
                   -2 * z_vec.T @ np.linalg.inv(A0) @ B[a].T @ Ainv @ x_vec
                   +   x_vec.T @ Ainv @ x_vec
                   +   x_vec.T @ Ainv @ B[a] @ np.linalg.inv(A0) @ B[a].T @ Ainv @ x_vec)
            sigma = np.sqrt(float(s2))
            p_vals.append(mu + alpha * sigma)
        return int(np.argmax(p_vals))

    # ---------- helper: update arm ----------
    def update(a, x_vec, z_vec, reward):
        nonlocal A0, b0  # Declare A0 and b0 as nonlocal variables
        A[a] += x_vec @ x_vec.T
        B[a] += x_vec @ z_vec.T
        b[a] += reward * x_vec

        A0   += z_vec @ z_vec.T
        b0   += reward * z_vec

    # ---------- main loop ----------
    for i, (x_row, r_row) in enumerate(tqdm(zip(X, R), total=len(X), desc="Training LinUCB")):
        x  = x_row.reshape(-1,1)
        a  = select_arm(x)
        r  = float(r_row[a])

        z  = kron(x, arm_attrs[a])

        update(a, x, z, r)

        # stats
        rewards_history.append(r)
        arm_selections[a] += 1
        topic = df["subject_name"].iloc[i]
        topic_arm_selections[topic][a] += 1
        topic_rewards[topic].append(r)

    # expose objects we'll need later
    return dict(
        select_arm = select_arm,
        β_hat      = np.linalg.inv(A0) @ b0,
        θ_hat      = [np.linalg.inv(A[a]) @ (b[a] - B[a] @ (np.linalg.inv(A0) @ b0))
                      for a in range(K)],
        rewards_history      = rewards_history,
        arm_selections       = arm_selections,
        topic_arm_selections = topic_arm_selections,
        topic_rewards        = topic_rewards
    )

def plot_training_stats(stats, alpha):
    """Plot various statistics about the training process."""
    plt.figure(figsize=(20, 15))
    
    # 1. Cumulative rewards over time with moving average
    plt.subplot(3, 2, 1)
    cumulative_rewards = np.cumsum(stats['rewards_history'])
    window_size = max(1, len(stats['rewards_history']) // 20)  # 5% of data points
    moving_avg = np.convolve(stats['rewards_history'], np.ones(window_size)/window_size, mode='valid')
    plt.plot(cumulative_rewards, label='Cumulative Reward')
    plt.plot(range(window_size-1, len(stats['rewards_history'])), moving_avg, 
             label=f'Moving Average (window={window_size})', color='red')
    plt.title(f'Cumulative Rewards and Moving Average (α={alpha})')
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # 2. Arm selection distribution with percentages
    plt.subplot(3, 2, 2)
    arms = list(stats['arm_selections'].keys())
    counts = [stats['arm_selections'][a] for a in arms]
    total = sum(counts)
    percentages = [count/total*100 for count in counts]
    bars = plt.bar([MODELS[a] for a in arms], counts)
    plt.title('Arm Selection Distribution')
    plt.xlabel('Model')
    plt.ylabel('Number of Selections')
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%', ha='center', va='bottom')
    
    # 3. Average reward per topic with error bars
    plt.subplot(3, 2, 3)
    topics = list(stats['topic_rewards'].keys())
    avg_rewards = [np.mean(stats['topic_rewards'][t]) for t in topics]
    std_rewards = [np.std(stats['topic_rewards'][t]) for t in topics]
    plt.bar(topics, avg_rewards, yerr=std_rewards, capsize=5)
    plt.title('Average Reward per Topic with Standard Deviation')
    plt.xlabel('Topic')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    
    # 4. Model selection per topic (heatmap)
    plt.subplot(3, 2, 4)
    selection_matrix = np.zeros((len(topics), len(MODELS)))
    for i, topic in enumerate(topics):
        for j, model in enumerate(MODELS):
            selection_matrix[i, j] = stats['topic_arm_selections'][topic].get(j, 0)
    plt.imshow(selection_matrix, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Number of Selections')
    plt.title('Model Selection Heatmap by Topic')
    plt.xlabel('Model')
    plt.ylabel('Topic')
    plt.xticks(range(len(MODELS)), MODELS, rotation=45)
    plt.yticks(range(len(topics)), topics)
    
    # 5. Reward distribution histogram
    plt.subplot(3, 2, 5)
    plt.hist(stats['rewards_history'], bins=20, density=True, alpha=0.7)
    plt.title('Reward Distribution')
    plt.xlabel('Reward Value')
    plt.ylabel('Density')
    plt.grid(True)
    
    # 6. Learning curve (average reward over time)
    plt.subplot(3, 2, 6)
    window = max(1, len(stats['rewards_history']) // 20)
    avg_reward = np.convolve(stats['rewards_history'], np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(stats['rewards_history'])), avg_reward)
    plt.title('Learning Curve (Average Reward over Time)')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'linucb_stats_alpha{alpha}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print detailed statistics
    print(f"\nDetailed Statistics for alpha = {alpha}:")
    print("----------------------------------------")
    print(f"Total training steps: {len(stats['rewards_history'])}")
    print(f"Average reward: {np.mean(stats['rewards_history']):.4f}")
    print(f"Standard deviation of rewards: {np.std(stats['rewards_history']):.4f}")
    print("\nModel selection counts:")
    for model_idx, count in stats['arm_selections'].items():
        print(f"{MODELS[model_idx]}: {count} selections ({count/len(stats['rewards_history'])*100:.1f}%)")
    
    print("\nTopic-wise statistics:")
    for topic in topics:
        topic_rewards = stats['topic_rewards'][topic]
        print(f"\n{topic}:")
        print(f"  Average reward: {np.mean(topic_rewards):.4f}")
        print(f"  Standard deviation: {np.std(topic_rewards):.4f}")
        print("  Model selections:")
        for model_idx, count in stats['topic_arm_selections'][topic].items():
            print(f"    {MODELS[model_idx]}: {count} selections")

def get_best_models_for_topics(select_arm_func):
    """Return a dictionary mapping each topic to its best model."""
    topics = df["subject_name"].fillna("Unknown").unique()
    topic_to_model = {}
    
    for topic in topics:
        dummy_q = f"What is {topic}?"
        qv = embedder.encode([dummy_q])[0]
        tv = OneHotEncoder(sparse_output=False).fit(
                df["subject_name"].fillna("Unknown").to_numpy().reshape(-1,1)
             ).transform([[topic]]).flatten()
        x = np.hstack([qv, tv]).astype(np.float32).reshape(-1,1)
        best = select_arm_func(x)
        topic_to_model[topic] = MODELS[best]
    
    return topic_to_model

# ───────────────────────────────────────────────────────────────
# 6.  Train and evaluate with different alpha values
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    alphas = [0.5, 1.0, 1.5, 2.0]
    best_alpha = None
    best_avg_reward = -1
    
    for alpha in alphas:
        print(f"\nTraining with alpha = {alpha}")
        stats = train_hybrid_linucb(X, R, alpha=alpha)
        avg_reward = np.mean(stats['rewards_history'])
        print(f"Average reward: {avg_reward:.4f}")
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_alpha = alpha
            best_stats = stats
        
        plot_training_stats(stats, alpha)
    
    print(f"\nBest alpha value: {best_alpha} (avg reward: {best_avg_reward:.4f})")
    
    # Get best models for all topics using best alpha
    topic_to_model = get_best_models_for_topics(best_stats['select_arm'])
    print("\nBest models for each topic:")
    for topic, model in sorted(topic_to_model.items()):
        print(f"{topic}: {model}")
    
    # Test case
    test_q = "Which nerve supplies the biceps brachii?"
    test_topic = "Surgery"
    print(f"\nTest case - Question: '{test_q}'")
    print(f"Topic: {test_topic}")
    test_x = np.hstack([
        embedder.encode([test_q])[0],
        OneHotEncoder(sparse_output=False).fit(
            df["subject_name"].fillna("Unknown").to_numpy().reshape(-1,1)
        ).transform([[test_topic]]).flatten()
    ]).astype(np.float32).reshape(-1,1)
    best_model_idx = best_stats['select_arm'](test_x)
    print(f"Suggested model: {MODELS[best_model_idx]}")
