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

# Read JSONL file line by line and handle potential format issues
data = []
with open(SRC, 'r') as f:
    for line in f:
        try:
            # Clean the line by replacing problematic characters
            line = line.strip().replace('false', 'False').replace('true', 'True')
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping malformed line: {e}")
            continue

df = pd.DataFrame(data)

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
    plt.savefig(f'ucb_train_stats_alpha{alpha}.png', dpi=300, bbox_inches='tight')
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

def train_ucb(X, R, alpha=1.5, n_arms=None):
    """Train UCB with given parameters and return model and statistics."""
    if n_arms is None:
        n_arms = len(MODELS)
    d = X.shape[1]
    
    A = [np.eye(d, dtype=np.float32) for _ in range(n_arms)]
    b = [np.zeros((d,1), dtype=np.float32) for _ in range(n_arms)]
    
    # Statistics tracking
    rewards_history = []
    arm_selections = defaultdict(int)
    topic_arm_selections = defaultdict(lambda: defaultdict(int))
    topic_rewards = defaultdict(list)
    
    def select_arm(x_vec):
        p = []
        for a in range(n_arms):
            Ainv = np.linalg.inv(A[a])
            theta = Ainv @ b[a]
            exploit = float(theta.T @ x_vec)
            explore = alpha * np.sqrt(float(x_vec.T @ Ainv @ x_vec))
            p.append(exploit + explore)
        return int(np.argmax(p))
    
    def update_arm(a, x_vec, reward):
        A[a] += x_vec @ x_vec.T
        b[a] += reward * x_vec
    
    # Training loop with statistics collection
    for i, (x_row, rewards) in enumerate(tqdm(zip(X, R), total=len(X), desc="Training UCB")):
        x = x_row.reshape(-1,1)
        a = select_arm(x)
        reward = rewards[a]
        
        # Update statistics
        rewards_history.append(reward)
        arm_selections[a] += 1
        topic = df["subject_name"].iloc[i]
        topic_arm_selections[topic][a] += 1
        topic_rewards[topic].append(reward)
        
        update_arm(a, x, reward)
    
    return {
        'A': A,
        'b': b,
        'select_arm': select_arm,
        'rewards_history': rewards_history,
        'arm_selections': arm_selections,
        'topic_arm_selections': topic_arm_selections,
        'topic_rewards': topic_rewards
    }

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
        stats = train_ucb(X, R, alpha=alpha)
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
