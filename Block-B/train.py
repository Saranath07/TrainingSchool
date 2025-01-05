"""
train.py
---------
Trains an RL agent on multiple tasks (some classification, some regression).
Each episode randomly picks one dataset from a set of tasks.
The agent chooses transformations + model. 
Reward is based on accuracy (classification) or negative MSE (regression).
We save the final RL model to 'trained_rl_model_multi.zip'.
"""

import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

# Scikit-learn
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Preprocessing, Models
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import accuracy_score, mean_squared_error


# ---------------------------------------------------------------------
# A. Transformations usable by both classification + regression
# ---------------------------------------------------------------------
TRANSFORMATIONS = {
    "standard_scaler": StandardScaler,
    "pca": PCA,
}

# ---------------------------------------------------------------------
# B. Models: we store both classification and regression models
#    along with a "task_type" flag so we know which they support.
# ---------------------------------------------------------------------
MODELS = {
    # Classification
    "logreg": {
        "constructor": LogisticRegression,
        "task_type": "classification"
    },
    "random_forest_clf": {
        "constructor": RandomForestClassifier,
        "task_type": "classification"
    },
    # Regression
    "linear_regression": {
        "constructor": LinearRegression,
        "task_type": "regression"
    },
    "random_forest_reg": {
        "constructor": RandomForestRegressor,
        "task_type": "regression"
    },
}

# ---------------------------------------------------------------------
# C. Load multiple tasks (some classification, some regression)
# ---------------------------------------------------------------------
def load_multiple_tasks():
    """
    Returns a list of tuples:
       [
         ("iris", "classification", (X_train, X_test, y_train, y_test)),
         ("wine", "classification", (X_train, X_test, y_train, y_test)),
         ("diabetes", "regression", (X_train, X_test, y_train, y_test)),
         ...
       ]
    """
    tasks = []

    # 1. Iris (Classification)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    tasks.append(("iris", "classification", (X_train, X_test, y_train, y_test)))

    # 2. Wine (Classification)
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.3, random_state=42
    )
    tasks.append(("wine", "classification", (X_train, X_test, y_train, y_test)))

    # 3. Diabetes (Regression)
    #   This is a classic regression dataset (target is a continuous measure).
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.3, random_state=42
    )
    tasks.append(("diabetes", "regression", (X_train, X_test, y_train, y_test)))

    return tasks

# ---------------------------------------------------------------------
# D. Train & Evaluate the chosen pipeline on (X_train, y_train, X_test, y_test)
#    with either classification or regression metrics
# ---------------------------------------------------------------------
def train_and_evaluate_pipeline(transform_names, model_name,
                                task_type,  # "classification" or "regression"
                                X_train, y_train, X_test, y_test):
    """
    Builds a scikit-learn Pipeline with the chosen transforms + model.
    If classification => return accuracy.
    If regression => return negative MSE (the smaller MSE, the higher reward).
    """
    from sklearn.pipeline import Pipeline

    steps = []
    # Add transformations
    for t_name in transform_names:
        if t_name in TRANSFORMATIONS:
            steps.append( (t_name, TRANSFORMATIONS[t_name]()) )

    # Determine which model to use
    model_info = MODELS.get(model_name, None)
    if model_info is None:
        return 0.0  # invalid model name
    if model_info["task_type"] != task_type:
        # If model type doesn't match dataset task => penalty or zero reward
        return 0.0

    model_cls = model_info["constructor"]
    steps.append(("model", model_cls()))

    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Classification => accuracy
    if task_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        return acc

    # Regression => negative MSE as reward (the smaller the MSE, the bigger reward)
    elif task_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        # One simple approach: reward = 1 / (1 + MSE) to keep it in [0,1)
        # or negative MSE => reward = -mse
        # or 1 - min(1,mse/...)
        # We'll do a negative MSE approach:
        return -mse


# ---------------------------------------------------------------------
# E. RL Environment that handles multiple datasets (classification/regression)
# ---------------------------------------------------------------------
class MultiTaskPipelineSearchEnv(gym.Env):
    """
    On each episode, we randomly pick one dataset from tasks:
       (task_name, task_type, (X_train, X_test, y_train, y_test))
    The agent picks transformations + a model. 
    Reward:
     - If classification => accuracy
     - If regression => negative MSE
     - If model type doesn't match => reward=0
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, tasks, max_transforms=2):
        super().__init__()
        self.tasks = tasks
        self.max_transforms = max_transforms

        # We'll unify all model keys in a single list
        self.transform_options = list(TRANSFORMATIONS.keys())  # e.g. ["standard_scaler", "pca"]
        self.model_options = list(MODELS.keys())  # e.g. ["logreg", "random_forest_clf", "linear_regression", ...]

        # Discrete action space
        self.num_transform_actions = len(self.transform_options)
        self.num_model_actions = len(self.model_options)
        self.action_space = spaces.Discrete(self.num_transform_actions + self.num_model_actions)

        # Observations: [transform_flags..., transform_count, model_chosen_flag]
        self.observation_space = spaces.Box(
            low=0, 
            high=1,
            shape=(self.num_transform_actions + 2,),
            dtype=np.float32
        )

        # Internal states
        self.chosen_transforms = []
        self.transform_count = 0
        self.model_chosen = None

        # Current dataset info
        self.current_task_name = None
        self.current_task_type = None  # "classification" or "regression"
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None

    def _pick_random_task(self):
        task_name, task_type, (Xtr, Xte, ytr, yte) = random.choice(self.tasks)
        self.current_task_name = task_name
        self.current_task_type = task_type
        self.X_train = Xtr
        self.X_test  = Xte
        self.y_train = ytr
        self.y_test  = yte

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a new dataset
        self._pick_random_task()

        self.chosen_transforms = []
        self.transform_count = 0
        self.model_chosen = None

        obs = self._get_obs()
        info = {"task_name": self.current_task_name, "task_type": self.current_task_type}
        return obs, info

    def _get_obs(self):
        transform_flags = [
            1.0 if t in self.chosen_transforms else 0.0
            for t in self.transform_options
        ]
        count_val = float(self.transform_count)
        model_flag = 1.0 if self.model_chosen else 0.0
        return np.array(transform_flags + [count_val, model_flag], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if action < self.num_transform_actions:
            # pick a transformation
            t_name = self.transform_options[action]
            if self.transform_count < self.max_transforms:
                if t_name not in self.chosen_transforms:
                    self.chosen_transforms.append(t_name)
                self.transform_count += 1
                # small negative reward to discourage too many transforms
                reward -= 0.01
            else:
                # big penalty if exceeding max
                reward -= 0.5
        else:
            # pick a model => episode ends
            model_idx = action - self.num_transform_actions
            self.model_chosen = self.model_options[model_idx]

            # Evaluate pipeline
            reward = train_and_evaluate_pipeline(
                self.chosen_transforms,
                self.model_chosen,
                self.current_task_type,
                self.X_train, self.y_train,
                self.X_test,  self.y_test
            )
            terminated = True

        obs = self._get_obs()
        info = {"task_name": self.current_task_name, "task_type": self.current_task_type}
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------
# F. Main training script
# ---------------------------------------------------------------------
def main():
    # 1. Load tasks (Iris/Wine => classification, Diabetes => regression)
    tasks = load_multiple_tasks()
    print(f"[INFO] Found {len(tasks)} tasks:")
    for t in tasks:
        print("  -", t[0], "(", t[1], ")")

    # 2. Create environment
    env = MultiTaskPipelineSearchEnv(tasks, max_transforms=2)
    vec_env = DummyVecEnv([lambda: env])

    # 3. Create RL model
    model = PPO(
        MlpPolicy,
        vec_env,
        verbose=1,
        device="cpu",
        seed=42
    )

    # 4. Train RL model 
    #    Over many episodes, environment picks random tasks => agent learns generalized strategy
    model.learn(total_timesteps=10000)

    # 5. Save RL model
    model.save("trained_rl_model_multi.zip")
    print("\n[INFO] RL model saved to 'trained_rl_model_multi.zip'")

    # 6. (Optional) Test-run one final episode
    obs, info = env.reset()
    task_name = info["task_name"]
    task_type = info["task_type"]
    print(f"[TEST-RUN] Agent got dataset: {task_name} ({task_type})")

    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    # If classification => reward is accuracy, if regression => reward is negative MSE
    # e.g., for regression a "higher reward" means smaller MSE
    print("Chosen transforms:", env.chosen_transforms)
    print("Chosen model:     ", env.model_chosen)
    print(f"Final reward = {total_reward:.4f}  (acc or -mse)")

if __name__ == "__main__":
    main()
