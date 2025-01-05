"""
infer.py
---------
Loads the multi-task RL model from "trained_rl_model_multi.zip".
Demonstrates inference on two separate tasks:
  1) Classification (Wine)
  2) Regression (Diabetes)

For each task, we:
  - Create an environment with that single task
  - Run one episode to pick transformations + model
  - Build the final scikit-learn pipeline
  - Fit and save it (e.g., joblib.dump)

After this, you can load those pipelines directly for inference (no RL needed).
"""

import numpy as np
import random
import joblib

# Gymnasium
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Scikit-learn
from sklearn.datasets import load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Preprocessing / Models
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import accuracy_score, mean_squared_error


# ---------------------------------------------------------------------
# A. Same dictionary structures as in train.py
# ---------------------------------------------------------------------
TRANSFORMATIONS = {
    "standard_scaler": StandardScaler,
    "pca": PCA,
}

MODELS = {
    "logreg": {
        "constructor": LogisticRegression,
        "task_type": "classification",
    },
    "random_forest_clf": {
        "constructor": RandomForestClassifier,
        "task_type": "classification",
    },
    "linear_regression": {
        "constructor": LinearRegression,
        "task_type": "regression",
    },
    "random_forest_reg": {
        "constructor": RandomForestRegressor,
        "task_type": "regression",
    },
}


# ---------------------------------------------------------------------
# B. Utility: train & evaluate pipeline
# ---------------------------------------------------------------------
def train_and_evaluate_pipeline(transform_names, model_name, task_type, X_train, y_train, X_test, y_test):
    steps = []
    for t_name in transform_names:
        if t_name in TRANSFORMATIONS:
            steps.append( (t_name, TRANSFORMATIONS[t_name]()) )

    model_info = MODELS.get(model_name, None)
    if model_info is None:
        return 0.0
    if model_info["task_type"] != task_type:
        # model type doesn't match => 0 reward
        return 0.0

    steps.append(("model", model_info["constructor"]()))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if task_type == "classification":
        return accuracy_score(y_test, y_pred)
    elif task_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        return -mse  # negative MSE reward


# ---------------------------------------------------------------------
# C. Minimal environment for a SINGLE task (classification or regression)
# ---------------------------------------------------------------------
class SingleTaskPipelineEnv(gym.Env):
    """
    Environment that focuses on ONE dataset (either classification or regression).
    The RL agent picks transformations + model, gets the reward (accuracy or -MSE).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task_name, task_type, X_train, y_train, X_test, y_test, max_transforms=2):
        super().__init__()
        self.task_name = task_name
        self.task_type = task_type
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.transform_options = list(TRANSFORMATIONS.keys())
        self.model_options = list(MODELS.keys())
        self.max_transforms = max_transforms

        self.num_transform_actions = len(self.transform_options)
        self.num_model_actions = len(self.model_options)
        self.action_space = spaces.Discrete(self.num_transform_actions + self.num_model_actions)

        # Observation space
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_transform_actions + 2,),
            dtype=np.float32
        )

        self.chosen_transforms = []
        self.transform_count = 0
        self.model_chosen = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.chosen_transforms = []
        self.transform_count = 0
        self.model_chosen = None
        obs = self._get_obs()
        info = {"task_name": self.task_name, "task_type": self.task_type}
        return obs, info

    def _get_obs(self):
        flags = [1.0 if t in self.chosen_transforms else 0.0 for t in self.transform_options]
        count_val = float(self.transform_count)
        model_flag = 1.0 if self.model_chosen else 0.0
        return np.array(flags + [count_val, model_flag], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if action < self.num_transform_actions:
            # pick transform
            t_name = self.transform_options[action]
            if self.transform_count < self.max_transforms:
                if t_name not in self.chosen_transforms:
                    self.chosen_transforms.append(t_name)
                self.transform_count += 1
                reward -= 0.01  # small penalty for each transform
            else:
                reward -= 0.5  # big penalty if exceeding
        else:
            # pick model => end
            model_idx = action - self.num_transform_actions
            self.model_chosen = self.model_options[model_idx]
            reward = train_and_evaluate_pipeline(
                self.chosen_transforms,
                self.model_chosen,
                self.task_type,
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test
            )
            terminated = True

        obs = self._get_obs()
        info = {"task_name": self.task_name, "task_type": self.task_type}
        return obs, reward, terminated, truncated, info


def main():
    # -----------------------------------------------------------------
    # 1) Load the RL model trained on multiple tasks
    # -----------------------------------------------------------------
    model = PPO.load("trained_rl_model_multi.zip", device='cpu')
    print("[INFO] RL model loaded from 'trained_rl_model_multi.zip'")

    # -----------------------------------------------------------------
    # 2) Classification Task: Wine
    # -----------------------------------------------------------------
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # Create environment for single classification task
    env_clf = SingleTaskPipelineEnv(
        task_name="wine",
        task_type="classification",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        max_transforms=2
    )
    vec_env_clf = DummyVecEnv([lambda: env_clf])

    # Run a single episode => picks transforms + model
    obs, info = env_clf.reset()
    terminated = truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_clf.step(action)
        total_reward += reward

    print("\n=== CLASSIFICATION (WINE) ===")
    print("Chosen transforms:", env_clf.chosen_transforms)
    print("Chosen model:     ", env_clf.model_chosen)
    print(f"Final reward (accuracy): {total_reward:.4f}")

    # Build the final scikit-learn pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    steps = []
    for t_name in env_clf.chosen_transforms:
        steps.append((t_name, TRANSFORMATIONS[t_name]()))
    model_info = MODELS[env_clf.model_chosen]
    steps.append(("model", model_info["constructor"]()))
    pipeline_clf = Pipeline(steps)
    pipeline_clf.fit(X_train, y_train)

    # Save pipeline
    joblib.dump(pipeline_clf, "final_pipeline_wine.joblib")
    print("[INFO] Saved final Wine pipeline to 'final_pipeline_wine.joblib'")

    # -----------------------------------------------------------------
    # 3) Regression Task: Diabetes
    # -----------------------------------------------------------------
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=234)

    env_reg = SingleTaskPipelineEnv(
        task_name="diabetes",
        task_type="regression",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        max_transforms=2
    )
    vec_env_reg = DummyVecEnv([lambda: env_reg])

    obs, info = env_reg.reset()
    terminated = truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_reg.step(action)
        total_reward += reward

    print("\n=== REGRESSION (DIABETES) ===")
    print("Chosen transforms:", env_reg.chosen_transforms)
    print("Chosen model:     ", env_reg.model_chosen)
    print(f"Final reward (-MSE): {total_reward:.4f}")

    # Build final pipeline for Diabetes
    steps = []
    for t_name in env_reg.chosen_transforms:
        steps.append((t_name, TRANSFORMATIONS[t_name]()))
    model_info = MODELS[env_reg.model_chosen]
    steps.append(("model", model_info["constructor"]()))
    pipeline_reg = Pipeline(steps)
    pipeline_reg.fit(X_train, y_train)

    # Save pipeline
    joblib.dump(pipeline_reg, "final_pipeline_diabetes.joblib")
    print("[INFO] Saved final Diabetes pipeline to 'final_pipeline_diabetes.joblib'")


if __name__ == "__main__":
    main()
