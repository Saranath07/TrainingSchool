import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# Define the same transformations and models
TRANSFORMATIONS = {
    "standard_scaler": StandardScaler,
    "pca": PCA,
}

MODELS = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

class PipelineSearchEnv(gym.Env):
    """
    Simplified version of the environment just for inference
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, max_transforms=2):
        super(PipelineSearchEnv, self).__init__()
        self.transform_options = list(TRANSFORMATIONS.keys())
        self.model_options = list(MODELS.keys())
        self.max_transforms = max_transforms
        
        self.num_transform_actions = len(self.transform_options)
        self.num_model_actions = len(self.model_options)
        self.action_space = spaces.Discrete(self.num_transform_actions + self.num_model_actions)
        
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
        return self._get_obs(), {}

    def _get_obs(self):
        transform_flags = [1.0 if t in self.chosen_transforms else 0.0 
                           for t in self.transform_options]
        count_val = float(self.transform_count)
        model_flag = 1.0 if self.model_chosen else 0.0
        return np.array(transform_flags + [count_val, model_flag], dtype=np.float32)

    def step(self, action):
        terminated = truncated = False
        reward = 0.0

        if action < self.num_transform_actions:
            transform_name = self.transform_options[action]
            if self.transform_count < self.max_transforms:
                if transform_name not in self.chosen_transforms:
                    self.chosen_transforms.append(transform_name)
                self.transform_count += 1
        else:
            model_idx = action - self.num_transform_actions
            self.model_chosen = self.model_options[model_idx]
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

def get_pipeline_from_actions(model_path, env_class=PipelineSearchEnv):
    """
    Load the trained PPO model and get its pipeline configuration
    """
    env = env_class(max_transforms=2)
    model = PPO.load(model_path, device='cpu')
    
    obs, _ = env.reset()
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
    
    return env.chosen_transforms, env.model_chosen

def build_pipeline(transforms, model_name):
    """
    Build a scikit-learn pipeline from the configuration
    """
    steps = []
    
    # Add transforms
    for t_name in transforms:
        if t_name in TRANSFORMATIONS:
            if t_name == "pca":
                # For PCA, we'll adjust n_components to maintain most variance while reducing dimensions
                steps.append((t_name, PCA(n_components=0.95)))
            else:
                transform_cls = TRANSFORMATIONS[t_name]
                steps.append((t_name, transform_cls()))
    
    # If no transforms were selected, add StandardScaler by default
    if not steps:
        steps.append(("standard_scaler", StandardScaler()))
    
    # Add model with adjusted parameters
    if model_name == "logreg":
        steps.append(("model", LogisticRegression(max_iter=1000)))
    elif model_name == "random_forest":
        steps.append(("model", RandomForestClassifier()))
    
    return Pipeline(steps)

def main():
    # Path to your saved model
    MODEL_PATH = "ppo_pipeline_model"  # adjust this to your saved model path
    
    # Load both datasets
    print("\nLoading datasets...")
    from sklearn.datasets import load_iris, load_breast_cancer
    
    # Training dataset (Iris)
    iris = load_iris()
    X_iris_train, X_iris_val, y_iris_train, y_iris_val = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Test dataset (Breast Cancer)
    cancer = load_breast_cancer()
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )
    
    # Get pipeline configuration from trained agent
    transforms, model_name = get_pipeline_from_actions(MODEL_PATH)
    print("\nDiscovered Pipeline Configuration:")
    print(f"Transforms: {transforms}")
    print(f"Model: {model_name}")
    
    # Build the pipeline
    pipeline = build_pipeline(transforms, model_name)
    
    # Evaluate on both datasets
    print("\n1. Original Dataset (Iris) Performance:")
    pipeline.fit(X_iris_train, y_iris_train)
    iris_val_score = pipeline.score(X_iris_val, y_iris_val)
    print(f"Validation Accuracy: {iris_val_score:.4f}")
    
    print("\n2. New Dataset (Breast Cancer) Performance:")
    # Create a new pipeline instance for the cancer dataset
    new_pipeline = build_pipeline(transforms, model_name)
    new_pipeline.fit(X_cancer_train, y_cancer_train)
    cancer_test_score = new_pipeline.score(X_cancer_test, y_cancer_test)
    print(f"Test Accuracy: {cancer_test_score:.4f}")
    
    # Detailed report for cancer dataset
    y_pred = new_pipeline.predict(X_cancer_test)
    print("\nDetailed Classification Report (Breast Cancer Dataset):")
    print(classification_report(y_cancer_test, y_pred))

if __name__ == "__main__":
    main()