import numpy as np
import random

# Gymnasium
import gymnasium as gym
from gymnasium import spaces

# RL Algorithm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

# Scikit-learn / Data
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------
# A. Define transformations and models we allow
# ---------------------------------------------------------------------
TRANSFORMATIONS = {
    "standard_scaler": StandardScaler,
    "pca": PCA,
}

MODELS = {
    "logreg": LogisticRegression,         
    "random_forest": RandomForestClassifier,
}


# ---------------------------------------------------------------------
# B. Create multiple tasks (datasets) for meta-learning
#    For demonstration, we pick 3: Iris, Wine, Breast Cancer.
#    Each task has (X_train, X_test, y_train, y_test).
# ---------------------------------------------------------------------
def load_datasets_for_meta_learning():
    tasks = []

    # 1. Iris
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    tasks.append(("iris", (X_train, X_test, y_train, y_test)))

    # 2. Wine
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.3, random_state=42
    )
    tasks.append(("wine", (X_train, X_test, y_train, y_test)))

    # 3. Breast Cancer
    bc = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, test_size=0.3, random_state=42
    )
    tasks.append(("breast_cancer", (X_train, X_test, y_train, y_test)))

    return tasks


# ---------------------------------------------------------------------
# C. Train & Evaluate a Pipeline on a chosen dataset
# ---------------------------------------------------------------------
def train_and_evaluate_pipeline(transform_names, model_name,
                                X_train, y_train, X_test, y_test):
    """
    Given a list of transformations and a model name,
    build a scikit-learn Pipeline, train on (X_train, y_train),
    evaluate on (X_test, y_test), return accuracy.
    """
    steps = []
    
    # Add each transformation
    for t_name in transform_names:
        if t_name in TRANSFORMATIONS:
            transform_cls = TRANSFORMATIONS[t_name]
            steps.append((t_name, transform_cls()))

    # Finally, add the model
    if model_name in MODELS:
        model_cls = MODELS[model_name]
        steps.append(("model", model_cls()))
    else:
        return 0.0  # Invalid model, shouldn't happen

    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc


# ---------------------------------------------------------------------
# D. Define the RL Environment for Meta-Learning
# ---------------------------------------------------------------------
class MetaLearningPipelineEnv(gym.Env):
    """
    Environment that picks transformations + model. 
    On each new episode, we randomly pick a dataset from a set of tasks.
    The reward is accuracy on that dataset.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, tasks, max_transforms=2):
        """
        :param tasks: List of tuples (task_name, (X_train, X_test, y_train, y_test))
        :param max_transforms: Max number of transformations allowed per pipeline
        """
        super(MetaLearningPipelineEnv, self).__init__()

        self.tasks = tasks
        self.max_transforms = max_transforms

        self.transform_options = list(TRANSFORMATIONS.keys())
        self.model_options = list(MODELS.keys())

        # Action space: discrete set of transforms + models
        self.num_transform_actions = len(self.transform_options)
        self.num_model_actions = len(self.model_options)
        self.action_space = spaces.Discrete(self.num_transform_actions + self.num_model_actions)

        # Observation space:
        # [transform_flags..., transform_count, model_chosen_flag]
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

        # We'll store the current dataset each episode
        self.current_task_name = None
        self.current_X_train = None
        self.current_y_train = None
        self.current_X_test = None
        self.current_y_test = None

    def _pick_new_dataset(self):
        """
        Randomly select a dataset from self.tasks
        """
        task_name, (Xtr, Xte, ytr, yte) = random.choice(self.tasks)
        self.current_task_name = task_name
        self.current_X_train = Xtr
        self.current_y_train = ytr
        self.current_X_test = Xte
        self.current_y_test = yte

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # On each reset (new episode), pick a new dataset
        self._pick_new_dataset()

        self.chosen_transforms = []
        self.transform_count = 0
        self.model_chosen = None

        # Return observation + info with "task_name"
        info = {"task_name": self.current_task_name}
        return self._get_obs(), info

    def _get_obs(self):
        transform_flags = [
            1.0 if t in self.chosen_transforms else 0.0
            for t in self.transform_options
        ]
        count_val = float(self.transform_count)
        model_flag = 1.0 if self.model_chosen else 0.0
        obs = np.array(transform_flags + [count_val, model_flag], dtype=np.float32)
        return obs

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if action < self.num_transform_actions:
            # Pick a transformation
            transform_name = self.transform_options[action]
            if self.transform_count < self.max_transforms:
                if transform_name not in self.chosen_transforms:
                    self.chosen_transforms.append(transform_name)
                self.transform_count += 1
                # small negative reward to discourage many transforms
                reward -= 0.01
            else:
                # penalty for exceeding max transforms
                reward -= 0.5
        else:
            # Pick a model -> end episode
            model_idx = action - self.num_transform_actions
            self.model_chosen = self.model_options[model_idx]

            # Evaluate pipeline on the current dataset
            accuracy = train_and_evaluate_pipeline(
                self.chosen_transforms,
                self.model_chosen,
                self.current_X_train,
                self.current_y_train,
                self.current_X_test,
                self.current_y_test
            )
            reward = accuracy
            terminated = True

        obs = self._get_obs()
        info = {"task_name": self.current_task_name}  # Keep track of which dataset was used
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------
# E. Train the RL agent across multiple datasets (meta-learning)
# ---------------------------------------------------------------------
def main():
    # 1. Load tasks
    tasks = load_datasets_for_meta_learning()
    print(f"[INFO] Loaded {len(tasks)} tasks: {[t[0] for t in tasks]}")

    # 2. Create environment
    env = MetaLearningPipelineEnv(tasks, max_transforms=2)

    # 3. Wrap environment for Stable-Baselines3
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    # 4. Create RL model (PPO) - explicitly use CPU
    model = PPO(
        MlpPolicy,
        vec_env,
        verbose=1,
        seed=42,
        device="cpu"
    )

    # 5. Train the RL agent
    #    The agent will see multiple datasets *across episodes*.
    #    It should learn a pipeline strategy that works "generally well."
    model.learn(total_timesteps=1)

    # -----------------------------------------------------------------
    # 6. After training, let's do a single run (one episode) to see
    #    which pipeline it picks for a random dataset. 
    # -----------------------------------------------------------------
    obs, info = env.reset()
    task_name = info["task_name"]
    print(f"\n[TEST RUN] The environment picked dataset: {task_name}")

    terminated = False
    truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("[END OF EPISODE]")
    print("Chosen transforms:", env.chosen_transforms)
    print("Chosen model:     ", env.model_chosen)
    print(f"Final Reward (Accuracy on {task_name}): {total_reward:.4f}")

    # -----------------------------------------------------------------
    # 7. Evaluate the final pipeline on *all tasks* to see how well
    #    it generalizes. We'll just use the single pipeline the agent
    #    ended up with in this test run. For a thorough approach, you
    #    might sample multiple episodes or do a separate policy rollout
    #    for each task.
    # -----------------------------------------------------------------
    final_pipeline = Pipeline([
        (t_name, TRANSFORMATIONS[t_name]()) 
        for t_name in env.chosen_transforms
    ] + [
        ("model", MODELS[env.model_chosen]())
    ])

    for name, (Xtr, Xte, ytr, yte) in tasks:
        final_pipeline.fit(Xtr, ytr)
        acc = final_pipeline.score(Xte, yte)
        print(f"Pipeline accuracy on {name}: {acc:.4f}")


if __name__ == "__main__":
    main()
