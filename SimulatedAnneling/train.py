import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
import math
import random
import warnings
warnings.filterwarnings('ignore')

class TrainingSchool:
    def __init__(self, search_method="naive", max_iter=20):
        """
        :param search_method: 'naive', 'simulated_annealing', or 'a_star'
        :param max_iter: number of iterations for advanced searches (SA, A*)
        """
        self.search_method = search_method
        self.max_iter = max_iter
        
        # Possible scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # Possible regression models
        self.regression_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'rf': RandomForestRegressor(),
            'svr': SVR(),
            'xgb': xgb.XGBRegressor(),
            'mlp': MLPRegressor(max_iter=1000)
        }
        
        # Possible classification models
        self.classification_models = {
            'logistic': LogisticRegression(max_iter=1000),
            'rf': RandomForestClassifier(),
            'svc': SVC(),
            'xgb': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'mlp': MLPClassifier(max_iter=1000)
        }
        
        # Internal best
        self.best_model = None
        self.best_scaler = None
        self.best_score = float('-inf')
        self.best_model_name = None
        self.task_type = None
        
    #########################################################################
    # LSTM / Transformer creation
    #########################################################################
    def _create_lstm_model(self, input_shape, output_dim):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(output_dim)
        ])
        model.compile(
            optimizer='adam', 
            loss='mse' if self.task_type == 'regression' else 'binary_crossentropy',
            metrics=['mae'] if self.task_type == 'regression' else ['accuracy']
        )
        return model
    
    def _create_transformer_model(self, input_shape, output_dim):
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=64, dropout=0.1
        )(x, x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(output_dim)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='mse' if self.task_type == 'regression' else 'binary_crossentropy',
            metrics=['mae'] if self.task_type == 'regression' else ['accuracy']
        )
        return model
    
    #########################################################################
    # Basic Helpers
    #########################################################################
    def _detect_task_type(self, y):
        # If # unique values <= 10 and all are integers => classification
        unique_values = np.unique(y)
        if len(unique_values) <= 10 and all(isinstance(val, (int, np.integer)) for val in unique_values):
            return 'classification'
        return 'regression'
    
    def _evaluate_config(self, scaler_key, model_key, X, y):
        """
        Evaluate (scaler, model) config with cross-validation.
        Returns a 'score' which is higher-better.
        """
        # Build pipeline manually
        scaler = self.scalers[scaler_key]
        X_scaled = scaler.fit_transform(X)

        model = None
        if self.task_type == 'regression':
            model = self.regression_models[model_key]
            scores = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_squared_error')
            # We'll define "score" as -RMSE so that bigger is better
            # i.e. if RMSE=5 => -5 is the (neg) cost
            rmse_arr = np.sqrt(-scores)  
            return -np.mean(rmse_arr)
        else:
            model = self.classification_models[model_key]
            scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
            return np.mean(scores)

    def _get_all_configs(self):
        """Return all (scaler_key, model_key) combos for the task type."""
        if self.task_type == 'regression':
            scaler_keys = list(self.scalers.keys())
            model_keys = list(self.regression_models.keys())
        else:
            scaler_keys = list(self.scalers.keys())
            model_keys = list(self.classification_models.keys())
        
        configs = []
        for sk in scaler_keys:
            for mk in model_keys:
                configs.append((sk, mk))
        return configs

    #########################################################################
    # Simulated Annealing
    #########################################################################
    def search_simulated_annealing(self, X, y, max_iter=20, T=1.0, cooling=0.95):
        """
        Attempt to find best (scaler, model) combo using simulated annealing.
        """
        configs = self._get_all_configs()
        # Random initial choice
        current_config = random.choice(configs)
        current_score = self._evaluate_config(current_config[0], current_config[1], X, y)
        
        best_config = current_config
        best_score = current_score
        
        for i in range(max_iter):
            # pick a neighbor by changing either the scaler or the model
            neighbor = self._mutate_config(current_config, configs)
            neighbor_score = self._evaluate_config(neighbor[0], neighbor[1], X, y)
            
            if neighbor_score > current_score:
                # better => accept
                current_config = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best_config = current_config
                    best_score = current_score
            else:
                # maybe accept with probability e^( (neighbor_score - current_score)/T )
                delta = neighbor_score - current_score
                acceptance_prob = math.exp(delta / T)
                if random.random() < acceptance_prob:
                    current_config = neighbor
                    current_score = neighbor_score
            
            # reduce temperature
            T *= cooling
        
        return best_config, best_score
    
    def _mutate_config(self, current_config, all_configs):
        """
        For simplicity, pick a random config different from current_config.
        A more refined approach might only tweak scaler or model individually.
        """
        neighbor = random.choice(all_configs)
        while neighbor == current_config:
            neighbor = random.choice(all_configs)
        return neighbor

    #########################################################################
    # A* Search
    #########################################################################
    def search_a_star(self, X, y, max_iter=20):
        """
        Very simplistic approach to do A* on the discrete set of (scaler, model).
        Typically, A* is used for pathfinding, so we contrive an example:
        
        We'll treat each (scaler, model) as a 'state'.
        We use cost = 1 - accuracy (for classification) or RMSE for regression.
        Heuristic = 0 (or naive).
        """
        from heapq import heappush, heappop

        # Convert each config -> "state"
        # We'll just do a discrete search (no partial states).
        configs = self._get_all_configs()
        # precompute cost for each config
        cost_dict = {}
        for c in configs:
            score = self._evaluate_config(c[0], c[1], X, y)
            if self.task_type == 'classification':
                # cost = 1 - accuracy
                cost_dict[c] = (1 - score)
            else:
                # cost = RMSE => but we stored negative RMSE as "score"
                # so RMSE = -score
                rmse = -score
                cost_dict[c] = rmse
        
        # We'll define a trivial heuristic => 0
        # so it basically becomes a uniform-cost search (Dijkstra).
        def heuristic(_):
            return 0
        
        # We'll treat the entire set as states, no adjacency except everything is connected
        # because in a real pathfinding scenario you'd define edges.
        # For demonstration, we'll just pick the best config by cost using a priority queue.

        open_list = []
        closed_set = set()

        # We push all states in, with priority = cost + heuristic
        for c in configs:
            f_val = cost_dict[c] + heuristic(c)
            heappush(open_list, (f_val, c))
        
        best_config = None
        best_cost = float('inf')
        
        # We pop up to max_iter times
        # In a typical A*, you'd expand neighbors, but here we just pop from a big queue
        for i in range(min(max_iter, len(configs))):
            f_val, c = heappop(open_list)
            if c in closed_set:
                continue
            closed_set.add(c)

            if f_val < best_cost:
                best_cost = f_val
                best_config = c
        
        # best_config is the one with minimal cost => best performance
        # but we have cost = 1 - accuracy or RMSE
        # so let's get a "score" that is consistent with the rest of the code
        if self.task_type == 'classification':
            final_score = 1 - best_cost
        else:
            # best_cost was RMSE => we want negative RMSE
            final_score = -best_cost
        
        return best_config, final_score

    #########################################################################
    # Fit method with advanced search
    #########################################################################
    def fit(self, X, y, sequence_data=False):
        """
        If self.search_method == 'naive', do normal brute force over all combos.
        If self.search_method == 'simulated_annealing', do SA.
        If self.search_method == 'a_star', do A*.
        Then pick best config. Finally, if sequence_data, also try LSTM/Transformer.
        """
        self.task_type = self._detect_task_type(y)
        print(f"Detected task type: {self.task_type}")
        
        # 1. Find best (scaler, model) via chosen search method
        if self.search_method == "naive":
            self._naive_search(X, y)
        elif self.search_method == "simulated_annealing":
            config, score = self.search_simulated_annealing(X, y, max_iter=self.max_iter)
            self.best_scaler = self.scalers[config[0]]
            if self.task_type == 'regression':
                self.best_model = self.regression_models[config[1]]
            else:
                self.best_model = self.classification_models[config[1]]
            self.best_score = score
            self.best_model_name = config
        elif self.search_method == "a_star":
            config, score = self.search_a_star(X, y, max_iter=self.max_iter)
            self.best_scaler = self.scalers[config[0]]
            if self.task_type == 'regression':
                self.best_model = self.regression_models[config[1]]
            else:
                self.best_model = self.classification_models[config[1]]
            self.best_score = score
            self.best_model_name = config
        else:
            raise ValueError(f"Unknown search_method: {self.search_method}")

        # 2. If sequence_data, also try LSTM/Transformer quickly
        #    Compare their scores to see if they beat the best so far.
        if sequence_data:
            self._try_deep_learning_models(X, y)
        
        print(f"\nChosen best config => {self.best_model_name}, Score: {self.best_score:.4f}")
        
        # 3. Final fit on entire dataset with best model
        self._final_fit(X, y, sequence_data)
        
        return self

    def _naive_search(self, X, y):
        """Original brute force approach: loop over all (scaler, model) combos."""
        all_configs = self._get_all_configs()
        best_score = float('-inf')
        best_config = None
        for (s_key, m_key) in all_configs:
            score = self._evaluate_config(s_key, m_key, X, y)
            if score > best_score:
                best_score = score
                best_config = (s_key, m_key)
        
        self.best_score = best_score
        self.best_model_name = best_config
        self.best_scaler = self.scalers[best_config[0]]
        if self.task_type == 'regression':
            self.best_model = self.regression_models[best_config[1]]
        else:
            self.best_model = self.classification_models[best_config[1]]

    def _try_deep_learning_models(self, X, y):
        """
        Evaluate LSTM and Transformer. If either is better than self.best_score,
        update self.best_model, etc.
        """
        X_scaled = self.best_scaler.fit_transform(X)
        output_dim = 1 if self.task_type == 'regression' else len(np.unique(y))
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
        # 1) Evaluate LSTM
        lstm_model = self._create_lstm_model(
            input_shape=(X_scaled.shape[1], 1),
            output_dim=output_dim
        )
        lstm_score = self._evaluate_deep_model(lstm_model, X_reshaped, y)
        if lstm_score > self.best_score:
            self.best_score = lstm_score
            self.best_model = lstm_model
            self.best_model_name = "lstm"
        
        # 2) Evaluate Transformer
        transformer_model = self._create_transformer_model(
            input_shape=(X_scaled.shape[1], 1),
            output_dim=output_dim
        )
        transformer_score = self._evaluate_deep_model(transformer_model, X_reshaped, y)
        if transformer_score > self.best_score:
            self.best_score = transformer_score
            self.best_model = transformer_model
            self.best_model_name = "transformer"

    def _evaluate_deep_model(self, model, X, y):
        """
        Simple approach: Fit for a few epochs, use validation set, 
        and return val_accuracy (classification) or -val_mae (regression).
        """
        history = model.fit(
            X, y,
            epochs=5,       # fewer epochs for quick check
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        if self.task_type == 'classification':
            # higher val_accuracy => better => we return that
            return np.mean(history.history['val_accuracy'])
        else:
            # lower val_mae => better => we can return negative val_mae
            return -np.mean(history.history['val_mae'])

    def _final_fit(self, X, y, sequence_data):
        """
        After deciding on the best model, fit it fully on the entire dataset.
        """
        X_scaled = self.best_scaler.fit_transform(X)

        if isinstance(self.best_model, tf.keras.Model):
            # Must be deep learning => reshape if needed
            if sequence_data:
                X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            self.best_model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)
        else:
            # scikit-learn model
            self.best_model.fit(X_scaled, y)

    def predict(self, X):
        """Make predictions using the best model"""
        X_scaled = self.best_scaler.transform(X)
        if isinstance(self.best_model, tf.keras.Model):
            X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            return self.best_model.predict(X_scaled)
        else:
            return self.best_model.predict(X_scaled)
    
    def get_best_model(self):
        """Return the best model and scaler"""
        return {
            'model': self.best_model,
            'scaler': self.best_scaler,
            'score': self.best_score,
            'model_name': self.best_model_name
        }
