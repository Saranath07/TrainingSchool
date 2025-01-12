import numpy as np
import pandas as pd
import yaml
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
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except Exception as e:
    logging.warning(f"Error initializing TensorFlow: {e}")

def configure_gpu():
    try:
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU found. Running on CPU.")
            # Disable GPU usage
            tf.config.set_visible_devices([], 'GPU')
        else:
            # Configure memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). GPU acceleration enabled.")
    except Exception as e:
        print(f"GPU configuration failed. Running on CPU. Error: {str(e)}")
        # Disable GPU usage
        tf.config.set_visible_devices([], 'GPU')

class TrainingSchool:
    def __init__(self, config_path='config.yaml'):
        """
        Initialize TrainingSchool with configuration from YAML file
        """
        configure_gpu()
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scalers = self._initialize_scalers()
        self.models = {}  # Store all models and scalers here
        self.regression_models = {}
        self.classification_models = {}
        self.best_model = None
        self.best_scaler = None
        self.best_score = float('-inf')
        self.task_type = None
        self.sequence_data = self.config['sequence_data']['enabled']
        
    def _get_class_from_string(self, class_name):
        """Convert string class name to actual class"""
        class_mapping = {
            # Scalers
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'RobustScaler': RobustScaler,
            # Regression Models
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'RandomForestRegressor': RandomForestRegressor,
            'SVR': SVR,
            'XGBRegressor': xgb.XGBRegressor,
            'MLPRegressor': MLPRegressor,
            # Classification Models
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'SVC': SVC,
            'XGBClassifier': xgb.XGBClassifier,
            'MLPClassifier': MLPClassifier
        }
        return class_mapping.get(class_name)
    
    def _initialize_scalers(self):
        """Initialize scalers from config"""
        scalers = {}
        for scaler_name, scaler_config in self.config['scalers'].items():
            scaler_class = self._get_class_from_string(scaler_config['name'])
            if scaler_class:
                scalers[scaler_name] = scaler_class(**scaler_config['params'])
        return scalers
    
    def _initialize_models(self, task_type):
        """Initialize models based on task type from config"""
        models = {}
        model_configs = self.config['models'][task_type]
        
        for model_name, model_config in model_configs.items():
            model_class = self._get_class_from_string(model_config['name'])
            if model_class:
                models[model_name] = model_class(**model_config['params'])
        
        return models
    
    
    def _evaluate_model(self, model, X, y, is_deep_learning=False):
        """Evaluate model using configured metrics"""
        if is_deep_learning:
            dl_params = (self.config['deep_learning']['lstm'] if isinstance(model, Sequential)
                        else self.config['deep_learning']['transformer'])['params']
            history = model.fit(
                X, y,
                epochs=dl_params['epochs'],
                batch_size=dl_params['batch_size'],
                validation_split=0.2,
                verbose=0
            )
            return np.mean(history.history['val_accuracy' if self.task_type == 'classification'
                                        else 'val_mae'])
        
        metrics = self.config['evaluation']['metrics']
        cv_folds = self.config['evaluation']['cv_folds']
        
        if self.task_type == 'regression':
            scores = cross_val_score(model, X, y, cv=cv_folds,
                                   scoring=metrics['regression'][0])
            return np.mean(np.sqrt(-scores))
        else:
            scores = cross_val_score(model, X, y, cv=cv_folds,
                                   scoring=metrics['classification'][0])
            return np.mean(scores)
    
    def _detect_task_type(self, y):
        """Helper method to detect if the task is classification or regression"""
        unique_values = np.unique(y)
        if len(unique_values) <= 100 and all(isinstance(val, (int, np.integer)) for val in unique_values):
            return 'classification'
        return 'regression'

    def _create_lstm_model(self, input_shape, output_dim):
        """Create LSTM model with config parameters and proper output shape"""
        try:
            lstm_config = self.config['deep_learning']['lstm']
            if not lstm_config['enabled']:
                return None
                
            params = lstm_config['params']
            with tf.device('/CPU:0'):  # Force CPU usage for model creation
                model = Sequential()
                
                for i, units in enumerate(params['units']):
                    if i == 0:
                        model.add(LSTM(units, return_sequences=i < len(params['units'])-1,
                                    input_shape=input_shape))
                    else:
                        model.add(LSTM(units, return_sequences=i < len(params['units'])-1))
                    model.add(Dropout(params['dropout_rates'][i]))
                
                if self.task_type == 'classification':
                    model.add(Dense(output_dim, activation='softmax'))
                else:
                    model.add(Dense(output_dim))
                
                loss = 'sparse_categorical_crossentropy' if self.task_type == 'classification' else 'mse'
                metrics = ['accuracy'] if self.task_type == 'classification' else ['mae']
                
                model.compile(
                    optimizer=params['optimizer'],
                    loss=loss,
                    metrics=metrics
                )
            return model
        except Exception as e:
            print(f"Error creating LSTM model: {str(e)}")
            return None

    def _create_transformer_model(self, input_shape, output_dim):
        """Create Transformer model with correct output shape for classification"""
        try:
            transformer_config = self.config['deep_learning']['transformer']
            if not transformer_config['enabled']:
                return None
                
            params = transformer_config['params']
            with tf.device('/CPU:0'):  # Force CPU usage for model creation
                inputs = tf.keras.Input(shape=input_shape)
                
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
                attention_output = tf.keras.layers.MultiHeadAttention(
                    num_heads=params['num_heads'],
                    key_dim=params['key_dim'],
                    dropout=params['dropout_rate'])(x, x)
                
                x = tf.keras.layers.Add()([x, attention_output])
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
                x = tf.keras.layers.GlobalAveragePooling1D()(x)
                
                for units in params['dense_units']:
                    x = tf.keras.layers.Dense(units, activation='relu')(x)
                    x = tf.keras.layers.Dropout(params['dropout_rate'])(x)
                
                if self.task_type == 'classification':
                    outputs = tf.keras.layers.Dense(output_dim, activation='softmax')(x)
                else:
                    outputs = tf.keras.layers.Dense(output_dim)(x)
                
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                
                loss = 'sparse_categorical_crossentropy' if self.task_type == 'classification' else 'mse'
                metrics = ['accuracy'] if self.task_type == 'classification' else ['mae']
                
                model.compile(
                    optimizer=params['optimizer'],
                    loss=loss,
                    metrics=metrics
                )
            return model
        except Exception as e:
            print(f"Error creating Transformer model: {str(e)}")
            return None
    def fit(self, X, y):
        """
        Fit function for TrainingSchoolV2 with proper classification handling
        """
        # Detect task type and initialize appropriate models
        self.X_train = X
        self.y_train = y
        self.task_type = self._detect_task_type(y)
        print(f"Detected task type: {self.task_type}")
        
        # Initialize models based on task type from config
        if self.task_type == 'regression':
            self.regression_models = self._initialize_models('regression')
            active_models = self.regression_models
        else:
            self.classification_models = self._initialize_models('classification')
            active_models = self.classification_models
        
        # Initialize tracking variables for best model
        best_score = float('-inf')
        best_model = None
        best_scaler = None
        best_model_name = None
        
        # Try different scalers from config
        for scaler_name, scaler in self.scalers.items():
            try:
                X_scaled = scaler.fit_transform(X)
                
                # Evaluate traditional ML models
                for model_name, model in active_models.items():
                    try:
                        score = self._evaluate_model(model, X_scaled, y)
                        print(f"Score for {model_name} with {scaler_name} scaler: {score:.4f}")
                        
                        # Save models and scalers
                        self.models[model_name] = {'model': model, 'scaler': scaler}
                        
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_scaler = scaler
                            best_model_name = model_name
                    except Exception as e:
                        print(f"Error with {model_name}: {str(e)}")
                
                # Handle deep learning models for sequential data
                if self.sequence_data:
                    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
                    output_dim = len(np.unique(y)) if self.task_type == 'classification' else 1
                    
                    # LSTM evaluation
                    lstm_model = self._create_lstm_model(
                        input_shape=(X_scaled.shape[1], 1),
                        output_dim=output_dim
                    )
                    if lstm_model is not None:
                        try:
                            lstm_score = self._evaluate_model(lstm_model, X_reshaped, y, is_deep_learning=True)
                            print(f"Score for LSTM with {scaler_name} scaler: {lstm_score:.4f}")
                            
                            # Save LSTM model
                            self.models['lstm'] = {'model': lstm_model, 'scaler': scaler}
                            
                            if lstm_score > best_score:
                                best_score = lstm_score
                                best_model = lstm_model
                                best_scaler = scaler
                                best_model_name = 'lstm'
                        except Exception as e:
                            print(f"Error with LSTM: {str(e)}")
                    
                    # Transformer evaluation
                    transformer_model = self._create_transformer_model(
                        input_shape=(X_scaled.shape[1], 1),
                        output_dim=output_dim
                    )
                    if transformer_model is not None:
                        try:
                            transformer_score = self._evaluate_model(
                                transformer_model, X_reshaped, y, is_deep_learning=True
                            )
                            print(f"Score for Transformer with {scaler_name} scaler: {transformer_score:.4f}")
                            
                            # Save Transformer model
                            self.models['transformer'] = {'model': transformer_model, 'scaler': scaler}
                            
                            if transformer_score > best_score:
                                best_score = transformer_score
                                best_model = transformer_model
                                best_scaler = scaler
                                best_model_name = 'transformer'
                        except Exception as e:
                            print(f"Error with Transformer: {str(e)}")
                            
            except Exception as e:
                print(f"Error with {scaler_name} scaler: {str(e)}")

        
        # Store best results
        self.best_score = best_score
        self.best_model = best_model
        self.best_scaler = best_scaler
        self.best_model_name = best_model_name
        
        # Print final results
        print(f"\nBest model: {self.best_model_name}")
        print(f"Best scaler: {type(self.best_scaler).__name__}")
        print(f"Best score: {self.best_score:.4f}")
        
        # Final fitting of the best model
        X_scaled = self.best_scaler.fit_transform(X)
        
        if isinstance(self.best_model, tf.keras.Model):
            if self.sequence_data:
                X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            
            # Get deep learning parameters from config
            dl_config = (self.config['deep_learning']['lstm'] 
                        if isinstance(self.best_model, Sequential)
                        else self.config['deep_learning']['transformer'])
            dl_params = dl_config['params']
            
            self.best_model.fit(
                X_scaled, 
                y, 
                epochs=dl_params['epochs'],
                batch_size=dl_params['batch_size'],
                verbose=0
            )
        else:
            self.best_model.fit(X_scaled, y)
        
        return self
    def get_cv_scores(self, model_name):
        """
        Compute cross-validation scores for a given model.
        Args:
            model_name (str): The name of the model for which CV scores are computed.
        Returns:
            dict: A dictionary with cross-validation scores for the model.
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found in trained models.")
            
            model_info = self.models[model_name]
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Scale the data
            X_scaled = scaler.transform(self.X_train)
            
            # Use cross-validation to compute scores
            metrics = self.config['evaluation']['metrics']
            cv_folds = self.config['evaluation']['cv_folds']
            
            if self.task_type == 'regression':
                scoring_metric = metrics['regression'][0]  # e.g., 'neg_mean_squared_error'
            else:
                scoring_metric = metrics['classification'][0]  # e.g., 'accuracy'
            
            scores = cross_val_score(model, X_scaled, self.y_train, cv=cv_folds, scoring=scoring_metric)
            return {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores.tolist()
            }
        
        except Exception as e:
            print(f"Error computing CV scores for model '{model_name}': {str(e)}")
            return None
    def predict(self, X):
        """Make predictions using the best model"""
        X_scaled = self.best_scaler.transform(X)
        if isinstance(self.best_model, (tf.keras.Model)):
            X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        return self.best_model.predict(X_scaled)
    
    def get_best_model(self):
        """Return the best model and scaler"""
        return {
            'model': self.best_model,
            'scaler': self.best_scaler,
            'score': self.best_score,
            'model_name': self.best_model_name
        }