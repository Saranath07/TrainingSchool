import pandas as pd
import os
import tempfile
from processData import MLDataAnalyzer
from generateConfig import GenerateConfigLLM
from model_analysis import ModelAnalysisLLM  
from trainingSchool import TrainingSchool 
import logging
import time
from sklearn.model_selection import train_test_split
import sys
import numpy as np





class IterativeModelTrainer:
    def __init__(self, df, task_description, target_column="target", max_iterations=2):
        self.df = df
        self.task_description = task_description
        self.target_column = target_column
        self.max_iterations = max_iterations
        self.analyzer = MLDataAnalyzer(df, target_column, task_description)
        self.model_analyzer = ModelAnalysisLLM()
        self.config_generator = GenerateConfigLLM()  # Add initial config generator
        self.available_models = [
            'StandardScaler', 'MinMaxScaler', 'RobustScaler',
            'LinearRegression', 'Ridge', 'RandomForestRegressor',
            'SVR', 'XGBRegressor', 'MLPRegressor',
            'LogisticRegression', 'RandomForestClassifier',
            'SVC', 'XGBClassifier', 'MLPClassifier'
        ]
        
    def _train_and_evaluate(self, config_yaml, X_train, X_test, y_train, y_test):
        """Train models and collect performance metrics"""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml', delete=False) as temp_file:
            temp_file.write(config_yaml)
            temp_file.flush()
            
            trainer = TrainingSchool(config_path=temp_file.name)
            trainer.fit(X_train, y_train)
            
            results = {}
            for model_name, model_info in trainer.models.items():
                model = model_info['model']
                scaler = model_info['scaler']
                
                # Time the training
                start_time = time.time()
                model.fit(scaler.transform(X_train), y_train)
                training_time = time.time() - start_time
                
                # Time the prediction
                start_time = time.time()
                y_pred = model.predict(scaler.transform(X_test))
                prediction_time = time.time() - start_time
                
                # Get memory usage
                memory_usage = sys.getsizeof(model) + sys.getsizeof(scaler)
                
                # Collect CV scores
                cv_scores = trainer.get_cv_scores(model_name)
                
                # Get feature importance if available
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(
                        self.df.drop(self.target_column, axis=1).columns,
                        model.feature_importances_
                    ))
                
                results[model_name] = {
                    'model_name': model_name,
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'cv_scores': cv_scores,
                    'training_time': training_time,
                    'prediction_time': prediction_time,
                    'memory_usage': memory_usage,
                    'feature_importance': feature_importance,
                    'convergence_info': {
                        'n_iter_': getattr(model, 'n_iter_', None),
                        'loss_curve_': getattr(model, 'loss_curve_', None)
                    }
                }
            
            os.unlink(temp_file.name)
            return results, trainer.get_best_model()
    
    def train_iteratively(self):
        """Perform iterative training with model analysis and refinement"""
        # Initial analysis and data split
        analysis = self.analyzer.analyze_dataset()
        sample_data = self.analyzer.get_sample_data()
        X = self.df.drop(self.target_column, axis=1).values
        y = self.df[self.target_column].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Storage for tracking progress
        all_iterations = []
        best_model_info = None
        best_score = float('-inf')
        
        # Get initial configuration using GenerateConfigLLM
        initial_config = self.config_generator.generate_config(
            analysis, sample_data, self.task_description, self.available_models
        )
        yaml_content = initial_config.strip().strip('`').lstrip('yml').strip()

        print(self.task_description)

        print(yaml_content)
        
        # First iteration with initial configuration
        logging.info("Starting first iteration with GenerateConfigLLM configuration")
        results, iteration_best = self._train_and_evaluate(
            yaml_content, X_train, X_test, y_train, y_test
        )
 
        # Generate performance report
        performance_report = self.model_analyzer.generate_performance_report(
            results,
            'classification' if len(np.unique(y)) <= 10 else 'regression'
        )
        
        # Store first iteration results
        iteration_info = {
            'iteration': 1,
            'config': yaml_content,
            'results': results,
            'best_model': iteration_best,
            'performance_report': performance_report
        }
        all_iterations.append(iteration_info)
        
        # Update best model
        best_score = iteration_best['score']
        best_model_info = iteration_best
        
        logging.info(f"Completed first iteration with score: {best_score}")
        
        # Continue with remaining iterations using ModelAnalysisLLM
        current_config = yaml_content
        for iteration in range(1, self.max_iterations):
            logging.info(f"Starting iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate refined configuration
            current_config = self.model_analyzer.generate_refined_config(
                analysis,
                results,
                'classification' if len(np.unique(y)) <= 10 else 'regression',
                self.available_models
            )
            
            print(current_config)
            
            # Train and evaluate models
            results, iteration_best = self._train_and_evaluate(
                current_config, X_train, X_test, y_train, y_test
            )

           
            
            # Generate performance report
            performance_report = self.model_analyzer.generate_performance_report(
                results,
                'classification' if len(np.unique(y)) <= 10 else 'regression'
            )
            
            # Store iteration results
            iteration_info = {
                'iteration': iteration + 1,
                'config': current_config,
                'results': results,
                'best_model': iteration_best,
                'performance_report': performance_report
            }
            all_iterations.append(iteration_info)
            
            # Update best model if better
            if iteration_best['score'] > best_score:
                best_score = iteration_best['score']
                best_model_info = iteration_best
            
            logging.info(f"Completed iteration {iteration + 1} with best score: {best_score}")
        
        return all_iterations, best_model_info

df = pd.read_csv("example_data.csv")

task_description = "Using the dataframe I need a binary classification algorithm"

trainer = IterativeModelTrainer(
    df=df,
    task_description=task_description,
    max_iterations=2  
)

# Run training
iterations, best_model = trainer.train_iteratively()