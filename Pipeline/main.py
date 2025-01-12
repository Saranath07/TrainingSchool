# app.py
from flask import Flask, render_template, request, jsonify, send_file, g, make_response
import pandas as pd
import yaml
import os
import json
from werkzeug.utils import secure_filename
import tempfile
from processData import MLDataAnalyzer
from generateConfig import GenerateConfigLLM
from model_analysis import ModelAnalysisLLM  # New import
from trainingSchool import TrainingSchool
import joblib
import logging
import time
from sklearn.model_selection import train_test_split
import numpy as np
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_ITERATIONS'] = 2

# Ensure upload and model directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

def get_temp_file():
    if 'temp_file' not in g:
        g.temp_file = None
    return g.temp_file

@app.teardown_appcontext
def cleanup_temp_file(exception):
    temp_file = g.pop('temp_file', None)
    if temp_file:
        try:
            os.unlink(temp_file)
        except:
            pass

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    task_description = request.form.get('task_description', '')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            
            # Initialize iterative trainer
            trainer = IterativeModelTrainer(
                df,
                task_description,
                max_iterations=app.config['MAX_ITERATIONS']
            )
            
            # Perform iterative training
            iterations, best_model_info = trainer.train_iteratively()
            
            # Save best model and its information
            model_filename = f"{filename.rsplit('.', 1)[0]}_model.joblib"
            model_save_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
            
            joblib.dump(
                {
                    'model': best_model_info['model'],
                    'scaler': best_model_info['scaler']
                },
                model_save_path
            )
            
            # Save iterations information
            iterations_info = {
                'filename': filename,
                'task_description': task_description,
                'best_model': {
                    'model_type': type(best_model_info['model']).__name__,
                    'score': best_model_info['score'],
                    'model_path': model_filename
                },
                'feature_columns': list(df.drop("target", axis=1).columns),
                'iterations': [
                    {
                        'iteration': it['iteration'],
                        'performance_report': it['performance_report'],
                        'best_score': it['best_model']['score']
                    }
                    for it in iterations
                ]
            }
            
            # Save model info JSON
            info_path = os.path.join(
                app.config['MODEL_FOLDER'],
                f"{filename.rsplit('.', 1)[0]}_info.json"
            )
            with open(info_path, 'w') as f:
                json.dump(iterations_info, f)
            
            return jsonify({
                'success': True,
                'model_info': iterations_info
            })
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400
@app.route('/models', methods=['GET'])
def list_models():
    models = []
    for filename in os.listdir(app.config['MODEL_FOLDER']):
        if filename.endswith('_info.json'):
            with open(os.path.join(app.config['MODEL_FOLDER'], filename)) as f:
                models.append(json.load(f))
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint that loads a stored model and scaler to make predictions on uploaded CSV.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    model_name = request.form.get('model_name', '')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Suppose we only allow CSV
    if file and allowed_file(file.filename):
        try:
            # 1. Load test data
            test_df = pd.read_csv(file)
            
            # 2. Load model info (JSON)
            model_info_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_name}_info.json")
            with open(model_info_path) as f:
                model_info = json.load(f)
            
            # 3. Load the model+scaler dictionary from joblib
            model_path = os.path.join(app.config['MODEL_FOLDER'], model_info['best_model']['model_path'])
            loaded_dict = joblib.load(model_path)  # Contains { 'model': ..., 'scaler': ... }
            
            loaded_model = loaded_dict['model']
            loaded_scaler = loaded_dict['scaler']
            
            # 4. Ensure test data has required features
            required_features = model_info['feature_columns']
            if not all(feature in test_df.columns for feature in required_features):
                return jsonify({'error': 'Test data missing required features'}), 400
            
            # 5. Make predictions using the model after scaling
            X_test = test_df[required_features].values
            X_test_scaled = loaded_scaler.transform(X_test)
            predictions = loaded_model.predict(X_test_scaled)
            
            # 6. Create a new DataFrame with predictions
            result_df = test_df.copy()
            result_df['target'] = predictions
            
            # 7. Create a temporary file to store the CSV
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
            result_df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
            temp_file.close()
            
            # 8. Store the temp file path in Flask's g object for cleanup
            g.temp_file = temp_file_path
            
            # 9. Generate output filename
            output_filename = f"predictions_{model_name}_{os.path.splitext(file.filename)[0]}.csv"

            print(output_filename)
            
            response = make_response(send_file(
                temp_file_path,
                mimetype='text/csv',
                as_attachment=True,
                download_name=output_filename
            ))
            response.headers["Content-Disposition"] = f"attachment; filename={output_filename}"
            return response
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400
def after_this_request(func):
    if not hasattr(g, 'call_after_request'):
        g.call_after_request = []
    g.call_after_request.append(func)
    return func

@app.after_request
def per_request_callbacks(response):
    for func in getattr(g, 'call_after_request', ()):
        response = func(response)
    return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)