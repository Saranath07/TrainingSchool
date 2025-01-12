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
from trainingSchool import TrainingSchool
import joblib
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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

def save_model(model, filepath):
    
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def load_model(filepath):
   
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        print(f"Model successfully loaded from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

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
            # Load and process the data
            df = pd.read_csv(filepath)
            
            # Initialize analyzer and generate config
            analyzer = MLDataAnalyzer(df, "target", task_description)
            sample_data = analyzer.get_sample_data()
            analysis = analyzer.analyze_dataset()
            models = [
            'StandardScaler',
            'MinMaxScaler',
            'RobustScaler',
            'LinearRegression',
            'Ridge',
            'RandomForestRegressor',
            'SVR',
            'XGBRegressor',
            'MLPRegressor',
            'LogisticRegression',
            'RandomForestClassifier',
            'SVC',
            'XGBClassifier',
            'MLPClassifier'
        ]
            
            config_generator = GenerateConfigLLM()
            config = config_generator.generate_config(analysis, sample_data, task_description, models)
            
            # Save config and train model
            yaml_content = config.strip().strip('`').lstrip('yml').strip()
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml', delete=False) as temp_config_file:
                temp_config_file.write(yaml_content)
                temp_config_file.flush()
                config_path = temp_config_file.name

                # 1. Train the model
                trainer = TrainingSchool(config_path=config_path)
                X = df.drop("target", axis=1).values
                y = df["target"].values
                
                trainer.fit(X, y)
                best_info = trainer.get_best_model()
                
                # 2. Save model and scaler together in a single file
                model_filename = f"{filename.rsplit('.', 1)[0]}_model.joblib"
                model_save_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)

                # Dump a dictionary containing both the model and the scaler
                joblib.dump(
                    {
                        'model': best_info['model'],
                        'scaler': best_info['scaler']
                    },
                    model_save_path
                )
                
                # 3. Create serializable model info
                model_info = {
                    'filename': filename,
                    'task_description': task_description,
                    'best_model': {
                        'model_type': type(best_info['model']).__name__,
                        'score': best_info.get('score', None),
                        'model_path': model_filename  # relative filename in MODEL_FOLDER
                    },
                    'feature_columns': list(df.drop("target", axis=1).columns)
                }
                
                # Save model info JSON
                model_info_path = os.path.join(
                    app.config['MODEL_FOLDER'],
                    f"{filename.rsplit('.', 1)[0]}_info.json"
                )
                with open(model_info_path, 'w') as f:
                    json.dump(model_info, f)
                
                # Cleanup
                os.unlink(config_path)
                
                return jsonify({
                    'success': True,
                    'model_info': model_info
                })
                
        except Exception as e:
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