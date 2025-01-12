from typing import Optional, Dict, Any, List, Tuple
import json
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from generateConfig import GenerateConfigLLM

@dataclass
class ModelPerformanceMetrics:
    model_name: str
    metrics: Dict[str, float]
    training_time: float
    prediction_time: float
    memory_usage: float
    convergence_info: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]

class ModelAnalysisLLM(GenerateConfigLLM):
    def __init__(self, **config):
        super().__init__(**config)
    
    def _analyze_model_performance(self, results: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
        """Analyze performance metrics for all tried models"""
        performance_metrics = []
        
        for model_name, model_results in results.items():
            metrics = {}
            
            if task_type == "classification":
                # Extract classification metrics
                report = classification_report(
                    model_results['y_true'],
                    model_results['y_pred'],
                    output_dict=True
                )
                metrics = {
                    'accuracy': report['accuracy'],
                    'macro_f1': report['macro avg']['f1-score'],
                    'weighted_f1': report['weighted avg']['f1-score'],
                    'class_wise_performance': {
                        str(cls): {'precision': d['precision'], 'recall': d['recall'], 'f1': d['f1-score']}
                        for cls, d in report.items() if isinstance(cls, (str, int)) and cls not in ['accuracy', 'macro avg', 'weighted avg']
                    }
                }
            else:  # regression
                metrics = {
                    'mse': float(mean_squared_error(model_results['y_true'], model_results['y_pred'])),
                    'rmse': float(np.sqrt(mean_squared_error(model_results['y_true'], model_results['y_pred']))),
                    'r2': float(r2_score(model_results['y_true'], model_results['y_pred'])),
                    'mae': float(np.mean(np.abs(np.array(model_results['y_true']) - np.array(model_results['y_pred']))))
                }
            
            # Add cross-validation scores if available
            if 'cv_scores' in model_results:
                cv_scores = model_results['cv_scores']
                if isinstance(cv_scores, dict):
                    metrics['cv_scores'] = {
                        'mean': float(cv_scores.get('mean', 0)),
                        'std': float(cv_scores.get('std', 0)),
                        'scores': list(cv_scores.get('scores', []))  # Convert to list
                    }
            
            # Create performance metrics dictionary
            performance_dict = {
                'model_name': model_name,
                'metrics': metrics,
                'training_time': float(model_results.get('training_time', 0)),
                'prediction_time': float(model_results.get('prediction_time', 0)),
                'memory_usage': float(model_results.get('memory_usage', 0)),
                'convergence_info': model_results.get('convergence_info', {}),
                'feature_importance': model_results.get('feature_importance', None)
            }
            
            # Convert any NumPy arrays in convergence_info or feature_importance to lists
            if isinstance(performance_dict['convergence_info'], dict):
                for key, value in performance_dict['convergence_info'].items():
                    if isinstance(value, np.ndarray):
                        performance_dict['convergence_info'][key] = value.tolist()

            if isinstance(performance_dict['feature_importance'], dict):
                for key, value in performance_dict['feature_importance'].items():
                    if isinstance(value, np.ndarray):
                        performance_dict['feature_importance'][key] = value.tolist()
            
            performance_metrics.append(performance_dict)
        
        return performance_metrics

    def generate_refined_config(self, 
                              analysis: Dict,
                              initial_results: Dict[str, Any],
                              task_type: str,
                              available_models: List[str]) -> str:
        """Generate refined configuration based on initial model performance"""
        
        # Analyze model performance
        performance_metrics = self._analyze_model_performance(initial_results, task_type)
        
        # Create performance summary
        performance_summary = {
            metric['model_name']: {
                'metrics': metric['metrics'],
                'efficiency': {
                    'training_time': metric['training_time'],
                    'prediction_time': metric['prediction_time'],
                    'memory_usage': metric['memory_usage']
                },
                'feature_importance': metric['feature_importance'],
                'convergence': metric['convergence_info']
            }
            for metric in performance_metrics
        }

        template = '''You are an expert ML engineer analyzing model performance and suggesting improvements. Based on the initial results and data characteristics, recommend an optimized configuration.

Initial Performance Results:
{performance_summary}

Dataset Characteristics:
{data_characteristics}

Available Models:
{available_models}

Analysis Objectives:
1. Identify best performing models and their strengths
2. Detect potential issues (overfitting, underfitting, bias)
3. Suggest parameter adjustments or alternative models
4. Consider computational efficiency vs performance tradeoffs

The configuration must follow this exact structure:

scalers:
  <scaler_name>:
    name: <ScalerClass>
    params: {{}}

preprocessing:
  missing_values:
    strategy: <strategy>
  categorical:
    strategy: <strategy>
  feature_selection:
    enabled: <bool>
    method: <method>
    params: {{}}

models:
  <task_type>:
    <model_name>:
      name: <ModelClass>
      params: {{}}  # For MLP use PythonLists for hidden_layer_sizes

deep_learning:
  lstm:
    enabled: true
    params:
      units: <PythonList>[]
      dropout_rates: <PythonList>[]
      optimizer: <optimizer>
      batch_size: <int>
      epochs: <int>
  
  transformer:
    enabled: true
    params:
      num_heads: <int>
      key_dim: <int>
      dropout_rate: <float>
      dense_units: <PythonList>[]
      optimizer: <optimizer>
      batch_size: <int>
      epochs: <int>

evaluation:
  cv_folds: <int>
  stratify: <bool>
  metrics: 
  <task_type>:<PythonList>[str, str, str,]

sequence_data:
  enabled: <bool>
  

Provide only the YAML configuration without explanations without any markdown. Ensure all parameters are appropriate for the data characteristics described above.'''


        prompt = template.format(
            performance_summary=str(performance_summary),
            data_characteristics=self._analyze_data_characteristics(analysis),
            available_models=available_models
        )

        response = self.invoke(prompt)
        return response.content

    def generate_performance_report(self, results: Dict[str, Any], task_type: str) -> str:
        """Generate a detailed performance report with insights and recommendations"""
        
        # Convert results to performance metrics format
        performance_metrics = self._analyze_model_performance(results, task_type)
        
        template = '''As an ML expert, analyze the following model performance results and provide detailed insights:

Performance Metrics:
{performance_details}

Task Type: {task_type}

Provide a comprehensive analysis including:

1. Model Comparison
2. Detailed Analysis
3. Potential Issues
4. Recommendations'''

        prompt = template.format(
            performance_details=str(performance_metrics),
            task_type=task_type
        )

        response = self.invoke(prompt)
        return response.content