from typing import Optional, Dict, Any, List
import json
import yaml
import numpy as np
from dataclasses import dataclass
from enum import Enum
from __init__ import BaseLLM



@dataclass
class DataCharacteristics:
    has_missing_values: bool
    has_outliers: bool
    is_imbalanced: bool
    feature_correlation_strength: str  # "high", "medium", "low"
    data_size: str  # "small", "medium", "large"
    dimension_count: str  # "low", "medium", "high"
    target_skewness: float

class GenerateConfigLLM(BaseLLM):
    def __init__(self, **config):
        super().__init__(**config)
        
    def _analyze_data_characteristics(self, analysis: Dict) -> DataCharacteristics:
        """Extract key characteristics from data analysis"""
        # Check for missing values
        has_missing = any(analysis['basic_info']['missing_values'].values())
        
        # Check for outliers using IQR
        has_outliers = False
        for col_stats in analysis['numerical_analysis'].values():
            q1 = col_stats['quantiles']['25%']
            q3 = col_stats['quantiles']['75%']
            iqr = q3 - q1
            if col_stats['min'] < (q1 - 1.5 * iqr) or col_stats['max'] > (q3 + 1.5 * iqr):
                has_outliers = True
                break
        
        # Check for class imbalance
        is_imbalanced = False
        if 'target_analysis' in analysis and analysis['target_analysis']['type'] == 'categorical':
            value_counts = list(analysis['target_analysis']['statistics']['value_counts'].values())
            if min(value_counts) / max(value_counts) < 0.3:  # 30% threshold
                is_imbalanced = True
        
        # Analyze feature correlations
        correlation_strength = "low"
        if 'target_correlations' in analysis:
            correlations = abs(np.array(list(analysis['target_correlations'].values())))
            max_correlation = np.max(correlations)
            if max_correlation > 0.7:
                correlation_strength = "high"
            elif max_correlation > 0.4:
                correlation_strength = "medium"
        
        # Determine data size
        total_rows = analysis['basic_info']['total_rows']
        data_size = "small" if total_rows < 1000 else "medium" if total_rows < 10000 else "large"
        
        # Determine dimensionality
        feature_count = analysis['basic_info']['total_columns'] - 1
        dimension_count = "low" if feature_count < 10 else "medium" if feature_count < 50 else "high"
        
        # Get target skewness
        target_skewness = 0.0
        if 'target_analysis' in analysis and analysis['target_analysis']['type'] == 'numerical':
            target_skewness = analysis['target_analysis']['statistics']['skewness']
        
        return DataCharacteristics(
            has_missing_values=has_missing,
            has_outliers=has_outliers,
            is_imbalanced=is_imbalanced,
            feature_correlation_strength=correlation_strength,
            data_size=data_size,
            dimension_count=dimension_count,
            target_skewness=target_skewness
        )
    


    def generate_config(self, analysis: Dict, sample_data: Any, task_description: str, models: List[str]) -> str:
        """Generate configuration using enhanced prompt"""
        # Extract data characteristics
        characteristics = self._analyze_data_characteristics(analysis)

        
        template = '''You are an expert ML engineer tasked with creating an optimal machine learning pipeline configuration. Based on the following detailed analysis, recommend the best configuration.

Task Description: {task_description}

Dataset Characteristics:
1. Basic Information:
   - Rows: {rows}
   - Features: {columns}
   - Task Type: <task_type> regression/classification
   - Data Size: {data_size}
   - Dimensionality: {dimension_count}

2. Key Data Characteristics:
   - Missing Values: {missing_values}
   - Outliers: {outliers}
   - Class Imbalance: {imbalance}
   - Feature Correlation: {correlation}
   - Target Skewness: {skewness:.2f}

3. Detailed Analysis:
Target Variable Analysis:
{target_analysis}

Numerical Features:
{numerical_analysis}

Categorical Features:
{categorical_analysis}

Target Correlations:
{target_correlations}

4. Sample Data:
{sample_data}


The Models which you are supposed to use. (Use with these exact names). Use Python Lists [] for parameters like hidden layers.

{models}

If sequence data use only lstm and transformer models

Based on these characteristics, generate a YAML configuration that:

1. Addresses Data Quality:
   - Handle missing values if present
   - Address outliers if detected
   - Consider class imbalance in classification tasks

2. Optimizes Preprocessing:
   - Select appropriate scalers based on distributions
   - Handle categorical variables effectively
   - Consider dimensionality reduction if needed

3. Selects Models:
   - Choose models suitable for the data size
   - Consider model complexity vs. data size
   - Adjust hyperparameters for data characteristics

4. Configures Evaluation:
   - Select metrics appropriate for the task
   - Consider stratification for imbalanced data
   - Set appropriate cross-validation strategy

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
    <model_name>: # Give a generic name.
      name: <ModelClass>
      params: {{}}

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
        task_description=task_description,
        models=models,
        rows=analysis['basic_info']['total_rows'],
        columns=analysis['basic_info']['total_columns'],
        data_size=characteristics.data_size,
        dimension_count=characteristics.dimension_count,
        missing_values="Present" if characteristics.has_missing_values else "None",
        outliers="Present" if characteristics.has_outliers else "None",
        imbalance="Present" if characteristics.is_imbalanced else "None",
        correlation=characteristics.feature_correlation_strength,
        skewness=characteristics.target_skewness,
        target_analysis=json.dumps(analysis['target_analysis'], indent=2),
        numerical_analysis=json.dumps(analysis['numerical_analysis'], indent=2),
        categorical_analysis=json.dumps(analysis['categorical_analysis'], indent=2),
        target_correlations=json.dumps(analysis.get('target_correlations', {}), indent=2),
        sample_data=sample_data.to_string()
    )



        response = self.invoke(prompt)
        return response.content

def process_config_response(response: str) -> Dict:
    """Process and validate LLM response"""
    try:
        config = yaml.safe_load(response)
        
        # Validate essential sections
        required_sections = ['scalers', 'preprocessing', 'models', 'evaluation']
        if not all(section in config for section in required_sections):
            raise ValueError("Missing required configuration sections")
        
        return config
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {str(e)}")