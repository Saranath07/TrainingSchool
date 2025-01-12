import pandas as pd
import numpy as np
from scipy import stats
import json
import yaml
from typing import Dict, Any

class MLDataAnalyzer:
    def __init__(self, df: pd.DataFrame, target_column: str, task_description: str):
        self.df = df
        self.target_column = target_column
        self.task_description = task_description
        self.sample_size = 50

    def get_sample_data(self) -> pd.DataFrame:
        """Get random sample of data"""
        if len(self.df) > self.sample_size:
            return self.df.sample(n=self.sample_size, random_state=42)
        return self.df

    def analyze_dataset(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of the dataset"""
        analysis = {
            "basic_info": {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "column_types": self.df.dtypes.astype(str).to_dict(),
                "missing_values": self.df.isnull().sum().to_dict()
            },
            "numerical_analysis": {},
            "categorical_analysis": {},
            "target_analysis": {},
            "correlations": {}
        }
        
        # Analyze numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            analysis["numerical_analysis"][col] = {
                "mean": float(self.df[col].mean()),
                "median": float(self.df[col].median()),
                "std": float(self.df[col].std()),
                "skewness": float(stats.skew(self.df[col].dropna())),
                "kurtosis": float(stats.kurtosis(self.df[col].dropna())),
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "quantiles": {
                    "25%": float(self.df[col].quantile(0.25)),
                    "75%": float(self.df[col].quantile(0.75))
                }
            }
        
        # Analyze categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            analysis["categorical_analysis"][col] = {
                "unique_values": self.df[col].nunique(),
                "value_counts": self.df[col].value_counts().to_dict()
            }
        
        # Special analysis for target column
        if self.target_column in numerical_cols:
            analysis["target_analysis"] = {
                "type": "numerical",
                "statistics": analysis["numerical_analysis"][self.target_column]
            }
        else:
            analysis["target_analysis"] = {
                "type": "categorical",
                "statistics": analysis["categorical_analysis"][self.target_column]
            }
        
        # Calculate correlations with target for numerical columns
        if self.target_column in numerical_cols:
            target_correlations = {}
            for col in numerical_cols:
                if col != self.target_column:
                    correlation = self.df[col].corr(self.df[self.target_column])
                    target_correlations[col] = float(correlation)
            analysis["target_correlations"] = target_correlations

        return analysis