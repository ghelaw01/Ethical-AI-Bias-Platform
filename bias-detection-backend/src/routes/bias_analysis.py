from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import json
import io
import base64
from src.models.bias_analysis import BiasAnalysis, MitigationStrategy, FairnessMetric, db
from src.utils.bias_detector import BiasDetector
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

bias_bp = Blueprint('bias', __name__)

@bias_bp.route('/analyze', methods=['POST'])
@cross_origin()
def analyze_bias():
    """
    Perform bias analysis on uploaded dataset or generated sample data
    """
    try:
        data = request.json
        
        # Check if using sample data or uploaded data
        if data.get('use_sample_data', False):
            # Generate sample dataset
            X, y = make_classification(
                n_samples=1000,
                n_features=10,
                n_informative=5,
                n_redundant=2,
                n_clusters_per_class=1,
                random_state=42
            )
            
            # Create DataFrame with feature names
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = y
            
            # Add synthetic protected attributes
            np.random.seed(42)
            df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
            df['age_group'] = np.random.choice(['Young', 'Middle', 'Senior'], size=len(df))
            df['ethnicity'] = np.random.choice(['Group_A', 'Group_B', 'Group_C'], size=len(df))
            
            # Introduce some bias
            bias_mask = (df['gender'] == 'Female') & (df['age_group'] == 'Young')
            df.loc[bias_mask, 'target'] = np.random.choice([0, 1], size=np.sum(bias_mask), p=[0.7, 0.3])
            
            protected_attributes = {
                'gender': ['Male', 'Female'],
                'age_group': ['Young', 'Middle', 'Senior'],
                'ethnicity': ['Group_A', 'Group_B', 'Group_C']
            }
            
            X = df.drop(['target'], axis=1)
            y = df['target']
            
        else:
            # Handle uploaded CSV data
            csv_data = data.get('csv_data')
            if not csv_data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Parse CSV data
            df = pd.read_csv(io.StringIO(csv_data))
            
            target_column = data.get('target_column')
            protected_attrs = data.get('protected_attributes', {})
            
            if target_column not in df.columns:
                return jsonify({'error': f'Target column {target_column} not found'}), 400
            
            y = df[target_column]
            X = df.drop([target_column], axis=1)
            protected_attributes = protected_attrs
        
        # Initialize bias detector
        detector = BiasDetector()
        
        # Select model type
        model_type = data.get('model_type', 'RandomForest')
        if model_type == 'LogisticRegression':
            model = LogisticRegression(random_state=42)
        elif model_type == 'SVM':
            model = SVC(probability=True, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Perform comprehensive bias analysis
        results = detector.comprehensive_bias_analysis(X, y, protected_attributes, model)
        
        # Get mitigation strategies
        mitigation_strategies = detector.suggest_mitigation_strategies(results)
        
        # Save analysis to database
        analysis = BiasAnalysis(
            dataset_name=data.get('dataset_name', 'Sample Dataset'),
            model_type=model_type,
            protected_attributes=json.dumps(list(protected_attributes.keys())),
            fairness_metrics=json.dumps(results['fairness_metrics']),
            bias_score=results['overall_bias_score'],
            mitigation_strategies=json.dumps(mitigation_strategies)
        )
        
        db.session.add(analysis)
        db.session.commit()
        
        # Prepare response
        response = {
            'analysis_id': analysis.id,
            'overall_accuracy': results['overall_accuracy'],
            'overall_bias_score': results['overall_bias_score'],
            'model_type': results['model_type'],
            'sample_size': results['sample_size'],
            'fairness_metrics': results['fairness_metrics'],
            'mitigation_strategies': mitigation_strategies,
            'bias_level': 'Low' if results['overall_bias_score'] < 0.1 else 'Medium' if results['overall_bias_score'] < 0.2 else 'High'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bias_bp.route('/analyses', methods=['GET'])
@cross_origin()
def get_analyses():
    """
    Get all bias analyses
    """
    try:
        analyses = BiasAnalysis.query.order_by(BiasAnalysis.created_at.desc()).all()
        return jsonify([analysis.to_dict() for analysis in analyses])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bias_bp.route('/analyses/<int:analysis_id>', methods=['GET'])
@cross_origin()
def get_analysis(analysis_id):
    """
    Get specific bias analysis
    """
    try:
        analysis = BiasAnalysis.query.get_or_404(analysis_id)
        return jsonify(analysis.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bias_bp.route('/metrics', methods=['GET'])
@cross_origin()
def get_fairness_metrics():
    """
    Get available fairness metrics information
    """
    metrics_info = [
        {
            'name': 'Demographic Parity',
            'description': 'Measures whether positive prediction rates are equal across different groups',
            'type': 'group',
            'formula': 'P(Ŷ=1|A=a) = P(Ŷ=1|A=b) for all groups a,b',
            'interpretation': 'Difference should be close to 0. Values > 0.1 indicate potential bias.',
            'threshold_good': 0.1,
            'threshold_acceptable': 0.2
        },
        {
            'name': 'Equalized Odds',
            'description': 'Measures whether true positive and false positive rates are equal across groups',
            'type': 'group',
            'formula': 'P(Ŷ=1|Y=y,A=a) = P(Ŷ=1|Y=y,A=b) for y∈{0,1} and all groups a,b',
            'interpretation': 'Difference should be close to 0. Values > 0.1 indicate potential bias.',
            'threshold_good': 0.1,
            'threshold_acceptable': 0.2
        },
        {
            'name': 'Equality of Opportunity',
            'description': 'Measures whether true positive rates are equal across groups',
            'type': 'group',
            'formula': 'P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b) for all groups a,b',
            'interpretation': 'Difference should be close to 0. Values > 0.1 indicate potential bias.',
            'threshold_good': 0.1,
            'threshold_acceptable': 0.2
        },
        {
            'name': 'Calibration',
            'description': 'Measures whether predicted probabilities match actual outcomes across groups',
            'type': 'individual',
            'formula': 'P(Y=1|Ŷ=v,A=a) = P(Y=1|Ŷ=v,A=b) for all prediction values v and groups a,b',
            'interpretation': 'Difference should be close to 0. Values > 0.05 indicate potential bias.',
            'threshold_good': 0.05,
            'threshold_acceptable': 0.1
        }
    ]
    
    return jsonify(metrics_info)

@bias_bp.route('/mitigation-strategies', methods=['GET'])
@cross_origin()
def get_mitigation_strategies():
    """
    Get available mitigation strategies
    """
    strategies = [
        {
            'name': 'Data Preprocessing',
            'type': 'pre-processing',
            'description': 'Modify training data to reduce bias before model training',
            'techniques': [
                'Resampling (oversampling/undersampling)',
                'Synthetic data generation (SMOTE, ADASYN)',
                'Feature selection and engineering',
                'Data augmentation'
            ],
            'pros': ['Addresses root cause', 'Model-agnostic', 'Preserves model performance'],
            'cons': ['May lose information', 'Requires domain knowledge', 'Can be computationally expensive'],
            'complexity': 'Medium'
        },
        {
            'name': 'Fairness-Aware Training',
            'type': 'in-processing',
            'description': 'Incorporate fairness constraints directly into model training',
            'techniques': [
                'Adversarial debiasing',
                'Fairness constraints optimization',
                'Multi-objective optimization',
                'Regularization techniques'
            ],
            'pros': ['Integrated approach', 'Can achieve multiple fairness criteria', 'Theoretically grounded'],
            'cons': ['Model-specific', 'Complex implementation', 'May reduce accuracy'],
            'complexity': 'High'
        },
        {
            'name': 'Post-Processing',
            'type': 'post-processing',
            'description': 'Adjust model outputs to achieve fairness criteria',
            'techniques': [
                'Threshold optimization',
                'Calibration adjustment',
                'Output redistribution',
                'Equalized odds post-processing'
            ],
            'pros': ['Model-agnostic', 'Easy to implement', 'Preserves model training'],
            'cons': ['May not address root cause', 'Can reduce overall performance', 'Limited flexibility'],
            'complexity': 'Low'
        }
    ]
    
    return jsonify(strategies)

@bias_bp.route('/sample-data', methods=['GET'])
@cross_origin()
def generate_sample_data():
    """
    Generate sample dataset for demonstration
    """
    try:
        # Generate sample dataset
        X, y = make_classification(
            n_samples=500,
            n_features=8,
            n_informative=4,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Add protected attributes
        np.random.seed(42)
        df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
        df['age_group'] = np.random.choice(['Young', 'Middle', 'Senior'], size=len(df))
        
        # Convert to CSV string
        csv_string = df.to_csv(index=False)
        
        return jsonify({
            'csv_data': csv_string,
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'description': 'Synthetic dataset with potential bias in gender and age_group attributes'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

