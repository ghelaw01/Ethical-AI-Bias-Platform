import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class BiasDetector:
    """
    Comprehensive bias detection and fairness analysis toolkit
    """
    
    def __init__(self):
        self.fairness_metrics = {}
        self.protected_groups = {}
        
    def calculate_demographic_parity(self, y_pred, protected_attribute):
        """
        Calculate demographic parity (statistical parity)
        Measures if positive prediction rates are equal across groups
        """
        groups = np.unique(protected_attribute)
        positive_rates = {}
        
        for group in groups:
            group_mask = protected_attribute == group
            group_predictions = y_pred[group_mask]
            positive_rate = np.mean(group_predictions)
            positive_rates[str(group)] = positive_rate
            
        # Calculate maximum difference between groups
        rates = list(positive_rates.values())
        demographic_parity_diff = max(rates) - min(rates)
        
        return {
            'metric': 'demographic_parity',
            'value': demographic_parity_diff,
            'group_rates': positive_rates,
            'interpretation': 'Lower is better (0 = perfect parity)',
            'threshold_good': 0.1,
            'threshold_acceptable': 0.2
        }
    
    def calculate_equalized_odds(self, y_true, y_pred, protected_attribute):
        """
        Calculate equalized odds
        Measures if TPR and FPR are equal across groups
        """
        groups = np.unique(protected_attribute)
        group_metrics = {}
        
        for group in groups:
            group_mask = protected_attribute == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if len(np.unique(group_y_true)) > 1:
                tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            else:
                tpr, fpr = 0, 0
                
            group_metrics[str(group)] = {'tpr': tpr, 'fpr': fpr}
        
        # Calculate maximum difference in TPR and FPR
        tprs = [metrics['tpr'] for metrics in group_metrics.values()]
        fprs = [metrics['fpr'] for metrics in group_metrics.values()]
        
        tpr_diff = max(tprs) - min(tprs) if tprs else 0
        fpr_diff = max(fprs) - min(fprs) if fprs else 0
        
        equalized_odds_diff = max(tpr_diff, fpr_diff)
        
        return {
            'metric': 'equalized_odds',
            'value': equalized_odds_diff,
            'group_metrics': group_metrics,
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'interpretation': 'Lower is better (0 = perfect equality)',
            'threshold_good': 0.1,
            'threshold_acceptable': 0.2
        }
    
    def calculate_equality_of_opportunity(self, y_true, y_pred, protected_attribute):
        """
        Calculate equality of opportunity
        Measures if TPR is equal across groups
        """
        groups = np.unique(protected_attribute)
        tprs = {}
        
        for group in groups:
            group_mask = protected_attribute == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # Only consider positive cases
            positive_mask = group_y_true == 1
            if np.sum(positive_mask) > 0:
                group_positive_true = group_y_true[positive_mask]
                group_positive_pred = group_y_pred[positive_mask]
                tpr = np.mean(group_positive_pred)
            else:
                tpr = 0
                
            tprs[str(group)] = tpr
        
        # Calculate maximum difference in TPR
        tpr_values = list(tprs.values())
        tpr_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0
        
        return {
            'metric': 'equality_of_opportunity',
            'value': tpr_diff,
            'group_tprs': tprs,
            'interpretation': 'Lower is better (0 = perfect equality)',
            'threshold_good': 0.1,
            'threshold_acceptable': 0.2
        }
    
    def calculate_calibration(self, y_true, y_prob, protected_attribute, n_bins=10):
        """
        Calculate calibration across groups
        Measures if predicted probabilities match actual outcomes
        """
        groups = np.unique(protected_attribute)
        calibration_metrics = {}
        
        for group in groups:
            group_mask = protected_attribute == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            if len(group_y_true) > 0:
                # Create bins
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                calibration_error = 0
                total_samples = 0
                
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (group_y_prob > bin_lower) & (group_y_prob <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = group_y_true[in_bin].mean()
                        avg_confidence_in_bin = group_y_prob[in_bin].mean()
                        calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                        total_samples += np.sum(in_bin)
                
                calibration_metrics[str(group)] = calibration_error
            else:
                calibration_metrics[str(group)] = 0
        
        # Calculate maximum difference in calibration error
        cal_values = list(calibration_metrics.values())
        cal_diff = max(cal_values) - min(cal_values) if cal_values else 0
        
        return {
            'metric': 'calibration',
            'value': cal_diff,
            'group_calibration_errors': calibration_metrics,
            'interpretation': 'Lower is better (0 = perfect calibration)',
            'threshold_good': 0.05,
            'threshold_acceptable': 0.1
        }
    
    def comprehensive_bias_analysis(self, X, y, protected_attributes, model=None):
        """
        Perform comprehensive bias analysis on a dataset
        """
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        results = {
            'overall_accuracy': accuracy_score(y_test, y_pred),
            'model_type': type(model).__name__,
            'fairness_metrics': {},
            'protected_attributes': protected_attributes,
            'sample_size': len(X_test)
        }
        
        # Calculate fairness metrics for each protected attribute
        for attr_name, attr_values in protected_attributes.items():
            if attr_name in X_test.columns:
                protected_attr = X_test[attr_name].values
                
                metrics = {}
                metrics['demographic_parity'] = self.calculate_demographic_parity(y_pred, protected_attr)
                metrics['equalized_odds'] = self.calculate_equalized_odds(y_test, y_pred, protected_attr)
                metrics['equality_of_opportunity'] = self.calculate_equality_of_opportunity(y_test, y_pred, protected_attr)
                metrics['calibration'] = self.calculate_calibration(y_test, y_prob, protected_attr)
                
                results['fairness_metrics'][attr_name] = metrics
        
        # Calculate overall bias score
        bias_scores = []
        for attr_metrics in results['fairness_metrics'].values():
            for metric_data in attr_metrics.values():
                bias_scores.append(metric_data['value'])
        
        results['overall_bias_score'] = np.mean(bias_scores) if bias_scores else 0
        
        return results
    
    def suggest_mitigation_strategies(self, bias_analysis_results):
        """
        Suggest mitigation strategies based on bias analysis results
        """
        strategies = []
        
        overall_bias = bias_analysis_results.get('overall_bias_score', 0)
        
        if overall_bias > 0.2:
            strategies.append({
                'name': 'Data Preprocessing',
                'type': 'pre-processing',
                'description': 'Apply data preprocessing techniques like resampling, synthetic data generation, or feature selection to reduce bias in training data.',
                'priority': 'high',
                'complexity': 'medium'
            })
        
        if overall_bias > 0.15:
            strategies.append({
                'name': 'Fairness-Aware Training',
                'type': 'in-processing',
                'description': 'Use fairness-aware machine learning algorithms that incorporate fairness constraints during model training.',
                'priority': 'high',
                'complexity': 'high'
            })
        
        if overall_bias > 0.1:
            strategies.append({
                'name': 'Post-Processing Calibration',
                'type': 'post-processing',
                'description': 'Apply post-processing techniques to adjust model outputs to achieve fairness criteria.',
                'priority': 'medium',
                'complexity': 'low'
            })
        
        # Check specific metrics
        for attr_name, attr_metrics in bias_analysis_results.get('fairness_metrics', {}).items():
            if attr_metrics.get('demographic_parity', {}).get('value', 0) > 0.15:
                strategies.append({
                    'name': f'Demographic Parity Correction for {attr_name}',
                    'type': 'post-processing',
                    'description': f'Apply threshold optimization to achieve demographic parity for {attr_name}.',
                    'priority': 'medium',
                    'complexity': 'low'
                })
        
        if not strategies:
            strategies.append({
                'name': 'Continuous Monitoring',
                'type': 'monitoring',
                'description': 'Implement continuous bias monitoring and regular fairness audits.',
                'priority': 'low',
                'complexity': 'low'
            })
        
        return strategies

