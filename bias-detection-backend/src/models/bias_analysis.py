from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class BiasAnalysis(db.Model):
    __tablename__ = 'bias_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(100), nullable=False)
    protected_attributes = db.Column(db.Text, nullable=False)  # JSON string
    fairness_metrics = db.Column(db.Text, nullable=False)  # JSON string
    bias_score = db.Column(db.Float, nullable=False)
    mitigation_strategies = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'dataset_name': self.dataset_name,
            'model_type': self.model_type,
            'protected_attributes': json.loads(self.protected_attributes) if self.protected_attributes else [],
            'fairness_metrics': json.loads(self.fairness_metrics) if self.fairness_metrics else {},
            'bias_score': self.bias_score,
            'mitigation_strategies': json.loads(self.mitigation_strategies) if self.mitigation_strategies else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class MitigationStrategy(db.Model):
    __tablename__ = 'mitigation_strategies'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    strategy_type = db.Column(db.String(100), nullable=False)  # pre-processing, in-processing, post-processing
    effectiveness_score = db.Column(db.Float)
    implementation_complexity = db.Column(db.String(50))  # low, medium, high
    applicable_metrics = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'strategy_type': self.strategy_type,
            'effectiveness_score': self.effectiveness_score,
            'implementation_complexity': self.implementation_complexity,
            'applicable_metrics': json.loads(self.applicable_metrics) if self.applicable_metrics else [],
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class FairnessMetric(db.Model):
    __tablename__ = 'fairness_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    metric_type = db.Column(db.String(100), nullable=False)  # individual, group, counterfactual
    formula = db.Column(db.Text)
    interpretation = db.Column(db.Text)
    threshold_good = db.Column(db.Float)
    threshold_acceptable = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'metric_type': self.metric_type,
            'formula': self.formula,
            'interpretation': self.interpretation,
            'threshold_good': self.threshold_good,
            'threshold_acceptable': self.threshold_acceptable,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

