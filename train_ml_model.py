"""
AI Model Training for Valve Leak Detection

Trains a supervised machine learning model using features extracted from WRPM/XML files.
Uses XGBoost and Random Forest ensemble for robust predictions.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb
from datetime import datetime


class LeakDetectionMLTrainer:
    """
    Machine Learning trainer for valve leak detection.

    Uses ensemble of XGBoost and Random Forest with voting.
    """

    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.feature_columns = None
        self.training_stats = {}

    def load_training_data(self, csv_path: str, label_column: str = 'manual_label'):
        """
        Load training data from CSV.

        Args:
            csv_path: Path to CSV file with extracted features
            label_column: Column containing labels ('leak' or 'normal')

        Returns:
            X (features), y (labels)
        """
        print(f"Loading training data from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Check if label column exists
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found. Available columns: {list(df.columns)}")

        # Define feature columns (all numeric features)
        self.feature_columns = [
            'mean_amplitude', 'median_amplitude', 'std_amplitude',
            'max_amplitude', 'min_amplitude',
            'percentile_25', 'percentile_75', 'percentile_90', 'percentile_95', 'percentile_99',
            'above_1g_ratio', 'above_2g_ratio', 'above_3g_ratio', 'above_5g_ratio',
            'peaks_above_5g', 'peaks_above_10g',
            'amplitude_range', 'iqr', 'rms', 'crest_factor'
        ]

        # Check if all features exist
        missing_features = [f for f in self.feature_columns if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Extract features
        X = df[self.feature_columns].values

        # Convert labels to binary (0 = normal, 1 = leak)
        y = (df[label_column].str.lower() == 'leak').astype(int).values

        print(f"Loaded {len(X)} samples")
        print(f"  Leak samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"  Normal samples: {len(y) - y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
        print(f"  Features: {len(self.feature_columns)}")

        return X, y

    def train_models(self, X, y, test_size=0.2, random_state=42):
        """
        Train XGBoost and Random Forest models.

        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of test set
            random_state: Random seed

        Returns:
            Training statistics dict
        """
        print("\n" + "="*70)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*70)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")

        # Train XGBoost
        print("\n[1/2] Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss'
        )
        self.xgb_model.fit(X_train, y_train)

        # Train Random Forest
        print("[2/2] Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        self.rf_model.fit(X_train, y_train)

        # Evaluate models
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)

        stats = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': len(self.feature_columns),
            'models': {}
        }

        for name, model in [('XGBoost', self.xgb_model), ('Random Forest', self.rf_model)]:
            print(f"\n{name} Performance:")

            # Train predictions
            y_train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)

            # Test predictions
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]

            test_acc = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)

            # ROC AUC (if we have both classes in test set)
            try:
                auc = roc_auc_score(y_test, y_test_proba)
            except:
                auc = None

            print(f"  Training Accuracy: {train_acc:.3f}")
            print(f"  Test Accuracy: {test_acc:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1 Score: {f1:.3f}")
            if auc:
                print(f"  ROC AUC: {auc:.3f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            print(f"\n  Confusion Matrix:")
            print(f"    TN: {cm[0,0]}  FP: {cm[0,1]}")
            print(f"    FN: {cm[1,0]}  TP: {cm[1,1]}")

            stats['models'][name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': auc,
                'confusion_matrix': cm.tolist()
            }

        # Cross-validation
        print("\n" + "="*70)
        print("CROSS-VALIDATION (5-fold)")
        print("="*70)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for name, model in [('XGBoost', self.xgb_model), ('Random Forest', self.rf_model)]:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            print(f"\n{name}:")
            print(f"  CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            stats['models'][name]['cv_accuracy'] = scores.mean()
            stats['models'][name]['cv_std'] = scores.std()

        # Feature importance
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE (Top 10)")
        print("="*70)

        xgb_importance = self.xgb_model.feature_importances_
        rf_importance = self.rf_model.feature_importances_

        # Average importance
        avg_importance = (xgb_importance + rf_importance) / 2
        feature_importance = list(zip(self.feature_columns, avg_importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print("\nFeature                  Importance")
        print("-" * 40)
        for feat, imp in feature_importance[:10]:
            print(f"{feat:24s} {imp:.4f}")

        stats['feature_importance'] = {feat: float(imp) for feat, imp in feature_importance}

        self.training_stats = stats
        return stats

    def save_models(self, output_dir: str = '.'):
        """
        Save trained models and metadata.

        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save models
        model_data = {
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'feature_columns': self.feature_columns,
            'training_stats': self.training_stats,
            'timestamp': timestamp
        }

        model_file = output_path / f'leak_detection_model_{timestamp}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

        # Also save as latest
        latest_file = output_path / 'leak_detection_model_latest.pkl'
        with open(latest_file, 'wb') as f:
            pickle.dump(model_data, f)

        print("\n" + "="*70)
        print("MODELS SAVED")
        print("="*70)
        print(f"  Timestamped: {model_file}")
        print(f"  Latest: {latest_file}")

        # Save training report
        report_file = output_path / f'training_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("VALVE LEAK DETECTION - ML TRAINING REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"\nTimestamp: {timestamp}\n")
            f.write(f"\nTraining Set Size: {self.training_stats['train_size']}\n")
            f.write(f"Test Set Size: {self.training_stats['test_size']}\n")
            f.write(f"Features: {self.training_stats['feature_count']}\n")

            for model_name, stats in self.training_stats['models'].items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Test Accuracy: {stats['test_accuracy']:.3f}\n")
                f.write(f"  Precision: {stats['precision']:.3f}\n")
                f.write(f"  Recall: {stats['recall']:.3f}\n")
                f.write(f"  F1 Score: {stats['f1_score']:.3f}\n")
                if stats['roc_auc']:
                    f.write(f"  ROC AUC: {stats['roc_auc']:.3f}\n")
                f.write(f"  CV Accuracy: {stats['cv_accuracy']:.3f} (+/- {stats['cv_std']:.3f})\n")

        print(f"  Report: {report_file}")

        return model_file


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Train ML model for leak detection')
    parser.add_argument('data', help='Training data CSV file')
    parser.add_argument('--label-column', default='manual_label',
                       help='Column containing labels (default: manual_label)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for models (default: current directory)')

    args = parser.parse_args()

    # Initialize trainer
    trainer = LeakDetectionMLTrainer()

    # Load data
    X, y = trainer.load_training_data(args.data, args.label_column)

    # Train models
    stats = trainer.train_models(X, y, test_size=args.test_size)

    # Save models
    trainer.save_models(args.output_dir)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review training_report_*.txt for detailed metrics")
    print("  2. Use leak_detection_model_latest.pkl for predictions")
    print("  3. Integrate with leak detector using ml_leak_detector.py")


if __name__ == "__main__":
    main()
