#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhishingDetector - Unified detector supporting both simple and ensemble models
"""
import warnings
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

# Import advanced feature extractor
from advanced_features import AdvancedFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhishingDetector:
    """
    Unified phishing detection system supporting both simple and ensemble modes.

    Modes:
        - simple: Fast Random Forest (33 KB, <20ms)
        - ensemble: Accurate 7-model ensemble (~100ms, TF-IDF + 39 advanced features)
    """

    def __init__(self, mode: str = 'simple', models_dir: str = 'models'):
        """
        Initialize detector in specified mode.

        Args:
            mode: 'simple' or 'ensemble'
            models_dir: Path to models directory
        """
        self.mode = mode
        self.models_dir = Path(models_dir)

        # Model components
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.feature_extractor = None

        # Metadata
        self.loaded = False
        self.model_info = {}

        # Load appropriate model
        if mode == 'simple':
            self.load_simple_model()
        elif mode == 'ensemble':
            self.load_ensemble_model()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'simple' or 'ensemble'")

    def load_simple_model(self):
        """Load simple Random Forest model (fast, lightweight)."""
        try:
            logger.info("Loading simple model...")

            clf_path = self.models_dir / "phishing_clf.pkl"
            vec_path = self.models_dir / "tfidf_vec.pkl"

            if not clf_path.exists() or not vec_path.exists():
                raise FileNotFoundError(
                    f"Simple models not found in {self.models_dir}. "
                    "Run train_model.py first."
                )

            self.model = joblib.load(clf_path)
            self.vectorizer = joblib.load(vec_path)

            self.model_info = {
                'mode': 'simple',
                'type': 'RandomForest',
                'features': 2000,
                'size_mb': 0.033,
                'speed_ms': 15
            }

            self.loaded = True
            logger.info("✅ Simple model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load simple model: {e}")
            raise

    def load_ensemble_model(self):
        """Load ensemble model (accurate, resource-intensive)."""
        try:
            logger.info("Loading ensemble model (this may take a moment)...")

            ensemble_dir = self.models_dir / "ensemble"
            ensemble_path = ensemble_dir / "ensemble_model.pkl"
            vectorizer_path = ensemble_dir / "vectorizer.pkl"
            scaler_path = ensemble_dir / "scaler.pkl"

            # Check all components exist
            if not ensemble_path.exists():
                raise FileNotFoundError(
                    f"Ensemble model not found: {ensemble_path}. "
                    "Run train_ensemble_enhanced.py first."
                )

            # Load models
            logger.info("  Loading ensemble (101 MB)...")
            ensemble_data = joblib.load(ensemble_path)
            self.model = ensemble_data['calibrated_ensemble']

            logger.info("  Loading vectorizer...")
            self.vectorizer = joblib.load(vectorizer_path)

            logger.info("  Loading scaler...")
            self.scaler = joblib.load(scaler_path)

            logger.info("  Initializing feature extractor...")
            self.feature_extractor = AdvancedFeatureExtractor()

            # Metadata
            performance = ensemble_data.get('performance_metrics', {})
            n_features = self.scaler.n_features_in_
            self.model_info = {
                'mode': 'ensemble',
                'type': 'VotingClassifier (7 models)',
                'models': ['RandomForest', 'XGBoost', 'LightGBM', 'SVM', 'MLP', 'LogisticRegression', 'GradientBoosting'],
                'features': n_features,
                'speed_ms': 100,
                'performance': performance
            }

            self.loaded = True
            logger.info("✅ Ensemble model loaded successfully")
            logger.info(f"   Models: {', '.join(self.model_info['models'])}")

        except Exception as e:
            logger.error(f"Failed to load ensemble model: {e}")
            raise

    def _extract_features_simple(self, text: str) -> np.ndarray:
        """Extract features for simple model (TF-IDF only)."""
        return self.vectorizer.transform([text]).toarray()

    def _extract_features_ensemble(self, text: str) -> np.ndarray:
        """
        Extract features for ensemble model (TF-IDF + Advanced).

        Returns:
            np.ndarray: Scaled feature array matching the trained model's dimensions
        """
        # TF-IDF features
        tfidf_features = self.vectorizer.transform([text]).toarray()

        # Advanced features (39)
        advanced_features_dict = self.feature_extractor.extract_all_features(text)
        advanced_features_array = np.array(list(advanced_features_dict.values())).reshape(1, -1)

        # Combine
        combined_features = np.hstack([tfidf_features, advanced_features_array])

        # Verify dimension matches scaler expectations
        expected_features = self.scaler.n_features_in_
        if combined_features.shape[1] != expected_features:
            raise ValueError(
                f"Feature dimension mismatch! Expected {expected_features}, got {combined_features.shape[1]}"
            )

        # Scale
        scaled_features = self.scaler.transform(combined_features)

        return scaled_features

    def predict(self, text: str) -> Tuple[str, float, bool]:
        """
        Classify text as phishing or legitimate.

        Args:
            text: Email or message content

        Returns:
            Tuple of (classification, confidence, is_phishing)
            - classification: "phishing" or "legitimate"
            - confidence: float 0.0-1.0
            - is_phishing: bool
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        if not text or not text.strip():
            raise ValueError("Empty text provided")

        # Extract features based on mode
        if self.mode == 'simple':
            features = self._extract_features_simple(text)
        else:  # ensemble
            features = self._extract_features_ensemble(text)

        # Predict (suppress LGBMClassifier feature name warnings -- the ensemble
        # contains models trained with and without feature names, but the feature
        # dimensions are validated above so the warning is harmless)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            prediction = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]

        # Format response
        classification = "phishing" if prediction == 1 else "legitimate"
        confidence = float(proba[prediction])
        is_phishing = bool(prediction == 1)

        return classification, confidence, is_phishing

    def predict_with_details(self, text: str) -> Dict:
        """
        Get detailed prediction with metadata and confidence breakdown.

        Args:
            text: Email or message content

        Returns:
            Dict with prediction, confidence, metadata, and optional model votes
        """
        classification, confidence, is_phishing = self.predict(text)

        result = {
            'classification': classification,
            'confidence': confidence,
            'is_phishing': is_phishing,
            'model_mode': self.mode,
            'model_info': self.model_info
        }

        # Add probability breakdown
        if self.mode == 'simple':
            features = self._extract_features_simple(text)
        else:
            features = self._extract_features_ensemble(text)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            proba = self.model.predict_proba(features)[0]
            result['probabilities'] = {
                'legitimate': float(proba[0]),
                'phishing': float(proba[1])
            }

            # For ensemble mode, can add individual model votes
            if self.mode == 'ensemble':
                # Get individual model predictions if available
                # This requires accessing the ensemble's estimators
                try:
                    individual_predictions = {}
                    for name, estimator in self.model.calibrated_classifiers_[0].estimator.named_estimators_.items():
                        pred = estimator.predict(features)[0]
                        individual_predictions[name] = "phishing" if pred == 1 else "legitimate"

                    result['individual_votes'] = individual_predictions

                    # Calculate agreement
                    votes = list(individual_predictions.values())
                    phishing_votes = votes.count('phishing')
                    total_votes = len(votes)
                    result['agreement_rate'] = phishing_votes / total_votes if classification == 'phishing' else (total_votes - phishing_votes) / total_votes

                except Exception as e:
                    logger.debug(f"Could not extract individual votes: {e}")

        return result

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'loaded': self.loaded,
            'mode': self.mode,
            'info': self.model_info
        }


# Convenience functions for backward compatibility
def load_simple_detector() -> PhishingDetector:
    """Load simple model detector."""
    return PhishingDetector(mode='simple')


def load_ensemble_detector() -> PhishingDetector:
    """Load ensemble model detector."""
    return PhishingDetector(mode='ensemble')


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("PhishingDetector - Testing both modes")
    print("=" * 60)

    test_texts = [
        "URGENT! Your PayPal account has been suspended. Click here to verify immediately!",
        "Thank you for your Amazon order #123-4567890. Your package will arrive tomorrow.",
        "Congratulations! You've won $1,000,000! Click to claim now!",
        "Meeting scheduled for tomorrow at 2pm in conference room B."
    ]

    # Test simple model
    print("\n1️⃣ Testing Simple Model")
    print("-" * 60)
    simple_detector = load_simple_detector()
    print(f"Model info: {simple_detector.get_model_info()}")

    for i, text in enumerate(test_texts, 1):
        classification, confidence, is_phishing = simple_detector.predict(text)
        print(f"\nTest {i}: {text[:50]}...")
        print(f"  Result: {classification} ({confidence:.2%} confident)")

    # Test ensemble model
    print("\n\n2️⃣ Testing Ensemble Model")
    print("-" * 60)
    ensemble_detector = load_ensemble_detector()
    print(f"Model info: {ensemble_detector.get_model_info()}")

    for i, text in enumerate(test_texts, 1):
        result = ensemble_detector.predict_with_details(text)
        print(f"\nTest {i}: {text[:50]}...")
        print(f"  Result: {result['classification']} ({result['confidence']:.2%} confident)")
        print(f"  Probabilities: Phishing={result['probabilities']['phishing']:.2%}, Legitimate={result['probabilities']['legitimate']:.2%}")

        if 'individual_votes' in result:
            phishing_votes = list(result['individual_votes'].values()).count('phishing')
            total_votes = len(result['individual_votes'])
            print(f"  Individual votes: {phishing_votes}/{total_votes} voted phishing (agreement: {result.get('agreement_rate', 0):.1%})")

    print("\n" + "=" * 60)
    print("✅ Both modes tested successfully!")
    print("=" * 60)
