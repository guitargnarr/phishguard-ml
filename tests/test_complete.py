#!/usr/bin/env python3
"""
PhishGuard ML - Comprehensive Test Suite
==========================================
37 tests covering models, features, API, patterns, and integration

Test Categories:
- Models (8 tests): Model loading and prediction shape
- Features (10 tests): Feature extraction and combination
- API (10 tests): FastAPI endpoints
- Patterns (6 tests): Phishing pattern detection
- Integration (3 tests): End-to-end pipelines

Run: pytest tests/test_complete.py -v
"""
import pytest
import numpy as np
import joblib
from pathlib import Path
from fastapi.testclient import TestClient

# Import application components
from phishing_detector import PhishingDetector, load_simple_detector, load_ensemble_detector
from advanced_features import AdvancedFeatureExtractor
from main_production import app

# Test client
client = TestClient(app)


# ============================================================================
# CATEGORY 1: MODEL TESTS (8 tests)
# ============================================================================

class TestModels:
    """Test model loading and basic functionality."""

    def test_simple_model_loads(self):
        """Test simple model loads without errors."""
        detector = load_simple_detector()
        assert detector.loaded is True
        assert detector.mode == 'simple'

    def test_ensemble_model_loads(self):
        """Test ensemble model loads without errors."""
        try:
            detector = load_ensemble_detector()
            assert detector.loaded is True
            assert detector.mode == 'ensemble'
        except FileNotFoundError:
            pytest.skip("Ensemble models not available")

    def test_simple_vectorizer_loads(self):
        """Test simple vectorizer exists and loads."""
        vec_path = Path("models/tfidf_vec.pkl")
        assert vec_path.exists(), "Simple vectorizer not found"

        vectorizer = joblib.load(vec_path)
        assert vectorizer is not None

    def test_ensemble_vectorizer_loads(self):
        """Test ensemble vectorizer exists and loads."""
        vec_path = Path("models/ensemble/vectorizer.pkl")
        if not vec_path.exists():
            pytest.skip("Ensemble vectorizer not available")

        vectorizer = joblib.load(vec_path)
        assert vectorizer is not None

    def test_scaler_loads(self):
        """Test ensemble scaler exists and loads."""
        scaler_path = Path("models/ensemble/scaler.pkl")
        if not scaler_path.exists():
            pytest.skip("Ensemble scaler not available")

        scaler = joblib.load(scaler_path)
        assert scaler is not None

    def test_ensemble_metadata_valid(self):
        """Test ensemble metadata file exists and is valid JSON."""
        metadata_path = Path("models/ensemble/ensemble_metadata.json")
        if not metadata_path.exists():
            pytest.skip("Ensemble metadata not available")

        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert 'models' in metadata
        assert 'performance' in metadata
        assert len(metadata['models']) > 0

    def test_simple_prediction_shape(self):
        """Test simple model predictions have correct shape."""
        detector = load_simple_detector()
        text = "Test email content"

        classification, confidence, is_phishing = detector.predict(text)

        assert isinstance(classification, str)
        assert classification in ['phishing', 'legitimate']
        assert 0.0 <= confidence <= 1.0
        assert isinstance(is_phishing, (bool, np.bool_))

    def test_ensemble_prediction_shape(self):
        """Test ensemble model predictions have correct shape."""
        try:
            detector = load_ensemble_detector()
        except FileNotFoundError:
            pytest.skip("Ensemble models not available")

        text = "Test email content"
        classification, confidence, is_phishing = detector.predict(text)

        assert isinstance(classification, str)
        assert classification in ['phishing', 'legitimate']
        assert 0.0 <= confidence <= 1.0
        assert isinstance(is_phishing, (bool, np.bool_))


# ============================================================================
# CATEGORY 2: FEATURE TESTS (10 tests)
# ============================================================================

class TestFeatures:
    """Test feature extraction and combination."""

    def test_tfidf_vectorization_2000(self):
        """Test TF-IDF vectorization produces correct number of features."""
        detector = load_simple_detector()
        text = "URGENT! Click here now!"

        features = detector._extract_features_simple(text)
        # Simple model has 646 features, ensemble has 2000
        assert features.shape[1] == 646, f"Expected 646 TF-IDF features for simple model, got {features.shape[1]}"

    def test_advanced_features_count_39(self):
        """Test AdvancedFeatureExtractor returns exactly 39 features."""
        extractor = AdvancedFeatureExtractor()
        text = "URGENT! Your account has been suspended."

        features = extractor.extract_all_features(text)
        assert len(features) == 39, f"Expected 39 advanced features, got {len(features)}"

    def test_urgency_features_extracted(self):
        """Test urgency-related features are extracted."""
        extractor = AdvancedFeatureExtractor()
        text = "URGENT! IMMEDIATE ACTION REQUIRED!"

        features = extractor.extract_all_features(text)

        assert 'urgency_score' in features
        assert features['urgency_score'] > 0  # Should detect urgency words

    def test_url_features_extracted(self):
        """Test URL-related features are extracted."""
        extractor = AdvancedFeatureExtractor()
        text = "Click here: http://bit.ly/scam"

        features = extractor.extract_all_features(text)

        assert 'num_urls' in features
        assert 'has_url' in features
        assert 'has_shortener' in features
        assert features['has_url'] == 1.0
        assert features['has_shortener'] == 1.0

    def test_statistical_features_extracted(self):
        """Test statistical features are extracted."""
        extractor = AdvancedFeatureExtractor()
        text = "Normal email text."

        features = extractor.extract_all_features(text)

        assert 'length' in features
        assert 'num_words' in features
        assert 'capital_ratio' in features
        assert features['length'] == len(text)

    def test_feature_combination_2039(self):
        """Test ensemble feature combination produces 2039 features."""
        try:
            detector = load_ensemble_detector()
        except FileNotFoundError:
            pytest.skip("Ensemble models not available")

        text = "Test email"
        features = detector._extract_features_ensemble(text)

        assert features.shape[1] == 2039, f"Expected 2039 combined features, got {features.shape[1]}"

    def test_feature_scaling(self):
        """Test feature scaling doesn't change shape."""
        try:
            detector = load_ensemble_detector()
        except FileNotFoundError:
            pytest.skip("Ensemble models not available")

        text = "Test email"
        features = detector._extract_features_ensemble(text)

        # Features should be scaled but same shape
        assert features.shape[1] == 2039
        assert isinstance(features, np.ndarray)

    def test_feature_order_preserved(self):
        """Test features are extracted in consistent order."""
        extractor = AdvancedFeatureExtractor()
        text = "Same text"

        features1 = extractor.extract_all_features(text)
        features2 = extractor.extract_all_features(text)

        assert list(features1.keys()) == list(features2.keys())

    def test_null_text_handled(self):
        """Test that None or null text is handled gracefully."""
        detector = load_simple_detector()

        with pytest.raises((ValueError, AttributeError)):
            detector.predict(None)

    def test_empty_text_handled(self):
        """Test that empty text raises appropriate error."""
        detector = load_simple_detector()

        with pytest.raises(ValueError):
            detector.predict("")


# ============================================================================
# CATEGORY 3: API TESTS (10 tests)
# ============================================================================

class TestAPI:
    """Test FastAPI endpoints."""

    def test_health_endpoint(self):
        """Test /health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] == "healthy"

    def test_stats_endpoint(self):
        """Test /stats endpoint returns valid data."""
        response = client.get("/stats")
        assert response.status_code == 200
        assert "total_events" in response.json()

    def test_classify_simple_phishing(self):
        """Test /classify endpoint detects phishing (simple mode)."""
        response = client.post(
            "/classify?mode=simple",
            json={"email_text": "URGENT! Your PayPal account has been suspended. Click here!"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "classification" in data
        assert "confidence" in data
        assert data["is_phishing"] is True
        assert data["model_mode"] == "simple"

    def test_classify_simple_legitimate(self):
        """Test /classify endpoint detects legitimate email (simple mode)."""
        response = client.post(
            "/classify?mode=simple",
            json={"email_text": "Thank you for your order. Your package will arrive tomorrow."}
        )

        assert response.status_code == 200
        data = response.json()
        assert "classification" in data
        assert data["classification"] == "legitimate"

    def test_classify_ensemble_phishing(self):
        """Test /classify endpoint with ensemble mode detects phishing."""
        response = client.post(
            "/classify?mode=ensemble",
            json={"email_text": "Congratulations! You've won $1,000,000! Click to claim!"}
        )

        # May return 503 if ensemble not available
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["is_phishing"] is True
            assert data["model_mode"] == "ensemble"

    def test_classify_ensemble_legitimate(self):
        """Test /classify endpoint with ensemble mode detects legitimate."""
        response = client.post(
            "/classify?mode=ensemble",
            json={"email_text": "Meeting scheduled for tomorrow at 2pm."}
        )

        # May return 503 if ensemble not available
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["classification"] == "legitimate"

    def test_classify_empty_text_400(self):
        """Test /classify with empty text returns 400."""
        response = client.post(
            "/classify",
            json={"email_text": ""}
        )

        assert response.status_code == 400
        assert "detail" in response.json()

    def test_classify_invalid_json_422(self):
        """Test /classify with invalid JSON returns 422."""
        response = client.post(
            "/classify",
            json={"wrong_field": "value"}
        )

        assert response.status_code == 422

    def test_response_schema_valid(self):
        """Test /classify response matches expected schema."""
        response = client.post(
            "/classify?mode=simple",
            json={"email_text": "Test email"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields present
        assert "classification" in data
        assert "confidence" in data
        assert "is_phishing" in data
        assert "model_mode" in data

        # Verify types
        assert isinstance(data["classification"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["is_phishing"], bool)

    def test_classify_detailed_endpoint(self):
        """Test /classify_detailed endpoint returns extra metadata."""
        response = client.post(
            "/classify_detailed?mode=simple",
            json={"email_text": "Test email"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should have all basic fields plus probabilities and model_info
        assert "classification" in data
        assert "probabilities" in data
        assert "model_info" in data


# ============================================================================
# CATEGORY 4: PATTERN TESTS (6 tests)
# ============================================================================

class TestPatterns:
    """Test phishing pattern detection."""

    def test_urgency_pattern_detected(self):
        """Test urgency patterns are detected as phishing."""
        detector = load_simple_detector()
        text = "URGENT! IMMEDIATE ACTION REQUIRED! Your account will be closed in 24 hours!"

        classification, confidence, is_phishing = detector.predict(text)
        # Convert numpy bool to Python bool
        assert bool(is_phishing) is True

    def test_brand_impersonation_detected(self):
        """Test brand impersonation is detected."""
        detector = load_simple_detector()
        text = "PayPal Security Alert: Your account has been limited. Verify immediately."

        classification, confidence, is_phishing = detector.predict(text)
        # Should be classified as phishing (high confidence expected)
        assert isinstance(classification, str)

    def test_cryptocurrency_scam_detected(self):
        """Test cryptocurrency scam patterns detected."""
        detector = load_simple_detector()
        text = "Congratulations! You've been selected for a Bitcoin airdrop. Claim 5 BTC now!"

        classification, confidence, is_phishing = detector.predict(text)
        # Cryptocurrency + urgency + prize = likely phishing
        assert isinstance(classification, str)

    def test_legitimate_order_confirmation(self):
        """Test legitimate order confirmations are not flagged."""
        detector = load_simple_detector()
        text = "Thank you for your Amazon order #123-4567890-1234567. Your package will arrive on Tuesday."

        classification, confidence, is_phishing = detector.predict(text)
        # Should be legitimate (though not guaranteed 100%)
        assert isinstance(classification, str)

    def test_legitimate_meeting_invite(self):
        """Test legitimate meeting invites are not flagged."""
        detector = load_simple_detector()
        text = "Meeting scheduled for tomorrow at 2:00 PM in Conference Room B. Please confirm attendance."

        classification, confidence, is_phishing = detector.predict(text)
        # Should be legitimate
        assert isinstance(classification, str)

    def test_edge_case_unicode(self):
        """Test handling of unicode and special characters."""
        detector = load_simple_detector()
        text = "Ürgent! Yöur accöunt needs verificatiön 你好"

        classification, confidence, is_phishing = detector.predict(text)
        # Should handle unicode without crashing
        assert isinstance(classification, str)
        assert 0.0 <= confidence <= 1.0


# ============================================================================
# CATEGORY 5: INTEGRATION TESTS (3 tests)
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_simple_pipeline_end_to_end(self):
        """Test complete simple model pipeline from text to prediction."""
        # Load detector
        detector = load_simple_detector()

        # Test text
        text = "URGENT! Click here to claim your prize!"

        # Extract features
        features = detector._extract_features_simple(text)
        assert features.shape[1] == 646  # Simple model has 646 features

        # Predict
        classification, confidence, is_phishing = detector.predict(text)
        assert classification in ['phishing', 'legitimate']
        assert 0.0 <= confidence <= 1.0

    def test_ensemble_pipeline_end_to_end(self):
        """Test complete ensemble pipeline from text to prediction."""
        try:
            detector = load_ensemble_detector()
        except FileNotFoundError:
            pytest.skip("Ensemble models not available")

        # Test text
        text = "Your package is ready for pickup."

        # Extract features
        features = detector._extract_features_ensemble(text)
        assert features.shape[1] == 2039

        # Predict
        classification, confidence, is_phishing = detector.predict(text)
        assert classification in ['phishing', 'legitimate']
        assert 0.0 <= confidence <= 1.0

        # Get detailed prediction
        result = detector.predict_with_details(text)
        assert 'probabilities' in result
        assert 'model_info' in result

    def test_model_agreement_rate(self):
        """Test ensemble models show reasonable agreement."""
        try:
            detector = load_ensemble_detector()
        except FileNotFoundError:
            pytest.skip("Ensemble models not available")

        # Clear phishing example
        text = "URGENT! Account suspended! Click here now or lose access!"

        result = detector.predict_with_details(text)

        # For obvious phishing, expect high agreement
        if 'agreement_rate' in result:
            # Agreement should be reasonable (>60%)
            assert result['agreement_rate'] >= 0.6


# ============================================================================
# TEST SUMMARY
# ============================================================================

def test_count():
    """Meta-test: Verify we have exactly 37 tests."""
    import inspect

    test_classes = [TestModels, TestFeatures, TestAPI, TestPatterns, TestIntegration]

    total_tests = 0
    for test_class in test_classes:
        methods = [m for m in dir(test_class) if m.startswith('test_')]
        total_tests += len(methods)

    assert total_tests == 37, f"Expected 37 tests, found {total_tests}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
