#!/usr/bin/env python3
"""
Simple Phishing Email Classification Example

This script demonstrates basic usage of the PhishGuard ML API.
"""

import requests
import json

# API endpoint (adjust if running on different host/port)
API_URL = "http://localhost:8000/classify"

# Example emails to test
test_emails = [
    {
        "text": "URGENT! Your PayPal account has been suspended. Click here to verify: http://bit.ly/verify",
        "expected": "phishing"
    },
    {
        "text": "Thank you for your Amazon order #12345. Your package will arrive tomorrow.",
        "expected": "legitimate"
    },
    {
        "text": "Congratulations! You've won $1,000,000! Click to claim your prize NOW!",
        "expected": "phishing"
    },
    {
        "text": "Your Wells Fargo account requires immediate verification. Click here or your account will be closed.",
        "expected": "phishing"
    },
    {
        "text": "Meeting scheduled for tomorrow at 2pm in conference room B. Please confirm your attendance.",
        "expected": "legitimate"
    }
]

def classify_email(email_text):
    """Classify an email using the PhishGuard ML API"""
    try:
        response = requests.post(API_URL, json={"email_text": email_text})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API. Make sure the server is running:")
        print("  cd /path/to/security-phishing-detector")
        print("  ./run.sh")
        exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed: {e}")
        exit(1)

def main():
    print("=" * 70)
    print("PhishGuard ML - Simple Classification Example")
    print("=" * 70)
    print()

    correct = 0
    total = len(test_emails)

    for i, email in enumerate(test_emails, 1):
        print(f"Test {i}/{total}:")
        print(f"Email: {email['text'][:60]}...")
        print(f"Expected: {email['expected']}")

        result = classify_email(email['text'])

        print(f"Result: {result['classification']}")
        print(f"Confidence: {result['confidence']:.2%}")

        is_correct = result['classification'] == email['expected']
        print(f"Match: {'✅ Correct' if is_correct else '❌ Incorrect'}")

        if is_correct:
            correct += 1

        print()

    print("=" * 70)
    print(f"Accuracy: {correct}/{total} ({correct/total:.1%})")
    print("=" * 70)

if __name__ == "__main__":
    main()
