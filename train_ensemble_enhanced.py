#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Ensemble Training with Comprehensive Phishing Pattern Coverage
Includes modern tactics, sophisticated techniques, language diversity, and real-world noise
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import json
from datetime import datetime
import random

# Import our modules
from model_ensemble import EnsemblePhishingDetector, create_and_train_ensemble
from advanced_features import AdvancedFeatureExtractor


def generate_comprehensive_phishing_data(n_samples=5000):
    """
    Generate comprehensive training data covering modern phishing tactics.

    Includes:
    - Modern phishing tactics (crypto, COVID, job scams, romance, social media, 2FA, BEC, invoice)
    - Sophisticated techniques (spear phishing, multi-stage, QR codes, vishing, punycode, HTML)
    - Language diversity (typos, regional variations, grammatical errors)
    - Real-world noise (marketing emails, legitimate security alerts, forwarded emails)
    """
    print("🎲 Generating comprehensive phishing training data...")
    print(f"  Target: {n_samples} samples (expanded from 27 to 150+ templates)")

    # ==================== MODERN PHISHING TACTICS ====================

    # 1. Cryptocurrency Scams
    crypto_phishing = [
        "URGENT: Your {crypto} wallet has been {action}! Verify your seed phrase immediately: {url}",
        "Congratulations! You've been airdropped {amount} {crypto}. Claim now before it expires: {url}",
        "{exchange} Security Alert: Unusual login detected from {location}. Confirm your identity: {url}",
        "WARNING: {crypto} wallet migration required. Move your funds to secure wallet: {url}",
        "Free {amount} {crypto} giveaway! {celebrity} is giving away crypto. Claim yours: {url}",
        "Your {exchange} account shows unauthorized transactions. Freeze account: {url}",
        "{crypto} Hard Fork Alert! Claim your new tokens before {date}: {url}",
        "KYC Verification Required: {exchange} needs your documents to avoid account closure: {url}",
    ]

    # 2. COVID/Health-Related Phishing
    health_phishing = [
        "COVID-19 Vaccine Appointment Confirmation - Click to verify your slot: {url}",
        "Your COVID test results are ready. Access secure portal: {url}",
        "Health Insurance: Claim your pandemic relief benefit of ${amount}: {url}",
        "Urgent: Vaccine side effects reported. Check if you're affected: {url}",
        "COVID Relief Fund: You qualify for ${amount} stimulus payment. Apply: {url}",
        "Contact Tracing Alert: You may have been exposed. Get tested: {url}",
        "Healthcare.gov: Update your insurance information by {date}: {url}",
        "Pharmacy Notification: Your prescription is ready for pickup. Confirm: {url}",
    ]

    # 3. Job Offer Scams
    job_phishing = [
        "Job Offer: {company} wants to hire you as {position}! Salary: ${amount}/year. Apply: {url}",
        "Urgent: Complete your {company} background check to start Monday: {url}",
        "Work From Home Opportunity: Earn ${amount}/month. No experience needed: {url}",
        "LinkedIn: {name} wants to hire you. Review the job posting: {url}",
        "Indeed Job Alert: {position} at {company} - ${amount}/yr. Quick apply: {url}",
        "Your resume was selected for {position}. Schedule interview: {url}",
        "Mystery Shopper Opportunity: Earn ${amount} per assignment. Sign up: {url}",
        "Remote Position: {company} needs {position}. Training provided. Start today: {url}",
    ]

    # 4. Romance/Dating Scams
    romance_phishing = [
        "{name} sent you a message on {platform}. Read it here: {url}",
        "Someone liked your profile! See who's interested: {url}",
        "You have {number} new matches on {platform}! View them: {url}",
        "{name} wants to video chat with you. Join now: {url}",
        "Verify your {platform} profile to unlock premium features: {url}",
        "Your {platform} subscription is expiring. Renew to keep matches: {url}",
        "Someone sent you a gift! Click to accept: {url}",
        "Profile Verification Required: Prove you're not a bot: {url}",
    ]

    # 5. Social Media Impersonation
    social_media_phishing = [
        "{platform} Security: Unusual login from {location}. Was this you? {url}",
        "Your {platform} account has been reported for {violation}. Appeal here: {url}",
        "{platform} Copyright Claim: Your content violates policy. Dispute: {url}",
        "Congratulations! You've been selected for {platform} verification. Apply: {url}",
        "Your {platform} page will be deleted in 24 hours. Prevent this: {url}",
        "{number} people tagged you in a post. See who: {url}",
        "{platform} Prize: You won {prize}! Claim within {time}: {url}",
        "Action Required: {platform} Terms of Service updated. Accept to continue: {url}",
    ]

    # 6. Two-Factor Authentication Bypass
    twofa_phishing = [
        "{company} Security Code: {code}. Never share this code. If you didn't request this, secure your account: {url}",
        "Your {company} 2FA has been disabled due to suspicious activity. Re-enable: {url}",
        "New device login detected for {email}. Approve or deny: {url}",
        "Your authenticator app needs re-sync with {company}. Update now: {url}",
        "Security Alert: Someone attempted to disable your 2FA. Verify it's secure: {url}",
        "Backup codes for {company} have been regenerated. Download new codes: {url}",
        "{company}: Verification required to complete sign-in. Code: {code}. Confirm: {url}",
        "Your 2FA phone number {phone} will be removed in 24 hours. Keep it: {url}",
    ]

    # 7. Business Email Compromise (BEC)
    bec_phishing = [
        "CEO Urgent Request: Wire transfer needed for {company} acquisition. Details: {url}",
        "CFO: Update direct deposit information for payroll by {date}: {url}",
        "URGENT - from {executive}: Need you to purchase ${amount} in gift cards for client. Reply ASAP.",
        "HR: W2 forms required for all employees. Submit by EOD: {url}",
        "Legal: Confidential lawsuit settlement. Review NDA and wire ${amount}: {url}",
        "{executive} (via mobile): In meeting, need you to send wire transfer. Details attached.",
        "Accounting: Vendor payment details changed. Update records: {url}",
        "From {executive}'s Assistant: Boss needs {number} Amazon gift cards for employee rewards. Buy and send codes.",
    ]

    # 8. Invoice/Payment Redirection Scams
    invoice_phishing = [
        "Invoice #{invoice} Overdue: Payment of ${amount} required by {date}: {url}",
        "Updated Banking Information: {vendor} changed payment details. Update records: {url}",
        "FINAL NOTICE: Invoice #{invoice} unpaid. Avoid legal action, pay now: {url}",
        "Payment Failed: Your transaction of ${amount} declined. Update payment method: {url}",
        "{company} Accounting: Wire transfer instructions updated. See new details: {url}",
        "Urgent: Your {service} subscription payment failed. Retry payment: {url}",
        "Vendor Portal Update: {vendor} requires resubmission of payment info: {url}",
        "Tax Notice: IRS requires payment of ${amount} by {date}. Pay online: {url}",
    ]

    # ==================== SOPHISTICATED TECHNIQUES ====================

    # 9. Spear Phishing (Personalized)
    spear_phishing = [
        "Hi {firstname}, your {company} colleague {coworker} shared a document with you: {url}",
        "{firstname}, I noticed you work at {company}. I have an opportunity for you: {url}",
        "Dear {firstname} {lastname}, your {alma_mater} alumni network has a message: {url}",
        "{firstname}, someone from {company} viewed your LinkedIn profile. See who: {url}",
        "Hey {firstname}! Remember me from {event}? Let's connect: {url}",
        "{firstname}, your manager {manager} needs you to review this urgently: {url}",
        "Hi {firstname}, this is {client} from {client_company}. Updated proposal attached: {url}",
        "{firstname}, your PTO request for {date} needs approval. Review: {url}",
    ]

    # 10. Multi-Stage Phishing (Legitimate First Contact)
    multistage_phishing = [
        "Thank you for your {company} application. Next steps in onboarding: {url}",
        "Your {company} order #{order} shipped! Track your package: {url}",
        "Receipt for your ${amount} purchase at {company}. View details: {url}",
        "Your {service} trial is ending in 3 days. Continue your subscription: {url}",
        "Welcome to {company}! Complete your profile to get started: {url}",
        "Your appointment is confirmed for {date} at {time}. View details: {url}",
        "{company} Survey: Share your feedback and enter to win ${amount}: {url}",
        "Your {company} rewards points ({points}) expire soon. Redeem them: {url}",
    ]

    # 11. QR Code Phishing
    qr_phishing = [
        "Scan this QR code to verify your {company} account: [QR CODE] Alt link: {url}",
        "Restaurant Menu: Scan QR code to order and pay. Link: {url}",
        "Parking Payment: Scan to avoid ticket. Expires in 10 minutes. {url}",
        "Event Check-In: Scan QR code to register for {event}: {url}",
        "WiFi Access: Scan QR to connect to {location} network. {url}",
        "Product Authentication: Scan to verify your {product} is genuine: {url}",
        "COVID Vaccination Record: Scan QR to download proof: {url}",
        "Package Delivery: Scan QR to confirm delivery address: {url}",
    ]

    # 12. Voice Phishing (Vishing) Transcripts
    vishing_phishing = [
        "[VOICEMAIL] This is {name} from {company} fraud department. Call us immediately at {phone} regarding suspicious activity.",
        "[TRANSCRIPT] Your Social Security number has been suspended. Press 1 to speak with an agent or call {phone}.",
        "[VOICEMAIL] IRS Legal Department: You owe ${amount} in back taxes. Call {phone} to avoid arrest.",
        "[TRANSCRIPT] This is {bank} security. We've detected fraud. Verify your identity at {phone}.",
        "[VOICEMAIL] Your {service} account will be closed. Call {phone} to prevent suspension.",
        "[TRANSCRIPT] Microsoft support: Your computer has a virus. Call {phone} for remote assistance.",
        "[VOICEMAIL] Package delivery failed. Reschedule at {phone} or visit {url}.",
        "[TRANSCRIPT] Auto warranty expiring. Call {phone} today to renew coverage.",
    ]

    # 13. Unicode/Punycode Lookalike Domains
    punycode_phishing = [
        "Update your Apple ID: https://ạpple.com/account (Note: 'a' is unicode ạ)",
        "Paypal Security Alert: https://pаypal.com/verify (Note: 'a' is Cyrillic а)",
        "Amazon Order Confirmation: https://amazοn.com/orders (Note: 'o' is Greek ο)",
        "Google Security: https://gооgle.com/security (Note: 'o' is Cyrillic о)",
        "Microsoft Account: https://microsоft.com/login (Note: 'o' is Cyrillic о)",
        "Netflix Update: https://netfliх.com/account (Note: 'x' is Cyrillic х)",
        "Facebook Message: https://facеbook.com/messages (Note: 'e' is Cyrillic е)",
        "Your Αmazon package: https://αmazon.com/track (Note: 'A' is Greek Α)",
    ]

    # 14. HTML Email with Hidden Content
    html_phishing = [
        "Dear customer, <span style='display:none'>SCAM SCAM SCAM</span> your account needs verification: {url}",
        "Legitimate bank notice. <!--This is a phishing scam--> Update your details: {url}",
        "<div style='font-size:0px'>FAKE EMAIL</div>Your {company} security alert. Click: {url}",
        "From: {company} <span style='color:white'>fake@scammer.com</span> Verify account: {url}",
        "Your order shipped. <img src='{url}' width='1' height='1'/> Track here: {url}",
        "Security notice<!-- actually phishing --> from {company}: {url}",
        "<a href='{malicious_url}'>https://{legitimate_company}.com</a> Click to verify",
        "Email appears from {company} <div style='display:none'>but it's not</div>. Update account: {url}",
    ]

    # ==================== LANGUAGE DIVERSITY ====================

    # 15. Typos and Grammar Errors (Common in Phishing)
    typo_phishing = [
        "URGENT: You're account has been suspened! Verufy now: {url}",  # intentional typos
        "Dear costumer, yur {company} accont need update. Click hear: {url}",
        "Congratulation! You won prize of ${amount}. Clame here: {url}",
        "Securty alert: Someone try login you account. Confirm: {url}",
        "IMORTANT: You package coud not deliverd. Reshcedule: {url}",
        "You {bank} acount have suspicios activty. Verify: {url}",
        "Tax refund ${amount} aprooved. Submit form hear: {url}",
        "You're order #${order} has been shiped. Track: {url}",  # incorrect "you're"
    ]

    # 16. Regional Variations (UK English)
    regional_phishing = [
        "Your {company} account has been cancelled. Authorise access: {url}",  # UK: cancelled, authorise
        "HMRC Tax Refund: You're owed £{amount}. Claim here: {url}",  # UK tax agency
        "Royal Mail: Your parcel couldn't be delivered. Reschedule: {url}",
        "TV Licence payment failed. Update your details: {url}",  # UK specific
        "Your NHS appointment is confirmed for {date}. View details: {url}",  # UK healthcare
        "Barclays Security Alert: Unusual activity on your current account: {url}",  # UK bank
        "Council Tax arrears of £{amount}. Avoid court action: {url}",  # UK local tax
        "DWP Benefits: Update your Universal Credit details: {url}",  # UK benefits
    ]

    # ==================== REAL-WORLD NOISE ====================

    # 17. Marketing Emails (Hard to Distinguish from Spam)
    marketing_templates = [
        "FLASH SALE: {percent}% off everything! Limited time only: {url}",
        "You're invited! Exclusive early access to {product}. Shop now: {url}",
        "{company} Member Rewards: You have {points} points. Redeem: {url}",
        "Last Chance: Your cart items are selling out! Complete purchase: {url}",
        "NEW ARRIVALS: Spring collection just dropped. Shop first: {url}",
        "🔥 HOT DEAL: Save ${amount} on {product}. Ends midnight! {url}",
        "Your {company} wishlist items are on sale! Don't miss out: {url}",
        "VIP ACCESS: Be the first to know about our secret sale: {url}",
    ]

    # 18. Legitimate Security Alerts (Look Similar to Phishing)
    legitimate_security = [
        "Security Alert: New sign-in from {location}. If this wasn't you, secure your account: {url}",
        "Your password was changed on {date}. Didn't make this change? Reset it: {url}",
        "Two-factor authentication enabled on your account. Manage settings: {url}",
        "Unusual activity detected on your account. Review recent activity: {url}",
        "Your account was accessed from a new device. View devices: {url}",
        "Security checkup: Review your account security settings: {url}",
        "Recovery email added to your account. Remove it if not you: {url}",
        "Payment method updated successfully. View payment methods: {url}",
    ]

    # 19. Forwarded Emails (Multiple Layers)
    forwarded_templates = [
        "FWD: FWD: URGENT - Read this! Your {company} account issue: {url}",
        "Fwd: Important: CEO message about {topic}. All staff must review: {url}",
        "FW: Security alert from IT department. Action required: {url}",
        ">>> Forwarded message >>> From: {sender} Prize notification: {url}",
        "FWD: Invoice #{invoice} - Please process payment ASAP: {url}",
        "Fwd: Client request - needs response today: {url}",
        "FW: Tax document attached - review and sign: {url}",
        ">> Forwarded >> Partnership opportunity from {company}: {url}",
    ]

    # ==================== MORE LEGITIMATE TEMPLATES ====================

    legitimate_confirmations = [
        "Your {company} order #{order} has been confirmed. Estimated delivery: {date}.",
        "Thank you for your purchase! Receipt for ${amount} is attached.",
        "Your appointment with {provider} is scheduled for {date} at {time}.",
        "Welcome to {company}! Your account has been created successfully.",
        "Your subscription to {service} has been renewed. Next billing: {date}.",
        "Password changed successfully for your {company} account on {date}.",
        "Your return for order #{order} has been processed. Refund in 5-7 days.",
        "Reservation confirmed: {hotel} from {checkin} to {checkout}.",
    ]

    legitimate_notifications = [
        "Your {service} monthly statement is now available in your account.",
        "Reminder: Your bill of ${amount} is due on {date}.",
        "Thank you for your feedback! We appreciate your time.",
        "{company} scheduled maintenance on {date}. Services may be unavailable.",
        "Your support ticket #{ticket} has been updated. View response.",
        "New features available in your {company} account. Learn more.",
        "Your download is ready. File will be available for {days} days.",
        "Meeting reminder: {meeting} at {time} tomorrow via {platform}.",
    ]

    legitimate_transactional = [
        "Payment received: ${amount} for invoice #{invoice}. Thank you!",
        "Shipping notification: Your order #{order} is out for delivery today.",
        "Refund processed: ${amount} will appear in your account in 3-5 business days.",
        "E-ticket for {event} on {date}. Present this email at entrance.",
        "Your {company} trial has started. Full access for {days} days.",
        "Document shared: {sender} shared '{filename}' with you.",
        "Form submission received: {form} submitted successfully.",
        "Auto-reply: Out of office until {date}. For urgent matters, contact {backup}.",
    ]

    # ==================== COMPANY/URL/CONTENT VARIABLES ====================

    companies = [
        "PayPal", "Amazon", "Netflix", "Apple", "Microsoft", "Google", "Facebook",
        "Bank of America", "Chase", "Wells Fargo", "Capital One", "Citibank",
        "eBay", "Walmart", "Target", "Best Buy", "Costco", "LinkedIn",
        "Instagram", "Twitter", "TikTok", "Spotify", "Dropbox", "Adobe"
    ]

    crypto_exchanges = ["Coinbase", "Binance", "Kraken", "Gemini", "Crypto.com"]
    cryptocurrencies = ["Bitcoin", "Ethereum", "Dogecoin", "Cardano", "Solana"]
    dating_platforms = ["Tinder", "Bumble", "Hinge", "Match", "OkCupid"]
    social_platforms = ["Facebook", "Instagram", "Twitter", "TikTok", "LinkedIn"]

    suspicious_urls = [
        "bit.ly/verify", "tinyurl.com/secure", "goo.gl/account",
        "paypal-security.tk", "amazon-update.ml", "netflix-billing.ga",
        "192.168.1.1/login", "10.0.0.1/verify", "172.16.0.1/secure",
        "amaz0n.com", "paypaI.com", "netfIix.com", "goog1e.com",  # look-alikes
        "secure-login-verify.com", "account-update-center.net",
        "customer-service-support.org", "billing-payment-update.com"
    ]

    legitimate_urls = [
        "paypal.com", "amazon.com", "netflix.com", "apple.com",
        "microsoft.com", "google.com", "facebook.com", "chase.com",
        "support.company.com", "help.service.com", "account.business.com"
    ]

    # ==================== GENERATE DATASET ====================

    # Combine all phishing templates
    all_phishing = (
        crypto_phishing + health_phishing + job_phishing + romance_phishing +
        social_media_phishing + twofa_phishing + bec_phishing + invoice_phishing +
        spear_phishing + multistage_phishing + qr_phishing + vishing_phishing +
        punycode_phishing + html_phishing + typo_phishing + regional_phishing +
        forwarded_templates[:4]  # Some forwards are phishing
    )

    # Combine all legitimate templates
    all_legitimate = (
        marketing_templates + legitimate_security + forwarded_templates[4:] +
        legitimate_confirmations + legitimate_notifications + legitimate_transactional
    )

    print(f"  📧 Phishing templates: {len(all_phishing)}")
    print(f"  ✅ Legitimate templates: {len(all_legitimate)}")

    emails = []
    labels = []

    # Generate phishing emails
    target_phishing = n_samples // 2
    for _ in range(target_phishing):
        template = np.random.choice(all_phishing)

        # Fill in variables
        email = template.format(
            company=np.random.choice(companies),
            crypto=np.random.choice(cryptocurrencies),
            exchange=np.random.choice(crypto_exchanges),
            platform=np.random.choice(dating_platforms + social_platforms),
            url=np.random.choice(suspicious_urls),
            amount=np.random.choice([500, 1000, 2500, 5000, 10000]),
            code=f"{np.random.randint(100000, 999999)}",
            phone=f"({np.random.randint(200,999)}) {np.random.randint(200,999)}-{np.random.randint(1000,9999)}",
            email="user@email.com",
            action=np.random.choice(["suspended", "locked", "compromised", "flagged", "frozen"]),
            location=np.random.choice(["Russia", "China", "Nigeria", "Unknown Location"]),
            date=np.random.choice(["tomorrow", "within 24 hours", "by Friday", "immediately"]),
            time=np.random.choice(["10 minutes", "1 hour", "24 hours", "48 hours"]),
            number=np.random.choice([999999, 1000000, 5000]),
            points=np.random.choice([1000, 5000, 10000, 50000]),
            name=np.random.choice(["John", "Sarah", "Mike", "Emma", "David"]),
            firstname="User",
            lastname="Name",
            coworker="Colleague",
            manager="Manager",
            position=np.random.choice(["Software Engineer", "Manager", "Analyst"]),
            order=np.random.randint(100000, 999999),
            invoice=np.random.randint(10000, 99999),
            ticket=np.random.randint(1000, 9999),
            violation=np.random.choice(["spam", "harassment", "copyright infringement"]),
            prize=np.random.choice(["iPhone", "$1000", "Gift Card"]),
            executive=np.random.choice(["CEO", "CFO", "VP"]),
            vendor="Vendor Inc",
            service=np.random.choice(["cloud storage", "software", "hosting"]),
            percent=np.random.choice([25, 50, 70]),
            product="Product",
            event="Conference",
            celebrity=np.random.choice(["Elon Musk", "Bill Gates", "Warren Buffett"]),
            alma_mater="University",
            client="Client",
            client_company="Client Corp",
            bank=np.random.choice(["Chase", "Wells Fargo", "Bank of America"]),
            topic=np.random.choice(["security", "features", "updates"]),
            sender="sender@company.com",
            malicious_url=np.random.choice(suspicious_urls),
            legitimate_company=np.random.choice(companies)
        )

        # Randomly add urgency markers (50% chance)
        if np.random.random() > 0.5 and "URGENT" not in email:
            email = np.random.choice(["URGENT! ", "ACTION REQUIRED: ", "ALERT: "]) + email

        emails.append(email)
        labels.append(1)  # Phishing

    # Generate legitimate emails
    target_legitimate = n_samples - target_phishing
    for _ in range(target_legitimate):
        template = np.random.choice(all_legitimate)

        email = template.format(
            company=np.random.choice(companies),
            service=np.random.choice(["subscription", "software", "cloud storage"]),
            url=np.random.choice(legitimate_urls),
            amount=np.random.choice([9.99, 19.99, 49.99, 99.99]),
            order=np.random.randint(100000, 999999),
            date=np.random.choice(["March 15", "next Monday", "in 3 days"]),
            time=np.random.choice(["2:00 PM", "10:30 AM", "4:00 PM"]),
            ticket=np.random.randint(1000, 9999),
            points=np.random.randint(100, 10000),
            location=np.random.choice(["New York", "California", "London"]),
            days=np.random.choice([7, 14, 30]),
            meeting="Team Sync",
            platform="Zoom",
            provider="Dr. Smith",
            hotel="Hotel Name",
            checkin="May 1",
            checkout="May 5",
            invoice=np.random.randint(10000, 99999),
            sender="colleague@company.com",
            filename="document.pdf",
            form="Contact Form",
            backup="manager@company.com",
            event="Concert",
            percent=np.random.choice([20, 30, 40]),
            product="Product Name"
        )

        emails.append(email)
        labels.append(0)  # Legitimate

    print(f"  ✅ Generated {len(emails)} total samples")
    print(f"     - Phishing: {sum(labels)}")
    print(f"     - Legitimate: {len(labels) - sum(labels)}")
    print(f"\n  📊 Template Coverage:")
    print(f"     - Modern tactics: ✅ Crypto, COVID, jobs, romance, social media, 2FA, BEC, invoices")
    print(f"     - Sophisticated: ✅ Spear phishing, multi-stage, QR codes, vishing, punycode, HTML")
    print(f"     - Language diversity: ✅ Typos, regional (UK), grammar errors")
    print(f"     - Real-world noise: ✅ Marketing, legit alerts, forwarded emails")

    return emails, labels


def prepare_enhanced_features(texts, feature_extractor, vectorizer=None, fit=False):
    """Prepare enhanced features combining TF-IDF and advanced features."""
    print("🔧 Extracting enhanced features...")

    advanced_features = []
    for i, text in enumerate(texts):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(texts)} samples...")
        features = feature_extractor.extract_all_features(text)
        advanced_features.append(list(features.values()))

    advanced_features = np.array(advanced_features)
    print(f"  ✅ Extracted {advanced_features.shape[1]} advanced features")

    if fit:
        print("  📝 Fitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased from 1000
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=2
        )
        tfidf_features = vectorizer.fit_transform(texts).toarray()
    else:
        print("  📝 Transforming with TF-IDF...")
        tfidf_features = vectorizer.transform(texts).toarray()

    print(f"  ✅ Generated {tfidf_features.shape[1]} TF-IDF features")

    combined_features = np.hstack([tfidf_features, advanced_features])
    print(f"  ✅ Total features: {combined_features.shape[1]}")

    return combined_features, vectorizer


def train_comprehensive_ensemble():
    """Train ensemble with comprehensive phishing coverage."""
    print("=" * 80)
    print("🚀 COMPREHENSIVE ENSEMBLE TRAINING")
    print("=" * 80)
    print("\n📚 Training on 150+ templates covering:")
    print("   ✅ Modern phishing tactics (8 categories)")
    print("   ✅ Sophisticated techniques (6 categories)")
    print("   ✅ Language diversity (typos, regional)")
    print("   ✅ Real-world noise (marketing, forwards)")
    print()

    feature_extractor = AdvancedFeatureExtractor()

    # Generate comprehensive data
    emails, labels = generate_comprehensive_phishing_data(n_samples=5000)

    # Save training data
    data_path = Path("data/training_data_comprehensive.pkl")
    data_path.parent.mkdir(exist_ok=True)
    joblib.dump({'emails': emails, 'labels': labels}, data_path)
    print(f"\n  💾 Saved comprehensive training data to {data_path}")

    # Split data
    print("\n📂 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")

    # Extract features
    print("\n🔬 Feature Extraction...")
    X_train_features, vectorizer = prepare_enhanced_features(
        X_train, feature_extractor, fit=True
    )
    X_test_features, _ = prepare_enhanced_features(
        X_test, feature_extractor, vectorizer=vectorizer, fit=False
    )

    # Scale features
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)

    # Train ensemble
    print("\n" + "=" * 80)
    print("🎯 TRAINING ENSEMBLE MODEL")
    print("=" * 80)

    ensemble_detector = EnsemblePhishingDetector(models_dir="models/ensemble")
    ensemble_detector.train_individual_models(X_train_scaled, y_train, X_test_scaled, y_test)
    ensemble_detector.create_ensemble(voting='soft')
    ensemble_detector.train_ensemble(X_train_scaled, y_train)

    # Evaluate
    print("\n" + "=" * 80)
    print("📊 FINAL EVALUATION")
    print("=" * 80)

    metrics = ensemble_detector.evaluate_ensemble(X_test_scaled, y_test)

    # Save everything
    print("\n💾 Saving comprehensive model...")
    ensemble_detector.save_ensemble("ensemble_model.pkl")
    joblib.dump(vectorizer, "models/ensemble/vectorizer.pkl")
    joblib.dump(scaler, "models/ensemble/scaler.pkl")
    joblib.dump(feature_extractor, "models/ensemble/feature_extractor.pkl")

    metadata = {
        'training_date': datetime.now().isoformat(),
        'version': '2.0_comprehensive',
        'n_training_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_features': X_train_scaled.shape[1],
        'template_categories': {
            'modern_tactics': 8,
            'sophisticated_techniques': 6,
            'language_diversity': 2,
            'real_world_noise': 3
        },
        'total_templates': '150+',
        'test_metrics': {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.number))}
    }

    with open("models/ensemble/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE ENSEMBLE TRAINING COMPLETE!")
    print("=" * 80)
    print(f"  ✅ Accuracy: {metrics['accuracy']:.2%}")
    print(f"  ✅ Precision: {metrics['precision']:.2%}")
    print(f"  ✅ Recall: {metrics['recall']:.2%}")
    print(f"  ✅ F1 Score: {metrics['f1']:.2%}")
    print(f"  ✅ Templates: 150+ (vs 27 previously)")
    print(f"  ✅ Samples: 5000 (vs 2000 previously)")
    print("=" * 80)

    return ensemble_detector, metrics


if __name__ == "__main__":
    train_comprehensive_ensemble()
