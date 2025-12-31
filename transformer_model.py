#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-Based Phishing Detector with Attention Visualization
Adds modern NLP capabilities to the PhishGuard ensemble

This module implements:
1. DistilBERT-based email classification
2. Attention weight extraction for interpretability
3. Integration with existing ensemble system
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Check for transformers library
HAS_TORCH = False
HAS_TRANSFORMERS = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    pass

try:
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        DistilBertModel,
        get_linear_schedule_with_warmup
    )
    # AdamW moved to torch.optim in newer versions
    from torch.optim import AdamW
    HAS_TRANSFORMERS = True
except ImportError:
    pass

if not HAS_TORCH:
    print("PyTorch not installed. Install with: pip install torch")
if not HAS_TRANSFORMERS:
    print("Transformers not installed. Install with: pip install transformers")


@dataclass
class TransformerConfig:
    """Configuration for transformer model."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    device: str = "cpu"  # or "cuda" if available


class EmailDataset(Dataset):
    """PyTorch Dataset for email text classification."""

    def __init__(self, texts: List[str], labels: Optional[List[int]],
                 tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class TransformerPhishingDetector:
    """
    DistilBERT-based phishing detector with attention visualization.

    This model provides:
    - State-of-the-art NLP classification
    - Attention weight extraction for interpretability
    - Fine-tuning on phishing-specific data
    """

    def __init__(self, config: Optional[TransformerConfig] = None,
                 models_dir: str = "models/transformer"):
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            raise ImportError(
                "PyTorch and Transformers required. Install with:\n"
                "pip install torch transformers"
            )

        self.config = config or TransformerConfig()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.config.model_name)
        self.model = None
        self.is_trained = False

        # For attention extraction
        self.attention_model = None

    def _init_model(self, num_labels: int = 2):
        """Initialize or reset the model."""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels,
            output_attentions=True,
            output_hidden_states=True
        )
        self.model.to(self.device)

        # Also load base model for attention extraction
        self.attention_model = DistilBertModel.from_pretrained(
            self.config.model_name,
            output_attentions=True
        )
        self.attention_model.to(self.device)

    def train(self, texts: List[str], labels: List[int],
              val_texts: Optional[List[str]] = None,
              val_labels: Optional[List[int]] = None) -> Dict:
        """
        Fine-tune the transformer on phishing detection task.

        Args:
            texts: List of email texts
            labels: List of labels (0=legitimate, 1=phishing)
            val_texts: Optional validation texts
            val_labels: Optional validation labels

        Returns:
            Dict with training metrics
        """
        print(f"🚀 Training Transformer on {len(texts)} samples...")
        print(f"   Device: {self.device}")

        self._init_model()

        # Create datasets
        train_dataset = EmailDataset(
            texts, labels, self.tokenizer, self.config.max_length
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        if val_texts and val_labels:
            val_dataset = EmailDataset(
                val_texts, val_labels, self.tokenizer, self.config.max_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        else:
            val_loader = None

        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        training_history = {
            'train_loss': [],
            'val_accuracy': [],
            'epoch_metrics': []
        }

        self.model.train()

        for epoch in range(self.config.num_epochs):
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )

                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / num_batches
            training_history['train_loss'].append(avg_loss)

            # Validation
            if val_loader:
                val_acc = self._evaluate(val_loader)
                training_history['val_accuracy'].append(val_acc)
                print(f"   Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
            else:
                print(f"   Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"Loss={avg_loss:.4f}")

            training_history['epoch_metrics'].append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_accuracy': training_history['val_accuracy'][-1] if val_loader else None
            })

        self.is_trained = True
        print("✅ Training complete!")

        return training_history

    def _evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)

        self.model.train()
        return correct / total

    def predict(self, text: str) -> Dict:
        """
        Predict phishing probability with confidence score.

        Args:
            text: Email text to classify

        Returns:
            Dict with prediction, confidence, and probabilities
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")

        self.model.eval()

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

        return {
            'prediction': prediction,
            'is_phishing': bool(prediction == 1),
            'confidence': confidence,
            'probabilities': {
                'legitimate': probabilities[0][0].item(),
                'phishing': probabilities[0][1].item()
            }
        }

    def predict_with_attention(self, text: str) -> Dict:
        """
        Predict with attention weights for interpretability.

        This method extracts attention weights to show which parts
        of the email the model focused on when making its prediction.

        Args:
            text: Email text to classify

        Returns:
            Dict with prediction, confidence, and attention analysis
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()

        # Tokenize with special tokens
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )

        # Get prediction
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

        # Extract attention weights
        # outputs.attentions is tuple of (batch, num_heads, seq_len, seq_len)
        # Average across all layers and heads
        all_attentions = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
        avg_attention = all_attentions.mean(dim=(0, 2))  # Average over layers and heads

        # Get attention from CLS token (first token) to all other tokens
        cls_attention = avg_attention[0, 0, :].cpu().numpy()

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Find non-padding tokens
        actual_length = attention_mask[0].sum().item()
        tokens = tokens[:actual_length]
        cls_attention = cls_attention[:actual_length]

        # Normalize attention weights
        cls_attention = cls_attention / cls_attention.sum()

        # Get top attended tokens (excluding special tokens)
        token_attention_pairs = []
        for i, (token, attn) in enumerate(zip(tokens, cls_attention)):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_attention_pairs.append({
                    'token': token,
                    'attention': float(attn),
                    'position': i
                })

        # Sort by attention weight
        token_attention_pairs.sort(key=lambda x: x['attention'], reverse=True)

        # Identify high-attention phrases (potential phishing indicators)
        high_attention_tokens = [
            t for t in token_attention_pairs
            if t['attention'] > np.mean(cls_attention) * 1.5
        ]

        return {
            'prediction': prediction,
            'is_phishing': bool(prediction == 1),
            'confidence': confidence,
            'probabilities': {
                'legitimate': probabilities[0][0].item(),
                'phishing': probabilities[0][1].item()
            },
            'attention_analysis': {
                'top_tokens': token_attention_pairs[:10],
                'high_attention_tokens': high_attention_tokens[:5],
                'attention_summary': self._summarize_attention(token_attention_pairs)
            }
        }

    def _summarize_attention(self, token_attention_pairs: List[Dict]) -> str:
        """Generate human-readable attention summary."""
        if not token_attention_pairs:
            return "No significant attention patterns detected."

        top_tokens = [t['token'] for t in token_attention_pairs[:5]]

        # Check for phishing-related patterns
        phishing_indicators = ['click', 'urgent', 'verify', 'suspend', 'account',
                               'password', 'link', 'expire', 'immediately', 'confirm']

        found_indicators = [t for t in top_tokens if any(
            ind in t.lower() for ind in phishing_indicators
        )]

        if found_indicators:
            return f"High attention on potential phishing indicators: {', '.join(found_indicators)}"
        else:
            return f"Top attention tokens: {', '.join(top_tokens)}"

    def save(self, path: Optional[str] = None):
        """Save the trained model."""
        if path is None:
            path = self.models_dir / "transformer_model"
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_name': self.config.model_name,
                'max_length': self.config.max_length,
                'is_trained': self.is_trained
            }, f)

        print(f"✅ Model saved to {path}")

    def load(self, path: Optional[str] = None):
        """Load a saved model."""
        if path is None:
            path = self.models_dir / "transformer_model"
        else:
            path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.is_trained = config.get('is_trained', True)

        # Load model and tokenizer
        self.model = DistilBertForSequenceClassification.from_pretrained(
            str(path),
            output_attentions=True,
            output_hidden_states=True
        )
        self.model.to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(str(path))

        print(f"✅ Model loaded from {path}")


def create_attention_visualization_html(result: Dict, text: str) -> str:
    """
    Create HTML visualization of attention weights.

    Args:
        result: Output from predict_with_attention()
        text: Original email text

    Returns:
        HTML string for visualization
    """
    attention_data = result.get('attention_analysis', {})
    top_tokens = attention_data.get('top_tokens', [])

    # Create token highlight map
    token_weights = {t['token'].replace('##', ''): t['attention'] for t in top_tokens}

    # Normalize weights for color intensity
    if token_weights:
        max_weight = max(token_weights.values())
        min_weight = min(token_weights.values())
        range_weight = max_weight - min_weight if max_weight != min_weight else 1

    # Build HTML
    html_parts = ['<div style="font-family: monospace; line-height: 1.8;">']

    words = text.split()
    for word in words:
        clean_word = word.lower().strip('.,!?";:')
        weight = token_weights.get(clean_word, 0)

        if weight > 0:
            # Normalize to 0-1 range
            normalized = (weight - min_weight) / range_weight
            # Color from yellow (low) to red (high attention)
            r = 255
            g = int(255 * (1 - normalized))
            b = 0
            opacity = 0.3 + (normalized * 0.5)

            html_parts.append(
                f'<span style="background-color: rgba({r},{g},{b},{opacity}); '
                f'padding: 2px 4px; border-radius: 3px;" '
                f'title="Attention: {weight:.4f}">{word}</span> '
            )
        else:
            html_parts.append(f'{word} ')

    html_parts.append('</div>')

    # Add legend
    html_parts.append('''
    <div style="margin-top: 20px; padding: 10px; background: #f5f5f5; border-radius: 5px;">
        <strong>Legend:</strong>
        <span style="background: rgba(255,255,0,0.5); padding: 2px 8px;">Low Attention</span>
        <span style="background: rgba(255,128,0,0.6); padding: 2px 8px;">Medium Attention</span>
        <span style="background: rgba(255,0,0,0.7); padding: 2px 8px;">High Attention</span>
    </div>
    ''')

    # Add prediction summary
    pred_class = "Phishing" if result['is_phishing'] else "Legitimate"
    pred_color = "#dc3545" if result['is_phishing'] else "#28a745"

    html_parts.insert(0, f'''
    <div style="margin-bottom: 20px; padding: 15px; background: {pred_color}22;
                border-left: 4px solid {pred_color}; border-radius: 4px;">
        <strong>Prediction:</strong> <span style="color: {pred_color};">{pred_class}</span>
        <br>
        <strong>Confidence:</strong> {result['confidence']:.1%}
        <br>
        <strong>Analysis:</strong> {attention_data.get('attention_summary', 'N/A')}
    </div>
    ''')

    return '\n'.join(html_parts)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("PhishGuard Transformer Model - Test Run")
    print("=" * 60)

    if not HAS_TORCH or not HAS_TRANSFORMERS:
        print("\n❌ Required libraries not installed.")
        print("Install with: pip install torch transformers")
        exit(1)

    # Create detector
    config = TransformerConfig(
        num_epochs=1,  # Quick test
        batch_size=4
    )
    detector = TransformerPhishingDetector(config=config)

    # Sample training data
    train_texts = [
        "URGENT: Your account has been suspended! Click here to verify: http://suspicious.com",
        "Your package couldn't be delivered. Click to reschedule: http://fake-ups.com",
        "Congratulations! You've won $1,000,000! Claim now: http://scam.com",
        "Security alert: Unusual login detected. Verify your identity: http://phish.com",
        "Thank you for your order. Your receipt is attached.",
        "Meeting reminder: Team sync at 3pm today in Conference Room B",
        "Your monthly statement is ready. Log in to view.",
        "Thanks for subscribing to our newsletter!",
    ]
    train_labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1=phishing, 0=legitimate

    print("\n📚 Training on sample data...")
    history = detector.train(train_texts, train_labels)

    # Test prediction with attention
    test_email = "URGENT! Your PayPal account has been limited. Click here to verify your identity immediately: http://paypa1-verify.com"

    print(f"\n📧 Test Email:\n{test_email}\n")

    result = detector.predict_with_attention(test_email)

    print("📊 Prediction Results:")
    print(f"   Classification: {'PHISHING' if result['is_phishing'] else 'LEGITIMATE'}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Phishing Probability: {result['probabilities']['phishing']:.1%}")

    print("\n🔍 Attention Analysis:")
    print(f"   Summary: {result['attention_analysis']['attention_summary']}")
    print("   Top Attended Tokens:")
    for token_info in result['attention_analysis']['top_tokens'][:5]:
        print(f"      - '{token_info['token']}': {token_info['attention']:.4f}")

    # Generate visualization HTML
    html = create_attention_visualization_html(result, test_email)

    # Save visualization
    viz_path = Path("models/transformer/attention_visualization.html")
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    with open(viz_path, 'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>PhishGuard Attention Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
        h1 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>🛡️ PhishGuard Attention Visualization</h1>
    <h2>Email Analysis</h2>
    {html}
</body>
</html>
        """)

    print(f"\n✅ Visualization saved to: {viz_path}")
    print("\n" + "=" * 60)
