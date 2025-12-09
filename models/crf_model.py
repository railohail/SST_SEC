"""
BERT + CRF Model for Sequence Labeling

Architecture: BERT (768d) → Linear(768→512) → LayerNorm → GELU → Dropout(0.3)
                          → Linear(512→256) → LayerNorm → GELU → Dropout(0.2)
                          → Linear(256→3) → CRF
"""
import torch
from torch import nn
from transformers import AutoModel

try:
    from torchcrf import CRF
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-crf"])
    from torchcrf import CRF


class TokenClassificationCRFEnhanced(nn.Module):
    """
    Enhanced CRF model with deep hidden layers for sequence labeling.

    Labels:
        - 0: O (Outside - normal text)
        - 1: B-Modify (position to modify)
        - 2: B-Filling (replacement content)
    """

    def __init__(
        self,
        pretrained_model_name: str = "google-bert/bert-base-multilingual-cased",
        num_labels: int = 3,
        hidden_size_1: int = 512,
        hidden_size_2: int = 256,
        dropout_rate_1: float = 0.3,
        dropout_rate_2: float = 0.2
    ):
        """
        Initialize the enhanced CRF model.

        Args:
            pretrained_model_name: BERT model name
            num_labels: Number of output labels (3: O, B-Modify, B-Filling)
            hidden_size_1: First hidden layer size
            hidden_size_2: Second hidden layer size
            dropout_rate_1: Dropout rate for first layer
            dropout_rate_2: Dropout rate for second layer
        """
        super(TokenClassificationCRFEnhanced, self).__init__()

        # Load BERT backbone
        self.bert = AutoModel.from_pretrained(
            pretrained_model_name,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )

        bert_hidden_size = self.bert.config.hidden_size  # 768 for base BERT
        self.num_labels = num_labels

        # First hidden layer: 768 → 512
        self.hidden_layer_1 = nn.Linear(bert_hidden_size, hidden_size_1)
        self.layer_norm_1 = nn.LayerNorm(hidden_size_1)
        self.gelu_1 = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_rate_1)

        # Second hidden layer: 512 → 256
        self.hidden_layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer_norm_2 = nn.LayerNorm(hidden_size_2)
        self.gelu_2 = nn.GELU()
        self.dropout_2 = nn.Dropout(dropout_rate_2)

        # Output layer: 256 → 3
        self.classifier = nn.Linear(hidden_size_2, num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Label IDs [batch_size, seq_len] (optional, for training)

        Returns:
            dict with 'loss' (if labels provided) and 'logits'
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # First hidden layer with LayerNorm + GELU
        hidden_1 = self.hidden_layer_1(sequence_output)
        hidden_1 = self.layer_norm_1(hidden_1)
        hidden_1 = self.gelu_1(hidden_1)
        hidden_1 = self.dropout_1(hidden_1)

        # Second hidden layer with LayerNorm + GELU
        hidden_2 = self.hidden_layer_2(hidden_1)
        hidden_2 = self.layer_norm_2(hidden_2)
        hidden_2 = self.gelu_2(hidden_2)
        hidden_2 = self.dropout_2(hidden_2)

        # Get logits
        logits = self.classifier(hidden_2)

        if labels is not None:
            # Training mode: compute CRF loss
            crf_mask = attention_mask.bool()

            # Prepare labels for CRF
            labels_for_crf = labels.clone()
            labels_for_crf[~crf_mask] = 0
            labels_for_crf[labels == -100] = 0

            # CRF negative log-likelihood
            loss = -self.crf(logits, labels_for_crf, mask=crf_mask, reduction='mean')

            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

    def decode(self, input_ids, attention_mask=None):
        """
        Viterbi decoding for best label sequence.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            List of predicted label sequences
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state

        # Pass through hidden layers
        hidden_1 = self.hidden_layer_1(sequence_output)
        hidden_1 = self.layer_norm_1(hidden_1)
        hidden_1 = self.gelu_1(hidden_1)
        hidden_1 = self.dropout_1(hidden_1)

        hidden_2 = self.hidden_layer_2(hidden_1)
        hidden_2 = self.layer_norm_2(hidden_2)
        hidden_2 = self.gelu_2(hidden_2)
        hidden_2 = self.dropout_2(hidden_2)

        logits = self.classifier(hidden_2)

        # CRF decode
        predictions = self.crf.decode(logits, mask=attention_mask.bool())
        return predictions
