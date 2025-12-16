"""Sequence Labeling Service using BERT+CRF model"""
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer

from config import config
from models.crf_model import TokenClassificationCRFEnhanced


class SequenceLabeler:
    """
    Sequence labeling service for detecting correction commands.

    Labels:
        - O: Normal text
        - B-Modify: Position in text that needs modification
        - B-Filling: The replacement/insertion content
    """

    def __init__(
        self,
        model_path: Path = None,
        bert_model_name: str = None,
        device: str = None
    ):
        """
        Initialize the sequence labeler.

        Args:
            model_path: Path to model weights
            bert_model_name: BERT model name for tokenizer
            device: Device to use (cpu, cuda, mps)
        """
        self.model_path = model_path or config.CRF_MODEL_PATH
        self.bert_model_name = bert_model_name or config.BERT_MODEL_NAME

        # Determine device
        # NOTE: MPS has bugs with model loading, use CPU for now
        if device:
            self.device = torch.device(device)
        else:
            # MPS disabled due to PyTorch bug with unaligned blit
            self.device = torch.device("cpu")

        self.model: Optional[TokenClassificationCRFEnhanced] = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        print(f"Loading sequence labeling model from {self.model_path}...")
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bert_model_name,
            use_fast=True
        )

        # Load model
        self.model = TokenClassificationCRFEnhanced(
            pretrained_model_name=self.bert_model_name,
            num_labels=config.NUM_LABELS
        )

        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        print("Model loaded!")

    def predict(self, text: str) -> List[str]:
        """
        Predict labels for each character in the text.

        Args:
            text: Input text (may contain [SEP] for correction commands)

        Returns:
            List of labels for each character (O, B-Modify, B-Filling)
        """
        if not self._loaded:
            self.load()

        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        # Get predictions
        with torch.no_grad():
            predictions_list = self.model.decode(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Map token predictions back to characters
        predictions = predictions_list[0]
        word_ids = tokenized.word_ids()

        # Convert to labels per character
        char_labels = []
        previous_word_idx = None

        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue  # Skip special tokens
            if word_idx != previous_word_idx:
                # New character
                label_id = predictions[i] if i < len(predictions) else 0
                label = config.ID_TO_LABEL.get(label_id, 'O')
                char_labels.append(label)
            previous_word_idx = word_idx

        return char_labels

    def predict_with_positions(self, text: str) -> Tuple[List[str], List[int], List[int]]:
        """
        Predict labels and return positions of B-Modify and B-Filling.

        Args:
            text: Input text

        Returns:
            Tuple of (labels, modify_positions, filling_positions)
        """
        labels = self.predict(text)

        modify_positions = [i for i, label in enumerate(labels) if label == 'B-Modify']
        filling_positions = [i for i, label in enumerate(labels) if label == 'B-Filling']

        return labels, modify_positions, filling_positions
