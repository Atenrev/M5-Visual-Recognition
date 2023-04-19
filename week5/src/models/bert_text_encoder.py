import torch

from typing import List
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding
    

class BertTextEncoder(torch.nn.Module):
    """
    Wrapper around BERT. Adds a projection layer to the output of the BERT model.
    """
    def __init__(self, embed_size: int = 256, freeze_backbone: bool = True):
        super(BertTextEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_model = AutoModel.from_pretrained("bert-base-uncased")

        if freeze_backbone:
            for param in self.text_model.parameters():
                param.requires_grad = False

        self.projector = torch.nn.Linear(self.text_model.config.hidden_size, embed_size)

    def tokenizer_encode_text(self, text: List[str]) -> BatchEncoding:
        """
        Use the tokenizer to encode the text.

        Args:
            text: str
        Returns:
            torch.Tensor of shape (batch_size, max_seq_len)
        """
        tokenized = self.tokenizer(text, return_tensors="pt", padding=True)
        return tokenized

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text: torch.Tensor of shape (batch_size, max_seq_len)
        Returns:
            torch.Tensor of shape (batch_size, embed_size)
        """
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_features.last_hidden_state[:, 0, :]
        text_features = self.projector(text_features)
        return text_features