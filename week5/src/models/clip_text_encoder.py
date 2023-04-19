import torch

from typing import List
from transformers import CLIPTextModel, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
    

class CLIPTextEncoder(torch.nn.Module):
    """
    Wrapper around CLIPTextModel. Adds a projection layer to the output of the CLIPTextModel.
    """
    def __init__(self, embed_size: int = 256, freeze_backbone: bool = True):
        super(CLIPTextEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

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
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[1]
        return self.projector(text_features)