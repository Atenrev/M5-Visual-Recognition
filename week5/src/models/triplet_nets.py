import torch

from torch import nn
from typing import Tuple, List
from transformers.tokenization_utils_base import BatchEncoding
    

class TripletModel(torch.nn.Module):
    def __init__(self, image_encoder: torch.nn.Module, text_encoder: torch.nn.Module):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def tokenize(self, text: List[str]) -> BatchEncoding:
        return self.text_encoder.tokenizer_encode_text(text)


class ImageToTextTripletModel(TripletModel):
    def __init__(self, image_encoder: torch.nn.Module, text_encoder: torch.nn.Module):
        super().__init__(image_encoder, text_encoder)

    def forward(
            self, 
            anchor_image: torch.Tensor, 
            positive_input_ids: torch.Tensor,
            positive_attention_mask: torch.Tensor,
            negative_input_ids: torch.Tensor,
            negative_attention_mask: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_image_embedding = self.image_encoder(anchor_image)
        positive_text_embedding = self.text_encoder(positive_input_ids, positive_attention_mask)
        negative_text_embedding = self.text_encoder(negative_input_ids, negative_attention_mask)
        return anchor_image_embedding, positive_text_embedding, negative_text_embedding    
    

class TextToImageTripletModel(TripletModel):
    def __init__(self, image_encoder: torch.nn.Module, text_encoder: torch.nn.Module):
        super().__init__(image_encoder, text_encoder)

    def forward(
            self, 
            anchor_input_ids: torch.Tensor,
            anchor_attention_mask: torch.Tensor,
            positive_image: torch.Tensor,
            negative_image: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_text_embedding = self.text_encoder(anchor_input_ids, anchor_attention_mask)
        positive_image_embedding = self.image_encoder(positive_image)
        negative_image_embedding = self.image_encoder(negative_image)
        return anchor_text_embedding, positive_image_embedding, negative_image_embedding


class ImageToTextWithTempModel(torch.nn.Module):
    def __init__(self, image_encoder: torch.nn.Module, text_encoder: torch.nn.Module):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.parameter.Parameter(torch.tensor(1.0))

    def tokenize(self, text: List[str]) -> BatchEncoding:
        return self.text_encoder.tokenizer_encode_text(text)

    def forward(
            self, 
            anchor_image: torch.Tensor, 
            text_input_ids: torch.Tensor,
            text_attention_mask: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_embedding = self.image_encoder(anchor_image)
        text_embedding = self.text_encoder(text_input_ids, text_attention_mask)
        logits = torch.matmul(image_embedding, text_embedding.T) * torch.exp(self.temperature)
        return logits, image_embedding, text_embedding