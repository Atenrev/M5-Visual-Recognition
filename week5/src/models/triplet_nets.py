import torch

from torch import nn
from torch.nn import functional as F
from typing import List, Optional
from transformers.tokenization_utils_base import BatchEncoding
from dataclasses import dataclass

from src.losses import SymmetricCrossEntropyLoss
    

@dataclass
class TripletOutput:
    loss: torch.Tensor
    anchor_embedding: torch.Tensor
    positive_embedding: torch.Tensor
    negative_embedding: Optional[torch.Tensor] = None


class TripletModel(torch.nn.Module):
    def __init__(self, image_encoder: torch.nn.Module, text_encoder: torch.nn.Module, args):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.loss_fn = torch.nn.TripletMarginLoss(
            margin=args.triplet_margin,
            p=args.triplet_norm
        )

    def calculate_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        return self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

    def tokenize(self, text: List[str]) -> BatchEncoding:
        return self.text_encoder.tokenizer_encode_text(text)
    
    def forward(self, *args, **kwargs) -> TripletOutput:
        raise NotImplementedError()


class ImageToTextTripletModel(TripletModel):
    def __init__(self, image_encoder: torch.nn.Module, text_encoder: torch.nn.Module, args):
        super().__init__(image_encoder, text_encoder, args)

    def forward(
            self, 
            anchor_image: torch.Tensor, 
            positive_input_ids: torch.Tensor,
            positive_attention_mask: torch.Tensor,
            negative_input_ids: torch.Tensor,
            negative_attention_mask: torch.Tensor,
            ) -> TripletOutput:
        anchor_image_embedding = self.image_encoder(anchor_image)
        positive_text_embedding = self.text_encoder(positive_input_ids, positive_attention_mask)
        negative_text_embedding = self.text_encoder(negative_input_ids, negative_attention_mask)
        loss = self.calculate_loss(anchor_image_embedding, positive_text_embedding, negative_text_embedding)
        return TripletOutput(
            loss=loss,
            anchor_embedding=anchor_image_embedding,
            positive_embedding=positive_text_embedding,
            negative_embedding=negative_text_embedding,
        )
    

class TextToImageTripletModel(TripletModel):
    def __init__(self, image_encoder: torch.nn.Module, text_encoder: torch.nn.Module, args):
        super().__init__(image_encoder, text_encoder, args)

    def forward(
            self, 
            anchor_input_ids: torch.Tensor,
            anchor_attention_mask: torch.Tensor,
            positive_image: torch.Tensor,
            negative_image: torch.Tensor,
            ) -> TripletOutput:
        anchor_text_embedding = self.text_encoder(anchor_input_ids, anchor_attention_mask)
        positive_image_embedding = self.image_encoder(positive_image)
        negative_image_embedding = self.image_encoder(negative_image)
        loss = self.calculate_loss(anchor_text_embedding, positive_image_embedding, negative_image_embedding)
        return TripletOutput(
            loss=loss,
            anchor_embedding=anchor_text_embedding,
            positive_embedding=positive_image_embedding,
            negative_embedding=negative_image_embedding,
        )


class SymmetricSiameseModel(torch.nn.Module):
    def __init__(self, image_encoder: torch.nn.Module, text_encoder: torch.nn.Module, args):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.parameter.Parameter(torch.tensor(1.0))
        self.loss_fn = SymmetricCrossEntropyLoss()

    def tokenize(self, text: List[str]) -> BatchEncoding:
        return self.text_encoder.tokenizer_encode_text(text)

    def forward(
            self, 
            anchor_image: torch.Tensor, 
            text_input_ids: torch.Tensor,
            text_attention_mask: torch.Tensor,
            ) -> TripletOutput:
        # Compute the embeddings
        image_embedding = self.image_encoder(anchor_image)
        text_embedding = self.text_encoder(text_input_ids, text_attention_mask)
        # L2-normalize the embeddings
        image_embedding_norm = F.normalize(image_embedding, dim=1, p=2)
        text_embedding_norm = F.normalize(text_embedding, dim=1, p=2)
        # Compute the logits
        logits = torch.mm(image_embedding_norm, text_embedding_norm.T) * torch.exp(self.temperature)
        # Compute the symmetric cross entropy loss
        loss = self.loss_fn(logits)
        return TripletOutput(
            loss=loss,
            anchor_embedding=image_embedding,
            positive_embedding=text_embedding,
        )