import os
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

# Default text prompts for medical image classification
DEFAULT_TEXT_PROMPTS = [
    "a chest x-ray image showing COVID-19 pneumonia with ground-glass opacities",
    "a healthy normal chest x-ray image with clear lung fields", 
    "a chest x-ray image showing non-COVID pneumonia infiltrates"
]

class CLIPClassifier(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", num_classes=3, 
                 text_prompts=None, device=None):
        super(CLIPClassifier, self).__init__()
        
        # Load CLIP model and processor
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.num_classes = num_classes
        
        # If text prompts provided, encode them for zero-shot classification
        if text_prompts is not None:
            with torch.no_grad():
                text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                text_outputs = self.clip_model.get_text_features(**text_inputs)
                # Normalize text embeddings
                self.text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        else:
            self.text_features = None
    
    def forward(self, pixel_values):
        # Get image features
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        
        # Normalize image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if self.text_features is not None:
            # Zero-shot classification using text embeddings
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ self.text_features.T
        else:
            # If no text features, you'd need to add a classification head
            # This is a placeholder - implement if needed
            raise NotImplementedError("Classification head not implemented for non-zero-shot mode")
        
        return logits
    
    def preprocess_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)
    
    def predict(self, image):
        self.eval()
        with torch.no_grad():
            pixel_values = self.preprocess_image(image)
            logits = self.forward(pixel_values)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
            
        return pred_class.item(), confidence.item()
    
    def freeze_text_encoder(self):
        """Freeze text encoder parameters."""
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
    
    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters."""
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False
    
    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder parameters for fine-tuning."""
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self, vision_only=True):
        if vision_only:
            return filter(lambda p: p.requires_grad, 
                         self.clip_model.vision_model.parameters())
        else:
            return filter(lambda p: p.requires_grad, self.parameters())


def create_clip_model(num_classes=3, text_prompts=None, device="cuda", 
                      model_name="openai/clip-vit-base-patch32"):
    if text_prompts is None:
        text_prompts = DEFAULT_TEXT_PROMPTS
    
    model = CLIPClassifier(
        model_name=model_name,
        num_classes=num_classes,
        text_prompts=text_prompts,
        device=device
    )
    
    return model


def load_clip_model(checkpoint_path, num_classes=3, text_prompts=None, 
                    device="cuda", model_name="openai/clip-vit-base-patch32"):
    model = create_clip_model(num_classes, text_prompts, device, model_name)
    model.clip_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    return model