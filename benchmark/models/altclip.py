from torch import nn
import torch.nn.functional as F

from transformers import AltCLIPModel as HfAltCLIPModel, AltCLIPProcessor

from . import CLIP


class AltCLIPModel(nn.Module, CLIP):
    def __init__(self, from_pretrained):
        super(AltCLIPModel, self).__init__()
        self.model = HfAltCLIPModel.from_pretrained(from_pretrained)
        self.processor = AltCLIPProcessor.from_pretrained(from_pretrained)

    def encode_images(self, images):
        image_feat = self.model.get_image_features(images)
        image_feat = F.normalize(image_feat)
        return image_feat

    def encode_text(self, **kwargs):
        text_feat = self.model.get_text_features(**kwargs)
        text_feat = F.normalize(text_feat)
        return text_feat

    def transform(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def tokenize(self, captions):
        return self.processor(text=captions, padding=True, truncation=True, return_tensors="pt")

