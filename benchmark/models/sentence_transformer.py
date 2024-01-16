from sentence_transformers import SentenceTransformer
from torch import nn

from . import CLIP
from transformers import CLIPImageProcessor


class SentenceTransformerCLIPModel(nn.Module, CLIP):
    def __init__(self, from_pretrained):
        super(SentenceTransformerCLIPModel, self).__init__()
        self.img_model = SentenceTransformer('clip-ViT-B-32')
        self.text_model = SentenceTransformer(from_pretrained)
        self.preprocess = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.tokenizer = self.text_model.tokenizer

    def encode_images(self, images):
        features = self.img_model(dict(pixel_values=images, image_text_info=[0]*len(images)))["sentence_embedding"]
        return features

    def encode_text(self, **kwargs):
        features = self.text_model(**kwargs)["sentence_embedding"]
        return features

    def transform(self, image):
        return self.preprocess(image, return_tensors="pt")["pixel_values"].squeeze(0)

    def tokenize(self, captions):
        return self.text_model.tokenize(captions)