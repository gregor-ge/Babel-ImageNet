import open_clip
from torch import nn
import torch.nn.functional as F

from . import CLIP


class OpenCLIPModel(nn.Module, CLIP):
    def __init__(self, from_pretrained, dataset):
        super(OpenCLIPModel, self).__init__()

        model, _, transform = open_clip.create_model_and_transforms(from_pretrained, pretrained=dataset)
        self.preprocess = transform
        self.tokenizer = open_clip.get_tokenizer(from_pretrained)
        self.model = model

    def encode_images(self, images):
        image_feat = self.model.encode_image(images)
        image_feat = F.normalize(image_feat)
        return image_feat

    def encode_text(self, input_ids, **kwargs):
        text_feat = self.model.encode_text(input_ids)
        text_feat = F.normalize(text_feat)
        return text_feat

    def transform(self, image):
        return self.preprocess(image)

    def tokenize(self, captions):
        return dict(input_ids=self.tokenizer(captions))
