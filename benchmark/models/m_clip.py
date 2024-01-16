from transformers import AutoTokenizer

import open_clip
from torch import nn

from multilingual_clip import pt_multilingual_clip, legacy_multilingual_clip

from . import CLIP


class MCLIPModel(nn.Module, CLIP):
    def __init__(self, from_pretrained, **kwargs):
        super(MCLIPModel, self).__init__()
        self.legacy = False
        if "BERT" in from_pretrained:
            self.legacy = True
            config={
            'M-CLIP/M-BERT-Distil-40': {
                'model_name': 'M-CLIP/M-BERT-Distil-40',
                'tokenizer_name': 'M-CLIP/M-BERT-Distil-40',
                'head_name': 'M-BERT Distil 40 Linear Weights.pkl'
            },

            'M-CLIP/M-BERT-Base-69': {
                'model_name': 'M-CLIP/M-BERT-Base-69',
                'tokenizer_name': 'M-CLIP/M-BERT-Base-69',
                'head_name': 'M-BERT-Base-69 Linear Weights.pkl'
            },

            'M-CLIP/Swe-CLIP-500k': {
                'model_name': 'M-CLIP/Swedish-500k',
                'tokenizer_name': 'M-CLIP/Swedish-500k',
                'head_name': 'Swedish-500k Linear Weights.pkl'
            },

            'M-CLIP/Swe-CLIP-2M': {
                'model_name': 'M-CLIP/Swedish-2M',
                'tokenizer_name': 'M-CLIP/Swedish-2M',
                'head_name': 'Swedish-2M Linear Weights.pkl'
            },

            'M-CLIP/M-BERT-Base-ViT-B': {
                'model_name': 'M-CLIP/M-BERT-Base-ViT-B',
                'tokenizer_name': 'M-CLIP/M-BERT-Base-ViT-B',
                'head_name': 'M-BERT-Base-69-ViT Linear Weights.pkl'
            },
            }
            config = config[from_pretrained]
            self.text_model = legacy_multilingual_clip.MultilingualClip(**config, weights_dir=kwargs["weights_dir"])
        else:
            self.text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(from_pretrained)

        text2image = {
            "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus": ('ViT-B-16-plus-240', "laion400m_e32"),
            "M-CLIP/XLM-Roberta-Large-Vit-L-14": ("ViT-L-14", "openai"),
            "M-CLIP/LABSE-Vit-L-14": ("ViT-L-14", "openai"),
            "M-CLIP/XLM-Roberta-Large-Vit-B-32": ("ViT-B-32", "openai"),
            "M-CLIP/M-BERT-Base-ViT-B": ("ViT-B-32", "openai"),
            "M-CLIP/M-BERT-Distil-40": ("RN50x4", "openai"),
            "M-CLIP/M-BERT-Base-69": ("RN50x4", "openai")
        }
        name, pretrained = text2image[from_pretrained]

        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self.img_model = model
        self.preprocess = preprocess
        self.tokenizer = AutoTokenizer.from_pretrained(from_pretrained)

    def encode_images(self, images):
        features = self.img_model.encode_image(images)
        return features

    def encode_text(self, **kwargs):
        attention_mask = kwargs["attention_mask"]
        embs = self.text_model.transformer(**kwargs)[0]
        embs = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1)[:, None]
        if self.legacy:
            return self.text_model.clip_head(embs)
        else:
            return self.text_model.LinearTransformation(embs)

    def transform(self, image):
        return self.preprocess(image)

    def tokenize(self, captions):
        return self.tokenizer(text=captions, padding=True, truncation=True, return_tensors="pt")
