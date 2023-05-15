import torch.nn.functional as F
from torch import nn
from transformers import CLIPModel

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass
try:
    from multilingual_clip import pt_multilingual_clip, legacy_multilingual_clip
except ImportError:
    pass
try:
    import open_clip
except ImportError:
    pass
try:
    # No package; we import from the downloaded files:
    # https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP/hf_altclip
    from hf_altclip.modeling_altclip import AltCLIP
except ImportError:
    pass


def get_model(args):

    source = args.source
    from_pretrained = args.from_pretrained

    if source == "sentence-transformers":
        model = SentenceTransformerCLIPModel(from_pretrained)
    elif source == "m-clip":
        model = MCLIPModel(from_pretrained, weights_dir=args.mclip_legay_weights_folder)
    elif source == "altclip":
        model = AltCLIPModel.from_pretrained(from_pretrained)
    elif source == "openclip":
        model = OpenCLIPModel(*from_pretrained.split("@"))
    else:
        model = HuggingfaceCLIPModel.from_pretrained(from_pretrained)
    return model


class CLIP:
    def encode_images(self, images):
        raise NotImplementedError

    def encode_text(self, **kwargs):
        raise NotImplementedError


class HuggingfaceCLIPModel(CLIPModel, CLIP):
    def encode_images(self, images):
        image_feat = self.get_image_features(images)
        image_feat = F.normalize(image_feat)
        return image_feat

    def encode_text(self, **kwargs):
        text_feat = self.get_text_features(**kwargs)
        text_feat = F.normalize(text_feat)
        return text_feat


class AltCLIPModel(AltCLIP, CLIP):
    def encode_images(self, images):
        image_feat = self.get_image_features(images)
        image_feat = F.normalize(image_feat)
        return image_feat

    def encode_text(self, **kwargs):
        text_feat = self.get_text_features(**kwargs)
        text_feat = F.normalize(text_feat)
        return text_feat


class OpenCLIPModel(nn.Module, CLIP):
    def __init__(self, from_pretrained, dataset):
        super(OpenCLIPModel, self).__init__()

        model, _, _ = open_clip.create_model_and_transforms(from_pretrained, pretrained=dataset)
        self.model = model

    def encode_images(self, images):
        image_feat = self.model.encode_image(images)
        image_feat = F.normalize(image_feat)
        return image_feat

    def encode_text(self, **kwargs):
        text_feat = self.model.encode_text(kwargs["input_ids"])
        text_feat = F.normalize(text_feat)
        return text_feat


class SentenceTransformerCLIPModel(nn.Module, CLIP):
    def __init__(self, from_pretrained):
        super(SentenceTransformerCLIPModel, self).__init__()
        self.img_model = SentenceTransformer('clip-ViT-B-32')
        self.text_model = SentenceTransformer(from_pretrained)

    def encode_images(self, images):
        features = self.img_model(dict(pixel_values=images, image_text_info=[0]*len(images)))["sentence_embedding"]
        return features

    def encode_text(self, **kwargs):
        features = self.text_model(**kwargs)["sentence_embedding"]
        return features


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
