class CLIP:
    def encode_images(self, images):
        raise NotImplementedError

    def encode_text(self, **kwargs):
        raise NotImplementedError

    def transform(self, image):
        raise NotImplementedError

    def tokenize(self, captions):
        raise NotImplementedError


def get_model(args):

    source = args.source
    from_pretrained = args.from_pretrained

    if source == "sentence-transformers":
        from .sentence_transformer import SentenceTransformerCLIPModel
        model = SentenceTransformerCLIPModel(from_pretrained)
    elif source == "m-clip":
        from .m_clip import MCLIPModel
        model = MCLIPModel(from_pretrained, weights_dir=args.mclip_legay_weights_folder)
    elif source == "altclip":
        from .altclip import AltCLIPModel
        model = AltCLIPModel(from_pretrained)
    elif source == "openclip":
        from .open_clip import OpenCLIPModel
        model = OpenCLIPModel(*from_pretrained.split("@"))
    elif source == "nllb-openclip":
        from .nllb_clip import NLLBCLIPModel
        model = NLLBCLIPModel(*from_pretrained.split("@"))
    else:
        from .hugging_face import HuggingfaceCLIPModel
        model = HuggingfaceCLIPModel(from_pretrained)
    return model
