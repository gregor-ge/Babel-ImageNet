import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *


def get_dataset(args, model):
    dataset = json.load(open(args.dataset_file))
    image_dataset = RetrievalImageDataset(images=dataset["images"], image_root=args.image_folder, transform=model.transform)

    text_collate = model.tokenize
    text_datasets = []

    for lang, captions in dataset["captions"].items():
        ds = RetrievalTextDataset(captions_per_image=captions, language=lang)
        text_datasets.append(ds)
    return image_dataset, text_datasets, text_collate



class RetrievalTextDataset(Dataset):
    def __init__(self, captions_per_image, language):
        self.lang = language

        text2image_idxs = []
        data = []
        for i, caps in enumerate(captions_per_image):
            for cap in caps:
                text2image_idxs.append(i)
                data.append(cap)

        self.text2image_idxs = text2image_idxs
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

class RetrievalImageDataset(Dataset):
    def __init__(self, image_root, images, transform):
        self.transform = transform
        self.image_root = image_root
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        img = Image.open(os.path.join(self.image_root, img))
        img = self.transform(img)
        return img
