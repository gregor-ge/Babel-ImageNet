import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import get_babel_imagenet
from .models import get_model


def compute_accuracy(image_embeddings, text_embeddings, num_prompts=1, images_per_class=50):
    # Prompt ensembles are averaged for the final prompt embedding
    if num_prompts > 1:
        text_embeddings = text_embeddings.view(len(text_embeddings)//num_prompts, num_prompts, -1)
        text_embeddings = torch.mean(text_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

    scores = image_embeddings @ text_embeddings.t()
    target = torch.arange(0, len(text_embeddings)).repeat_interleave(images_per_class, 0).to("cuda")
    prediction = scores.argmax(dim=1)
    accuracy = (target == prediction).sum().item() / target.size(0)

    return accuracy


def compute_image_embeddings(image_dataset, model, args):
    dataloader = DataLoader(image_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    all_embeddings = None
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, desc="Encoding images"):
            batch = batch[0].to("cuda")
            embeddings = model.encode_images(batch)
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
    return all_embeddings


def compute_text_embeddings(text_dataset, collate, model, args):
    if "nllb" in args.source:
        model.set_language(text_dataset.lang)
    dataloader = DataLoader(text_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    all_embeddings = None
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, desc="Encoding text"):
            batch = {k: v.to("cuda") for k,v in batch.items()}
            embeddings = model.encode_text(**batch)
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
    return all_embeddings


def main(args):
    images_per_class = 50  # True for ImageNet validation split but hard-coding is not ideal?

    model = get_model(args)
    model.eval()
    model.to("cuda")

    image_dataset, text_datasets, text_collate = get_babel_imagenet(args, model)

    print("Computing image embeddings")
    image_embeddings = compute_image_embeddings(image_dataset, model, args)

    results = []
    for text_dataset in tqdm(text_datasets, desc="Evaluating languages and prompts"):
        text_embeddings = compute_text_embeddings(text_dataset, text_collate, model, args)
        subset_image_embeddings_mask = [idx*images_per_class + i for idx in text_dataset.class_idxs for i in range(images_per_class)]
        accuracy = compute_accuracy(image_embeddings[subset_image_embeddings_mask], text_embeddings,
                                    num_prompts=text_dataset.num_prompts, images_per_class=images_per_class)
        results.append({
            "lang": text_dataset.lang,
            "prompt": text_dataset.prompt,
            "num_classes": len(text_dataset.class_idxs),
            "accuracy": accuracy
        })

    meta = {
        "model": args.from_pretrained,
        "source": args.source,
        "prompts": args.prompts,
        "parameters": sum(p.numel() for p in model.parameters())
    }

    final_results = {
        "meta": meta,
        "results": results
    }

    out_file = args.out_file
    if not out_file.endswith(".json"):
        out_file = out_file + ".json"
    out_path, _ = os.path.split(out_file)
    os.makedirs(out_path, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)




