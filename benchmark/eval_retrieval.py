import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import get_dataset
from .models import get_model


def compute_r_at_k(image_embeddings, text_embeddings, text2image_idxs, ks=(1,5,10)):
    scores = text_embeddings @ image_embeddings.t()
    positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
    positive_pairs[torch.arange(len(scores)), text2image_idxs] = True
    metrics = {}

    relevant = torch.gather(positive_pairs, 1, torch.topk(scores, k=max(ks), dim=-1, sorted=True)[1])
    for k in ks:
        r_at_k = (relevant[:,:k].sum(dim=1) > 0).float().mean()
        metrics[f"t2i_r@{k}"] = r_at_k.cpu().item()

    relevant = torch.gather(positive_pairs.T, 1, torch.topk(scores.T, k=max(ks), dim=-1, sorted=True)[1])
    for k in ks:
        r_at_k = (relevant[:,:k].sum(dim=1) > 0).float().mean()
        metrics[f"i2t_r@{k}"] = r_at_k.cpu().item()

    return metrics

def compute_image_embeddings(image_dataset, model, args):
    dataloader = DataLoader(image_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    all_embeddings = None
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, desc="Encoding images"):
            batch = batch.to("cuda")
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
    model = get_model(args)
    model.eval()
    model.to("cuda")

    image_dataset, text_datasets, text_collate = get_dataset(args, model)

    print("Computing image embeddings")
    image_embeddings = compute_image_embeddings(image_dataset, model, args)

    results = []
    for text_dataset in tqdm(text_datasets, desc="Evaluating languages and prompts"):
        text_embeddings = compute_text_embeddings(text_dataset, text_collate, model, args)
        metrics = compute_r_at_k(image_embeddings, text_embeddings, text2image_idxs=text_dataset.text2image_idxs)
        metrics["lang"] = text_dataset.lang
        results.append(metrics)

    meta = {
        "model": args.from_pretrained,
        "source": args.source,
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




