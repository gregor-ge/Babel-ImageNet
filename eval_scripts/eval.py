import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import get_data
from model import get_model


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
    dataloader = DataLoader(text_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    all_embeddings = None
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, desc="Encoding text"):
            batch = batch.to("cuda")
            embeddings = model.encode_text(**batch)
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
    return all_embeddings


def main(args):
    images_per_class = 50 # True for ImageNet validation split
    image_dataset, text_datasets, text_collate = get_data(args)

    model = get_model(args)
    model.eval()
    model.to("cuda")

    results = []

    image_embeddings = compute_image_embeddings(image_dataset, model, args)
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

    out_path, _ = os.path.split(args.out_file)
    os.makedirs(out_path, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(results, f)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser Example')
    parser.add_argument('--out_file', type=str, default="../results/mymodel_results.json")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--source', type=str, default="openclip", help="openclip, altclip, m-clip, sentence-transformers, or ''. See model.py get_model()")
    parser.add_argument('--from_pretrained', type=str, default="xlm-roberta-base-ViT-B-32@laion5b_s13b_b90k", help="Model name (OpenCLIP combines model and dataset with @)")
    parser.add_argument('--mclip_legay_weights_folder', type=str, default="", help="See M-CLIP repository for download link to weights and save them in this folder.")

    parser.add_argument('--imagenet_folder', type=str, help="Path to your ImageNet images ready for torchvision dataset")
    parser.add_argument('--babelimagenet_folder', type=str, default="../data")
    parser.add_argument('--languages', type=str, default="xlmr", help="'xlmr' to use all 92 XLM-R languages or a comma-joined list of language codes")
    parser.add_argument('--tokenizer', type=str, default="xlm-roberta-base", help="Name of the tokenizer")
    parser.add_argument('--prompts', type=str, default="label,openai_en,nllb_dist13b_prompts",
                        help="(Comma-joined list of) prompt style to use. 'label' uses only labels, 'openai_en' uses OpenAI prompts and otherwise try to load JSON with the name from babelimagenet_folder.")
    parser.add_argument('--normalization_mode', type=str, default="imagenet", help="Set image normalization to 'imagenet' or 'inception' style mean & std")
    parser.add_argument('--image_size', type=int, default=224)


    args = parser.parse_args()
    main(args)
