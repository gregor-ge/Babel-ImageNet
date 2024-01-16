import argparse
from benchmark.eval_babel_imagenet import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', type=str, default="./results/mymodel_results.json")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--source', type=str, default="openclip", help="Model source: openclip, nllb-openclip, altclip, m-clip, sentence-transformers, or '' for HuggingFace")
    parser.add_argument('--from_pretrained', type=str, default="xlm-roberta-base-ViT-B-32@laion5b_s13b_b90k", help="Model name (for OpenCLIP, we combine model and dataset with @)")
    parser.add_argument('--mclip_legay_weights_folder', type=str, default="", help="See M-CLIP repository for download link to weights and save them in this folder.")

    parser.add_argument('--imagenet_folder', type=str, help="Path to your ImageNet images ready for torchvision dataset")
    parser.add_argument('--babelimagenet_folder', type=str, default="./data")
    parser.add_argument('--languages', type=str, default="xlmr", help="'xlmr' to use all 92 XLM-R languages, '298' for all 298 langugages with >=10 classes, or a comma-joined list of language codes")
    parser.add_argument('--prompts', type=str, default="label,openai_en,nllb_dist13b_prompts",
                        help="(Comma-joined list of) prompt style to use. 'label' uses only labels, 'openai_en' uses OpenAI prompts and otherwise try to load JSON with the name from babelimagenet_folder.")


    args = parser.parse_args()
    main(args)
