image_folder="/media/gregor/DATA/projects/wuerzburg/iglue/datasets/Crossmodal3600/images"

cd ..

# OpenCLIP

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="openclip" --from_pretrained="xlm-roberta-base-ViT-B-32@laion5b_s13b_b90k" \
  --out_file="results/retrieval/xm3600/openclip-xlmrb-vitb32"

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=256 \
  --source="openclip" --from_pretrained="xlm-roberta-large-ViT-H-14@frozen_laion5b_s13b_b90k" \
  --out_file="results/retrieval/xm3600/openclip-xlmrl-vith14"

# Sentence-Transformers

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="sentence-transformers" --from_pretrained="sentence-transformers/clip-ViT-B-32-multilingual-v1" \
  --out_file="results/retrieval/xm3600/st-mbert-vitb32"

# M-CLIP

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="m-clip" --from_pretrained="M-CLIP/M-BERT-Base-ViT-B" --mclip_legay_weights_folder="/media/gregor/DATA/projects/wuerzburg/zeroshot-image/babel/data/mclip_legacy_weights/" \
  --out_file="results/retrieval/xm3600/mclip-mbert-vitb32"

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="m-clip" --from_pretrained="M-CLIP/XLM-Roberta-Large-Vit-B-32" \
  --out_file="results/retrieval/xm3600/mclip-xlmrl-vitb32"

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="m-clip" --from_pretrained="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus" \
  --out_file="results/retrieval/xm3600/mclip-xlmrl-vitb16plus"

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="m-clip" --from_pretrained="M-CLIP/XLM-Roberta-Large-Vit-L-14" \
  --out_file="results/retrieval/xm3600/mclip-xlmrl-vitl14"

# AltCLIP

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="altclip" --from_pretrained="BAAI/AltCLIP-m9" \
  --out_file="results/retrieval/xm3600/altclip-xlmrl-vitl14"

### New ###

# SigLIP
python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="openclip" --from_pretrained="ViT-B-16-SigLIP-i18n-256@webli" \
  --out_file="results/retrieval/xm3600/siglip-vitb16"

# NLLB
python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="nllb-openclip" --from_pretrained="nllb-clip-base@v1" \
  --out_file="results/retrieval/xm3600/nllb-base"

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="nllb-openclip" --from_pretrained="nllb-clip-base-siglip@v1" \
  --out_file="results/retrieval/xm3600/nllb-siglip-base"

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=256 \
  --source="nllb-openclip" --from_pretrained="nllb-clip-large@v1" \
  --out_file="results/retrieval/xm3600/nllb-large"

python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=256 \
  --source="nllb-openclip" --from_pretrained="nllb-clip-large-siglip@v1" \
  --out_file="results/retrieval/xm3600/nllb-siglip-large"