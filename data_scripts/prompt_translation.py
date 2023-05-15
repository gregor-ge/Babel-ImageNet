import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Template source: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

xlmr_langs = {'kn', 'sk', 'et', 'gu', 'sl', 'ka', 'gl', 'hi', 'ja', 'no', 'ms', 'my', 'eo', 'fi', 'ar', 'lv', 'de', 'ha', 'mn', 'sa', 'fr', 'br', 'or', 'ta', 'bs', 'lo', 'he', 'si', 'te', 'es', 'el', 'pt', 'km', 'ro', 'sv', 'bg', 'vi', 'az', 'la', 'th', 'af', 'om', 'eu', 'ga', 'ca', 'nl', 'ps', 'ml', 'uk', 'hy', 'jv', 'gd', 'sd', 'tl', 'zh', 'mk', 'am', 'kk', 'da', 'pa', 'ug', 'sq', 'fy', 'su', 'mg', 'is', 'ku', 'lt', 'yi', 'be', 'uz', 'id', 'sw', 'as', 'cy', 'ru', 'sr', 'mr', 'ko', 'fa', 'ur', 'xh', 'bn', 'hr', 'pl', 'cs', 'tr', 'ne', 'it', 'ky', 'hu', 'so'}
xlmr2nllb = {'ja': 'jpn_Jpan', 'om': 'gaz_Latn', 'de': 'deu_Latn', 'or': 'ory_Orya', 'mk': 'mkd_Cyrl', 'et': 'est_Latn', 'lv': 'lvs_Latn', 'ku': 'ckb_Arab', 'ru': 'rus_Cyrl', 'fr': 'fra_Latn', 'eo': 'epo_Latn', 'gd': 'gla_Latn', 'af': 'afr_Latn', 'id': 'ind_Latn', 'sv': 'swe_Latn', 'bg': 'bul_Cyrl', 'ps': 'pbt_Arab', 'fy': '??????', 'si': 'sin_Sinh', 'fi': 'fin_Latn', 'mr': 'mar_Deva', 'th': 'tha_Thai', 'kk': 'kaz_Cyrl', 'fa': 'pes_Arab', 'ur': 'urd_Arab', 'sk': 'slk_Latn', 'cy': 'cym_Latn', 'ar': 'arb_Arab', 'pl': 'pol_Latn', 'la': '??????', 'hi': 'hin_Deva', 'tl': 'tgl_Latn', 'az': 'azj_Latn', 'br': '??????', 'pt': 'por_Latn', 'ug': 'uig_Arab', 'te': 'tel_Telu', 'pa': 'pan_Guru', 'eu': 'eus_Latn', 'no': 'nob_Latn', 'mg': 'plt_Latn', 'ko': 'kor_Hang', 'ne': 'npi_Deva', 'my': 'mya_Mymr', 'so': 'som_Latn', 'mn': 'khk_Cyrl', 'sl': 'slv_Latn', 'ta': 'tam_Taml', 'su': 'sun_Latn', 'bn': 'ben_Beng', 'cs': 'ces_Latn', 'sd': 'snd_Arab', 'ca': 'cat_Latn', 'km': 'khm_Khmr', 'es': 'spa_Latn', 'hy': 'hye_Armn', 'yi': 'ydd_Hebr', 'zh': 'zho_Hant', 'jv': 'jav_Latn', 'it': 'ita_Latn', 'tr': 'tur_Latn', 'ky': 'kir_Cyrl', 'is': 'isl_Latn', 'gu': 'guj_Gujr', 'ms': 'zsm_Latn', 'uk': 'ukr_Cyrl', 'gl': 'glg_Latn', 'sq': 'als_Latn', 'sa': 'san_Deva', 'lo': 'lao_Laoo', 'hr': 'hrv_Latn', 'ka': 'kat_Geor', 'da': 'dan_Latn', 'sw': 'swh_Latn', 'hu': 'hun_Latn', 'nl': 'nld_Latn', 'lt': 'lit_Latn', 'ml': 'mal_Mlym', 'en': 'eng_Latn', 'he': 'heb_Hebr', 'as': 'asm_Beng', 'xh': 'xho_Latn', 'ha': 'hau_Latn', 'be': 'bel_Cyrl', 'ro': 'ron_Latn', 'bs': 'bos_Latn', 'vi': 'vie_Latn', 'el': 'ell_Grek', 'uz': 'uzn_Latn', 'ga': 'gle_Latn', 'am': 'amh_Ethi', 'sr': 'srp_Cyrl', 'kn': 'kan_Knda'}

# Attempts to preserve {} through MT
replacements = ["{c}", "<a>{c}</a>", "<b>{c}</b>", "<i>{c}</i>", "<div>{c}</div>", "<a>{}</a>", "<b>{}</b>", "<i>{}</i>", "<div>{}</div>", "°", "*", "#", "§", "※"]

## Translate all templates with all replacements for all languages
nllb_model = "facebook/nllb-200-distilled-1.3B" #"facebook/nllb-200-3.3B" #"facebook/nllb-200-distilled-1.3B"
tokenizer = AutoTokenizer.from_pretrained(nllb_model)
model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model)
model = model.to("cuda")
nllb_results = dict()
for lang, nllb_lang in tqdm(xlmr2nllb.items(), desc="Languages"):
    if "?" in nllb_lang:
        continue
    mt_templates = []
    total_retained = 0
    for prompt in tqdm(templates, desc="Templates"):
        mt_prompts = []
        retained = False
        for replacement in replacements:
            p = prompt.replace("{}", replacement)
            inputs = tokenizer(p, return_tensors="pt")
            inputs = inputs.to("cuda")
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[nllb_lang])
            mt_prompt = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            if replacement in mt_prompt:
                retained = True
            mt_prompts.append(mt_prompt)
        if retained:
            total_retained += 1
        mt_templates.append(mt_prompts)
    print(f"{lang}: {total_retained}/{len(templates)}")
    nllb_results[lang.upper()] = mt_templates

## Save interim results
with open("../data/nllb_dist13b_templates.json", "w", encoding="utf-8") as f:
    json.dump(nllb_results, f, ensure_ascii=False, indent=2)

## Select first replacement method that was preserved. If nothing worked, we simply append {} to the first one.
nllbselected = {lang: [] for lang in nllb_results}
for lang, translations in nllb_results.items():
    retained = 0
    for template in translations:
        chosen = template[0] + " {}" # Fallback choice
        for translation, replacement in zip(template, replacements):
            if replacement in translation:
                chosen = translation.replace(replacement, " {} ", 1).replace(replacement, "")
                retained += 1
                break
        nllbselected[lang].append(chosen)

    print(f"{lang}: {retained}")

## Save final prompts
with open("../data/nllb_dist13b_prompts.json", "w", encoding="utf-8") as f:
    json.dump(nllbselected, f, ensure_ascii=False, indent=2)