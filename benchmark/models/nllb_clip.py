import open_clip
from torch import nn
import torch.nn.functional as F

from . import CLIP


class NLLBCLIPModel(nn.Module, CLIP):
    def __init__(self, from_pretrained, dataset):
        super(NLLBCLIPModel, self).__init__()

        model, _, transform = open_clip.create_model_and_transforms(from_pretrained, pretrained=dataset)
        self.preprocess = transform
        self.tokenizer = open_clip.get_tokenizer(from_pretrained)
        self.model = model

    def encode_images(self, images):
        image_feat = self.model.encode_image(images)
        image_feat = F.normalize(image_feat)
        return image_feat

    def encode_text(self, input_ids, **kwargs):
        text_feat = self.model.encode_text(input_ids)
        text_feat = F.normalize(text_feat)
        return text_feat

    def transform(self, image):
        return self.preprocess(image)

    def tokenize(self, captions):
        return dict(input_ids=self.tokenizer(captions))

# adapted from https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/models/nllb_clip.py
    def set_language(self, lang_code):
        lang_code = lang_code.lower()
        lang = lang_map.get(lang_code, lang_map["en"])
        if lang == lang_map["en"] and lang_code != "en":
            print(f"WARNING: {lang_code} not covered by NLLB tokenizer. Defaulting to 'en'.")
        self.tokenizer.tokenizer.set_src_lang_special_tokens(lang)

lang_map = {
"af": "afr_Latn",
"am": "amh_Ethi",
"ar": "arb_Arab",
"az": "azj_Latn",
"be": "bel_Cyrl",
"bn": "ben_Beng",
"bg": "bul_Cyrl",
"ca": "cat_Latn",
"ceb": "ceb_Latn",
"cs": "ces_Latn",
"cy": "cym_Latn",
"da": "dan_Latn",
"de": "deu_Latn",
"el": "ell_Grek",
"en": "eng_Latn",
"eo": "epo_Latn",
"et": "est_Latn",
"eu": "eus_Latn",
"fa": "pes_Arab",
"fil": "tgl_Latn",
"fi": "fin_Latn",
"fr": "fra_Latn",
"gd": "gla_Latn",
"ga": "gle_Latn",
"gl": "glg_Latn",
"gu": "guj_Gujr",
"ht": "hat_Latn",
"ha": "hau_Latn",
"iw": "heb_Hebr",
"hi": "hin_Deva",
"hu": "hun_Latn",
"hy": "hye_Armn",
"ig": "ibo_Latn",
"id": "ind_Latn",
"is": "isl_Latn",
"it": "ita_Latn",
"jv": "jav_Latn",
"ja": "jpn_Jpan",
"kn": "kan_Knda",
"ka": "kat_Geor",
"kk": "kaz_Cyrl",
"km": "khm_Khmr",
"ky": "kir_Cyrl",
"ko": "kor_Hang",
"ku": "ckb_Arab",
"lo": "lao_Laoo",
"lv": "lvs_Latn",
"lt": "lit_Latn",
"lb": "ltz_Latn",
"ml": "mal_Mlym",
"mr": "mar_Deva",
"mk": "mkd_Cyrl",
"mg": "plt_Latn",
"mt": "mlt_Latn",
"mn": "khk_Cyrl",
"mi": "mri_Latn",
"ms": "zsm_Latn",
"my": "mya_Mymr",
"ne": "npi_Deva",
"nl": "nld_Latn",
"no": "nno_Latn",
"ny": "nya_Latn",
"pa": "pan_Guru",
"pl": "pol_Latn",
"pt": "por_Latn",
"ps": "pbt_Arab",
"ro": "ron_Latn",
"ru": "rus_Cyrl",
"si": "sin_Sinh",
"sk": "slk_Latn",
"sl": "slv_Latn",
"sm": "smo_Latn",
"sn": "sna_Latn",
"sd": "snd_Arab",
"so": "som_Latn",
"st": "sot_Latn",
"es": "spa_Latn",
"sq": "als_Latn",
"sr": "srp_Cyrl",
"su": "sun_Latn",
"sw": "swh_Latn",
"sv": "swe_Latn",
"ta": "tam_Taml",
"te": "tel_Telu",
"tg": "tgk_Cyrl",
"th": "tha_Thai",
"tr": "tur_Latn",
"uk": "ukr_Cyrl",
"ur": "urd_Arab",
"uz": "uzn_Latn",
"vi": "vie_Latn",
"xh": "xho_Latn",
"yi": "ydd_Hebr",
"yo": "yor_Latn",
"zh": "zho_Hant",
"zu": "zul_Latn",
"om": "gaz_Latn",
"or": "ory_Orya",
"tl": "tgl_Latn",
"ug": "uig_Arab",
"sa": "san_Deva",
"hr": "hrv_Latn",
"he": "heb_Hebr",
"as": "asm_Beng",
"bs": "bos_Latn",
"jp": "jpn_Jpan",
"cn": "zho_Hant",
"zhm": "yue_Hant",
"quz": "quy_Latn",
"ace_Arab": "ace_Arab",
"ace_Latn": "ace_Latn",
"acm_Arab": "acm_Arab",
"acq_Arab": "acq_Arab",
"aeb_Arab": "aeb_Arab",
"afr_Latn": "afr_Latn",
"ajp_Arab": "ajp_Arab",
"aka_Latn": "aka_Latn",
"amh_Ethi": "amh_Ethi",
"apc_Arab": "apc_Arab",
"arb_Arab": "arb_Arab",
"ars_Arab": "ars_Arab",
"ary_Arab": "ary_Arab",
"arz_Arab": "arz_Arab",
"asm_Beng": "asm_Beng",
"ast_Latn": "ast_Latn",
"awa_Deva": "awa_Deva",
"ayr_Latn": "ayr_Latn",
"azb_Arab": "azb_Arab",
"azj_Latn": "azj_Latn",
"bak_Cyrl": "bak_Cyrl",
"bam_Latn": "bam_Latn",
"ban_Latn": "ban_Latn",
"bel_Cyrl": "bel_Cyrl",
"bem_Latn": "bem_Latn",
"ben_Beng": "ben_Beng",
"bho_Deva": "bho_Deva",
"bjn_Arab": "bjn_Arab",
"bjn_Latn": "bjn_Latn",
"bod_Tibt": "bod_Tibt",
"bos_Latn": "bos_Latn",
"bug_Latn": "bug_Latn",
"bul_Cyrl": "bul_Cyrl",
"cat_Latn": "cat_Latn",
"ceb_Latn": "ceb_Latn",
"ces_Latn": "ces_Latn",
"cjk_Latn": "cjk_Latn",
"ckb_Arab": "ckb_Arab",
"crh_Latn": "crh_Latn",
"cym_Latn": "cym_Latn",
"dan_Latn": "dan_Latn",
"deu_Latn": "deu_Latn",
"dik_Latn": "dik_Latn",
"dyu_Latn": "dyu_Latn",
"dzo_Tibt": "dzo_Tibt",
"eng_Latn": "eng_Latn",
"ell_Grek": "ell_Grek",
"epo_Latn": "epo_Latn",
"est_Latn": "est_Latn",
"eus_Latn": "eus_Latn",
"ewe_Latn": "ewe_Latn",
"fao_Latn": "fao_Latn",
"fij_Latn": "fij_Latn",
"fin_Latn": "fin_Latn",
"fon_Latn": "fon_Latn",
"fra_Latn": "fra_Latn",
"fur_Latn": "fur_Latn",
"fuv_Latn": "fuv_Latn",
"gla_Latn": "gla_Latn",
"gle_Latn": "gle_Latn",
"glg_Latn": "glg_Latn",
"grn_Latn": "grn_Latn",
"guj_Gujr": "guj_Gujr",
"hat_Latn": "hat_Latn",
"hau_Latn": "hau_Latn",
"heb_Hebr": "heb_Hebr",
"hin_Deva": "hin_Deva",
"hne_Deva": "hne_Deva",
"hrv_Latn": "hrv_Latn",
"hun_Latn": "hun_Latn",
"hye_Armn": "hye_Armn",
"ibo_Latn": "ibo_Latn",
"ilo_Latn": "ilo_Latn",
"ind_Latn": "ind_Latn",
"isl_Latn": "isl_Latn",
"ita_Latn": "ita_Latn",
"jav_Latn": "jav_Latn",
"jpn_Jpan": "jpn_Jpan",
"kab_Latn": "kab_Latn",
"kac_Latn": "kac_Latn",
"kam_Latn": "kam_Latn",
"kan_Knda": "kan_Knda",
"kas_Arab": "kas_Arab",
"kas_Deva": "kas_Deva",
"kat_Geor": "kat_Geor",
"knc_Arab": "knc_Arab",
"knc_Latn": "knc_Latn",
"kaz_Cyrl": "kaz_Cyrl",
"kbp_Latn": "kbp_Latn",
"kea_Latn": "kea_Latn",
"khm_Khmr": "khm_Khmr",
"kik_Latn": "kik_Latn",
"kin_Latn": "kin_Latn",
"kir_Cyrl": "kir_Cyrl",
"kmb_Latn": "kmb_Latn",
"kmr_Latn": "kmr_Latn",
"kon_Latn": "kon_Latn",
"kor_Hang": "kor_Hang",
"lao_Laoo": "lao_Laoo",
"lij_Latn": "lij_Latn",
"lim_Latn": "lim_Latn",
"lin_Latn": "lin_Latn",
"lit_Latn": "lit_Latn",
"lmo_Latn": "lmo_Latn",
"ltg_Latn": "ltg_Latn",
"ltz_Latn": "ltz_Latn",
"lua_Latn": "lua_Latn",
"lug_Latn": "lug_Latn",
"luo_Latn": "luo_Latn",
"lus_Latn": "lus_Latn",
"lvs_Latn": "lvs_Latn",
"mag_Deva": "mag_Deva",
"mai_Deva": "mai_Deva",
"mal_Mlym": "mal_Mlym",
"mar_Deva": "mar_Deva",
"min_Latn": "min_Latn",
"mkd_Cyrl": "mkd_Cyrl",
"plt_Latn": "plt_Latn",
"mlt_Latn": "mlt_Latn",
"mni_Beng": "mni_Beng",
"khk_Cyrl": "khk_Cyrl",
"mos_Latn": "mos_Latn",
"mri_Latn": "mri_Latn",
"mya_Mymr": "mya_Mymr",
"nld_Latn": "nld_Latn",
"nno_Latn": "nno_Latn",
"nob_Latn": "nob_Latn",
"npi_Deva": "npi_Deva",
"nso_Latn": "nso_Latn",
"nus_Latn": "nus_Latn",
"nya_Latn": "nya_Latn",
"oci_Latn": "oci_Latn",
"gaz_Latn": "gaz_Latn",
"ory_Orya": "ory_Orya",
"pag_Latn": "pag_Latn",
"pan_Guru": "pan_Guru",
"pap_Latn": "pap_Latn",
"pes_Arab": "pes_Arab",
"pol_Latn": "pol_Latn",
"por_Latn": "por_Latn",
"prs_Arab": "prs_Arab",
"pbt_Arab": "pbt_Arab",
"quy_Latn": "quy_Latn",
"ron_Latn": "ron_Latn",
"run_Latn": "run_Latn",
"rus_Cyrl": "rus_Cyrl",
"sag_Latn": "sag_Latn",
"san_Deva": "san_Deva",
"scn_Latn": "scn_Latn",
"shn_Mymr": "shn_Mymr",
"sin_Sinh": "sin_Sinh",
"slk_Latn": "slk_Latn",
"slv_Latn": "slv_Latn",
"smo_Latn": "smo_Latn",
"sna_Latn": "sna_Latn",
"snd_Arab": "snd_Arab",
"som_Latn": "som_Latn",
"sot_Latn": "sot_Latn",
"spa_Latn": "spa_Latn",
"als_Latn": "als_Latn",
"srd_Latn": "srd_Latn",
"srp_Cyrl": "srp_Cyrl",
"ssw_Latn": "ssw_Latn",
"sun_Latn": "sun_Latn",
"swe_Latn": "swe_Latn",
"swh_Latn": "swh_Latn",
"szl_Latn": "szl_Latn",
"tam_Taml": "tam_Taml",
"tat_Cyrl": "tat_Cyrl",
"tel_Telu": "tel_Telu",
"tgk_Cyrl": "tgk_Cyrl",
"tgl_Latn": "tgl_Latn",
"tha_Thai": "tha_Thai",
"tir_Ethi": "tir_Ethi",
"taq_Latn": "taq_Latn",
"taq_Tfng": "taq_Tfng",
"tpi_Latn": "tpi_Latn",
"tsn_Latn": "tsn_Latn",
"tso_Latn": "tso_Latn",
"tuk_Latn": "tuk_Latn",
"tum_Latn": "tum_Latn",
"tur_Latn": "tur_Latn",
"twi_Latn": "twi_Latn",
"tzm_Tfng": "tzm_Tfng",
"uig_Arab": "uig_Arab",
"ukr_Cyrl": "ukr_Cyrl",
"umb_Latn": "umb_Latn",
"urd_Arab": "urd_Arab",
"uzn_Latn": "uzn_Latn",
"vec_Latn": "vec_Latn",
"vie_Latn": "vie_Latn",
"war_Latn": "war_Latn",
"wol_Latn": "wol_Latn",
"xho_Latn": "xho_Latn",
"ydd_Hebr": "ydd_Hebr",
"yor_Latn": "yor_Latn",
"yue_Hant": "yue_Hant",
"zho_Hans": "zho_Hans",
"zho_Hant": "zho_Hant",
"zsm_Latn": "zsm_Latn",
"zul_Latn": "zul_Latn",
"afr": "afr_Latn",
"ak": "aka_Latn",
"aka": "aka_Latn",
"als": "als_Latn",
"amh": "amh_Ethi",
"ary": "ary_Arab",
"arz": "arz_Arab",
"asm": "asm_Beng",
"ast": "ast_Latn",
"azb": "azb_Arab",
"ba": "bak_Cyrl",
"bak": "bak_Cyrl",
"ban": "ban_Latn",
"bel": "bel_Cyrl",
"bul": "bul_Cyrl",
"bho": "bho_Deva",
"bjn": "bjn_Latn",
"bm": "bam_Latn",
"bam": "bam_Latn",
"ben": "ben_Beng",
"bo": "bod_Tibt",
"bod": "bod_Tibt",
"bos": "bos_Latn",
"cat": "cat_Latn",
"ckb": "ckb_Arab",
"crh": "crh_Latn",
"ces": "ces_Latn",
"cym": "cym_Latn",
"dan": "dan_Latn",
"deu": "deu_Latn",
"dz": "dzo_Tibt",
"dzo": "dzo_Tibt",
"ee": "ewe_Latn",
"ewe": "ewe_Latn",
"ell": "ell_Grek",
"eng": "eng_Latn",
"epo": "epo_Latn",
"spa": "spa_Latn",
"est": "est_Latn",
"eus": "eus_Latn",
"fin": "fin_Latn",
"fj": "fij_Latn",
"fij": "fij_Latn",
"fo": "fao_Latn",
"fao": "fao_Latn",
"fra": "fra_Latn",
"fur": "fur_Latn",
"gle": "gle_Latn",
"gla": "gla_Latn",
"glg": "glg_Latn",
"gn": "grn_Latn",
"grn": "grn_Latn",
"guj": "guj_Gujr",
"hau": "hau_Latn",
"heb": "heb_Hebr",
"hin": "hin_Deva",
"hrv": "hrv_Latn",
"hat": "hat_Latn",
"hun": "hun_Latn",
"hye": "hye_Armn",
"ind": "ind_Latn",
"ibo": "ibo_Latn",
"ilo": "ilo_Latn",
"isl": "isl_Latn",
"ita": "ita_Latn",
"jpn": "jpn_Jpan",
"jav": "jav_Latn",
"kat": "kat_Geor",
"kab": "kab_Latn",
"kg": "kon_Latn",
"kon": "kon_Latn",
"ki": "kik_Latn",
"kik": "kik_Latn",
"kaz": "kaz_Cyrl",
"khm": "khm_Khmr",
"kmr": "kmr_Latn",
"kan": "kan_Knda",
"kor": "kor_Hang",
"ks": "kas_Deva",
"kas": "kas_Deva",
"kir": "kir_Cyrl",
"ltz": "ltz_Latn",
"lg": "lug_Latn",
"lug": "lug_Latn",
"li": "lim_Latn",
"lim": "lim_Latn",
"lij": "lij_Latn",
"lmo": "lmo_Latn",
"ln": "lin_Latn",
"lin": "lin_Latn",
"lao": "lao_Laoo",
"lit": "lit_Latn",
"ltg": "ltg_Latn",
"mai": "mai_Deva",
"mri": "mri_Latn",
"min": "min_Latn",
"mkd": "mkd_Cyrl",
"mal": "mal_Mlym",
"mar": "mar_Deva",
"mlt": "mlt_Latn",
"mya": "mya_Mymr",
"nld": "nld_Latn",
"nn": "nno_Latn",
"nno": "nno_Latn",
"nso": "nso_Latn",
"nya": "nya_Latn",
"oc": "oci_Latn",
"oci": "oci_Latn",
"pan": "pan_Guru",
"pap": "pap_Latn",
"pol": "pol_Latn",
"por": "por_Latn",
"ron": "ron_Latn",
"rus": "rus_Cyrl",
"san": "san_Deva",
"sat": "sat_Olck",
"sc": "srd_Latn",
"srd": "srd_Latn",
"scn": "scn_Latn",
"snd": "snd_Arab",
"shn": "shn_Mymr",
"sin": "sin_Sinh",
"slk": "slk_Latn",
"slv": "slv_Latn",
"smo": "smo_Latn",
"sna": "sna_Latn",
"som": "som_Latn",
"srp": "srp_Cyrl",
"sot": "sot_Latn",
"sun": "sun_Latn",
"swe": "swe_Latn",
"szl": "szl_Latn",
"tam": "tam_Taml",
"tel": "tel_Telu",
"tgk": "tgk_Cyrl",
"tha": "tha_Thai",
"ti": "tir_Ethi",
"tir": "tir_Ethi",
"tk": "tuk_Latn",
"tuk": "tuk_Latn",
"tgl": "tgl_Latn",
"tn": "tsn_Latn",
"tsn": "tsn_Latn",
"tpi": "tpi_Latn",
"tur": "tur_Latn",
"tt": "tat_Cyrl",
"tat": "tat_Cyrl",
"uig": "uig_Arab",
"ukr": "ukr_Cyrl",
"urd": "urd_Arab",
"vec": "vec_Latn",
"vie": "vie_Latn",
"war": "war_Latn",
"wo": "wol_Latn",
"wol": "wol_Latn",
"xho": "xho_Latn",
"yor": "yor_Latn",
"zho": "zho_Hant",
"zul": "zul_Latn"
}
# Missing XLMR languages
#  br bre Breton
#  fy fry Western Frisian
#  la lat Latin
# Total  3
# Missing Babel-ImageNet-298 languages
#  ab Abkhazian
#  acw Hijazi Arabic
#  ady Adyghe
#  afb Gulf Arabic
#  aii Assyrian Neo-Aramaic
#  alt Southern Altai
#  an Aragonese
#  ang Old English (ca. 450-1100)
#  ar Arabic
#  arc Official Aramaic (700-300 BCE)
#  av Avaric
#  ay Aymara
#  az Azerbaijani
#  bar Bavarian
#  bat_smg Samogitian
#  bcl Central Bikol
#  be_x_old Belarusian (Taraškievica)
#  bh Bihari
#  br Breton
#  bxr Russia Buriat
#  ccc Chamicuro
#  cdo Min Dong Chinese
#  ce Chechen
#  chr Cherokee
#  chy Cheyenne
#  co Corsican
#  cr Cree
#  csb Kashubian
#  cu Church Slavic
#  cv Chuvash
#  din Dinka
#  diq Dimli (individual language)
#  dlm Dalmatian
#  dng Dungan
#  dsb Lower Sorbian
#  dv Dhivehi
#  eml Emilian-Romagnol
#  enm Middle English (1100-1500)
#  ext Extremaduran
#  fiu_vro Võro
#  fro Old French (842-ca. 1400)
#  frp Arpitan
#  frr Northern Frisian
#  fy Western Frisian
#  gan Gan Chinese
#  gmq_bot Westrobothnian
#  got Gothic
#  grc Ancient Greek (to 1453)
#  gv Manx
#  hak Hakka Chinese
#  haw Hawaiian
#  hrx Hunsrik
#  hsb Upper Sorbian
#  ia Interlingua (International Auxiliary Language Association)
#  ik Inupiaq
#  inh Ingush
#  io Ido
#  iu Inuktitut
#  jam Jamaican Creole English
#  jbo Lojban
#  kaa Kara-Kalpak
#  kbd Kabardian
#  khb Lü
#  kl Kalaallisut
#  koi Komi-Permyak
#  krc Karachay-Balkar
#  ksh Kölsch
#  ku Kurdish
#  kum Kumyk
#  kv Komi
#  kw Cornish
#  kxd Brunei
#  la Latin
#  lad Ladino
#  lbe Lak
#  lez Lezghian
#  lv Latvian
#  lzz Laz
#  mdf Moksha
#  mel Central Melanau
#  mg Malagasy
#  mhr Eastern Mari
#  mic Mi'kmaq
#  mn Mongolian
#  mnc Manchu
#  mnw Mon
#  mrj Western Mari
#  ms Malay (macrolanguage)
#  mwl Mirandese
#  myv Erzya
#  mzn Mazanderani
#  nah Nahuatl
#  nap Neapolitan
#  nci Classical Nahuatl
#  nds Low German
#  nds_nl Dutch Low Saxon
#  ne Nepali (macrolanguage)
#  new Newari
#  no Norwegian
#  nog Nogai
#  non Old Norse
#  nov Novial
#  nrm Narom
#  nv Navajo
#  oj Ojibwa
#  olo Livvi
#  om Oromo
#  or Oriya (macrolanguage)
#  orv Old Russian
#  os Ossetian
#  ota Ottoman Turkish (1500-1928)
#  ovd Elfdalian
#  pam Pampanga
#  pcd Picard
#  pdc Pennsylvania German
#  pdt Plautdietsch
#  pms Piemontese
#  pnb Western Panjabi
#  ps Pushto
#  qu Quechua
#  rm Romansh
#  rmy Vlax Romani
#  roa_rup Aromanian
#  rue Rusyn
#  sah Yakut
#  sco Scots
#  se Northern Sami
#  sh Code element for 639-1 has been deprecated
#  shi Tachelhit
#  simple Simple English
#  smj Lule Sami
#  smn Inari Sami
#  sms Skolt Sami
#  sne Bau Bidayuh
#  sq Albanian
#  srn Sranan Tongo
#  stq Saterfriesisch
#  sva Svan
#  sw Swahili (macrolanguage)
#  syc Classical Syriac
#  tcy Tulu
#  to Tonga (Tonga Islands)
#  tyv Tuvinian
#  udm Udmurt
#  uga Ugaritic
#  uz Uzbek
#  vep Veps
#  vls Vlaams
#  vo Volapük
#  wa Walloon
#  wuu Wu Chinese
#  xal Kalmyk
#  xmf Mingrelian
#  yi Yiddish
#  yua Yucateco
#  za Zhuang
#  zdj Ngazidja Comorian
#  zh_classical Classical Chinese
#  zh_min_nan Min Nan
#  zh_yue Cantonese
# Total  161