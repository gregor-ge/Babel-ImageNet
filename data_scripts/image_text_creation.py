import json
import os



def create_xtd(root="/media/gregor/DATA/projects/wuerzburg/iglue/datasets/XTD"):
    languages = ["de", "en", "es", "fr", "it", "jp", "ko", "pl", "ru", "tr", "zh"]

    images = [line.strip() for line in open(os.path.join(root, "test_image_names.txt")).readlines()]

    captions = dict()
    for lang in languages:
        caps = [[line.strip()] for line in open(os.path.join(root, f"test_1kcaptions_{lang}.txt")).readlines()]
        captions[lang] = caps


    dataset = dict(images=images, captions=captions)
    json.dump(dataset, open("../data/xtd.json", "w"))


def create_xflickrco(root="/media/gregor/DATA/projects/wuerzburg/iglue/datasets/xFlickrCO/annotations"):
    languages = ["de", "en", "es", "id", "ja", "ru", "tr", "zh"]

    images = []

    captions = dict()
    for lang in languages:
        data = [json.loads(line) for line in open(os.path.join(root, lang, "test.jsonl")).readlines()]
        caps = []
        for i, d in enumerate(data):
            caps.append(d["sentences"])
            if len(images) > i:
                assert images[i] == d["img_path"]
            else:
                images.append(d["img_path"])
        captions[lang] = caps

    dataset = dict(images=images, captions=captions)
    json.dump(dataset, open("../data/xflickrco.json", "w"))


def create_xm3600(root="/media/gregor/DATA/projects/wuerzburg/iglue/datasets/Crossmodal3600"):
    languages = ['fa', 'te', 'ko', 'fi', 'fil', 'mi', 'hu', 'id', 'hr', 'fr', 'quz', 'sv', 'zh', 'sw', 'no', 'vi', 'da', 'ja', 'nl', 'he', 'th', 'ru', 'it', 'hi', 'uk', 'de', 'pt', 'tr', 'cs', 'pl', 'bn', 'ar', 'ro', 'en', 'es', 'el']

    data = [json.loads(line) for line in open(os.path.join(root, "captions.jsonl")).readlines()]

    images = []

    captions = {lang: [] for lang in languages}

    for d in data:
        img = d["image/key"] + ".jpg"
        images.append(img)

        for lang in languages:
            caps = d[lang]["caption"]
            captions[lang].append(caps)

    dataset = dict(images=images, captions=captions)
    json.dump(dataset, open("../data/xm3600.json", "w"))

def create_xm3600_single(root="/media/gregor/DATA/projects/wuerzburg/iglue/datasets/Crossmodal3600"):
    languages = ['fa', 'te', 'ko', 'fi', 'fil', 'mi', 'hu', 'id', 'hr', 'fr', 'quz', 'sv', 'zh', 'sw', 'no', 'vi', 'da', 'ja', 'nl', 'he', 'th', 'ru', 'it', 'hi', 'uk', 'de', 'pt', 'tr', 'cs', 'pl', 'bn', 'ar', 'ro', 'en', 'es', 'el']

    data = [json.loads(line) for line in open(os.path.join(root, "captions.jsonl")).readlines()]

    images = []

    captions = {lang: [] for lang in languages}

    for d in data:
        img = d["image/key"] + ".jpg"
        images.append(img)

        for lang in languages:
            caps = [d[lang]["caption"][0]]
            captions[lang].append(caps)

    dataset = dict(images=images, captions=captions)
    json.dump(dataset, open("../data/xm3600_single.json", "w"))

if __name__ == "__main__":
    create_xtd()
    create_xflickrco()
    create_xm3600()
    create_xm3600_single()