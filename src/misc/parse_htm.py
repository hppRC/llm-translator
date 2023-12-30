import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

import datasets as ds


def preprocess(text: str):
    text = re.sub(r"＜注[0-9]+＞", "", text.strip())
    text = re.sub(r"［＃.*に傍点］", "", text)
    text = re.sub(r"（[\u3040-\u309F]+）", "", text)
    text = re.sub(r" − (.+) − ", "――\\1――", text)
    text = re.sub(r"_(.+)_", "\\1", text)
    return text.strip()


def parse_html_table(path: Path):
    try:
        with path.open(encoding="shift_jis") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with path.open(encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with path.open(encoding="cp932") as f:
                    content = f.read()
            except UnicodeDecodeError:
                return [], []

    soup = BeautifulSoup(content, "lxml")
    tables = soup.find_all("table")

    texts_en, texts_ja = [], []
    cur_text_en, cur_text_ja = "", ""
    cur_left_parens, cur_right_parens = 0, 0
    cur_left_parens_ja, cur_right_parens_ja = 0, 0

    for table in tables:
        for tr in table.find_all("tr"):
            text_en, _, text_ja = (preprocess(td.text) for td in tr.find_all("td"))
            text_ja = unicodedata.normalize("NFKC", text_ja)

            cur_left_parens += text_en.count("(")
            cur_right_parens += text_en.count(")")
            cur_left_parens_ja += text_ja.count("「")
            cur_right_parens_ja += text_ja.count("」")

            if (
                text_ja.strip().endswith("。")
                and text_en.strip().endswith(".")
                and cur_left_parens == cur_right_parens
                and cur_left_parens_ja == cur_right_parens_ja
            ):
                texts_en.append((cur_text_en + " " + text_en).strip())
                texts_ja.append((cur_text_ja + text_ja).strip())
                cur_text_en, cur_text_ja = "", ""
                cur_left_parens, cur_right_parens = 0, 0
                cur_left_parens_ja, cur_right_parens_ja = 0, 0
            else:
                cur_text_en += " " + text_en
                cur_text_ja += text_ja

        texts_en.append(cur_text_en.strip())
        texts_ja.append(cur_text_ja.strip())

    return texts_en, texts_ja


def main():
    data = []
    for path in tqdm(list(Path("datasets/align/htmPages").glob("*.htm"))):
        texts_en, texts_ja = parse_html_table(path)
        for text_en, text_ja in zip(texts_en, texts_ja, strict=True):
            data.append({"en": text_en, "ja": text_ja, "source": path.name})
    dataset = ds.Dataset.from_list(data)
    dataset.save_to_disk("datasets/en-ja-alignment")


if __name__ == "__main__":
    main()
