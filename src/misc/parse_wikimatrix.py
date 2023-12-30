from pathlib import Path

import datasets as ds

data = []

with Path("datasets/WikiMatrix.en-ja.tsv").open() as f:
    for line in f:
        score, text_en, text_ja, *_ = line.strip().split("\t")
        data.append({"score": float(score), "text_en": text_en.strip(), "text_ja": text_ja.strip()})

    dataset = ds.Dataset.from_list(data)
    dataset.save_to_disk("datasets/wikimatrix")
