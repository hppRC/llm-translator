import os
import pdb

import datasets as ds

dataset = ds.load_from_disk("datasets/wikimatrix")
dataset = dataset.filter(lambda x: 1.075 < x["score"] < 1.15, num_proc=os.cpu_count())
pdb.set_trace()
