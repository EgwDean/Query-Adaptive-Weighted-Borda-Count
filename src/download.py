"""download.py -- fetch a single BEIR dataset named in config.yaml.

Downloads the dataset's zip from the BEIR mirror into
data/datasets/<dataset>/ and verifies it by loading corpus/queries/qrels.
Re-running is a no-op if the corpus is already present.

Progress: BEIR's `download_and_unzip` streams the archive with a tqdm bar.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from utils import load_config, get_paths, dataset_dir

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def main():
    config = load_config()
    paths = get_paths(config)
    name = config["dataset"]
    split = config.get("split", "test")

    target = dataset_dir(paths, name)
    corpus_file = os.path.join(target, "corpus.jsonl")

    if os.path.exists(corpus_file):
        print(f"[download] '{name}' already present at {target} -- skipping download.")
    else:
        url = BEIR_URL.format(name=name)
        print(f"[download] downloading '{name}' from:\n           {url}")
        data_path = util.download_and_unzip(url, paths["datasets"])  # tqdm bar inside
        print(f"[download] unzipped to {data_path}")

    # Verify + report sizes so you know what you just pulled.
    corpus, queries, qrels = GenericDataLoader(data_folder=target).load(split=split)
    print(
        f"[download] '{name}' ready: "
        f"corpus={len(corpus):,} docs | queries[{split}]={len(queries):,} | "
        f"qrels={len(qrels):,}"
    )


if __name__ == "__main__":
    main()
