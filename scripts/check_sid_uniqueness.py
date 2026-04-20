"""Check SID uniqueness for a dataset's item features.

Usage on SageMaker:
    python scripts/check_sid_uniqueness.py --dataset sports --rqvae-ckpt <path>
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from data.amazon import AmazonReviews
from modules.quantize import QuantizeForwardMode
from modules.rqvae import RqVae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["beauty", "sports"])
    p.add_argument("--rqvae-ckpt", required=True)
    p.add_argument("--dataset-folder", default="dataset/amazon")
    p.add_argument("--output", default="/opt/ml/model/")
    args = p.parse_args()

    ckpt = torch.load(args.rqvae_ckpt, map_location="cpu", weights_only=False)
    rq = RqVae(
        input_dim=768, embed_dim=32, hidden_dims=[512, 256, 128],
        codebook_size=256, n_layers=3, n_cat_features=0,
        codebook_mode=QuantizeForwardMode.STE, codebook_kmeans_init=False,
    )
    rq.load_state_dict(ckpt["model"])
    rq.eval()

    d = AmazonReviews(root=args.dataset_folder, split=args.dataset)
    x = d.data["item"].x
    with torch.no_grad():
        sids = rq.get_semantic_ids(x).sem_ids.numpy()

    unique = np.unique(sids, axis=0)
    c = Counter(map(tuple, sids.tolist()))
    result = {
        "dataset": args.dataset,
        "total_items": int(len(sids)),
        "unique_sid_tuples": int(len(unique)),
        "unique_ratio": float(len(unique) / len(sids)),
        "max_collision_size": int(c.most_common(1)[0][1]),
        "top_10_collisions": [[list(k), v] for k, v in c.most_common(10)],
    }
    print(json.dumps(result, indent=2))
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, f"sid_uniqueness_{args.dataset}.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
