"""Steam game reviews dataset loader.

Source: https://cseweb.ucsd.edu/~jmcauley/datasets/steam/
Expected stats after 5-core filtering: ~334K users, ~13K items, ~3M interactions.

Mirrors the AmazonReviews interface for compatibility with existing
ItemData and SeqData loaders in data/processed.py.
"""
import gzip
import json
import os
import urllib.request
from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData, InMemoryDataset

STEAM_REVIEWS_URL = "https://huggingface.co/datasets/recommender-system/steam-review-and-bundle-dataset/resolve/main/steam_reviews.json.gz"
STEAM_GAMES_URL = "https://huggingface.co/datasets/recommender-system/steam-review-and-bundle-dataset/resolve/main/steam_games.json.gz"


def _parse_json_gz(path: str):
    """Yield dicts from a gzipped JSON-lines file."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Some McAuley Lab files use Python literal syntax — fall back to eval
                try:
                    yield eval(line)  # noqa: S307
                except Exception:
                    continue


def _download_file(url: str, dest: str) -> None:
    """Download *url* to *dest* with a simple progress indicator."""
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {dest}")


class RawSteam(InMemoryDataset):
    """Steam game reviews dataset with 5-core filtering and leave-one-out split.

    Mirrors the AmazonReviews interface so that the existing ItemData /
    SeqData wrappers in data/processed.py work unchanged.

    Directory layout after processing::

        <root>/
            raw/
                steam_reviews.json.gz
                steam_games.json.gz
            processed/
                data_steam.pt
    """

    def __init__(
        self,
        root: str,
        split: str = "steam",  # kept for API compatibility — only one split exists
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        force_reload: bool = False,
    ) -> None:
        self.split = split
        super().__init__(root, transform, pre_transform, force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    # ------------------------------------------------------------------
    # InMemoryDataset interface
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self) -> list[str]:
        return ["steam_reviews.json.gz", "steam_games.json.gz"]

    @property
    def processed_file_names(self) -> str:
        return f"data_{self.split}.pt"

    def download(self) -> None:
        reviews_dest = os.path.join(self.raw_dir, "steam_reviews.json.gz")
        games_dest = os.path.join(self.raw_dir, "steam_games.json.gz")
        if not os.path.exists(reviews_dest):
            _download_file(STEAM_REVIEWS_URL, reviews_dest)
        if not os.path.exists(games_dest):
            _download_file(STEAM_GAMES_URL, games_dest)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, max_seq_len: int = 20) -> None:  # noqa: C901
        data = HeteroData()

        # ---- Load reviews ------------------------------------------------
        print("Loading Steam reviews …")
        reviews_path = os.path.join(self.raw_dir, "steam_reviews.json.gz")
        records = []
        for rec in _parse_json_gz(reviews_path):
            uid = rec.get("username") or rec.get("user_id")
            iid = rec.get("product_id") or rec.get("item_id")
            ts = rec.get("date") or rec.get("timestamp") or 0
            if uid is not None and iid is not None:
                records.append({"raw_user": str(uid), "raw_item": str(iid), "timestamp": ts})

        reviews_df = pd.DataFrame(records)
        print(f"Raw interactions: {len(reviews_df):,}")

        # ---- 5-core filtering (iterative) --------------------------------
        reviews_df = self._apply_k_core(reviews_df, k=5)
        print(
            f"After 5-core: {reviews_df['raw_user'].nunique():,} users, "
            f"{reviews_df['raw_item'].nunique():,} items, "
            f"{len(reviews_df):,} interactions"
        )

        # ---- Remap IDs to consecutive integers ---------------------------
        user_ids = {u: i for i, u in enumerate(sorted(reviews_df["raw_user"].unique()))}
        item_ids = {it: i for i, it in enumerate(sorted(reviews_df["raw_item"].unique()))}

        reviews_df["user_id"] = reviews_df["raw_user"].map(user_ids)
        reviews_df["item_id"] = reviews_df["raw_item"].map(item_ids)

        num_items = len(item_ids)
        print(f"num_items={num_items:,}, num_users={len(user_ids):,}")

        # ---- Leave-one-out split -----------------------------------------
        sequences = self._leave_one_out_split(reviews_df, max_seq_len=max_seq_len)
        data[("user", "rated", "item")]["history"] = sequences

        # ---- Item features: title embeddings -----------------------------
        print("Loading Steam game metadata …")
        games_path = os.path.join(self.raw_dir, "steam_games.json.gz")
        title_map: dict[str, str] = {}
        for rec in _parse_json_gz(games_path):
            app_id = str(rec.get("id") or rec.get("app_id") or rec.get("item_id") or "")
            title = str(rec.get("title") or rec.get("app_name") or "Unknown")
            if app_id:
                title_map[app_id] = title

        # Build ordered title list aligned with item_id index
        id_to_raw = {v: k for k, v in item_ids.items()}
        sentences = [
            f"Title: {title_map.get(id_to_raw[i], 'Unknown')}"
            for i in range(num_items)
        ]

        print("Encoding item text features …")
        model = SentenceTransformer("sentence-transformers/sentence-t5-xxl")
        item_emb = model.encode(
            sentences,
            batch_size=2,
            show_progress_bar=True,
            convert_to_tensor=True,
        ).cpu()  # shape: [num_items, 768]

        data["item"].x = item_emb
        data["item"].text = np.array(sentences)

        gen = torch.Generator()
        gen.manual_seed(42)
        data["item"].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        self.save([data], self.processed_paths[0])
        print("Processing complete.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_k_core(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """Iteratively remove users and items with fewer than k interactions."""
        while True:
            n_before = len(df)
            item_counts = df.groupby("raw_item")["raw_item"].transform("count")
            df = df[item_counts >= k]
            user_counts = df.groupby("raw_user")["raw_user"].transform("count")
            df = df[user_counts >= k]
            if len(df) == n_before:
                break
        return df.reset_index(drop=True)

    @staticmethod
    def _leave_one_out_split(df: pd.DataFrame, max_seq_len: int = 20) -> dict:
        """Leave-one-out split: test = last item, val = second-to-last, train = rest.

        The returned dict matches the structure expected by SeqData in
        data/processed.py::

            {
                "train": {"itemId": [...], "itemId_fut": tensor, "userId": tensor},
                "eval":  {"itemId": tensor, "itemId_fut": tensor, "userId": tensor},
                "test":  {"itemId": tensor, "itemId_fut": tensor, "userId": tensor},
            }
        """
        # Sort by timestamp (best-effort; ties broken by original order)
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df = df.sort_values(["user_id", "timestamp"])
        else:
            df = df.sort_values("user_id")

        train_item_seqs, train_item_futs, train_uids = [], [], []
        eval_item_seqs, eval_item_futs, eval_uids = [], [], []
        test_item_seqs, test_item_futs, test_uids = [], [], []

        for uid, group in df.groupby("user_id", sort=True):
            items = group["item_id"].tolist()
            if len(items) < 3:
                continue  # need at least 3 interactions for a valid split

            train_seq = items[:-2]
            val_target = items[-2]
            test_target = items[-1]

            # Train: full history (variable length — stored as list, padded at load time)
            train_item_seqs.append(train_seq)
            train_item_futs.append(val_target)
            train_uids.append(uid)

            # Eval: last max_seq_len items of train history as context, target = val
            eval_ctx = items[-(max_seq_len + 2):-2]
            padded_eval = eval_ctx + [-1] * (max_seq_len - len(eval_ctx))
            eval_item_seqs.append(padded_eval)
            eval_item_futs.append(val_target)
            eval_uids.append(uid)

            # Test: last max_seq_len items including val, target = test
            test_ctx = items[-(max_seq_len + 1):-1]
            padded_test = test_ctx + [-1] * (max_seq_len - len(test_ctx))
            test_item_seqs.append(padded_test)
            test_item_futs.append(test_target)
            test_uids.append(uid)

        train_seqs_pl = pl.from_dict(
            {
                "itemId": train_item_seqs,
                "itemId_fut": train_item_futs,
                "userId": train_uids,
            }
        )
        eval_seqs_pl = pl.from_dict(
            {
                "itemId": eval_item_seqs,
                "itemId_fut": eval_item_futs,
                "userId": eval_uids,
            }
        )
        test_seqs_pl = pl.from_dict(
            {
                "itemId": test_item_seqs,
                "itemId_fut": test_item_futs,
                "userId": test_uids,
            }
        )

        def _to_tensor_dict(df_pl: pl.DataFrame) -> dict:
            item_fut = torch.tensor(df_pl["itemId_fut"].to_list(), dtype=torch.long).unsqueeze(1)
            user_ids_t = torch.tensor(df_pl["userId"].to_list(), dtype=torch.long).unsqueeze(1)
            return {
                "itemId": df_pl["itemId"].to_list(),
                "itemId_fut": item_fut,
                "userId": user_ids_t,
            }

        return {
            "train": _to_tensor_dict(train_seqs_pl),
            "eval": _to_tensor_dict(eval_seqs_pl),
            "test": _to_tensor_dict(test_seqs_pl),
        }
