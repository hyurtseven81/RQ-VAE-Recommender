from modules.decoding.dbs import DiverseBeamSearch
from modules.decoding.gumbel import GumbelTopKBeamSearch
from modules.decoding.hybrid import HybridBeamSearch
from modules.decoding.level_aware_mix import AlphaParams as AlphaParams
from modules.decoding.level_aware_mix import LevelAwareHybridDecoding
from modules.decoding.sasrec_reranker import SASRecReranker
from modules.decoding.vanilla import VanillaBeamSearch

# level_aware_mix_grid and level_aware_mix_learned both use LevelAwareHybridDecoding;
# run_eval.py distinguishes them by loading alpha from a CSV or .pt file respectively.
DECODING_STRATEGIES = {
    "vanilla": VanillaBeamSearch,
    "dbs": DiverseBeamSearch,
    "gumbel_topk": GumbelTopKBeamSearch,
    "hybrid": HybridBeamSearch,
    "level_aware_mix": LevelAwareHybridDecoding,
    "level_aware_mix_grid": LevelAwareHybridDecoding,
    "level_aware_mix_learned": LevelAwareHybridDecoding,
    "sasrec_rerank": SASRecReranker,
}
