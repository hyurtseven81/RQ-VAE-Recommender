from modules.decoding.vanilla import VanillaBeamSearch
from modules.decoding.dbs import DiverseBeamSearch
from modules.decoding.gumbel import GumbelTopKBeamSearch
from modules.decoding.hybrid import HybridBeamSearch
from modules.decoding.level_aware_mix import LevelAwareHybridDecoding, AlphaParams

DECODING_STRATEGIES = {
    "vanilla": VanillaBeamSearch,
    "dbs": DiverseBeamSearch,
    "gumbel_topk": GumbelTopKBeamSearch,
    "hybrid": HybridBeamSearch,
    "level_aware_mix": LevelAwareHybridDecoding,
}
