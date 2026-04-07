from modules.decoding.vanilla import VanillaBeamSearch
from modules.decoding.dbs import DiverseBeamSearch
from modules.decoding.gumbel import GumbelTopKBeamSearch
from modules.decoding.hybrid import HybridBeamSearch

DECODING_STRATEGIES = {
    "vanilla": VanillaBeamSearch,
    "dbs": DiverseBeamSearch,
    "gumbel_topk": GumbelTopKBeamSearch,
    "hybrid": HybridBeamSearch,
}
