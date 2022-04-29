import numpy as np
from .metropolis_fast import MetropolisFastSampler
from netket.sampler.metropolis import MetropolisSampler

def MetropolisFastHopping(hilbert, *args, clusters=None, graph=None, d_max=1, hop_probability=1.0, **kwargs) -> MetropolisFastSampler:
    from .rules.fermionic_hopping import FermionicHoppingRuleWithUpdates

    hoppingrule = FermionicHoppingRuleWithUpdates(hop_probability=hop_probability)

    return MetropolisFastSampler(hilbert, hoppingrule, *args, **kwargs, dtype=np.uint8)

def MetropolisHopping(hilbert, *args, clusters=None, graph=None, d_max=1, hop_probability=1.0, **kwargs) -> MetropolisFastSampler:
    from .rules.fermionic_hopping import FermionicHoppingRule

    hoppingrule = FermionicHoppingRule(hop_probability=hop_probability)

    return MetropolisSampler(hilbert, hoppingrule, *args, **kwargs, dtype=np.uint8)