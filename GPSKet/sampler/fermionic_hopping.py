import numpy as np
from .metropolis_fast import MetropolisFastSampler
from netket.sampler.metropolis import MetropolisSampler


def MetropolisFastHopping(
    hilbert,
    *args,
    clusters=None,
    graph=None,
    hop_probability=1.0,
    transition_probs=None,
    **kwargs
) -> MetropolisFastSampler:
    from .rules.fermionic_hopping import FermionicHoppingRuleWithUpdates

    if transition_probs is not None:
        assert not np.any(transition_probs - transition_probs.T)

    hoppingrule = FermionicHoppingRuleWithUpdates(
        hop_probability=hop_probability, transition_probs=transition_probs
    )

    return MetropolisFastSampler(hilbert, hoppingrule, *args, dtype=np.uint8, **kwargs)


def MetropolisHopping(
    hilbert,
    *args,
    clusters=None,
    graph=None,
    hop_probability=1.0,
    transition_probs=None,
    **kwargs
) -> MetropolisSampler:
    from .rules.fermionic_hopping import FermionicHoppingRule

    if transition_probs is not None:
        assert not np.any(transition_probs - transition_probs.T)

    hoppingrule = FermionicHoppingRule(
        hop_probability=hop_probability, transition_probs=transition_probs
    )

    return MetropolisSampler(hilbert, hoppingrule, *args, dtype=np.uint8, **kwargs)
