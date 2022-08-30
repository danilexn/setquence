from setquence.distributed.distribution import Distributed
from setquence.distributed.setquence_dual_distributed import SetQuenceDualDistributed
from setquence.distributed.setquence_epigenome_450k import SetQuenceEpigenome450kDistributed

DISTR_STR = {
    "base": Distributed,
    "setquence_dual_distributed": SetQuenceDualDistributed,
    "setquence_epigenome_450k": SetQuenceEpigenome450kDistributed,
}


def available_distributers():
    return DISTR_STR.keys()


def get_distributer(name: str) -> Distributed:
    try:
        return DISTR_STR[name]
    except KeyError:
        raise KeyError(f"Could not find '{name}' distribution strategy")
