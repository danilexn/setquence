from typing import Union

from torch import nn as nn

from setquence.base.base import BaseModule
from setquence.base.config import Config, Environment
from setquence.base.encoder import SetQuenceBERTLarge
from setquence.distributed.distribution import NoDistributed
from setquence.models import SetQuenceDual
from setquence.models.setquence_dual import SetQuencePool


class SetQuenceDualLarge(SetQuenceDual):
    model_name = "setquence_dual_large"

    def __init__(
        self,
        config: Config,
        encoder: Union[BaseModule, nn.Module] = SetQuenceBERTLarge,
        decoder: Union[BaseModule, nn.Module] = SetQuencePool,
        env: Environment = None,
    ):
        super().__init__(config, encoder, decoder, env)

    def _dist_alloc(self):
        """
        This is overriden wrt SetQuenceDual to avoid the DDP error
        when trying to update self.encoder (BERT) pooler modules
        """
        self.encoder = (
            self.distribution_strategy(self.encoder, device_ids=self.environment.local_device_ids)
            if not isinstance(self.encoder, self.distribution_strategy) and self.distribution_strategy != NoDistributed
            else self.encoder
        )

        self.decoder = (
            self.distribution_strategy(self.decoder, device_ids=self.environment.local_device_ids)
            if not isinstance(self.decoder, self.distribution_strategy) and self.distribution_strategy != NoDistributed
            else self.decoder
        )
