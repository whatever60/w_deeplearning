from abc import ABCMeta
import warnings

from torch import nn


class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super().__init__()
        self._is_init = False
        self.init_cfg = init_cfg

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights"""
        from ..cnn import initialize

        if not self._is_init:
            if self.init_cfg:
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    if self.init_cfg["type"] == "pretrained":
                        return

            for m in self.children():
                if hasattr(m, "init_weights"):
                    m.init_weights()
            self._is_init = True
        else:
            warnings.warn(
                f"init_weights of {self.__class__.__name__} has been called more than once."
            )

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += "\n"
            s += str(self.init_cfg)
        return s


class Sequential(BaseModule, nn.Sequential):
    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList in openmmlab.
    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)
