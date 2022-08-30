from types import SimpleNamespace

from torch.optim import SGD, Adam

from setquence.base.optimizer import SAM

try:
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from torch.optim._multi_tensor.adam import Adam as Adam_multitensor
    from torch.optim._multi_tensor.sgd import SGD as SGD_multitensor

    ZERO_OPTIMIZER = True
    OPTIMIZERS_STR = {
        "adam": Adam,
        "sgd": SGD,
        "sam": SAM,
        "adam_multitensor": Adam_multitensor,
        "sgd_multitensor": SGD_multitensor,
    }
except ImportError:
    ZERO_OPTIMIZER = False
    OPTIMIZERS_STR = {"adam": Adam, "sgd": SGD}


def available_optimizers():
    return OPTIMIZERS_STR.keys()


def get_optimizer(model, optimizer: SimpleNamespace):
    if optimizer.distributed and ZERO_OPTIMIZER:
        _ret_opt = ZeroRedundancyOptimizer(
            model.parameters(optimizer=False),
            optimizer_class=OPTIMIZERS_STR[optimizer.name],
            parameters_as_bucket_view=True,
            **ns_to_dict(optimizer.config),
        )
    else:
        _ret_opt = OPTIMIZERS_STR[optimizer.name](model.parameters(optimizer=True), **ns_to_dict(optimizer.config))
        if model.model_basetype == "BaseDual":
            for g in _ret_opt.param_groups:
                g["lr"] = optimizer.config.lr * g["lr"]

    return _ret_opt


def ns_to_dict(namespace):
    _ret_dict = namespace.__dict__
    for k, v in _ret_dict.items():
        if isinstance(v, SimpleNamespace):
            _ret_dict[k] = ns_to_dict(v)
    return _ret_dict


def dict_to_namespace(d):
    x = SimpleNamespace()
    _ = [setattr(x, k, dict_to_namespace(v)) if isinstance(v, dict) else setattr(x, k, v) for k, v in d.items()]
    return x
