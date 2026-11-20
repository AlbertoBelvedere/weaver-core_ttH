"""Three event tokens: angular observables, b coefficients, and era flags."""

from pathlib import Path

from weaver.utils.import_tools import import_module

_base = import_module(str(Path(__file__).with_name("cpv_part_2lss_eventtoken_model.py")), name="_cpv_eventtoken_base")
get_loss = _base.get_loss
get_train_fn = _base.get_train_fn
get_evaluate_fn = _base.get_evaluate_fn


def get_model(data_config, **kwargs):
    names = data_config.input_dicts["event_features"]
    angular = tuple(i for i, name in enumerate(names) if name.startswith(("dEta", "dPhi", "dR")))
    b_coefficients = tuple(i for i, name in enumerate(names) if name in {f"b{j}_ttH" for j in range(1, 5)})
    eras = tuple(i for i, name in enumerate(names) if name.startswith("era_is_"))
    groups = (angular, b_coefficients, eras)
    if any(not group for group in groups) or sorted(i for group in groups for i in group) != list(range(len(names))):
        raise ValueError(f"Expected only angular, b1_ttH--b4_ttH, and era event features; got {names}")
    return _base.get_model(data_config, event_feature_groups=groups, **kwargs)
