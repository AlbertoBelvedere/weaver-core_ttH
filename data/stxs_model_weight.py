import torch
import torch.nn.functional as F

from weaver.nn.model.ParticleTransformer import ParticleTransformerTagger


def get_model(data_config, **kwargs):
    # class_weights is an option for the loss (get_loss), not for the model
    kwargs.pop("class_weights", None)
    model = ParticleTransformerTagger(
        pf_input_dim=len(data_config.input_dicts["jet_features"]),
        sv_input_dim=len(data_config.input_dicts["lep_features"]),
        num_classes=len(data_config.label_value),
        pair_input_type="pp",
        pair_input_dim=4,
        pair_extra_dim=0,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        fc_params=((256, 0.1, "gelu"), (128, 0.1, "gelu"), (64, 0.1, "gelu")),
        **kwargs,
    )

    input_names = list(data_config.input_names)
    dynamic_axes = {"softmax": {0: "N"}}
    for name in input_names:
        axes = {0: "N"}
        if name.startswith("jet_"):
            axes[2] = "n_jet"
        elif name.startswith("lep_"):
            axes[2] = "n_lep"
        dynamic_axes[name] = axes

    model_info = {
        "input_names": input_names,
        "input_shapes": {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        "output_names": ["softmax"],
        "dynamic_axes": dynamic_axes,
    }
    return model, model_info


class WeightedCrossEntropyLoss(torch.nn.Module):
    """Cross entropy with one weight per class.

    The weights are moved to the device of the model output at every call, because
    Weaver does not move the loss function to the GPU.
    """

    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, logits, target):
        weight = self.class_weights.to(logits.device)
        return F.cross_entropy(logits, target, weight=weight)

    def extra_repr(self):
        return "class_weights=" + str([round(w, 4) for w in self.class_weights.tolist()])


def get_loss(data_config, **kwargs):
    # set on the command line with:  -o class_weights "[w0,w1,w2]"
    # without the option, every class has weight 1 (same as the plain CrossEntropyLoss)
    n_classes = len(data_config.label_value)
    class_weights = [float(w) for w in kwargs.get("class_weights", [1.0] * n_classes)]
    if len(class_weights) != n_classes:
        raise ValueError(f"class_weights needs {n_classes} values, got {class_weights}")
    return WeightedCrossEntropyLoss(class_weights)