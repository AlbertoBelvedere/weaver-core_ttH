import math

import numpy as np
import torch

from weaver.nn.model.ParticleTransformer import ParticleTransformerTagger

# options read by get_loss; they are removed before building the model
LOSS_OPTIONS = ("loss", "huber_delta", "bin_weights", "weight_edges", "weight_file", "target_center", "var_penalty")


def get_model(data_config, **kwargs):
    for option in LOSS_OPTIONS:
        kwargs.pop(option, None)
    # dropout settings, e.g.  -o dropout 0.0  -o block_dropout 0.0  (default 0.1 = previous behaviour)
    #   dropout:       the three fully connected layers before the output
    #   block_dropout: the 8 particle-attention blocks (output, attention weights and feed-forward layer)
    # Dropout has no parameters, so a model trained with any setting is evaluated with the same model file.
    fc_dropout = float(kwargs.pop("dropout", 0.1))
    block_dropout = float(kwargs.pop("block_dropout", 0.1))
    model = ParticleTransformerTagger(
        pf_input_dim=len(data_config.input_dicts["jet_features"]),
        sv_input_dim=len(data_config.input_dicts["lep_features"]),
        num_classes=1,  # regression: one output = predicted log(pT(H))
        pair_input_type="pp",
        pair_input_dim=4,
        pair_extra_dim=0,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        fc_params=((256, fc_dropout, "gelu"), (128, fc_dropout, "gelu"), (64, fc_dropout, "gelu")),
        block_params={"dropout": block_dropout, "attn_dropout": block_dropout, "activation_dropout": block_dropout},
        **kwargs,
    )

    input_names = list(data_config.input_names)
    dynamic_axes = {"output": {0: "N"}}
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
        "output_names": ["output"],
        "dynamic_axes": dynamic_axes,
    }
    return model, model_info


class BinWeightedRegressionLoss(torch.nn.Module):
    """MSE or Huber loss in which every event is weighted by the true pT(H) bin it belongs to.

    Same normalization as the weighted cross entropy of the classifier:
        loss = sum_i w_i * l_i / sum_i w_i
    (l_i = squared error or Huber loss of event i, w_i = weight of the true bin of event i).

    The true bin is found from the label: the label is log(pT) - target_center, so
    log(pT) = label + target_center, and the bin edges (GeV) are compared in log(pT).
    """

    def __init__(self, loss, huber_delta, bin_weights, weight_edges, target_center):
        super().__init__()
        self.loss = loss
        self.huber_delta = huber_delta
        self.target_center = target_center
        self.weight_edges = list(weight_edges)
        self.bin_weights = torch.tensor(bin_weights, dtype=torch.float32)
        # bin edges in units of the label
        self.label_edges = torch.tensor([math.log(e) - target_center for e in weight_edges], dtype=torch.float32)

    def forward(self, preds, label):
        preds = preds.float()
        label = label.float()
        if self.loss == "mse":
            per_event = torch.nn.functional.mse_loss(preds, label, reduction="none")
        elif self.loss == "mae":
            per_event = torch.nn.functional.l1_loss(preds, label, reduction="none")
        else:
            per_event = torch.nn.functional.huber_loss(preds, label, reduction="none", delta=self.huber_delta)
        # bin index: 0 below the first edge, 1 between the first and second edge, ... (same as np.digitize)
        true_bin = torch.bucketize(label, self.label_edges.to(label.device), right=True)
        weight = self.bin_weights.to(label.device)[true_bin]
        return (weight * per_event).sum() / weight.sum()

    def extra_repr(self):
        n = len(self.bin_weights)
        if n <= 10:
            weights = f"weight_edges={self.weight_edges} GeV, bin_weights={[round(w, 4) for w in self.bin_weights.tolist()]}"
        else:  # fine bins from a weight file: summary only
            weights = (f"{n} bins, edges {self.weight_edges[0]:g} to {self.weight_edges[-1]:g} GeV, "
                       f"weights {self.bin_weights.min().item():.4g} to {self.bin_weights.max().item():.4g}")
        return f"loss={self.loss}, huber_delta={self.huber_delta}, {weights}, target_center={self.target_center}"


class VariancePenaltyLoss(torch.nn.Module):
    """Adds a penalty on the difference of the spreads of prediction and label (as in CMS HIG-23-015):

        loss = base_loss + var_penalty * (Var(prediction) - Var(label))**2

    Both variances are computed over the events of one batch (population variance, dividing by the number of
    events), in the units of the label, i.e. log(pT). The base loss is the MSE, Huber or bin-weighted loss.
    The penalty is 0 when the predictions in a batch are spread as much as the true values, and grows when the
    predictions are narrower (or wider) than the truth.
    """

    def __init__(self, base_loss, var_penalty):
        super().__init__()
        self.base_loss = base_loss
        self.var_penalty = var_penalty

    def forward(self, preds, label):
        preds = preds.float()
        label = label.float()
        variance_difference = preds.var(correction=0) - label.var(correction=0)
        return self.base_loss(preds, label) + self.var_penalty * variance_difference.square()

    def extra_repr(self):
        return f"var_penalty={self.var_penalty}"


def get_loss(data_config, **kwargs):
    base_loss = get_base_loss(**kwargs)
    # variance penalty (CMS HIG-23-015):  -o var_penalty 1.0   (default 0 = no penalty, same as before)
    var_penalty = float(kwargs.get("var_penalty", 0.0))
    if var_penalty < 0:
        raise ValueError(f"var_penalty must be >= 0, got {var_penalty}")
    if var_penalty == 0:
        return base_loss
    return VariancePenaltyLoss(base_loss, var_penalty)


def get_base_loss(**kwargs):
    # choose on the command line with:  -o loss "'mse'"   or   -o loss "'huber'" -o huber_delta 0.3
    # mse:   mean squared error of log(pT)
    # huber: squared error for |error| < huber_delta, linear above; large errors (tails) count less
    # mae:   mean absolute error of log(pT)  (-o loss "'mae'")
    #
    # weights per true pT(H) bin (like the class weights of the classifier), two ways:
    #   -o bin_weights "[w0,w1,w2]"   a few bins, edges given by weight_edges in GeV
    #                                 (default [120,200] = the STXS bins: <120, 120-200, >200)
    #   -o weight_file "'data/pt_weights_1gev_cap4.npz'"
    #                                 many bins, written by tools/plot_pt_1gev_weights.py --save-weights
    #                                 (arrays "edges": all bin edges in GeV, "weights": one weight per bin)
    #   target_center is the number subtracted from log(pT) in the label; the launcher sets it from TARGET_CENTER
    # without bin_weights and weight_file every event has weight 1 (same as before)
    loss = kwargs.get("loss", "mse")
    huber_delta = float(kwargs.get("huber_delta", 0.3))
    if loss not in ("mse", "huber", "mae"):
        raise ValueError(f"Unknown loss '{loss}', use 'mse', 'huber' or 'mae'")

    bin_weights = kwargs.get("bin_weights", None)
    weight_edges = kwargs.get("weight_edges", [120.0, 200.0])
    weight_file = kwargs.get("weight_file", None)
    if weight_file is not None:
        if bin_weights is not None:
            raise ValueError("Give either weight_file or bin_weights, not both")
        content = np.load(weight_file)
        # the file has all bin edges including the outer ones; the loss needs only the edges between bins:
        # labels below the first inner edge go to bin 0, labels above the last inner edge to the last bin
        weight_edges = content["edges"][1:-1].tolist()
        bin_weights = content["weights"].tolist()
        if weight_edges and weight_edges[0] <= 0:
            raise ValueError(f"weight_file edges must be > 0 GeV (log is taken), got {weight_edges[0]}")

    if bin_weights is None:
        if loss == "mse":
            return torch.nn.MSELoss()
        if loss == "mae":
            return torch.nn.L1Loss()
        return torch.nn.HuberLoss(delta=huber_delta)

    weight_edges = [float(e) for e in weight_edges]
    bin_weights = [float(w) for w in bin_weights]
    if len(bin_weights) != len(weight_edges) + 1:
        raise ValueError(f"bin_weights needs {len(weight_edges) + 1} values for {len(weight_edges)} edges, "
                         f"got {len(bin_weights)}")
    if any(b <= a for a, b in zip(weight_edges, weight_edges[1:])):
        raise ValueError(f"weight_edges must be increasing, got {weight_edges}")
    return BinWeightedRegressionLoss(loss, huber_delta, bin_weights, weight_edges,
                                     float(kwargs.get("target_center", 0.0)))