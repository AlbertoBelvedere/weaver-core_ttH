"""Event features enter object attention as tokens, without pairwise kinematics."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from weaver.nn.model.ParticleTransformer import ParticleTransformerTagger
from weaver.utils.import_tools import import_module


# Keep the same weighted loss and train/evaluate behavior as the event-MLP model.
_baseline = import_module(str(Path(__file__).with_name("cpv_part_2lss_eventmlp_model.py")), name="_cpv_eventmlp_base")
get_loss = _baseline.get_loss
get_train_fn = _baseline.get_train_fn
get_evaluate_fn = _baseline.get_evaluate_fn


class ObjectPairEmbed(nn.Module):
    """Calculate the physics pair bias for object tokens only."""

    def __init__(self, pair_embed, num_event_tokens=1):
        super().__init__()
        self.pair_embed = pair_embed
        self.num_event_tokens = num_event_tokens

    def forward(self, v, uu=None, mask=None):
        n = self.num_event_tokens
        object_uu = uu[..., :-n, :-n] if uu is not None else None
        object_mask = mask[..., :-n] if mask is not None else None
        object_pairs = self.pair_embed(v[..., :-n], uu=object_uu, mask=object_mask)
        # Event tokens have no physical four-vectors, so their pair bias is zero.
        return F.pad(object_pairs, (0, n, 0, n))


class EventTokenParticleTransformerTagger(ParticleTransformerTagger):
    def __init__(self, num_event_tokens=1, **kwargs):
        super().__init__(**kwargs)
        self.part.pair_embed = ObjectPairEmbed(self.part.pair_embed, num_event_tokens)

    def forward(self, pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask, event_token):
        with torch.no_grad():
            pf_x, pf_v, pf_mask, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
            sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            object_v = torch.cat((pf_v, sv_v), dim=2)
            object_mask = torch.cat((pf_mask, sv_mask), dim=2)
            num_event_tokens = event_token.shape[1]
            event_v = object_v.new_zeros((*object_v.shape[:2], num_event_tokens))
            event_mask = object_mask.new_ones((*object_mask.shape[:2], num_event_tokens))
            v = torch.cat((object_v, event_v), dim=2)
            mask = torch.cat((object_mask, event_mask), dim=2)

        x = torch.cat((self.pf_embed(pf_x), self.sv_embed(sv_x), event_token), dim=1)
        return self.part(x, v, mask)


class CPVParticleTransformerEventTokenTagger(nn.Module):
    def __init__(
        self,
        jet_input_dim,
        lep_input_dim,
        event_input_dim,
        num_classes,
        event_feature_groups=None,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params=None,
        event_embed_dims=(64, 64),
        event_dropout=0.1,
        fc_params=((128, 0.1), (64, 0.1)),
        activation="gelu",
        for_inference=False,
        use_amp=False,
        compile_model=False,
        **kwargs,
    ):
        super().__init__()
        self.for_inference = for_inference
        self.event_feature_groups = event_feature_groups or (tuple(range(event_input_dim)),)
        if sorted(i for group in self.event_feature_groups for i in group) != list(range(event_input_dim)):
            raise ValueError("Event feature groups must partition every event input exactly once")
        if any(not group for group in self.event_feature_groups):
            raise ValueError("Event feature groups must be nonempty")
        embed_dim = embed_dims[-1]
        self.tagger = EventTokenParticleTransformerTagger(
            num_event_tokens=len(self.event_feature_groups),
            pf_input_dim=jet_input_dim,
            sv_input_dim=lep_input_dim,
            num_classes=None,
            pair_input_type="pp",
            pair_input_dim=4,
            pair_extra_dim=0,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            block_params=block_params,
            cls_block_params=cls_block_params,
            fc_params=None,
            activation=activation,
            for_inference=for_inference,
            use_amp=use_amp,
            compile_model=compile_model,
        )

        self.event_embeds = nn.ModuleList()
        for group in self.event_feature_groups:
            event_layers = []
            event_dim = len(group)
            for out_dim in event_embed_dims:
                event_layers.extend((
                    nn.Linear(event_dim, out_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(event_dropout),
                ))
                event_dim = out_dim
            if len(self.event_feature_groups) == 1:
                # Retain the original parameter names for existing checkpoints.
                self.event_embed = nn.Sequential(*event_layers)
                self.event_to_token = nn.Linear(event_dim, embed_dim)
            else:
                event_layers.append(nn.Linear(event_dim, embed_dim))
                self.event_embeds.append(nn.Sequential(*event_layers))

        classifier_layers = []
        in_dim = embed_dim
        for out_dim, dropout in fc_params:
            classifier_layers.extend((
                nn.Linear(in_dim, out_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
            ))
            in_dim = out_dim
        classifier_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"tagger.part.cls_token"}

    def forward(
        self,
        jet_features,
        jet_vectors=None,
        jet_mask=None,
        lep_features=None,
        lep_vectors=None,
        lep_mask=None,
        event_features=None,
        loss_weight=None,
    ):
        if event_features.ndim == 3 and event_features.shape[-1] == 1:
            event_features = event_features.squeeze(-1)
        event_features = torch.nan_to_num(event_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if len(self.event_feature_groups) == 1:
            event_token = self.event_to_token(self.event_embed(event_features)).unsqueeze(1)
        else:
            event_token = torch.stack(
                [embed(event_features[:, group]) for embed, group in zip(self.event_embeds, self.event_feature_groups)],
                dim=1,
            )
        event_repr = self.tagger(
            jet_features, jet_vectors, jet_mask,
            lep_features, lep_vectors, lep_mask, event_token,
        )
        logits = self.classifier(event_repr)
        if self.for_inference:
            logits = torch.softmax(logits, dim=1)
        return logits


def get_model(data_config, **kwargs):
    model = CPVParticleTransformerEventTokenTagger(
        jet_input_dim=len(data_config.input_dicts["jet_features"]),
        lep_input_dim=len(data_config.input_dicts["lep_features"]),
        event_input_dim=len(data_config.input_dicts["event_features"]),
        num_classes=len(data_config.label_value),
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
