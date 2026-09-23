from weaver.nn.model.ParticleTransformer import ParticleTransformerTagger


def get_model(data_config, **kwargs):
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
