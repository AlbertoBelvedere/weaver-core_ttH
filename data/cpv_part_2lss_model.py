import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import time
import tqdm
import numpy as np

from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score

from weaver.nn.model.ParticleTransformer import ParticleTransformerTagger
from weaver.utils.logger import _logger


def _concat(arrays):
    if not arrays:
        return arrays
    try:
        import awkward as ak

        if isinstance(arrays[0], ak.Array):
            return ak.concatenate(arrays)
    except Exception:
        pass
    return np.concatenate(arrays)


def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    return label


def _flatten_preds(model_output, label=None, mask=None, label_axis=1):
    if isinstance(model_output, tuple):
        if len(model_output) == 2:
            preds, mask = model_output
        elif len(model_output) == 3:
            preds, label, mask = model_output
        else:
            raise RuntimeError("Unexpected model output tuple length: %d" % len(model_output))
    else:
        preds = model_output

    if preds.ndim > 2:
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]

    if label is not None:
        label = _flatten_label(label, mask)
    return preds, label, mask


def get_autocast_config(args):
    enable_autocast = args.use_amp
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else None
    return enable_autocast, autocast_dtype


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class CPVParticleTransformerTagger(nn.Module):
    def __init__(
        self,
        jet_input_dim,
        lep_input_dim,
        event_input_dim,
        num_classes,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params=None,
        fc_params=((256, 0.1), (128, 0.1), (64, 0.1)),
        activation="gelu",
        for_inference=False,
        use_amp=False,
        compile_model=False,
        **kwargs,
    ):
        super().__init__()
        self.for_inference = for_inference

        self.tagger = ParticleTransformerTagger(
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

        embed_dim = embed_dims[-1] if len(embed_dims) else jet_input_dim
        classifier_layers = []
        in_dim = embed_dim + event_input_dim
        for out_dim, dropout in fc_params:
            classifier_layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
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
        event_repr = self.tagger(jet_features, jet_vectors, jet_mask, lep_features, lep_vectors, lep_mask)

        if event_features is not None:
            if event_features.ndim == 3 and event_features.shape[-1] == 1:
                event_features = event_features.squeeze(-1)
            event_features = torch.nan_to_num(event_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
            event_repr = torch.cat([event_repr, event_features], dim=1)

        logits = self.classifier(event_repr)
        if self.for_inference:
            logits = torch.softmax(logits, dim=1)
        return logits


class ClassBalancedEventWeightedCrossEntropy(nn.Module):
    def forward(self, logits, label, weight=None):
        per_event_loss = F.cross_entropy(logits, label, reduction="none")
        if weight is None:
            weight = torch.ones_like(per_event_loss)
        else:
            weight = torch.nan_to_num(weight.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
            weight = torch.clamp(weight, min=0.0)

        class_losses = []
        for cls in torch.unique(label.detach()):
            class_mask = label == cls
            class_weight = weight[class_mask]
            class_weight_sum = class_weight.sum()
            if class_weight_sum > 0:
                class_loss = (per_event_loss[class_mask] * class_weight).sum() / class_weight_sum
                class_losses.append(class_loss)

        if not class_losses:
            return per_event_loss.mean()

        return torch.stack(class_losses).mean()


def _batch_loss_weight(X, dev):
    weight = X["loss_weight"].to(dev).float()
    if weight.ndim == 3 and weight.shape[-1] == 1:
        weight = weight.squeeze(-1)
    if weight.ndim == 2 and weight.shape[-1] == 1:
        weight = weight.squeeze(-1)
    if weight.ndim > 1:
        weight = weight.reshape(weight.shape[0], -1)[:, 0]
    return torch.nan_to_num(weight.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def train_weighted_classification(
    model,
    loss_func,
    opt,
    scheduler,
    train_loader,
    dev,
    epoch,
    steps_per_epoch=None,
    grad_scaler=None,
    tb_helper=None,
    extra_args=None,
):
    model.train()

    data_config = train_loader.dataset.config
    clip_grad_norm = getattr(opt, "_clip_grad_norm", float("inf"))

    enable_autocast, autocast_dtype = (
        get_autocast_config(extra_args["args"]) if extra_args and "args" in extra_args else (False, None)
    )

    label_counter = Counter()
    weighted_label_sum = Counter()
    total_loss_num = 0.0
    total_weight = 0.0
    total_correct_weight = 0.0
    total_correct = 0
    entry_count = 0
    count = 0
    num_batches = 0
    grad_norm_max_val = 0.0

    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev)
            loss_weight = _batch_loss_weight(X, dev)
            entry_count += label.shape[0]
            try:
                mask = y[data_config.label_names[0] + "_mask"].bool().to(dev)
            except KeyError:
                mask = None

            if tb_helper:
                tb_helper.global_step += 1
            opt.zero_grad()
            with torch.autocast("cuda", enabled=enable_autocast, dtype=autocast_dtype):
                model_output = model(*inputs)
                logits, label, mask = _flatten_preds(model_output, label=label, mask=mask)
                if mask is not None:
                    loss_weight = loss_weight[mask.view(-1)]
                loss = loss_func(logits, label, loss_weight)

            if grad_scaler is None:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm).item()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm).item()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, "_update_per_step", False):
                scheduler.step()

            _, preds = logits.max(1)
            correct = preds == label
            batch_weight_sum = loss_weight.sum().item()
            batch_weighted_correct = (loss_weight * correct.float()).sum().item()
            batch_loss_num = loss.item()

            labels_np = label.numpy(force=True)
            weights_np = loss_weight.numpy(force=True)
            label_counter.update(labels_np)
            for cls in np.unique(labels_np):
                weighted_label_sum[int(cls)] += float(weights_np[labels_np == cls].sum())

            num_batches += 1
            count += label.shape[0]
            total_weight += batch_weight_sum
            total_loss_num += batch_loss_num
            total_correct_weight += batch_weighted_correct
            total_correct += correct.sum().item()
            grad_norm_max_val = max(grad_norm_max_val, grad_norm)

            avg_loss = total_loss_num / max(num_batches, 1)
            avg_wacc = total_correct_weight / max(total_weight, 1.0e-12)
            tq.set_postfix(
                {
                    "lr": "%.2e" % scheduler.get_last_lr()[0] if scheduler else opt.defaults["lr"],
                    "Loss": "%.5f" % loss.item(),
                    "AvgLoss": "%.5f" % avg_loss,
                    "WAcc": "%.5f" % (batch_weighted_correct / max(batch_weight_sum, 1.0e-12)),
                    "AvgWAcc": "%.5f" % avg_wacc,
                }
            )

            if tb_helper:
                tb_helper.write_scalars(
                    [
                        ("Loss/train", loss.item(), tb_helper.global_step),
                        ("Acc/train_weighted", batch_weighted_correct / max(batch_weight_sum, 1.0e-12), tb_helper.global_step),
                        ("Acc/train_unweighted", correct.float().mean().item(), tb_helper.global_step),
                    ]
                )

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info("Processed %d entries in total (avg. speed %.1f entries/s)" % (entry_count, entry_count / time_diff))
    _logger.info(
        "Train AvgLoss: %.5f, WeightedAvgAcc: %.5f, RawAvgAcc: %.5f"
        % (total_loss_num / max(num_batches, 1), total_correct_weight / max(total_weight, 1.0e-12), total_correct / count)
    )
    _logger.info("Train class distribution: \n    %s", str(sorted(label_counter.items())))
    _logger.info("Train weighted class sums: \n    %s", str(sorted(weighted_label_sum.items())))
    _logger.info("Max Grad Norm: %.5f" % (grad_norm_max_val,))
    _logger.info("Max CUDA memory: %.1f MB" % (torch.cuda.max_memory_allocated(dev) / 1024.0**2,))

    if tb_helper:
        tb_helper.write_scalars(
            [
                ("Loss/train (epoch)", total_loss_num / max(num_batches, 1), epoch),
                ("Acc/train_weighted (epoch)", total_correct_weight / max(total_weight, 1.0e-12), epoch),
                ("Acc/train_unweighted (epoch)", total_correct / count, epoch),
            ]
        )
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, "_update_per_step", False):
        scheduler.step()


def evaluate_weighted_classification(
    model,
    test_loader,
    dev,
    epoch,
    for_training=True,
    loss_func=None,
    steps_per_epoch=None,
    eval_metrics=None,
    tb_helper=None,
    extra_args=None,
):
    model.eval()

    data_config = test_loader.dataset.config
    enable_autocast, autocast_dtype = (
        get_autocast_config(extra_args["args"]) if extra_args and "args" in extra_args else (False, None)
    )

    label_counter = Counter()
    weighted_label_sum = Counter()
    total_loss_num = 0.0
    total_weight = 0.0
    total_correct_weight = 0.0
    total_correct = 0
    entry_count = 0
    count = 0
    num_batches = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    weights = []

    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                loss_weight = AllGather.apply(_batch_loss_weight(X, dev))
                y = {k: AllGather.apply(v.to(dev)) for k, v in y.items()}
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                try:
                    mask = y[data_config.label_names[0] + "_mask"].bool().to(dev)
                except KeyError:
                    mask = None

                with torch.autocast("cuda", enabled=enable_autocast, dtype=autocast_dtype):
                    model_output = AllGather.apply(model(*inputs))
                logits, label, mask = _flatten_preds(model_output, label=label, mask=mask)
                if mask is not None:
                    loss_weight = loss_weight[mask.view(-1)]
                loss = loss_func(logits, label, loss_weight) if loss_func is not None else torch.zeros((), device=dev)

                score = torch.softmax(logits.float(), dim=1)
                _, preds = logits.max(1)
                correct = preds == label

                batch_weight_sum = loss_weight.sum().item()
                batch_weighted_correct = (loss_weight * correct.float()).sum().item()
                batch_loss_num = loss.item()

                scores.append(score.numpy(force=True))
                weights.append(loss_weight.numpy(force=True))
                if mask is not None:
                    mask = mask.cpu()
                for k, v in y.items():
                    labels[k].append(_flatten_label(v, mask).numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                labels_np = label.numpy(force=True)
                weights_np = loss_weight.numpy(force=True)
                label_counter.update(labels_np)
                for cls in np.unique(labels_np):
                    weighted_label_sum[int(cls)] += float(weights_np[labels_np == cls].sum())

                num_batches += 1
                count += label.shape[0]
                total_weight += batch_weight_sum
                total_loss_num += batch_loss_num
                total_correct_weight += batch_weighted_correct
                total_correct += correct.sum().item()

                tq.set_postfix(
                    {
                        "Loss": "%.5f" % loss.item(),
                        "AvgLoss": "%.5f" % (total_loss_num / max(num_batches, 1)),
                        "WAcc": "%.5f" % (batch_weighted_correct / max(batch_weight_sum, 1.0e-12)),
                        "AvgWAcc": "%.5f" % (total_correct_weight / max(total_weight, 1.0e-12)),
                    }
                )

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    weighted_acc = total_correct_weight / max(total_weight, 1.0e-12)
    raw_acc = total_correct / count
    avg_loss = total_loss_num / max(num_batches, 1)
    _logger.info("Processed %d entries in total (avg. speed %.1f entries/s)" % (entry_count, entry_count / time_diff))
    _logger.info("Eval AvgLoss: %.5f, WeightedAvgAcc: %.5f, RawAvgAcc: %.5f" % (avg_loss, weighted_acc, raw_acc))
    _logger.info("Evaluation class distribution: \n    %s", str(sorted(label_counter.items())))
    _logger.info("Evaluation weighted class sums: \n    %s", str(sorted(weighted_label_sum.items())))

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    weights = np.concatenate(weights)
    truth = labels[data_config.label_names[0]]
    pred = scores.argmax(1)

    try:
        weighted_auc = roc_auc_score(truth, scores[:, 1], sample_weight=weights)
    except Exception as err:
        weighted_auc = None
        _logger.warning("Cannot compute weighted roc_auc_score: %s", str(err))

    try:
        weighted_confusion = confusion_matrix(truth, pred, sample_weight=weights, normalize="true")
    except Exception as err:
        weighted_confusion = None
        _logger.warning("Cannot compute weighted confusion_matrix: %s", str(err))

    _logger.info(
        "Weighted evaluation metrics: \n%s",
        "\n".join(
            [
                "    - weighted_roc_auc_score: \n%s" % str(weighted_auc),
                "    - weighted_confusion_matrix: \n%s" % str(weighted_confusion),
            ]
        ),
    )

    if tb_helper:
        tb_mode = "eval" if for_training else "test"
        tb_helper.write_scalars(
            [
                ("Loss/%s (epoch)" % tb_mode, avg_loss, epoch),
                ("Acc/%s_weighted (epoch)" % tb_mode, weighted_acc, epoch),
                ("Acc/%s_unweighted (epoch)" % tb_mode, raw_acc, epoch),
            ]
        )
        if weighted_auc is not None:
            tb_helper.write_scalars([("AUC/%s_weighted (epoch)" % tb_mode, weighted_auc, epoch)])

    metric_value = weighted_auc if weighted_auc is not None else weighted_acc
    if for_training:
        return metric_value

    observers = {k: _concat(v) for k, v in observers.items()}
    return metric_value, scores, labels, observers


def get_model(data_config, **kwargs):
    jet_input_dim = len(data_config.input_dicts["jet_features"])
    lep_input_dim = len(data_config.input_dicts["lep_features"])
    event_input_dim = len(data_config.input_dicts["event_features"])
    num_classes = len(data_config.label_value)

    model = CPVParticleTransformerTagger(
        jet_input_dim=jet_input_dim,
        lep_input_dim=lep_input_dim,
        event_input_dim=event_input_dim,
        num_classes=num_classes,
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


def get_loss(data_config, **kwargs):
    return ClassBalancedEventWeightedCrossEntropy()


def get_train_fn(data_config, **kwargs):
    return train_weighted_classification


def get_evaluate_fn(data_config, **kwargs):
    return evaluate_weighted_classification
