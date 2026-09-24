#!/usr/bin/env python3
"""Teaching implementation: fine-tune Laya on typed decisions, then calibrate.

This is independently written example code, NOT the official Laya trainer.
It uses the public checkpoint loader and the currently exposed model/tokenizer
attributes. Install matching Laya source and dependencies before running.

Input: JSONL case records with id, optional group_id, state, questions, gold.
Keep train/valid/calib/test disjoint at the case/customer/conversation level.
The script validates IDs, group IDs and identical serialized states across splits.

Real checkpoint/GPU training has NOT been executed for this article. --self-test
checks the loss, gradients, calibration and data-validation logic on CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit("tqdm is required for training progress bars; install it with: python -m pip install tqdm") from exc

TYPE_IDS = {"choice": 0, "score": 1, "noul": 2}
TEMP_LO, TEMP_HI = 0.5, 5.0


class TqdmLoggingHandler(logging.Handler):
    """Keep timestamped console logs from overwriting the active progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def configure_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("laya_finetune")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def json_field(value: Any) -> Any:
    """Accept both nested JSON objects and dataset columns containing JSON text."""
    return json.loads(value) if isinstance(value, str) else value


def read_cases(path: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "state", "questions", "gold"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"{path}:{line_no}: missing fields {required}")
            row["id"] = str(row["id"])
            row["group_id"] = str(row.get("group_id", row["id"]))
            if row["id"] in seen:
                raise ValueError(f"Duplicate case id in {path}: {row['id']}")
            seen.add(row["id"])
            for field in ("questions", "gold"):
                row[field] = json_field(row[field])
                if not isinstance(row[field], dict) or not row[field]:
                    raise ValueError(f"{path}:{line_no}: {field} must be a nonempty object")
            # Raw text states are legitimate; only decode apparent JSON objects/lists.
            if isinstance(row["state"], str) and row["state"].lstrip().startswith(("{", "[")):
                try:
                    row["state"] = json.loads(row["state"])
                except json.JSONDecodeError:
                    pass
            cases.append(row)
    if not cases:
        raise ValueError(f"No cases in {path}")
    return cases


def check_split_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    owners: dict[tuple[str, str], str] = {}
    for name, cases in splits.items():
        for row in cases:
            state = json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
            keys = (("id", row["id"]), ("group", row["group_id"]),
                    ("state", hashlib.sha256(state.encode()).hexdigest()))
            for key in keys:
                prior = owners.setdefault(key, name)
                if prior != name:
                    raise ValueError(f"Data leakage: {key[0]} is shared by {prior} and {name}")


def ordered_target(question: dict[str, Any], gold: dict[str, Any]) -> list[float]:
    qtype = question.get("type")
    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice requires a criteria object with at least 2 options")
        keys = list(criteria)  # Exact insertion order must match option rendering.
    elif qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score requires an ordered criteria list of length >= 2")
        keys = [str(i) for i in range(len(criteria))]
    elif qtype == "noul":
        keys = ["false", "true"]
    else:
        raise ValueError(f"Unknown question type: {qtype}")
    probs = gold.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(keys):
        raise ValueError(f"Gold probability keys must exactly match {keys}")
    values = [float(probs[key]) for key in keys]
    if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise ValueError("Targets must be finite, nonnegative and have a positive sum")
    total = sum(values)
    return [v / total for v in values]


def encode_cases(cases, tokenizer, max_len: int, head_max_len: int):
    from laya.common import build_sequence
    items = []
    for row in tqdm(cases, desc="Encoding", unit="case", dynamic_ncols=True, leave=False):
        for qid, question in row["questions"].items():
            if qid not in row["gold"]:
                raise ValueError(f"Missing gold for {row['id']}/{qid}")
            if not question.get("instructions"):
                raise ValueError(f"Missing instructions for {row['id']}/{qid}")
            target = ordered_target(question, row["gold"][qid])
            internal = {"t": question["type"], "ins": question["instructions"],
                        "crit": question.get("criteria", {})}
            if "labels" in question:
                internal["labels"] = question["labels"]
            ids, markers = build_sequence(tokenizer, row["state"], internal,
                                          max_len=max_len, head_max_len=head_max_len)
            if len(markers) != len(target):
                raise ValueError(f"Options truncated for {row['id']}/{qid}; increase token budget")
            items.append({"ids": ids, "markers": markers,
                          "qtype": TYPE_IDS[question["type"]], "target": target})
    if not items:
        raise ValueError("No encoded questions")
    return items


def collate(items, pad_id: int):
    batch, length = len(items), max(len(x["ids"]) for x in items)
    kmax = max(len(x["target"]) for x in items)
    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    positions = torch.zeros((batch, kmax), dtype=torch.long)
    mask = torch.zeros((batch, kmax), dtype=torch.bool)
    target = torch.zeros((batch, kmax), dtype=torch.float32)
    for i, item in enumerate(items):
        n, k = len(item["ids"]), len(item["target"])
        ids[i, :n] = torch.tensor(item["ids"])
        attention[i, :n] = 1
        positions[i, :k] = torch.tensor(item["markers"])
        mask[i, :k] = True
        target[i, :k] = torch.tensor(item["target"])
    return {"input_ids": ids, "attention_mask": attention, "marker_pos": positions,
            "marker_mask": mask, "qtype": torch.tensor([x["qtype"] for x in items]),
            "target": target}


def forward_model(model, batch):
    return model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                 batch["marker_mask"], batch["qtype"])[0].float()


def distribution_reward(probs, target, mask, qtype):
    """Log + 0.75*spherical - ordinal RPS. The log floor follows the recipe.

    Numerical clipping means this implemented reward should NOT be advertised as
    a proof that finite-data training produces perfectly calibrated probabilities.
    """
    probs = probs * mask
    log_score = (target * probs.clamp_min(1e-12).log().clamp_min(-9.21)).sum(-1)
    spherical = (target * probs).sum(-1) / probs.square().sum(-1).sqrt().clamp_min(1e-9)
    count = mask.sum(-1)
    cdf_error = (probs.cumsum(-1) - target.cumsum(-1)).square()
    indices = torch.arange(probs.size(-1), device=probs.device)
    ordinal_mask = indices < (count - 1).unsqueeze(-1)
    rps = (cdf_error * ordinal_mask).sum(-1) / (count - 1).clamp_min(1)
    return log_score + 0.75 * spherical - torch.where(qtype == 1, rps, 0.0)


def decision_loss(logits, target, mask, qtype, sigma: float, group: int, rl_weight: float):
    logits = logits.masked_fill(~mask, -1e4).float()
    ce = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
    if rl_weight == 0:
        return ce
    b, k = logits.shape
    expanded_mask = mask[:, None, :].expand(b, group, k)
    noise = torch.randn((b, group, k), device=logits.device) * expanded_mask
    noise -= noise.sum(-1, keepdim=True) / expanded_mask.sum(-1, keepdim=True) * expanded_mask
    # A REINFORCE sample must be held fixed when evaluating its log density.
    actions = (logits.detach()[:, None, :] + sigma * noise).masked_fill(~expanded_mask, -1e4)
    probs = actions.softmax(-1)
    with torch.no_grad():
        reward = distribution_reward(probs, target[:, None, :].expand_as(probs),
                                     expanded_mask, qtype[:, None].expand(b, group))
        advantage = reward - reward.mean(-1, keepdim=True)
        advantage /= advantage.std(unbiased=False).clamp_min(1e-6)
    # Density on the zero-sum logit subspace; parameter-independent constants omitted.
    delta = (actions - logits[:, None, :]) * expanded_mask
    log_density = -delta.square().sum(-1) / (2 * sigma * sigma)
    policy = -(advantage * log_density).mean()
    return ce + rl_weight * policy


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device, desc: str):
    model.eval()
    records = []
    progress = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False)
    for batch in progress:
        moved = {k: v.to(device) for k, v in batch.items()}
        context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with context:
            logits = forward_model(model, moved)
        for i, k in enumerate(batch["marker_mask"].sum(-1).tolist()):
            records.append((int(batch["qtype"][i]), logits[i, :k].float().cpu(),
                            batch["target"][i, :k].float().cpu()))
    return records


def fit_temperatures(records) -> list[float]:
    """Independent bounded grid-search implementation; no training-set fitting."""
    temperatures = []
    grid = torch.exp(torch.linspace(math.log(TEMP_LO), math.log(TEMP_HI), 256)).clamp(TEMP_LO, TEMP_HI)
    grid = torch.sort(torch.cat([grid, torch.ones(1)])).values
    for qtype in range(3):
        rows = [(z, y) for t, z, y in records if t == qtype]
        if not rows:
            temperatures.append(1.0)
            print(f"Warning: no calibration data for question type {qtype}; keeping T=1", file=sys.stderr)
            continue
        scores = torch.zeros_like(grid)
        for z, target in rows:
            scores += -(F.log_softmax(z[None, :] / grid[:, None], -1) * target).sum(-1)
        temperatures.append(float(grid[int(scores.argmin())]))
    return temperatures


def metrics(records, temperatures=None):
    temperatures = temperatures or [1.0, 1.0, 1.0]
    nll, brier, hits, confidence, score_mae = [], [], [], [], []
    for qtype, z, target in records:
        logp = F.log_softmax(z / temperatures[qtype], -1)
        p = logp.exp()
        nll.append(float(-(target * logp).sum()))
        brier.append(float((p - target).square().sum()))
        hits.append(float(p.argmax() == target.argmax()))
        confidence.append(float(p.max()))
        if qtype == 1:
            levels = torch.arange(len(p), dtype=p.dtype)
            score_mae.append(float(abs((p * levels).sum() - (target * levels).sum())))
    if not hits:
        raise ValueError("Empty evaluation set")
    conf, correct = torch.tensor(confidence), torch.tensor(hits)
    bin_ids = (conf * 15).long().clamp(max=14)
    ece = 0.0
    for index in range(15):
        chosen = bin_ids == index
        if chosen.any():
            ece += float(chosen.float().mean() * abs(conf[chosen].mean() - correct[chosen].mean()))
    return {"questions": len(hits), "accuracy_vs_target_argmax": sum(hits) / len(hits),
            "soft_target_nll": sum(nll) / len(nll), "squared_error_to_target": sum(brier) / len(brier),
            "ece_vs_target_argmax": ece,
            "score_expected_value_mae": sum(score_mae) / len(score_mae) if score_mae else None}


def save_checkpoint(model, tokenizer, config, directory: Path):
    from safetensors.torch import save_file
    directory.mkdir(parents=True, exist_ok=True)
    weights = {name: value.detach().cpu().contiguous().clone()
               for name, value in model.state_dict().items()}
    save_file(weights, str(directory / "model.safetensors"))
    model.encoder.config.save_pretrained(str(directory / "encoder"))
    tokenizer.save_pretrained(str(directory / "tokenizer"))
    (directory / "rl_agent_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def self_test():
    torch.manual_seed(19)
    logits = torch.tensor([[1.0, -0.2, 0.1], [-0.1, 0.7, -1e4]], requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    target = torch.tensor([[0., 1., 0.], [0.25, 0.75, 0.]])
    loss = decision_loss(logits, target, mask, torch.tensor([1, 2]), 0.3, 4, 1.0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0 and logits.grad[1, 2] == 0
    assert ordered_target({"type": "noul"}, {"probabilities": {"true": .7, "false": .3}}) == [.3, .7]
    records = [(t, torch.tensor([4., 0.]), torch.tensor([.7, .3])) for t in range(3)]
    temperatures = fit_temperatures(records)
    assert metrics(records, temperatures)["soft_target_nll"] < metrics(records)["soft_target_nll"]
    assert all(TEMP_LO <= t <= TEMP_HI for t in temperatures)
    example = {"id": "one", "group_id": "group", "state": {"text": "same"}}
    try:
        check_split_leakage({"train": [example], "calib": [example]})
    except ValueError:
        pass
    else:
        raise AssertionError("Leakage detection failed")
    print("PASS: target order, finite loss/gradients, masked gradients, temperature search, leakage detection")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", default="convaiinnovations/laya")
    for name in ("train", "valid", "calib", "test"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--out", default="./laya-business")
    parser.add_argument("--log-file", help="Training log path (default: <out>/training.log)")
    parser.add_argument("--log-every", type=int, default=500, help="Write a batch summary every N batches")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--head-max-len", type=int, default=192)
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--rl-weight", type=float, default=1.0)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.train, args.valid, args.calib)):
        parser.error("--train, --valid and --calib are required; --test is strongly recommended")
    if min(args.epochs, args.batch_size, args.grad_accum, args.log_every) < 1 or args.group < 2:
        parser.error("epochs/batch-size/grad-accum/log-every must be positive; group must be >= 2")
    if min(args.sigma_start, args.sigma_end, args.lr_encoder, args.lr_head) <= 0 or args.rl_weight < 0:
        parser.error("Learning rates and sigmas must be positive; rl-weight must be nonnegative")
    if not 16 <= args.head_max_len < args.max_len:
        parser.error("Require 16 <= head-max-len < max-len")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is nonempty; use a new --out: {output}")
    output.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else output / "training.log"
    logger = configure_logger(log_file)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("This training example supports CPU/CUDA only (MPS inference is a separate SDK capability)")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    logger.info("Starting training; log_file=%s", log_file.resolve())
    logger.info("Arguments: %s", json.dumps(vars(args), ensure_ascii=False, default=str))
    if device.type == "cuda":
        logger.info("Device: %s (%s)", device, torch.cuda.get_device_name(device))
    else:
        logger.info("Device: %s", device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = {name: read_cases(path) for name in ("train", "valid", "calib", "test")
              if (path := getattr(args, name))}
    check_split_leakage(splits)
    for name, cases in splits.items():
        question_count = sum(len(row["questions"]) for row in cases)
        logger.info("Loaded %s: %d cases, %d questions", name, len(cases), question_count)
    import laya
    from safetensors.torch import load_file
    logger.info("Loading Laya checkpoint: %s", args.model)
    agent = laya.load(args.model, device=str(device))
    logger.info("Checkpoint loaded")
    if not all(hasattr(agent, name) for name in ("model", "tok", "cfg")):
        raise RuntimeError("Laya SDK attributes changed; review this example against your installed revision")
    model, tokenizer, config = agent.model.float(), agent.tok, dict(agent.cfg)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id")
    config.update(max_len=args.max_len, head_max_len=args.head_max_len, temperature=[1., 1., 1.])
    config.pop("temperature_by_options", None)
    if hasattr(model, "temperature"):
        model.temperature.fill_(1.0)
    # No act/escalate supervision is provided: do not train or trust this head here.
    for parameter in model.act_head.parameters():
        parameter.requires_grad_(False)
    if not args.no_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    loaders = {}
    for name, cases in splits.items():
        logger.info("Encoding %s cases", name)
        items = encode_cases(cases, tokenizer, args.max_len, args.head_max_len)
        loaders[name] = DataLoader(items, batch_size=args.batch_size, shuffle=name == "train",
                                   collate_fn=partial(collate, pad_id=tokenizer.pad_token_id), num_workers=0)
        logger.info("Encoded %s: %d question sequences", name, len(items))
    encoder_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder_parameters if name.startswith("encoder.") else head_parameters).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": args.lr_encoder},
        {"params": head_parameters, "lr": args.lr_head}], weight_decay=0.01)
    updates = args.epochs * math.ceil(len(loaders["train"]) / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, history, global_update = math.inf, [], 0
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        loader = loaders["train"]
        progress_bar = tqdm(
            loader,
            desc=f"Train {epoch + 1}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for step, batch in enumerate(progress_bar):
            batch = {key: value.to(device) for key, value in batch.items()}
            progress = global_update / max(updates - 1, 1)
            sigma = args.sigma_start + progress * (args.sigma_end - args.sigma_start)
            chunk_start = step // args.grad_accum * args.grad_accum
            accumulate = min(args.grad_accum, len(loader) - chunk_start)
            context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with context:
                logits = forward_model(model, batch)
                loss = decision_loss(logits, batch["target"], batch["marker_mask"], batch["qtype"],
                                     sigma, args.group, args.rl_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss: inspect targets and lower LR/switch off AMP for diagnosis")
            scaler.scale(loss / accumulate).backward()
            batch_loss = float(loss.detach())
            epoch_loss += batch_loss
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:  # No skipped overflow update.
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
            average_loss = epoch_loss / (step + 1)
            lr_encoder = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[1]["lr"]
            progress_bar.set_postfix(
                loss=f"{batch_loss:.4f}",
                avg=f"{average_loss:.4f}",
                lr=f"{lr_encoder:.2e}/{lr_head:.2e}",
                sigma=f"{sigma:.3f}",
                refresh=False,
            )
            if (step + 1) % args.log_every == 0 or step + 1 == len(loader):
                logger.info(
                    "epoch=%d/%d batch=%d/%d loss=%.6f avg_loss=%.6f "
                    "lr_encoder=%.3g lr_head=%.3g sigma=%.4f",
                    epoch + 1,
                    args.epochs,
                    step + 1,
                    len(loader),
                    batch_loss,
                    average_loss,
                    lr_encoder,
                    lr_head,
                    sigma,
                )
        epoch_loss_avg = epoch_loss / len(loader)
        logger.info(
            "Epoch %d training complete in %.1f sec; average_loss=%.6f",
            epoch + 1,
            time.perf_counter() - epoch_started,
            epoch_loss_avg,
        )
        report = metrics(
            collect_predictions(
                model, loaders["valid"], device, desc=f"Valid {epoch + 1}/{args.epochs}"
            )
        )
        history.append({"epoch": epoch + 1, "training_loss": epoch_loss_avg, "valid": report})
        logger.info("Epoch %d validation: %s", epoch + 1, json.dumps(report, ensure_ascii=False))
        if report["soft_target_nll"] < best:
            best = report["soft_target_nll"]
            logger.info("New best validation soft-target NLL=%.6f; saving checkpoint", best)
            save_checkpoint(model, tokenizer, config, output)
    logger.info("Reloading best checkpoint before calibration")
    model.load_state_dict(load_file(str(output / "model.safetensors"), device=str(device)), strict=True)
    calibration = collect_predictions(model, loaders["calib"], device, desc="Calibrate")
    temperatures = fit_temperatures(calibration)
    config.update(temperature=temperatures, fine_tuned=True, model_name="laya-business-example")
    config.pop("temperature_by_options", None)
    if hasattr(model, "temperature"):
        model.temperature.copy_(torch.tensor(temperatures, device=device, dtype=model.temperature.dtype))
    save_checkpoint(model, tokenizer, config, output)
    report = {"laya_version": getattr(laya, "__version__", "unknown"), "torch_version": torch.__version__,
              "arguments": vars(args), "history": history, "temperatures": temperatures,
              "training_log": str(log_file.resolve()),
              "warning": "Act/escalate head was not trained. Soft targets imply teacher agreement, not ground-truth calibration."}
    if "test" in loaders:
        test_records = collect_predictions(model, loaders["test"], device, desc="Test")
        report["test_uncalibrated"] = metrics(test_records)
        report["test_calibrated"] = metrics(test_records, temperatures)
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Calibration temperatures: %s", temperatures)
    if "test_calibrated" in report:
        logger.info("Calibrated test metrics: %s", json.dumps(report["test_calibrated"], ensure_ascii=False))
    logger.info("Saved checkpoint and training report to %s", output.resolve())
    logger.info("No automatic routing threshold has been selected; validate it on held-out business outcomes.")


if __name__ == "__main__":
    main()
