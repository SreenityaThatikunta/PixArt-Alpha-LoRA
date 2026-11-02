"""Argument-parser contracts for the CLI entry points.

The defaults documented in the README are asserted here, so the two cannot
drift apart silently.
"""
import pytest

torch = pytest.importorskip("torch", reason="requires the ML stack")
pytest.importorskip("diffusers", reason="requires the ML stack")
pytest.importorskip("peft", reason="requires the ML stack")
pytest.importorskip("cleanfid", reason="requires the ML stack")


def test_train_defaults_match_the_documented_config(monkeypatch):
    import train

    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--data_dir", "data", "--caption_file", "captions.json"],
    )
    args = train.parse_args()

    assert args.pretrained_model_name == "PixArt-alpha/PixArt-XL-2-1024-MS"
    assert args.lora_rank == 16
    assert args.lora_alpha == 2 * args.lora_rank, "alpha should follow alpha = 2 x rank"
    assert args.lora_dropout == 0.1
    assert args.learning_rate == 5e-6
    assert args.lr_scheduler == "cosine"
    assert args.lr_warmup_steps == 200
    assert args.gradient_accumulation_steps == 4
    assert args.resolution == 512
    assert args.train_batch_size == 2
    assert args.save_steps == 500
    assert args.merge_weight == 1.0


def test_train_requires_data_and_captions(monkeypatch):
    import train

    monkeypatch.setattr("sys.argv", ["train.py"])
    with pytest.raises(SystemExit):
        train.parse_args()


def test_mixed_precision_is_constrained(monkeypatch):
    import train

    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--data_dir", "d", "--caption_file", "c.json",
         "--mixed_precision", "int4"],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_evaluate_skip_flags_default_off(monkeypatch):
    import evaluate

    monkeypatch.setattr("sys.argv", ["evaluate.py"])
    args = evaluate.parse_args()

    assert args.skip_fid is False
    assert args.skip_compbench is False
    assert args.skip_gen is False
    assert args.seed == 42
