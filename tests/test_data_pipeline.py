"""Dataset and caption-handling behaviour, exercised on tiny synthetic images."""
import json

import pytest

torch = pytest.importorskip("torch", reason="requires the ML stack")
pytest.importorskip("diffusers", reason="requires the ML stack")
pytest.importorskip("peft", reason="requires the ML stack")
pytest.importorskip("cleanfid", reason="requires the ML stack")
from PIL import Image  # noqa: E402  (after importorskip)


@pytest.fixture
def tiny_dataset(tmp_path):
    """Three images on disk, four captions - one caption is deliberately orphaned."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("hand_a.jpg", "hand_b.jpg", "hand_c.png"):
        Image.new("RGB", (80, 120), color=(120, 90, 80)).save(image_dir / name)

    captions = {
        "hand_a.jpg": "A hand with five clearly separated fingers.",
        "hand_b.jpg": {"caption": "An open palm facing the camera."},
        "hand_c.png": "A closed fist seen from the side.",
        "missing.jpg": "This image is not on disk.",
    }
    caption_file = tmp_path / "captions.json"
    caption_file.write_text(json.dumps(captions))
    return image_dir, caption_file


def test_dataset_drops_captions_with_no_image(tiny_dataset):
    import train

    image_dir, caption_file = tiny_dataset
    ds = train.HandsDataset(str(image_dir), str(caption_file), resolution=64)

    assert len(ds) == 3, "the orphaned caption should be filtered out"
    assert "missing.jpg" not in ds.image_files


def test_dataset_accepts_both_caption_formats(tiny_dataset):
    """Captions are either a bare string or a {'caption': ...} dict."""
    import train

    image_dir, caption_file = tiny_dataset
    ds = train.HandsDataset(str(image_dir), str(caption_file), resolution=64)

    by_name = {ds.image_files[i]: ds[i] for i in range(len(ds))}
    assert by_name["hand_a.jpg"]["caption"].startswith("A hand with five")
    assert by_name["hand_b.jpg"]["caption"] == "An open palm facing the camera."


def test_dataset_normalises_to_unit_range(tiny_dataset):
    import train

    image_dir, caption_file = tiny_dataset
    ds = train.HandsDataset(str(image_dir), str(caption_file), resolution=64)

    pixels = ds[0]["pixel_values"]
    assert pixels.shape == (3, 64, 64), "images are centre-cropped to the resolution"
    assert pixels.min() >= -1.0 and pixels.max() <= 1.0, "expected the [-1, 1] range"


def test_collate_fn_batches_pixels_and_captions(tiny_dataset):
    import train

    image_dir, caption_file = tiny_dataset
    ds = train.HandsDataset(str(image_dir), str(caption_file), resolution=64)

    batch = train.collate_fn([ds[0], ds[1]])
    assert batch["pixel_values"].shape == (2, 3, 64, 64)
    assert len(batch["captions"]) == 2
    assert all(isinstance(c, str) for c in batch["captions"])


def test_build_caption_json_parses_the_llava_delimiter(tmp_path):
    import train

    raw = tmp_path / "llava_out.txt"
    raw.write_text(
        "hand_a.jpg||| A hand with five fingers. \n"
        "\n"
        "hand_b.jpg|||An open palm, palm facing forward.\n"
        "this line has no delimiter and should be skipped\n"
    )
    out = tmp_path / "captions.json"

    captions = train.build_caption_json_from_folder(
        str(tmp_path), str(raw), out_json=str(out)
    )

    assert captions == {
        "hand_a.jpg": "A hand with five fingers.",
        "hand_b.jpg": "An open palm, palm facing forward.",
    }
    assert json.loads(out.read_text()) == captions


def test_load_captions_flattens_the_metadata_dict(tmp_path):
    import generate_images

    path = tmp_path / "captions.json"
    path.write_text(json.dumps({
        "hand_a.jpg": {"caption": "A hand with five fingers."},
        "hand_b.jpg": {"caption": "An open palm.", "source": "11k"},
    }))

    entries = generate_images.load_captions(str(path))

    assert entries == [
        {"filename": "hand_a.jpg", "caption": "A hand with five fingers."},
        {"filename": "hand_b.jpg", "caption": "An open palm."},
    ]


def test_vqa_average_is_the_mean_of_answers(tmp_path):
    import evaluate

    path = tmp_path / "vqa_result.json"
    path.write_text(json.dumps([
        {"question_id": 0, "answer": "0.80"},
        {"question_id": 1, "answer": "0.60"},
        {"question_id": 2, "answer": "0.40"},
    ]))

    assert evaluate._read_vqa_json(path) == pytest.approx(0.6)


def test_vqa_average_of_nothing_is_zero(tmp_path):
    import evaluate

    path = tmp_path / "vqa_result.json"
    path.write_text("[]")

    assert evaluate._read_vqa_json(path) == 0.0
