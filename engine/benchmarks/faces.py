#!/usr/bin/env python3
"""Compare face detectors at clipforge's sampling size, using public domain NASA portraits at known spots.

Usage: faces.py prepare | faces.py score haar | yunet | blaze
Photos stay in the work folder, never the repo. Portraits are chosen with YuNet at full size, which
favours it a little, so the harder variants matter most. These are formal portraits: read the results
as a floor for ideal faces, not a promise about real footage.
"""

import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WORK = Path.home() / "clipforge-work"
DATA = WORK / "bench" / "faces"
MODELS = WORK / "models" / "face"
WIDTH, HEIGHT = 1920, 1080
SAMPLE = (320, 180)


def download(count: int = 70) -> None:
    import httpx

    out = DATA / "raw"
    out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=40, follow_redirects=True, headers={"User-Agent": "clipforge-benchmark"})
    saved = 0
    seen: set[str] = set()
    for query in ("astronaut official portrait", "official crew portrait astronaut", "astronaut portrait"):
        items = client.get(
            "https://images-api.nasa.gov/search", params={"q": query, "media_type": "image", "page_size": 60}
        ).json()["collection"]["items"]
        for item in items:
            nasa_id = item["data"][0]["nasa_id"]
            link = next((x["href"] for x in item.get("links", []) if x.get("rel") == "preview"), None)
            if nasa_id in seen or not link or saved >= count:
                continue
            seen.add(nasa_id)
            body = client.get(link.replace("~thumb", "~medium"))
            if body.status_code == 200 and len(body.content) > 20000:
                (out / f"{re.sub(r'[^A-Za-z0-9_-]', '_', nasa_id)[:50]}.jpg").write_bytes(body.content)
                saved += 1


def prepare() -> None:
    if not (DATA / "raw").exists():
        download()
    frames = DATA / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    detector = cv2.FaceDetectorYN.create(
        str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.8, 0.3, 50
    )
    portraits = []
    for path in sorted((DATA / "raw").glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        if faces is None or len(faces) != 1 or faces[0][3] < height * 0.2 or faces[0][-1] < 0.9:
            continue
        x, y, w, h = faces[0][:4]
        portraits.append((path, float(x + w / 2), float(y + h / 2), float(h)))
    rng = random.Random(5)
    backgrounds = [
        lambda: np.full((HEIGHT, WIDTH, 3), 40, np.uint8),
        lambda: np.random.default_rng(1).integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8),
        lambda: cv2.GaussianBlur(
            np.random.default_rng(2).integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8), (0, 0), 25
        ),
    ]
    truth = []
    index = 0
    for path, center_x, center_y, face_height in portraits:
        image = cv2.imread(str(path))
        for size_name, share in (("large", 0.45), ("medium", 0.22), ("small", 0.11)):
            scale = share * HEIGHT / face_height
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            target_x, target_y = rng.uniform(0.15, 0.85) * WIDTH, rng.uniform(0.3, 0.6) * HEIGHT
            frame = backgrounds[index % 3]().copy()
            x0, y0 = int(target_x - center_x * scale), int(target_y - center_y * scale)
            ax0, ay0, ax1, ay1 = (
                max(0, x0),
                max(0, y0),
                min(WIDTH, x0 + small.shape[1]),
                min(HEIGHT, y0 + small.shape[0]),
            )
            frame[ay0:ay1, ax0:ax1] = small[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0]
            name = f"{index:04d}.jpg"
            cv2.imwrite(str(frames / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            truth.append({"file": name, "kind": size_name, "cx": target_x / WIDTH})
            if size_name != "small":
                for variant, make in (
                    (
                        "tilt 20",
                        lambda f, x=target_x: cv2.warpAffine(
                            f,
                            cv2.getRotationMatrix2D((x, HEIGHT * 0.45), 20, 1.0),
                            (WIDTH, HEIGHT),
                            borderValue=(40, 40, 40),
                        ),
                    ),
                    (
                        "tilt 40",
                        lambda f, x=target_x: cv2.warpAffine(
                            f,
                            cv2.getRotationMatrix2D((x, HEIGHT * 0.45), 40, 1.0),
                            (WIDTH, HEIGHT),
                            borderValue=(40, 40, 40),
                        ),
                    ),
                    ("dark", lambda f, x=0: np.clip(f.astype(np.float32) * 0.28, 0, 255).astype(np.uint8)),
                    ("blur", lambda f, x=0: cv2.GaussianBlur(f, (0, 0), 9)),
                ):
                    variant_name = f"{variant.replace(' ', '')}-{name}"
                    cv2.imwrite(str(frames / variant_name), make(frame), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    truth.append({"file": variant_name, "kind": variant, "cx": target_x / WIDTH})
            index += 1
    for i in range(30):
        frame = backgrounds[i % 3]().copy()
        for j in range(12):
            color = tuple(int(v) for v in np.random.default_rng(i + j).integers(0, 255, 3))
            x, y = rng.randrange(WIDTH), rng.randrange(HEIGHT)
            if j % 2:
                cv2.circle(frame, (x, y), rng.randrange(20, 160), color, -1)
            else:
                cv2.rectangle(
                    frame, (x, y), (x + rng.randrange(40, 400), y + rng.randrange(40, 300)), color, -1
                )
        cv2.imwrite(str(frames / f"none{i:03d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        truth.append({"file": f"none{i:03d}.jpg", "kind": "none"})
    (DATA / "truth.json").write_text(json.dumps(truth))
    print(len(portraits), "portraits,", len(truth), "frames")


def load(entry: dict) -> np.ndarray:
    return cv2.resize(cv2.imread(str(DATA / "frames" / entry["file"])), SAMPLE, interpolation=cv2.INTER_AREA)


def biggest(boxes: list) -> tuple | None:
    return max(boxes, key=lambda b: b[2] * b[3]) if len(boxes) else None


def haar():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return lambda f: biggest(
        list(
            cascade.detectMultiScale(
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=6, minSize=(24, 24)
            )
        )
    )


def yunet():
    net = cv2.FaceDetectorYN.create(
        str(MODELS / "face_detection_yunet_2023mar.onnx"), "", SAMPLE, 0.7, 0.3, 50
    )

    def detect(frame):
        _, boxes = net.detect(frame)
        return biggest([b[:4] for b in boxes]) if boxes is not None else None

    return detect


def blaze():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "blaze_face_short_range.tflite")),
        min_detection_confidence=0.5,
    )
    net = vision.FaceDetector.create_from_options(options)

    def detect(frame):
        found = net.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        )
        return biggest(
            [
                (
                    d.bounding_box.origin_x,
                    d.bounding_box.origin_y,
                    d.bounding_box.width,
                    d.bounding_box.height,
                )
                for d in found.detections
            ]
        )

    return detect


def score(name: str) -> dict:
    detect = {"haar": haar, "yunet": yunet, "blaze": blaze}[name]()
    results: dict = {}
    false_alarms = negatives = 0
    start = time.time()
    truth = json.loads((DATA / "truth.json").read_text())
    for entry in truth:
        box = detect(load(entry))
        if entry["kind"] == "none":
            negatives += 1
            false_alarms += box is not None
            continue
        row = results.setdefault(entry["kind"], {"found": 0, "of": 0, "errors": []})
        row["of"] += 1
        if box is not None and abs((box[0] + box[2] / 2) / SAMPLE[0] - entry["cx"]) < 0.08:
            row["found"] += 1
            row["errors"].append(abs((box[0] + box[2] / 2) / SAMPLE[0] - entry["cx"]) * 100)
    for row in results.values():
        errors = row.pop("errors")
        row["median_center_error_pct"] = round(float(np.median(errors)), 2) if errors else None
    return {
        **results,
        "false_alarms": f"{false_alarms}/{negatives}",
        "ms_per_frame": round((time.time() - start) / len(truth) * 1000, 1),
    }


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        print(json.dumps(score(sys.argv[2]), indent=1))
