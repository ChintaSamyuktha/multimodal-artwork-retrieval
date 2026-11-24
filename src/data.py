import csv, json
from pathlib import Path
from config import IMAGE_DIR, DESCRIPTIONS_FILE, AUDIO_DIR, MANIFEST_FILE

def load_catalog():
    if not DESCRIPTIONS_FILE.exists(): raise FileNotFoundError(f"Description JSON not found: {DESCRIPTIONS_FILE}")
    descriptions = json.loads(DESCRIPTIONS_FILE.read_text(encoding="utf-8"))
    if not isinstance(descriptions, dict):
        raise ValueError('Description JSON must map image filenames to descriptions.')
    records = []
    for image_path in sorted(IMAGE_DIR.glob("*.jpg"), key=lambda p: p.name.casefold()):
        value = descriptions.get(image_path.name, [])
        if isinstance(value, str): value = [value]
        if not isinstance(value, list): value = []
        records.append({"filename": image_path.name, "descriptions": [x.strip() for x in value if isinstance(x, str) and x.strip()]})
    return records

def build_mapping(force=False):
    if MANIFEST_FILE.exists() and not force:
        with MANIFEST_FILE.open(encoding='utf-8', newline='') as stream:
            return sum(1 for _ in csv.DictReader(stream))
    images = sorted(IMAGE_DIR.glob("*.jpg"), key=lambda p: p.name.casefold())
    audios = {p.stem.lower(): p.name for p in AUDIO_DIR.glob("*") if p.is_file()}
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_number", "image_filename", "audio_filename"])
        writer.writeheader()
        for number, image in enumerate(images, 1): writer.writerow({"image_number": number, "image_filename": image.name, "audio_filename": audios.get(f"image{number:04d}", "")})
    return len(images)

def get_mapping_lookup():
    if not MANIFEST_FILE.exists(): build_mapping()
    with MANIFEST_FILE.open(encoding="utf-8", newline="") as f: rows = list(csv.DictReader(f))
    by_number = {}
    for row in rows:
        try: by_number[int(row["image_number"])] = {"filename": row["image_filename"], "descriptions": []}
        except (ValueError, KeyError): continue
    descriptions = {item["filename"]: item["descriptions"] for item in load_catalog()}
    for entry in by_number.values():
        if entry['filename'] not in descriptions:
            raise ValueError('The number mapping refers to an unavailable artwork. Check the mapping CSV.')
        entry["descriptions"] = descriptions.get(entry["filename"], [])
    return by_number
