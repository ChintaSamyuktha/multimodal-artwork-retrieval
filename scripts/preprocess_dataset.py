"""Inspect the real dataset; preserve an existing mapping unless explicitly rebuilt."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from config import IMAGE_DIR, AUDIO_DIR, DESCRIPTIONS_FILE, MANIFEST_FILE, ROOT
from src.data import build_mapping
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-images',action='store_true')
    parser.add_argument('--rebuild-mapping',action='store_true',help='Explicitly replace the CSV with the sorted-filename convention.')
    args = parser.parse_args()
    descriptions = json.loads(DESCRIPTIONS_FILE.read_text(encoding='utf-8'))
    if not isinstance(descriptions,dict):
        raise ValueError('The description file must map image filenames to caption lists.')
    images = sorted(IMAGE_DIR.glob('*.jpg'),key=lambda p:p.name.casefold())
    audio = sorted(AUDIO_DIR.glob('*.mp3')) if AUDIO_DIR.exists() else []
    names = {p.name for p in images}
    missing = sorted(set(descriptions)-names)
    unmatched = sorted(names-set(descriptions))
    malformed = [name for name,value in descriptions.items()
                 if not isinstance(value,list) or not value or any(not isinstance(x,str) or not x.strip() for x in value)]
    unreadable = []
    if args.verify_images:
        for path in images:
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError,ValueError):
                unreadable.append(path.name)
    summary = {'images':len(images),'description_records':len(descriptions),'audio_files':len(audio),
               'images_without_descriptions':len(unmatched),'descriptions_without_images':len(missing),
               'invalid_descriptions':len(malformed),'unreadable_images':unreadable if args.verify_images else None,
               'mapping_file':str(MANIFEST_FILE),'mapping_rows':build_mapping(force=args.rebuild_mapping),
               'mapping_convention':'Generated sorted-filename numbering; original audio-image alignment is unverified.',
               'missing_images':missing,'missing_description_records':unmatched,'invalid_description_records':malformed}
    out = ROOT/'artifacts/dataset_summary.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if missing or unmatched or malformed or unreadable:
        raise SystemExit('Dataset validation found issues; review artifacts/dataset_summary.json.')

if __name__ == '__main__':
    main()
