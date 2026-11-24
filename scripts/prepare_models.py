"""Explicit model downloads with OS certificate validation enabled."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os


def main():
    # The university/antivirus CA is trusted by Windows, not old certifi bundles.
    # Keep certificate verification on; use the operating system's trust store.
    import truststore
    truststore.inject_into_ssl()
    from config import CLIP_MODEL, CACHE_DIR, WHISPER_CACHE, WHISPER_MODEL, MINILM_MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['clip', 'whisper', 'minilm', 'all'], default='all')
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '180')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    if args.only in ('clip', 'all'):
        from huggingface_hub import snapshot_download
        print('Preparing CLIP weights and processor...', flush=True)
        snapshot_download(CLIP_MODEL, cache_dir=str(CACHE_DIR),
                          allow_patterns=['*.json', 'vocab.json', 'merges.txt', 'pytorch_model.bin'], max_workers=2)
        from src.models import clip_components
        clip_components()
        print('CLIP loaded successfully.', flush=True)
    if args.only in ('minilm', 'all'):
        from huggingface_hub import snapshot_download
        print('Preparing MiniLM sentence encoder...', flush=True)
        snapshot_download(MINILM_MODEL, cache_dir=str(CACHE_DIR),
                          allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json',
                                          'special_tokens_map.json', 'vocab.txt', 'model.safetensors'], max_workers=2)
        from src.comparison import MiniLMEncoder
        MiniLMEncoder().encode(['model preparation check'])
        print('MiniLM loaded successfully.', flush=True)
    if args.only in ('whisper', 'all'):
        import whisper
        from src.models import configure_torch
        configure_torch()
        print(f'Preparing Whisper {WHISPER_MODEL}...', flush=True)
        WHISPER_CACHE.mkdir(parents=True, exist_ok=True)
        whisper.load_model(WHISPER_MODEL, device='cpu', download_root=str(WHISPER_CACHE))
        print('Whisper checksum validated and model loaded.', flush=True)


if __name__ == '__main__':
    main()
