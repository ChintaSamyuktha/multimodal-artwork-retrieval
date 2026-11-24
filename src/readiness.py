"""Status checks without model loading or network access."""
import hashlib
import ast
import importlib.util
from functools import lru_cache
from config import WHISPER_CACHE, WHISPER_MODEL


@lru_cache(maxsize=1)
def whisper_checksums():
    # Reading package metadata avoids importing Torch on the status page.
    source = importlib.util.find_spec('whisper').origin
    with open(source, encoding='utf-8') as stream:
        tree = ast.parse(stream.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_MODELS' for t in node.targets):
            return {name: url.split('/')[-2] for name, url in ast.literal_eval(node.value).items()}
    raise RuntimeError('Whisper model metadata is unavailable.')


@lru_cache(maxsize=8)
def _valid_whisper(path, size, modified, model):
    expected = whisper_checksums()[model]
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == expected


def whisper_ready():
    path = WHISPER_CACHE / f'{WHISPER_MODEL}.pt'
    if not path.is_file():
        return False
    stat = path.stat()
    return _valid_whisper(str(path), stat.st_size, stat.st_mtime_ns, WHISPER_MODEL)


def status(catalog):
    from src.indexing import index_path
    return {'artworks': len(catalog),
            'indexes': {mode: index_path(catalog, mode).exists() for mode in ('clip', 'pixels', 'tfidf', 'bm25', 'lsa', 'minilm')},
            'speech_model': WHISPER_MODEL, 'speech_ready': whisper_ready()}
