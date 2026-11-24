"""Versioned indexes persisted atomically and reused after application restarts."""
import hashlib
import json
import logging
import os
from functools import lru_cache
from uuid import uuid4
import numpy as np
from filelock import FileLock
from config import IMAGE_DIR, INDEX_DIR, CLIP_MODEL, MINILM_MODEL, BATCH_SIZE

log = logging.getLogger(__name__)


class IndexNotReady(RuntimeError):
    pass


def fingerprint(catalog, mode):
    rows = []
    for item in catalog:
        if mode in ('tfidf', 'bm25', 'lsa', 'minilm'):
            rows.append((item['filename'], item['descriptions']))
        else:
            stat = (IMAGE_DIR / item['filename']).stat()
            rows.append((item['filename'], stat.st_size, stat.st_mtime_ns))
    config = ('index-v2', mode, CLIP_MODEL if mode == 'clip' else MINILM_MODEL if mode == 'minilm' else '', str(IMAGE_DIR.resolve()), rows)
    return hashlib.sha256(json.dumps(config, ensure_ascii=False).encode()).hexdigest()


def index_path(catalog, mode):
    suffix = '.joblib' if mode in ('tfidf', 'bm25', 'lsa', 'minilm') else '.npz'
    return INDEX_DIR / f'{mode}-{fingerprint(catalog, mode)[:20]}{suffix}'


@lru_cache(maxsize=4)
def _load_vectors(path, modified):
    with np.load(path, allow_pickle=False) as data:
        return data['vectors'].copy(), tuple(data['names'].tolist())


def image_index(catalog, mode='clip', build=False):
    if mode not in {'clip', 'pixels'} or not catalog:
        raise ValueError('Provide a valid image index and a non-empty artwork collection.')
    path = index_path(catalog, mode)
    if not path.exists():
        if not build:
            raise IndexNotReady(f'The {mode} index is not prepared. Run python scripts/build_index.py --mode {mode}.')
        _build_images(catalog, mode, path)
    return _load_vectors(str(path), path.stat().st_mtime_ns)


def _build_images(catalog, mode, path):
    from src.retrieval import open_image, _embed_images, _visual_feature
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.lock', timeout=0):
        if path.exists():
            return
        vectors, names, pending, pending_names = [], [], [], []

        def flush():
            if not pending:
                return
            result = _embed_images(pending) if mode == 'clip' else np.stack([_visual_feature(im) for im in pending])
            result = result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
            vectors.append(result.astype(np.float32))
            names.extend(pending_names)
            for image in pending:
                image.close()
            pending.clear()
            pending_names.clear()
            log.info('%s index: %d / %d images', mode, len(names), len(catalog))

        for item in catalog:
            try:
                image = open_image(IMAGE_DIR / item['filename'])
            except ValueError:
                log.warning('Skipped unreadable image: %s', item['filename'])
                continue
            pending.append(image)
            pending_names.append(item['filename'])
            if len(pending) >= BATCH_SIZE:
                flush()
        flush()
        if not vectors:
            raise ValueError('No readable artwork images were found.')
        temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
        try:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, vectors=np.concatenate(vectors), names=np.array(names))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@lru_cache(maxsize=4)
def _load_text(path, modified):
    import joblib
    # Only load a project-generated local cache, never an uploaded file.
    return joblib.load(path)


def text_index(catalog):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    path = index_path(catalog, 'tfidf')
    if not path.exists():
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + '.lock', timeout=30):
            if not path.exists():
                names = tuple(item['filename'] for item in catalog)
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
                try:
                    matrix = vectorizer.fit_transform([' '.join(item['descriptions']) for item in catalog])
                except ValueError as exc:
                    raise ValueError('No searchable descriptions were found.') from exc
                temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
                try:
                    joblib.dump((vectorizer, matrix, names), temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
    return _load_text(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=3)
def _load_comparison(path, modified):
    import joblib
    return joblib.load(path)


def comparison_index(catalog, mode, build=False):
    """Persist full-catalogue interactive indexes separately from held-out evaluation."""
    import joblib
    from copy import copy
    if mode not in ('bm25', 'lsa', 'minilm') or not catalog:
        raise ValueError('Choose a valid caption model and a non-empty catalogue.')
    path = index_path(catalog, mode)
    if not path.exists():
        if not build:
            raise IndexNotReady(f'The {mode} index is not prepared. Run python scripts/build_index.py --mode {mode}.')
        from src.comparison import BM25Retriever, LSARetriever, MiniLMRetriever
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + '.lock', timeout=0):
            if not path.exists():
                documents = [' '.join(item['descriptions']) for item in catalog]
                if mode == 'bm25':
                    retriever = BM25Retriever(documents)
                elif mode == 'lsa':
                    retriever = LSARetriever(documents)
                else:
                    retriever = MiniLMRetriever([item['descriptions'] for item in catalog])
                saved = copy(retriever)
                if mode == 'minilm':
                    # Persist vectors, not a pickled Torch model or device state.
                    saved.encoder = None
                temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
                try:
                    joblib.dump((saved, tuple(item['filename'] for item in catalog)), temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
    return _load_comparison(str(path), path.stat().st_mtime_ns)
