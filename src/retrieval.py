"""Reusable CLIP retrieval and explicitly labelled baseline methods."""
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from src.models import clip_components, device, normalized_features, MODEL_LOCK, configure_torch

def open_image(stream):
    try:
        with Image.open(stream) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError('The image dimensions exceed the supported limit.')
            image.load()
            return ImageOps.exif_transpose(image).convert('RGB')
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError('The uploaded file is not a readable image.') from exc

def _visual_feature(image):
    return (np.asarray(image.convert('RGB').resize((32, 32)), dtype=np.float32) / 255).ravel()

def _embed_images(images):
    torch = configure_torch()
    with MODEL_LOCK, torch.inference_mode():
        processor, model = clip_components()
        batch = processor(images=images, return_tensors='pt').to(device())
        return normalized_features(model.get_image_features(**batch))

def embed_texts(texts):
    torch = configure_torch()
    with MODEL_LOCK, torch.inference_mode():
        processor, model = clip_components()
        batch = processor(text=list(texts), return_tensors='pt', padding=True, truncation=True, max_length=77).to(device())
        return normalized_features(model.get_text_features(**batch))

def _embed_text(text):
    return embed_texts([text])[0]

def rank_scores(scores, names, catalog, top_k=5, positive_only=False):
    scores = np.asarray(scores).ravel()
    by_name = {item['filename']: item for item in catalog}
    results = []
    for i in np.argsort(-scores, kind='stable'):
        if names[i] not in by_name or not np.isfinite(scores[i]):
            continue
        if positive_only and scores[i] <= 0:
            continue
        results.append({**by_name[names[i]], 'score': float(np.clip(scores[i], -1, 1)), 'match_type': 'Cosine similarity'})
        if len(results) >= top_k:
            break
    return results

def _rank(vector, catalog, top_k=5, mode='clip'):
    from src.indexing import image_index
    matrix, names = image_index(catalog, mode)
    vector = np.asarray(vector, dtype=np.float32)
    vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return rank_scores(matrix @ vector, names, catalog, top_k)

def search_text(query, catalog, top_k=5):
    from src.indexing import image_index
    image_index(catalog, 'clip')
    return _rank(_embed_text(query), catalog, top_k)

def search_text_fast(query, catalog, top_k=5):
    from src.indexing import text_index
    vectorizer, matrix, names = text_index(catalog)
    scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    return rank_scores(scores, names, catalog, top_k, positive_only=True)

def search_image(stream, catalog, top_k=5):
    from src.indexing import image_index
    image = open_image(stream)
    image_index(catalog, 'clip')
    return _rank(_embed_images([image])[0], catalog, top_k)

def search_image_fast(stream, catalog, top_k=5):
    return _rank(_visual_feature(open_image(stream)), catalog, top_k, mode='pixels')

def search_multimodal(query, stream, catalog, image_weight=0.5, top_k=5):
    if not query.strip() or len(query) > 2000 or not 0 <= image_weight <= 1:
        raise ValueError('Enter a description and an image weight between zero and one.')
    from src.indexing import image_index
    image_index(catalog, 'clip')
    image = open_image(stream)
    mixed = image_weight * _embed_images([image])[0] + (1-image_weight) * _embed_text(query)
    return _rank(mixed, catalog, top_k)
