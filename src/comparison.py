"""Caption models shared by held-out evaluation and full-catalogue search."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class BM25Retriever:
    """Okapi BM25 with positive Robertson IDF, k1=1.5 and b=0.75."""
    positive_only = True

    def __init__(self, documents, k1=1.5, b=0.75):
        self.vectorizer = CountVectorizer()
        counts = self.vectorizer.fit_transform(documents).astype(np.float64)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((len(documents) - df + 0.5) / (df + 0.5))
        weights = counts.tocoo()
        norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1e-12))
        weights.data = (weights.data * (k1 + 1) /
                        (weights.data + norm[weights.row])) * idf[weights.col]
        self.matrix = weights.tocsr()

    def score(self, texts):
        # Binary query terms avoid an unintended extra query-frequency factor.
        query = self.vectorizer.transform(texts)
        query.data[:] = 1
        return (query @ self.matrix.T).toarray()


class LSARetriever:
    positive_only = False

    def __init__(self, documents, dimensions=256):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        matrix = self.vectorizer.fit_transform(documents)
        components = min(dimensions, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if components < 1:
            raise ValueError('LSA requires at least two documents and two vocabulary terms.')
        self.svd = TruncatedSVD(n_components=components, random_state=42)
        self.matrix = normalize(self.svd.fit_transform(matrix))

    def score(self, texts):
        query = normalize(self.svd.transform(self.vectorizer.transform(texts)))
        scores = query @ self.matrix.T
        # An OOV query carries no retrieval evidence, even when ties are stable.
        scores[np.linalg.norm(query, axis=1) == 0] = -np.inf
        return scores


class MiniLMEncoder:
    """Model-card mean pooling, masked padding, L2 normalization, 256 tokens."""
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        from config import CACHE_DIR, MINILM_MODEL
        from src.models import configure_torch, device
        configure_torch()
        self.device = device()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MINILM_MODEL, cache_dir=str(CACHE_DIR), local_files_only=True)
            self.model = AutoModel.from_pretrained(
                MINILM_MODEL, cache_dir=str(CACHE_DIR), local_files_only=True,
                use_safetensors=True).to(self.device).eval()
        except OSError as exc:
            raise RuntimeError('Run python scripts/prepare_models.py --only minilm first.') from exc

    def encode(self, texts, batch_size=32):
        import torch
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(list(texts[start:start + batch_size]), padding=True,
                                       truncation=True, max_length=256,
                                       return_tensors='pt').to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vectors.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        return np.concatenate(vectors)


class MiniLMRetriever:
    positive_only = False

    def __init__(self, caption_groups):
        self.encoder = MiniLMEncoder()
        # Encode captions separately to avoid truncating concatenated documents.
        captions = [caption for group in caption_groups for caption in group]
        if not captions or any(not group for group in caption_groups):
            raise ValueError('MiniLM requires a remaining caption for every artwork.')
        vectors = self.encoder.encode(captions)
        documents, offset = [], 0
        for group in caption_groups:
            documents.append(vectors[offset:offset + len(group)].mean(axis=0))
            offset += len(group)
        self.matrix = normalize(np.asarray(documents))

    def score(self, texts):
        return self.encoder.encode(texts) @ self.matrix.T


def search_caption_model(query, catalog, mode, top_k=5):
    from src.indexing import comparison_index
    from src.models import MODEL_LOCK
    retriever, names = comparison_index(catalog, mode)
    with MODEL_LOCK:
        if mode == 'minilm' and retriever.encoder is None:
            retriever.encoder = MiniLMEncoder()
        scores = retriever.score([query])[0]
    by_name = {item['filename']: item for item in catalog}
    results = []
    for index in np.argsort(-scores, kind='stable'):
        score = float(scores[index])
        if not np.isfinite(score) or (retriever.positive_only and score <= 0):
            continue
        results.append({**by_name[names[index]], 'score':score,
                        'match_type':'BM25 relevance' if mode == 'bm25' else 'Cosine similarity'})
        if len(results) >= top_k:
            break
    return results
