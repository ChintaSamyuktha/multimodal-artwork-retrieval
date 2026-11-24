"""Compare five retrieval models on identical held-out captions and candidates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
import numpy as np
from config import ROOT
from src.data import load_catalog
from src.indexing import image_index, text_index
from src.retrieval import embed_texts, _embed_text


def held_out_split(items):
    queries = [(item['filename'], item['descriptions'][-1]) for item in items
               if len(set(item['descriptions'])) >= 2]
    held = dict(queries)
    training = [{**item, 'descriptions': [d for d in item['descriptions']
                                        if d != held.get(item['filename'])]} for item in items]
    return queries, training


def save_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(output)

def ranking_metrics(scores, expected, names, positive_only=False):
    # Stable tie-breaking matches interactive retrieval.
    target = tuple(names).index(expected)
    if not np.isfinite(scores[target]):
        return float('inf')
    if positive_only and scores[target] <= 0:
        return float('inf')  # The UI returns no zero-overlap TF-IDF match.
    order = np.argsort(-scores, kind='stable')
    return int(np.flatnonzero(np.asarray(names)[order] == expected)[0]) + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['clip', 'tfidf', 'bm25', 'lsa', 'minilm', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=0, help='Optional query cap; candidate collection remains complete.')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/retrieval_metrics.json')
    parser.add_argument('--per-query', type=Path, default=None,
                        help='CSV output; defaults to <output stem>_ranks.csv.')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    items = load_catalog()
    queries, training = held_out_split(items)
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        raise ValueError('Evaluation needs artworks with at least two distinct captions.')
    report = {'created_at':datetime.now(timezone.utc).isoformat(), 'candidates':len(items), 'queries':len(queries),
              'protocol':'The final supplied caption per artwork with at least two distinct captions is held out, including every exact repetition in that artwork. TF-IDF, BM25, LSA and MiniLM index remaining captions; frozen CLIP compares query text with artwork image embeddings.',
              'note':'Collection-specific caption retrieval, not an independent generalization test. All methods use the same queries and candidate identities but CLIP indexes images and the other methods index text. Remaining captions may share wording with queries. Warm query time includes encoding, scoring and sorting; excludes model loading, index construction, HTTP and speech transcription. No project-specific neural fine-tuning or hyperparameter search is performed.',
              'environment':{'python':platform.python_version(), 'platform':platform.platform()},
              'query_sha256':hashlib.sha256(json.dumps(queries).encode()).hexdigest(), 'methods':{}}
    report['gallery_caption_sha256'] = hashlib.sha256(json.dumps(training).encode()).hexdigest()
    all_ranks = {}
    modes = ['tfidf', 'bm25', 'lsa', 'minilm', 'clip'] if args.mode == 'all' else [args.mode]
    report['requested_methods'] = modes
    report['complete'] = False
    for mode in modes:
        print(f'Evaluating {mode}: {len(queries)} queries against {len(items)} artworks',flush=True)
        if mode == 'tfidf':
            vectorizer, matrix, names = text_index(training)
            def single(text):
                return (matrix @ vectorizer.transform([text]).T).toarray().ravel()
            def batch(texts):
                return (matrix @ vectorizer.transform(texts).T).toarray().T
            positive_only = True
            details = {'gallery':'remaining captions', 'model':'TF-IDF unigram/bigram, sublinear TF, L2 cosine'}
        elif mode == 'clip':
            matrix, names = image_index(items, 'clip')
            if len(names) != len(items):
                raise ValueError('Some candidate images were unreadable. Fix dataset validation before comparing methods.')
            def single(text):
                return matrix @ _embed_text(text)
            def batch(texts):
                return embed_texts(texts) @ matrix.T
            positive_only = False
            from config import CLIP_MODEL
            details = {'gallery':'images', 'model':CLIP_MODEL, 'fine_tuned':False}
        else:
            from src.comparison import BM25Retriever, LSARetriever, MiniLMRetriever
            documents = [' '.join(item['descriptions']) for item in training]
            if mode == 'bm25':
                retriever = BM25Retriever(documents)
                details = {'gallery':'remaining captions', 'model':'Okapi BM25', 'k1':1.5, 'b':0.75,
                           'idf':'log(1 + (N - df + 0.5)/(df + 0.5))'}
            elif mode == 'lsa':
                retriever = LSARetriever(documents)
                details = {'gallery':'remaining captions', 'model':'TF-IDF + truncated SVD + L2 cosine',
                           'dimensions':retriever.svd.n_components, 'random_state':42}
            else:
                retriever = MiniLMRetriever([item['descriptions'] for item in training])
                from config import MINILM_MODEL
                details = {'gallery':'remaining captions', 'model':MINILM_MODEL, 'fine_tuned':False,
                           'max_tokens':256, 'pooling':'masked token mean; normalized caption vectors averaged per artwork and renormalized'}
            names = tuple(item['filename'] for item in items)
            batch = retriever.score
            def single(text):
                return batch([text])[0]
            positive_only = retriever.positive_only
        single(queries[0][1])  # warm model and caches
        ranks = []
        for start in range(0, len(queries), 64):
            chunk = queries[start:start+64]
            texts = [text for _,text in chunk]
            scores = batch(texts)
            ranks.extend(ranking_metrics(row, target, names, positive_only=positive_only) for row,(target,_) in zip(scores,chunk))
            if start % 640 == 0:
                print(f'{mode}: {min(start+64,len(queries))}/{len(queries)} queries',flush=True)
        times = []
        for _,text in queries[:30]:
            start = time.perf_counter()
            np.argsort(-single(text), kind='stable')[:5]
            times.append(time.perf_counter()-start)
        r = np.asarray(ranks)
        report['methods'][mode] = {'recall_at_1':float(np.mean(r<=1)), 'recall_at_5':float(np.mean(r<=5)),
                                   'mrr':float(np.mean(1/r)), 'mean_query_seconds':float(np.mean(times)),
                                   'latency_samples':len(times), 'details':details,
                                   'median_query_seconds':float(np.median(times))}
        all_ranks[mode] = ranks
        report['complete'] = len(report['methods']) == len(modes)
        save_report(report, args.output)
        rank_path = args.per_query or args.output.with_name(args.output.stem + '_ranks.csv')
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        with rank_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['target', 'query'] + list(all_ranks))
            for index, (target, query) in enumerate(queries):
                writer.writerow([target, query] + [values[index] if np.isfinite(values[index]) else 'no_match'
                                                  for values in all_ranks.values()])
        print(json.dumps(report['methods'][mode],indent=2),flush=True)
    print(f'Saved actual measurements to {args.output}',flush=True)

if __name__ == '__main__':
    main()
