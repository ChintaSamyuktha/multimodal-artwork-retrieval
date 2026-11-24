"""Export a comparison table and a bounded discussion from measured results."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'artifacts/retrieval_metrics.json')
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding='utf-8'))
    required = {'tfidf', 'bm25', 'lsa', 'minilm', 'clip'}
    if not report.get('complete') or set(report['methods']) != required:
        raise ValueError('A complete five-method evaluation is required.')
    rows = report['methods']
    table = args.input.with_name('five_model_comparison.csv')
    with table.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'Recall@1 (%)', 'Recall@5 (%)', 'MRR', 'warm mean query (ms)'])
        for name, row in rows.items():
            writer.writerow([name.upper(), f"{100 * row['recall_at_1']:.2f}",
                             f"{100 * row['recall_at_5']:.2f}", f"{row['mrr']:.4f}",
                             f"{1000 * row['mean_query_seconds']:.3f}"])
    best = max(rows, key=lambda name: rows[name]['mrr'])
    lines = [
        'Five-model artwork retrieval comparison',
        f"Measured: {report['created_at']}",
        f"Queries: {report['queries']}; candidate artworks: {report['candidates']}",
        f"Query SHA256: {report['query_sha256']}", '',
        'Model         Recall@1     Recall@5       MRR     Warm query (ms)',
    ]
    for name, row in rows.items():
        lines.append(f"{name.upper():<12} {100*row['recall_at_1']:>8.2f}% {100*row['recall_at_5']:>11.2f}%"
                     f" {row['mrr']:>9.4f} {1000*row['mean_query_seconds']:>16.3f}")
    lines.extend(['', 'Observed differences from TF-IDF'])
    baseline = rows['tfidf']
    for name, row in rows.items():
        if name == 'tfidf':
            continue
        lines.append(f"{name.upper()}: Recall@1 {100*(row['recall_at_1']-baseline['recall_at_1']):+.2f} "
                     f"percentage points; Recall@5 {100*(row['recall_at_5']-baseline['recall_at_5']):+.2f} "
                     f"percentage points; MRR {row['mrr']-baseline['mrr']:+.4f}.")
    lines.extend(['', 'Discussion',
        f"{best.upper()} has the highest observed MRR ({rows[best]['mrr']:.4f}) in this run. "
        'This is an aggregate result on the supplied caption collection, not a statistical significance claim.',
        'BM25 applies term-frequency saturation and document-length normalization. '
        'TF-IDF ranks normalized sparse word/bigram vectors. LSA reduces TF-IDF to 256 latent dimensions. '
        'MiniLM compares frozen neural sentence embeddings. CLIP compares a text query with frozen image embeddings.',
        'Higher lexical scores can be consistent with shared vocabulary among descriptions of the same artwork. '
        'This experiment does not identify the cause of individual improvements or failures. '
        'Caption methods and CLIP receive different gallery information, so the table does not isolate representation quality.',
        'Latency uses the first 30 individually timed warm queries. Loading, gallery construction, HTTP and '
        'speech transcription are excluded. One timing run is sensitive to CPU scheduling and numerical-library threading; '
        'do not interpret small timing differences as a robust speed advantage.',
        'BM25 and LSA are classical retrieval models; the table contains five retrieval methods, not five neural networks. '
        'There is no project-specific neural training. The experiment does not measure image-query, audio-query '
        'or combined-query accuracy, independent descriptions, speech WER or end-to-end application latency.',
        '', 'Protocol', report['protocol'], report['note'], '', 'Implementation settings'])
    for name, row in rows.items():
        lines.append(f"{name.upper()}: {json.dumps(row['details'], sort_keys=True)}")
    lines.extend(['', 'Model/method references',
        'MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
        'LSA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html',
        'CLIP: https://proceedings.mlr.press/v139/radford21a.html',
        '', 'Next validation: inspect saved per-query ranks, collect independently written queries, '
        'run paired uncertainty estimates and repeat timings under controlled CPU conditions.'])
    discussion = args.input.with_name('five_model_discussion.txt')
    discussion.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved {table}\nSaved {discussion}')


if __name__ == '__main__':
    main()
