"""Build indexes once, before starting Flask."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import logging
from src.data import load_catalog
from src.indexing import image_index, text_index, comparison_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['clip', 'pixels', 'tfidf', 'bm25', 'lsa', 'minilm', 'all'], default='all')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    items = load_catalog()
    for mode in (['tfidf', 'pixels', 'clip', 'bm25', 'lsa', 'minilm'] if args.mode == 'all' else [args.mode]):
        if mode == 'tfidf':
            _, matrix, names = text_index(items)
        elif mode in ('bm25', 'lsa', 'minilm'):
            retriever, names = comparison_index(items, mode, build=True)
            matrix = retriever.matrix
        else:
            matrix, names = image_index(items, mode, build=True)
        print(f'{mode}: {len(names)} artworks indexed; matrix {matrix.shape}', flush=True)


if __name__ == '__main__':
    main()
