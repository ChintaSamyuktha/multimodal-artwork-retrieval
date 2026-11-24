"""Measure speech WER only against independently supplied reference transcripts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import re
import time
from config import AUDIO_DIR, ROOT
from src.audio import transcribe


def word_error_counts(reference, hypothesis):
    ref = re.findall(r'\w+', reference.lower())
    hyp = re.findall(r'\w+', hypothesis.lower())
    if not ref:
        raise ValueError('Reference transcripts must contain words.')
    row = list(range(len(hyp)+1))
    for i, word in enumerate(ref,1):
        nxt = [i]
        for j, other in enumerate(hyp,1):
            nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(word!=other)))
        row = nxt
    return row[-1],len(ref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--references',type=Path,required=True,help='JSON object mapping audio filenames to verified spoken transcripts.')
    args = parser.parse_args()
    refs = json.loads(args.references.read_text(encoding='utf-8'))
    if not isinstance(refs,dict) or not refs:
        raise ValueError('Provide a non-empty filename-to-transcript JSON object.')
    rows,errors,words = [],0,0
    for name,reference in refs.items():
        path = (AUDIO_DIR/name).resolve()
        if path.parent != AUDIO_DIR.resolve() or not path.is_file() or not isinstance(reference,str):
            raise ValueError(f'Invalid reference entry: {name}')
        start = time.perf_counter()
        with path.open('rb') as stream:
            text = transcribe(stream,name)
        edits,length = word_error_counts(reference,text)
        errors += edits
        words += length
        rows.append({'filename':name,'reference':reference,'transcript':text,'seconds':time.perf_counter()-start})
    result = {'word_error_rate':errors/words,'word_errors':errors,'reference_words':words,'recordings':rows,
              'note':'WER uses lowercased word tokens; filenames are not reference transcripts.'}
    out = ROOT/'artifacts/audio_metrics.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
