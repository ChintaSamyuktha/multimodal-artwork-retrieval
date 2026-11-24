"""Decode speech content; filenames never imply recognized words."""
import re
from functools import lru_cache
from threading import RLock
import numpy as np
from config import WHISPER_MODEL, WHISPER_CACHE, MAX_AUDIO_SECONDS
from src.models import configure_torch

_lock = RLock()

def decode_audio(stream):
    try:
        import av
    except ImportError as exc:
        raise RuntimeError('Audio decoding is unavailable. Install the project requirements (including PyAV).') from exc
    frames, count = [], 0
    try:
        with av.open(stream) as container:
            if not container.streams.audio:
                raise ValueError('The uploaded file contains no audio stream.')
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    count += converted.samples
                    if count > 16000 * MAX_AUDIO_SECONDS:
                        raise ValueError(f'Use a recording no longer than {MAX_AUDIO_SECONDS} seconds.')
                    frames.append(converted.to_ndarray().reshape(-1))
            for converted in resampler.resample(None):
                frames.append(converted.to_ndarray().reshape(-1))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('The uploaded file is not a readable audio recording.') from exc
    if not frames:
        raise ValueError('The recording contains no audio samples.')
    audio = np.concatenate(frames).astype(np.float32)
    if len(audio) > 16000 * MAX_AUDIO_SECONDS:
        raise ValueError(f'Use a recording no longer than {MAX_AUDIO_SECONDS} seconds.')
    if np.max(np.abs(audio)) < 1e-5:
        raise ValueError('The recording is silent. Please record speech and try again.')
    return audio

@lru_cache(maxsize=1)
def _model():
    import whisper
    from src.readiness import whisper_ready
    if not whisper_ready():
        raise RuntimeError('The speech model is not prepared. Run python scripts/prepare_models.py.')
    torch = configure_torch()
    return whisper.load_model(WHISPER_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu',
                              download_root=str(WHISPER_CACHE))

def transcribe(stream, filename=None):
    audio = decode_audio(stream)
    torch = configure_torch()
    with _lock:
        result = _model().transcribe(audio, language='en' if WHISPER_MODEL.endswith('.en') else None,
                                    fp16=torch.cuda.is_available(), temperature=0, beam_size=1,
                                    condition_on_previous_text=False, verbose=None)
    text = result['text'].strip()
    if not text:
        raise ValueError('No speech was recognized. Try a clearer recording.')
    return text

_SMALL = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
_TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))

def extract_image_number(text):
    match = re.search(r'\b(?:image|picture|artwork)\s*(?:number\s*)?(.+)', text, re.I)
    if not match:
        return None
    phrase = match.group(1).strip().lower().rstrip('.!?')
    if re.fullmatch(r'\d+|\d{1,3}(?:,\d{3})+', phrase):
        return int(phrase.replace(',', ''))
    tokens = phrase.replace('-', ' ').split()
    if not tokens or any(t not in {*_SMALL, *_TENS, 'and', 'hundred', 'thousand'} for t in tokens):
        return None
    def under_thousand(words):
        words = [word for word in words if word != 'and']
        hundreds = 0
        if 'hundred' in words:
            if words.count('hundred') != 1 or len(words) < 2 or words[1] != 'hundred' or words[0] not in _SMALL or not 1 <= _SMALL[words[0]] <= 9:
                return None
            hundreds = _SMALL[words[0]] * 100
            words = words[2:]
        if not words:
            return hundreds
        if len(words) == 1 and words[0] in _SMALL:
            return hundreds + _SMALL[words[0]]
        if len(words) == 1 and words[0] in _TENS:
            return hundreds + _TENS[words[0]]
        if len(words) == 2 and words[0] in _TENS and words[1] in _SMALL and 1 <= _SMALL[words[1]] <= 9:
            return hundreds + _TENS[words[0]] + _SMALL[words[1]]
        return None
    if tokens.count('thousand') > 1:
        return None
    if 'thousand' in tokens:
        split = tokens.index('thousand')
        left, right = under_thousand(tokens[:split]), under_thousand(tokens[split+1:])
        return left * 1000 + right if left and right is not None else None
    return under_thousand(tokens)
