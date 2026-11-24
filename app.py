"""Flask routes for artwork retrieval with explicit modes and genuine transcription."""
import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from flask import Flask, flash, render_template, request, send_from_directory
from config import (SECRET_KEY, MAX_UPLOAD_MB, IMAGE_DIR, ROOT, TEXT_SEARCH_MODE,
                    IMAGE_SEARCH_MODE, MAX_AUDIO_SECONDS)
from src.data import load_catalog, get_mapping_lookup
from src.retrieval import search_text, search_text_fast, search_image, search_image_fast, search_multimodal
from src.audio import transcribe, extract_image_number
from src.comparison import search_caption_model

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

@lru_cache(maxsize=1)
def catalog():
    return load_catalog()

@app.errorhandler(413)
def too_large(error):
    return render_template('error.html', message=f'Upload exceeds {MAX_UPLOAD_MB} MB.'), 413

def attempt(operation):
    start = time.perf_counter()
    try:
        results = operation()
        if not results:
            flash('No matching descriptions were found. Try different words.', 'info')
        return results, round(time.perf_counter()-start, 3), 200
    except ValueError as exc:
        flash(str(exc), 'error')
        return [], None, 400
    except RuntimeError as exc:
        logging.warning('%s', exc)
        flash(str(exc), 'error')
        return [], None, 503
    except Exception:
        logging.exception('Retrieval failed')
        flash('Search failed. Check the application terminal for details.', 'error')
        return [], None, 500

def search_mode(default, kind='text'):
    selected = request.form.get('mode', default)
    allowed = ('clip', 'fast') if kind == 'image' else ('clip', 'fast', 'bm25', 'lsa', 'minilm')
    if selected not in allowed:
        raise ValueError('Choose one of the listed search methods.')
    return selected

def text_results(query, mode):
    if not query or len(query) > 2000:
        raise ValueError('Enter a description between 1 and 2,000 characters.')
    if mode in ('bm25', 'lsa', 'minilm'):
        return search_caption_model(query, catalog(), mode)
    return (search_text if mode == 'clip' else search_text_fast)(query, catalog())

def lookup_result(number):
    entry = get_mapping_lookup().get(number)
    if entry is None:
        raise ValueError(f'Artwork number {number} does not exist in the mapping.')
    return [{**entry, 'score': None, 'match_type': 'Number mapping'}]

@app.route('/')
def home():
    return render_template('home.html', count=len(catalog()))

@app.route('/search/text', methods=['GET', 'POST'])
def text_page():
    mode, query, results, elapsed, code = TEXT_SEARCH_MODE, '', [], None, 200
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        def search():
            nonlocal mode
            mode = search_mode(TEXT_SEARCH_MODE)
            return text_results(query, mode)
        results, elapsed, code = attempt(search)
    return render_template('text.html', results=results, elapsed=elapsed, mode=mode, query=query), code

@app.route('/search/image', methods=['GET', 'POST'])
def image_page():
    mode, results, elapsed, code = IMAGE_SEARCH_MODE, [], None, 200
    if request.method == 'POST':
        def search():
            nonlocal mode
            mode = search_mode(IMAGE_SEARCH_MODE, kind='image')
            file = request.files.get('image')
            if not file or not file.filename:
                raise ValueError('Choose an artwork image.')
            return (search_image if mode == 'clip' else search_image_fast)(file.stream, catalog())
        results, elapsed, code = attempt(search)
    return render_template('image.html', results=results, elapsed=elapsed, mode=mode), code

@app.route('/search/combined', methods=['GET', 'POST'])
def combined_page():
    results, elapsed, code = [], None, 200
    query = request.form.get('query', '')
    weight = request.form.get('image_weight', '0.5')
    if request.method == 'POST':
        def search():
            file = request.files.get('image')
            if not file or not file.filename:
                raise ValueError('Choose an image and enter a description.')
            return search_multimodal(query, file.stream, catalog(), float(weight))
        results, elapsed, code = attempt(search)
    return render_template('combined.html', results=results, elapsed=elapsed, query=query, weight=weight), code

@app.route('/search/audio', methods=['GET', 'POST'])
def audio_page():
    transcript, results, elapsed, code = None, [], None, 200
    mode = TEXT_SEARCH_MODE
    if request.method == 'POST':
        def search():
            nonlocal transcript, mode
            mode = search_mode(TEXT_SEARCH_MODE)
            file = request.files.get('audio')
            if not file or not file.filename:
                raise ValueError('Choose a recording or record using the microphone.')
            transcript = transcribe(file.stream, file.filename)
            number = extract_image_number(transcript)
            return lookup_result(number) if number is not None else text_results(transcript, mode)
        results, elapsed, code = attempt(search)
    return render_template('audio.html', transcript=transcript, results=results, elapsed=elapsed,
                           mode=mode, max_seconds=MAX_AUDIO_SECONDS), code

@app.route('/lookup', methods=['POST'])
def lookup_page():
    results, elapsed, code = attempt(lambda: lookup_result(int(request.form.get('number', ''))))
    return render_template('audio.html', transcript=None, results=results, elapsed=elapsed,
                           mode=TEXT_SEARCH_MODE, max_seconds=MAX_AUDIO_SECONDS, lookup=True), code

@app.route('/gallery')
def gallery():
    items = catalog()
    page = max(1, request.args.get('page', 1, type=int))
    return render_template('gallery.html', items=items[(page-1)*36:page*36], page=page, has_next=page*36<len(items))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/status')
def status_page():
    from src.readiness import status
    return render_template('status.html', status=status(catalog()))

@app.route('/api/status')
def api_status():
    from src.readiness import status
    return status(catalog())

@app.route('/evaluation')
def evaluation_page():
    path = ROOT / 'artifacts' / 'retrieval_metrics.json'
    metrics = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    return render_template('evaluation.html', metrics=metrics)

@app.route('/artwork/<path:filename>')
def artwork(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    if os.getenv('WARM_MODELS', '1') == '1':
        from src.readiness import status
        ready = status(catalog())
        if ready['indexes']['clip']:
            from src.models import clip_components
            logging.info('Warming CLIP before accepting searches...')
            clip_components()
        if ready['speech_ready']:
            from src.audio import _model
            logging.info('Warming speech recognition...')
            _model()
    app.run(host='127.0.0.1', port=int(os.getenv('PORT', '5000')),
            debug=os.getenv('FLASK_DEBUG', '0') == '1', use_reloader=False)
