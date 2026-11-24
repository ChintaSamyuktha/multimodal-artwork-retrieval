# Multimodal Artistic Recognition Using Image, Text and Audio

A local Flask application for retrieving artwork from an image, a written description, a spoken description, or a combined image and text query.

## Windows setup

Use Python 3.11, open this project folder in VS Code, and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/preprocess_dataset.py
python scripts/prepare_models.py
python scripts/build_index.py --mode all
python app.py
```

Open http://127.0.0.1:5000 after the terminal reports that Flask is listening. The server warms local models before accepting searches. Models download only during preparation, never during a web search. Downloading and building the CLIP image index can take time on a CPU; do this once before the demonstration.

If an older pip reports certificate errors on a Windows system that trusts the university or antivirus certificate, use `python -m pip install --use-feature=truststore -r requirements.txt`. Certificate verification stays enabled. The model preparation command uses the operating system trust store.

Audio decoding uses PyAV and its bundled FFmpeg libraries. A separate FFmpeg command on PATH is not required. Whisper defaults to `tiny.en` for shorter English recordings on CPU. Set `WHISPER_MODEL=base` or another supported Whisper model, rerun preparation, and restart if a different model is needed.

## Search methods

- **CLIP** is the default image and text retrieval model: `openai/clip-vit-base-patch32`.
- **TF-IDF baseline** compares a text query with the supplied artwork descriptions.
- **Pixel baseline** compares normalized 32 by 32 RGB images. It captures colour and coarse layout, not semantic meaning.
- **BM25, LSA and MiniLM** are available in text search and for transcribed audio descriptions. Their full-catalogue caption indexes are prepared with `python scripts/build_index.py --mode all` and reused across searches and restarts.
- **Audio** always decodes the actual recording and transcribes it with Whisper. Its filename is never used as recognized speech.
- **Combined search** averages normalized CLIP image and text representations with an adjustable image weight and normalizes the combined vector before ranking.
- **Number lookup** is an explicitly separate metadata operation. It has no transcription and no similarity score.

Each search form names its method. There is no silent substitution when CLIP is unavailable. Visit **Setup** to see which indexes and speech model are ready. A search with an unprepared image index fails promptly with preparation instructions.

Scores on result cards are cosine similarities or BM25 relevance scores, as labelled, not accuracy percentages. Query time is the measured time for that request, not a benchmark average.

## Dataset and number mapping

The inspected collection has 3,606 JPGs in `dataset/images/images`, 3,606 filename-keyed JSON records in `dataset/text.json`, and 3,606 MP3s in `dataset/audio/audio_data`. Each artwork has multiple descriptions.

No verified source mapping between audio numbers and artwork filenames was supplied. `dataset/image_number_mapping.csv` records a **project-defined convention**: sort artwork filenames case-insensitively and assign consecutive numbers starting at one. This is not evidence of the dataset creator's intended alignment. Add the original Kaggle dataset URL and license to the academic report.

Spoken references such as “This is image one” or “image 123” use this convention. Unmapped numbers produce an error rather than an unrelated description search. The separate lookup form can demonstrate the mapping without implying that speech was recognized.

## Persistent indexes and performance

`artifacts/indexes` stores image vectors in NumPy archives and caption model vectors/vectorizers in locally generated Joblib caches. Cache filenames depend on the image collection, model name and index version; description caches also depend on caption contents. Image size or modification-time changes invalidate the corresponding image index. MiniLM caches caption vectors and loads its local encoder for query inference.

Flask loads these indexes instead of reopening and embedding every artwork per query. Keep the server running to reuse model and memory caches. The model warmup at startup prevents a browser request from triggering the initial model load.

After adding/removing/changing artwork files, stop Flask, rebuild indexes, and restart. To prepare just one baseline:

```powershell
python scripts/build_index.py --mode pixels
python scripts/build_index.py --mode tfidf
```

Only load project-generated Joblib caches; never replace an index with an untrusted downloaded cache.

## Evaluation and actual results

```powershell
python scripts/evaluate_retrieval.py --mode all
python scripts/summarize_comparison.py
```

This evaluates five retrieval methods on the same candidate artwork identities and queries: **TF-IDF, BM25, LSA, MiniLM, and CLIP**. Prepare MiniLM once with `python scripts/prepare_models.py --only minilm` before evaluating (the normal preparation command also includes it). The final caption of each artwork with at least two distinct captions is the query. Every exact occurrence of that caption is excluded from that artwork's caption document for all four text methods. CLIP uses image embeddings, without project-specific fine-tuning.

BM25 uses Okapi term weighting with k1=1.5, b=0.75 and positive IDF. LSA applies 256-component truncated SVD (seed 42) to TF-IDF and ranks normalized latent vectors. MiniLM uses the frozen `sentence-transformers/all-MiniLM-L6-v2` encoder, masked mean pooling, a 256-token limit per caption and normalized caption embeddings averaged per artwork. These are three distinct additional retrieval approaches; BM25 and LSA are classical models, not additional neural networks. Parameters are fixed before evaluation, with no tuning on the held-out queries.

The resulting `artifacts/retrieval_metrics.json` contains Recall@1, Recall@5, MRR, candidate and query counts, query fingerprint, and mean query latency from up to 30 individually timed warm searches. The Evaluation page displays this file. Model loading and index creation are excluded from warm query latency. The remaining supplied captions and an individual held-out caption can still share wording, so this is a **collection-specific caption benchmark**, not an independent generalization test.

A single-method run is available with `--mode tfidf`, `--mode bm25`, `--mode lsa`, `--mode minilm` or `--mode clip`. An optional `--limit 200` caps queries while leaving all candidate artworks in the collection; the saved query count discloses that cap. Never label a partial run as evaluation on all queries. Single-method runs replace the output file, so use `--output artifacts/bm25_metrics.json` (for example) to preserve the five-method report. `artifacts/retrieval_metrics_ranks.csv` saves each target's rank for each method; `no_match` contributes zero to reciprocal rank and recalls. JSON checkpoints save completed methods and explicitly mark incomplete runs.

The additional models are implemented in `src/comparison.py` for comparative experiments and interactive text/audio description search. The text and audio dropdowns offer CLIP, TF-IDF, BM25, LSA and MiniLM; the image dropdown offers CLIP and the pixel baseline. Interactive caption indexes use the full catalogue; evaluation constructs a separate held-out caption gallery. Caption retrieval has access to text information that CLIP's image gallery does not; differences in these scores do not isolate encoder quality or establish image, speech or combined-query performance. BM25 result scores are unbounded relevance scores and are labelled accordingly.

`scripts/summarize_comparison.py` requires a completed five-method run and exports `artifacts/five_model_comparison.csv` and `artifacts/five_model_discussion.txt` from the actual metrics. The Evaluation page automatically displays all five methods, with separate ranking and timing charts. The original two-method measurements are preserved in `artifacts/retrieval_metrics_two_method_20261121.json`; warm timings in a fresh run can differ.

For real speech WER, supply independently checked transcripts as a JSON object mapping audio filenames to what is actually spoken, then run:

```powershell
python scripts/evaluate_audio.py --references path/to/verified_transcripts.json
```

WER uses lowercased word tokens and edit distance. Filenames and generated number mappings are not speech references. No WER is reported when reference transcripts are unavailable.

## Microphone recording

Use a current Chrome or Edge browser at localhost or HTTPS and allow microphone access. Click Start recording, speak, click Stop, listen to the preview, and submit. Recordings are limited to 120 seconds by default. Uploading MP3, WAV, WebM, OGG or M4A is also supported when the container is decodable by PyAV.

## Configuration

Copy `.env.example` to `.env` and edit the required settings. Paths default relative to the project directory. Available settings include `DATASET_DIR`, `IMAGE_DIR`, `AUDIO_DIR`, `DESCRIPTIONS_FILE`, `MANIFEST_FILE`, `MODEL_CACHE`, `INDEX_DIR`, `CLIP_MODEL`, `WHISPER_MODEL`, `TEXT_SEARCH_MODE`, `IMAGE_SEARCH_MODE`, `BATCH_SIZE`, `TORCH_THREADS`, `MAX_AUDIO_SECONDS`, `MAX_UPLOAD_MB`, `WARM_MODELS`, `SECRET_KEY` and `PORT`.

Text search mode values are `clip`, `fast` (TF-IDF), `bm25`, `lsa` or `minilm`. Image search mode values are `clip` or `fast` (pixels). Methods must be selected explicitly in the interface or configuration. The development debugger and automatic reloader are disabled by default to prevent duplicate model loads. Restart Flask after changing Python code.

## Architecture

`app.py` supplies the Flask routes. `src/data.py` loads the catalogue and mapping; `src/models.py` loads local CLIP weights and adapts Transformers 4/5 outputs; `src/indexing.py` persists indexes; `src/retrieval.py` encodes and ranks queries; `src/audio.py` decodes and transcribes recordings; `src/readiness.py` checks setup without loading models. Templates and `static/app.js` provide forms, recording controls and result rendering.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests cover invalid uploads, genuine transcription routing, English number parsing, lookup score semantics, cache reuse/invalidation, combined embedding weights, Transformers output compatibility, audio decoding, and WER calculations. Mocked model outputs and generated decoder fixtures are only unit-test fixtures; they are not reported as measured artwork or speech predictions.
