import io
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import app
from src.audio import decode_audio, extract_image_number
from src.indexing import image_index, text_index, comparison_index, _load_vectors, _load_text, _load_comparison, IndexNotReady
from src.retrieval import search_text_fast, search_image_fast, search_multimodal
from scripts.evaluate_audio import word_error_counts


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root/'images'
        self.images.mkdir()
        Image.new('RGB',(40,40),'red').save(self.images/'a.jpg')
        Image.new('RGB',(40,40),'blue').save(self.images/'b.jpg')
        self.items = [{'filename':'a.jpg','descriptions':['red cow in a pasture']},
                      {'filename':'b.jpg','descriptions':['blue sea and ships']}]
        self.patches = [patch('src.indexing.IMAGE_DIR',self.images),
                        patch('src.indexing.INDEX_DIR',self.root/'indexes'),
                        patch('app.catalog',return_value=self.items),
                        patch('app.get_mapping_lookup',return_value={1:self.items[0],2:self.items[1]})]
        for p in self.patches:
            p.start()
        app.app.config.update(TESTING=True)
        self.client = app.app.test_client()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        _load_vectors.cache_clear()
        _load_text.cache_clear()
        _load_comparison.cache_clear()
        self.temp.cleanup()

    def test_pixel_index_survives_memory_cache_reset(self):
        image_index(self.items,'pixels',build=True)
        _load_vectors.cache_clear()
        with patch('src.retrieval._visual_feature',side_effect=AssertionError('must reuse disk index')):
            matrix,names = image_index(self.items,'pixels')
        self.assertEqual(matrix.shape,(2,3072))
        self.assertEqual(names,('a.jpg','b.jpg'))

    def test_image_change_invalidates_index(self):
        image_index(self.items,'pixels',build=True)
        Image.new('RGB',(90,90),'green').save(self.images/'a.jpg')
        with self.assertRaises(IndexNotReady):
            image_index(self.items,'pixels')

    def test_image_search_returns_actual_uploaded_image(self):
        image_index(self.items,'pixels',build=True)
        with (self.images/'a.jpg').open('rb') as stream:
            result = search_image_fast(stream,self.items)
        self.assertEqual(result[0]['filename'],'a.jpg')

    def test_text_search_and_unknown_words(self):
        self.assertEqual(search_text_fast('cow pasture',self.items)[0]['filename'],'a.jpg')
        self.assertEqual(search_text_fast('xyzzyunseenword',self.items),[])

    def test_text_caption_change_invalidates_cache(self):
        search_text_fast('cow',self.items)
        changed = [{**self.items[0],'descriptions':['purple flowers']},self.items[1]]
        self.assertEqual(search_text_fast('cow',changed),[])

    def test_combined_query_uses_image_weight(self):
        with patch('src.indexing.image_index',return_value=(np.eye(2),('a.jpg','b.jpg'))), \
             patch('src.retrieval._embed_images',return_value=np.array([[0.,1.]])), \
             patch('src.retrieval._embed_text',return_value=np.array([1.,0.])):
            result = search_multimodal('a scene',self.images/'a.jpg',self.items,.8)
        self.assertEqual(result[0]['filename'],'b.jpg')

    def test_combined_invalid_weight(self):
        with self.assertRaises(ValueError):
            search_multimodal('query',None,self.items,1.1)

    def test_invalid_named_audio_cannot_create_transcript(self):
        with patch('src.audio._model',side_effect=AssertionError('invalid audio must be rejected before inference')):
            response = self.client.post('/search/audio',data={'audio':(io.BytesIO(b'not audio'),'image0001.mp3'),'mode':'fast'},content_type='multipart/form-data')
        self.assertEqual(response.status_code,400)
        self.assertNotIn(b'WHISPER TRANSCRIPT',response.data)
        self.assertNotIn(b'This is image 1.',response.data)

    def test_direct_lookup_has_no_transcript_or_similarity(self):
        response = self.client.post('/lookup',data={'number':'1'})
        self.assertEqual(response.status_code,200)
        self.assertIn(b'Number mapping',response.data)
        self.assertNotIn(b'WHISPER TRANSCRIPT',response.data)
        self.assertNotIn(b'1.000',response.data)

    def test_spoken_number_in_words_routes_to_mapping(self):
        with patch('app.transcribe',return_value='This is image two.'), \
             patch('app.search_text_fast',side_effect=AssertionError('number must not be a description')):
            response = self.client.post('/search/audio',data={'audio':(io.BytesIO(b'fixture'),'recording.wav'),'mode':'fast'},content_type='multipart/form-data')
        self.assertEqual(response.status_code,200)
        self.assertIn(b'b.jpg',response.data)
        self.assertIn(b'This is image two.',response.data)

    def test_unknown_spoken_number_does_not_fall_back(self):
        with patch('app.transcribe',return_value='This is image 99999.'),patch('app.search_text_fast') as fallback:
            response = self.client.post('/search/audio',data={'audio':(io.BytesIO(b'fixture'),'recording.wav'),'mode':'fast'},content_type='multipart/form-data')
        self.assertEqual(response.status_code,400)
        fallback.assert_not_called()

    def test_invalid_image_is_rejected(self):
        response = self.client.post('/search/image',data={'image':(io.BytesIO(b'bad'),'bad.jpg'),'mode':'fast'},content_type='multipart/form-data')
        self.assertEqual(response.status_code,400)
        self.assertIn(b'not a readable image',response.data)

    def test_mode_is_explicit_and_validated(self):
        response = self.client.post('/search/text',data={'query':'cow','mode':'magic'})
        self.assertEqual(response.status_code,400)

    def test_text_and_audio_list_five_methods_but_image_keeps_image_methods(self):
        for path in ('/search/text', '/search/audio'):
            html = self.client.get(path).data
            for mode in ('clip', 'fast', 'bm25', 'lsa', 'minilm'):
                self.assertIn(f'value="{mode}"'.encode(), html)
        self.assertNotIn(b'value="bm25"', self.client.get('/search/image').data)
        self.assertEqual(self.client.post('/search/image', data={'mode':'bm25'}).status_code, 400)

    def test_bm25_search_reuses_persisted_index_and_preserves_raw_score(self):
        comparison_index(self.items, 'bm25', build=True)
        _load_comparison.cache_clear()
        with patch('src.comparison.BM25Retriever.__init__', side_effect=AssertionError('must reuse index')):
            response = self.client.post('/search/text', data={'query':'red cow pasture', 'mode':'bm25'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'a.jpg', response.data)
        self.assertIn(b'BM25 relevance', response.data)
        from src.comparison import search_caption_model
        self.assertGreater(search_caption_model('red cow pasture', self.items, 'bm25')[0]['score'], 1.0)

    def test_missing_comparison_index_returns_preparation_instructions(self):
        response = self.client.post('/search/text', data={'query':'cow', 'mode':'lsa'})
        self.assertEqual(response.status_code, 503)
        self.assertIn(b'build_index.py --mode lsa', response.data)

    def test_lsa_unknown_words_produce_no_matches(self):
        comparison_index(self.items, 'lsa', build=True)
        response = self.client.post('/search/text', data={'query':'zzzzunknown', 'mode':'lsa'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'No matching descriptions', response.data)

    def test_minilm_text_and_audio_route_to_selected_model(self):
        with patch('app.search_caption_model', return_value=[]) as search:
            response = self.client.post('/search/text', data={'query':'cow', 'mode':'minilm'})
            self.assertEqual(response.status_code, 200)
            search.assert_called_once_with('cow', self.items, 'minilm')
        with patch('app.transcribe', return_value='a blue sea'), \
             patch('app.search_caption_model', return_value=[]) as search:
            response = self.client.post('/search/audio', data={'mode':'minilm',
                'audio':(io.BytesIO(b'fixture'), 'recording.wav')}, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)
            search.assert_called_once_with('a blue sea', self.items, 'minilm')

    def test_all_new_pages_render(self):
        for path in ['/search/combined','/status','/evaluation','/search/audio']:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code,200)


class AudioAndModelTests(unittest.TestCase):
    def test_decode_wave_without_external_ffmpeg(self):
        # A generated tone is a decoder fixture, never a speech or artwork result.
        samples = (np.sin(np.arange(3200)*.1)*5000).astype('<i2')
        stream = io.BytesIO()
        with wave.open(stream,'wb') as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(16000)
            recording.writeframes(samples.tobytes())
        stream.seek(0)
        result = decode_audio(stream)
        self.assertEqual(result.dtype,np.float32)
        self.assertEqual(len(result),3200)

    def test_number_parser(self):
        examples = {'This is image one.':1,'image 0012':12,'show me artwork number twenty-three':23,
                    'picture three thousand six hundred and six':3606,
                    'an image of two cows':None,'cow farmer field':None,'image 0':0,
                    'image 3,606':3606,'image one two':None,'image hundred two':None}
        for text,expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(extract_image_number(text),expected)

    def test_wer_counts_edits(self):
        self.assertEqual(word_error_counts('a red cow','a blue cow'),(1,3))
        self.assertEqual(word_error_counts('hello world','hello world'),(0,2))

    def test_evaluation_does_not_count_zero_overlap_as_a_match(self):
        from scripts.evaluate_retrieval import ranking_metrics
        self.assertEqual(ranking_metrics(np.zeros(2),'a.jpg',('a.jpg','b.jpg'),positive_only=True),float('inf'))
        self.assertEqual(ranking_metrics(np.array([.1,.9]),'a.jpg',('a.jpg','b.jpg')),2)

    def test_transformers_tensor_and_pooled_outputs(self):
        import torch
        from src.models import normalized_features
        value = torch.tensor([[3.,4.]])
        np.testing.assert_allclose(normalized_features(value),[[.6,.8]],rtol=1e-6)
        np.testing.assert_allclose(normalized_features(SimpleNamespace(pooler_output=value)),[[.6,.8]],rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
