import unittest
import numpy as np
from src.comparison import BM25Retriever, LSARetriever
from scripts.evaluate_retrieval import held_out_split, ranking_metrics


class ComparisonTests(unittest.TestCase):
    def test_holdout_removes_every_duplicate_and_is_independent_of_query_cap(self):
        items = [{'filename':'a', 'descriptions':['remaining', 'held', 'held']},
                 {'filename':'b', 'descriptions':['blue', 'sea']},
                 {'filename':'c', 'descriptions':['only']}]
        queries, training = held_out_split(items)
        self.assertEqual(queries, [('a', 'held'), ('b', 'sea')])
        self.assertEqual([item['descriptions'] for item in training],
                         [['remaining'], ['blue'], ['only']])
        self.assertEqual(items[0]['descriptions'], ['remaining', 'held', 'held'])

    def test_bm25_matches_hand_computed_scores_and_batch_scoring(self):
        model = BM25Retriever(['cat cat dog', 'dog', 'fish fish'])
        scores = model.score(['cat'])[0]
        expected = np.log(1 + 2.5 / 1.5) * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / 2))
        self.assertAlmostEqual(scores[0], expected)
        np.testing.assert_array_equal(scores[1:], [0, 0])
        np.testing.assert_allclose(model.score(['cat', 'dog']),
                                   np.vstack([model.score(['cat']), model.score(['dog'])]))
        np.testing.assert_array_equal(model.score(['unknown']), [[0, 0, 0]])

    def test_lsa_retrieves_exact_document_and_rejects_oov(self):
        model = LSARetriever(['red cow pasture', 'blue sea ships', 'white snow mountain'], dimensions=2)
        scores = model.score(['blue sea ships', 'zzzzunknown'])
        self.assertEqual(np.argmax(scores[0]), 1)
        self.assertTrue(np.all(np.isneginf(scores[1])))
        self.assertEqual(ranking_metrics(scores[1], 'b', ['a', 'b', 'c']), float('inf'))

    def test_ranking_ties_and_no_overlap(self):
        self.assertEqual(ranking_metrics(np.array([1, 1, 0]), 'b', ['a', 'b', 'c']), 2)
        self.assertEqual(ranking_metrics(np.zeros(3), 'a', ['a', 'b', 'c'], True), float('inf'))


if __name__ == '__main__':
    unittest.main()
