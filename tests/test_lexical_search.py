import unittest
from typing import Dict
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coir.beir.retrieval.search.lexical.jaccard_search import LexicalJaccardSearch
from coir.beir.retrieval.search.lexical.bm25_search import LexicalBM25Search


class TestLexicalJaccardSearch(unittest.TestCase):
    """Test cases for LexicalJaccardSearch class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search = LexicalJaccardSearch()
        
        # Sample corpus for testing
        self.corpus = {
            "doc1": {"title": "Machine Learning", "text": "machine learning algorithms are powerful"},
            "doc2": {"title": "Deep Learning", "text": "deep learning neural networks"},
            "doc3": {"title": "AI Research", "text": "artificial intelligence research methods"},
            "doc4": {"title": "Data Science", "text": "data science machine learning analytics"}
        }
        
        # Sample queries for testing
        self.queries = {
            "q1": "machine learning",
            "q2": "deep neural networks",
            "q3": "artificial intelligence"
        }
    
    def test_initialization(self):
        """Test proper initialization of LexicalJaccardSearch."""
        search = LexicalJaccardSearch(batch_size=64)
        self.assertEqual(search.batch_size, 64)
        self.assertEqual(search.results, {})
    
    def test_jaccard_similarity_calculation(self):
        """Test Jaccard similarity calculation."""
        results = self.search.search(self.corpus, self.queries, top_k=2)
        
        # Check that results are returned for all queries
        self.assertEqual(len(results), 3)
        self.assertIn("q1", results)
        self.assertIn("q2", results)
        self.assertIn("q3", results)
        
        # Check that scores are between 0 and 1 (Jaccard similarity range)
        for query_id, docs in results.items():
            for doc_id, score in docs.items():
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
    
    def test_self_match_prevention(self):
        """Test that self-matches are prevented."""
        # Create a corpus where query IDs match document IDs
        corpus_with_query_ids = {
            "q1": {"title": "Query Doc", "text": "machine learning algorithms"},
            "doc1": {"title": "Regular Doc", "text": "machine learning is great"},
            "doc2": {"title": "Another Doc", "text": "deep learning networks"}
        }
        
        queries = {"q1": "machine learning"}
        
        results = self.search.search(corpus_with_query_ids, queries, top_k=3)
        
        # Ensure q1 doesn't return itself in results
        self.assertNotIn("q1", results["q1"])
        self.assertIn("doc1", results["q1"])
    
    def test_top_k_limitation(self):
        """Test that top_k parameter limits results correctly."""
        results = self.search.search(self.corpus, self.queries, top_k=2)
        
        for query_id, docs in results.items():
            self.assertLessEqual(len(docs), 2)
    
    def test_empty_corpus(self):
        """Test behavior with empty corpus."""
        empty_corpus = {}
        results = self.search.search(empty_corpus, self.queries, top_k=5)
        
        for query_id, docs in results.items():
            self.assertEqual(len(docs), 0)
    
    def test_empty_queries(self):
        """Test behavior with empty queries."""
        empty_queries = {}
        results = self.search.search(self.corpus, empty_queries, top_k=5)
        
        self.assertEqual(len(results), 0)


class TestLexicalBM25Search(unittest.TestCase):
    """Test cases for LexicalBM25Search class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search = LexicalBM25Search()
        
        # Sample corpus for testing
        self.corpus = {
            "doc1": {"title": "Machine Learning", "text": "machine learning algorithms are powerful tools"},
            "doc2": {"title": "Deep Learning", "text": "deep learning neural networks process data"},
            "doc3": {"title": "AI Research", "text": "artificial intelligence research methods and techniques"},
            "doc4": {"title": "Data Science", "text": "data science machine learning analytics and statistics"}
        }
        
        # Sample queries for testing
        self.queries = {
            "q1": "machine learning algorithms",
            "q2": "deep neural networks",
            "q3": "artificial intelligence research"
        }
    
    def test_initialization(self):
        """Test proper initialization of LexicalBM25Search."""
        search = LexicalBM25Search(batch_size=64)
        self.assertEqual(search.batch_size, 64)
        self.assertEqual(search.results, {})
    
    def test_bm25_search_functionality(self):
        """Test BM25 search functionality."""
        results = self.search.search(self.corpus, self.queries, top_k=3)
        
        # Check that results are returned for all queries
        self.assertEqual(len(results), 3)
        self.assertIn("q1", results)
        self.assertIn("q2", results)
        self.assertIn("q3", results)
        
        # Check that scores are positive (BM25 scores should be positive)
        for query_id, docs in results.items():
            for doc_id, score in docs.items():
                self.assertGreaterEqual(score, 0.0)
                self.assertIsInstance(score, float)
    
    def test_self_match_prevention(self):
        """Test that self-matches are prevented."""
        # Create a corpus where query IDs match document IDs
        corpus_with_query_ids = {
            "q1": {"title": "Query Doc", "text": "machine learning algorithms are powerful"},
            "doc1": {"title": "Regular Doc", "text": "machine learning is great for analysis"},
            "doc2": {"title": "Another Doc", "text": "deep learning networks are complex"}
        }
        
        queries = {"q1": "machine learning algorithms"}
        
        results = self.search.search(corpus_with_query_ids, queries, top_k=3)
        
        # Ensure q1 doesn't return itself in results
        self.assertNotIn("q1", results["q1"])
    
    def test_top_k_limitation(self):
        """Test that top_k parameter limits results correctly."""
        results = self.search.search(self.corpus, self.queries, top_k=2)
        
        for query_id, docs in results.items():
            self.assertLessEqual(len(docs), 2)
    
    def test_corpus_smaller_than_top_k(self):
        """Test behavior when corpus is smaller than top_k."""
        small_corpus = {
            "doc1": {"title": "Test", "text": "machine learning"}
        }
        
        results = self.search.search(small_corpus, {"q1": "machine learning"}, top_k=10)
        
        # Should return at most the number of documents in corpus
        self.assertLessEqual(len(results["q1"]), 1)
    
    def test_empty_corpus(self):
        """Test behavior with empty corpus."""
        empty_corpus = {}
        results = self.search.search(empty_corpus, self.queries, top_k=5)
        
        for query_id, docs in results.items():
            self.assertEqual(len(docs), 0)
    
    def test_empty_queries(self):
        """Test behavior with empty queries."""
        empty_queries = {}
        results = self.search.search(self.corpus, empty_queries, top_k=5)
        
        self.assertEqual(len(results), 0)


class TestLexicalSearchComparison(unittest.TestCase):
    """Test cases comparing both lexical search implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.jaccard_search = LexicalJaccardSearch()
        self.bm25_search = LexicalBM25Search()
        
        # Sample corpus for testing
        self.corpus = {
            "doc1": {"title": "ML", "text": "machine learning algorithms"},
            "doc2": {"title": "DL", "text": "deep learning networks"},
            "doc3": {"title": "AI", "text": "artificial intelligence"}
        }
        
        self.queries = {"q1": "machine learning"}
    
    def test_result_format_consistency(self):
        """Test that both implementations return results in the same format."""
        jaccard_results = self.jaccard_search.search(self.corpus, self.queries, top_k=3)
        bm25_results = self.bm25_search.search(self.corpus, self.queries, top_k=3)
        
        # Both should return dict[str, dict[str, float]]
        self.assertIsInstance(jaccard_results, dict)
        self.assertIsInstance(bm25_results, dict)
        
        for query_id in self.queries.keys():
            self.assertIn(query_id, jaccard_results)
            self.assertIn(query_id, bm25_results)
            
            self.assertIsInstance(jaccard_results[query_id], dict)
            self.assertIsInstance(bm25_results[query_id], dict)
            
            for doc_id, score in jaccard_results[query_id].items():
                self.assertIsInstance(doc_id, str)
                self.assertIsInstance(score, float)
            
            for doc_id, score in bm25_results[query_id].items():
                self.assertIsInstance(doc_id, str)
                self.assertIsInstance(score, float)


if __name__ == '__main__':
    unittest.main()