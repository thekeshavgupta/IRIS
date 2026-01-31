from rank_bm25 import BM25Okapi
import math
import numpy as np
class BM25Ranker():
    
    def __init__(self, data, corpus, top_k_rank=20):
        self.data = data
        self.corpus = corpus
        self.k = top_k_rank
        self.ranker = BM25Okapi(self.corpus)
        # Create a map for faster index lookup using string representation of tokens
        self.doc_map = {" ".join(doc): i for i, doc in enumerate(self.corpus)}

    def __get_relevant_rank(self, text):
        goldAnswer = self.data[(self.data['question'] == text) & (self.data['relevance_label'] == 1)]['long_answers'].values[0]
        query = text
        # Using the pre-initialized ranker
        output = self.ranker.get_scores(query.split())
        
        # Optimization: Use the map if possible, but fallback to list search if needed or if goldAnswer is guaranteed to be in corpus
        # " ".join(goldAnswer.split()) should match one of the keys in doc_map
        gold_key = " ".join(goldAnswer.split())
        if gold_key in self.doc_map:
            indi = self.doc_map[gold_key]
        else:
            # Fallback (though it should be there)
            indi = self.corpus.index(goldAnswer.split())
            
        relevance_score = output[indi]
        
        sorted_output = sorted(output, reverse=True)
        sorted_relevant_output = sorted_output[:self.k]
        
        if(relevance_score in sorted_relevant_output):
            return sorted_relevant_output.index(relevance_score)+1
        else:
            return 0

    def __get_bm25_score(self, row):
        query = row['question'].split()
        doc = row['long_answers'].split()
        # We need the score of THIS doc for THIS query.
        # Since get_scores returns scores for ALL docs, we find the index of this doc.
        doc_key = " ".join(doc)
        if doc_key in self.doc_map:
            idx = self.doc_map[doc_key]
            # BM25Okapi get_scores calculates for all. Optimisation could be to calculate just for one,
            # but the library optimized get_scores. Accessing by index is fast.
            # However, if we call get_scores for every row, it recomputes for all docs every time.
            # This is slow (2000 docs * 100k rows).
            # Better approach: Pre-compute scores? No, query changes.
            # Correct approach for efficiency: 
            # The library accepts a single doc? No.
            # We can implement the BM25 scoring formula for a single doc here manually or use the library.
            # The library `get_scores` uses `self.doc_freqs`, `self.idf`, etc.
            # Let's trust the library for now, but cache the query scores?
            # Many rows have the same query!
            # Optimization: Group by query.
            return 0 # Placeholder, logic handled in batch method
        return 0

    def __calculate_bm25_scores_batch(self):
        # Efficiently calculate scores by grouping by question
        # For each question, compute scores for all docs ONCE.
        # Then map the scores to the rows.
        
        question_groups = self.data.groupby('question')
        
        # We need to assign scores back to the dataframe.
        # Create a dictionary to hold (question, doc) -> score
        # Or just iterate and update.
        
        scores_map = {} # (question, doc_str) -> score
        
        for question, group in question_groups:
            tokenized_query = question.split()
            doc_scores = self.ranker.get_scores(tokenized_query)
            
            # Now we have scores for all docs in corpus.
            # We need to pick the ones relevant to the current rows.
            # Iterate over the rows for this question
            for idx, row in group.iterrows():
                doc_content = row['long_answers']
                doc_key = " ".join(doc_content.split())
                if doc_key in self.doc_map:
                    doc_idx = self.doc_map[doc_key]
                    score = doc_scores[doc_idx]
                    self.data.at[idx, 'bm25_score'] = score
                    
    def __get_bm25_mrr(self, score):
        if score == 0:
            return 0
        else:
            return 1/score

    def __calculate_bm25_ndcg(self, rankIndex):
        relevant_score=6
        irrelevant_score=2
        
        idcg = relevant_score/math.log2(2)
        for i in range(1, self.k):
            idcg+=(irrelevant_score/math.log2(i+1))
            
        dcg = 0
        for i in range(1, self.k+1):
            if i == rankIndex:
                dcg+=(relevant_score/math.log2(i+1))
            else:
                dcg+=(irrelevant_score/math.log2(i+1))
        
        return dcg/idcg
    
    def __calculate_bm25_metrics_batch(self):
        # Efficiently calculate ranks, MRR, NDCG in batch
        self.data['bm25Rank'] = 0
        self.data['bm25_mrr'] = 0.0
        self.data['bm25_ndcg'] = 0.0
        
        # Pre-calculate scores if not already done (Wait, getFullScoringData calls __calculate_bm25_scores_batch later)
        # But we need scores to calculate rank.
        # Actually __calculate_bm25_scores_batch computes scores for specificity (Q, D).
        # Here we need rank of Gold Answer against ALL corpus.
        
        # Group by question
        grouped = self.data.groupby('question')
        
        # We need to update self.data
        # Iterating groups is fast provided N groups is small (2000).
        
        for question, group in grouped:
            # Find gold answer for this question
            gold_rows = group[group['relevance_label'] == 1]
            if gold_rows.empty:
                continue # Should not happen in training data
            
            gold_ans = gold_rows.iloc[0]['long_answers']
            gold_key = " ".join(gold_ans.split())
            
            if gold_key not in self.doc_map:
                continue # Should not happen
                
            gold_idx = self.doc_map[gold_key]
            
            # Get scores for all corpus
            tokenized_query = question.split()
            corpus_scores = self.ranker.get_scores(tokenized_query) # numpy array
            
            gold_score = corpus_scores[gold_idx]
            
            # Calculate rank: count how many have higher score
            # +1 for 1-based rank
            rank = np.sum(corpus_scores > gold_score) + 1
            
            # Check if rank is within k
            if rank > self.k:
                rank_val = 0
            else:
                rank_val = rank
                
            # MRR
            mrr = 0 if rank_val == 0 else 1/rank_val
            
            # NDCG - Using existing logic
            # Existing logic is complex?
            # __calculate_bm25_ndcg uses rankIndex
            ndcg = self.__calculate_bm25_ndcg(rank_val)
            
            # Assign to all rows for this question
            self.data.loc[group.index, 'bm25Rank'] = rank_val
            self.data.loc[group.index, 'bm25_mrr'] = mrr
            self.data.loc[group.index, 'bm25_ndcg'] = ndcg
            
    def __getBMRank(self):
        # Deprecated by batch method
        pass

    def __getBMMRRScore(self):
        pass
    
    def __getBMNDCGScore(self):
        pass
    
    def getFullScoringData(self):
        self.__calculate_bm25_metrics_batch() # Calculates Rank, MRR, NDCG
        self.__calculate_bm25_scores_batch() # Calculates raw scores for features
        return self.data