from sentence_transformers import util, SentenceTransformer
import math
import torch
class SBertRanker():
    def __init__(self, data, corpus, top_k_rank=20):
        self.k = top_k_rank
        # Ensure data is a copy to avoid SettingWithCopy warnings and issues
        self.data = data.copy()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus = corpus
        self.corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
        self.doc_map = {doc: i for i, doc in enumerate(corpus)}
    
    def __get_relevant_score(self, query):
        goldAnswer = self.data[(self.data['question'] == query) & (self.data['relevance_label'] == 1)]['long_answers'].values[0]
        query_embeddings = self.model.encode(query, convert_to_tensor=True)
        goldAnswerCorpusIndex = self.corpus.index(goldAnswer)
        hits = util.semantic_search(query_embeddings=query_embeddings, corpus_embeddings=self.corpus_embeddings, top_k=self.k)[0]
        for hit in range(len(hits)):
            if hits[hit]['corpus_id'] == goldAnswerCorpusIndex:
                return hit+1
        return 0
        
    def __calculate_sbert_scores_batch(self):
        # 1. Encode all unique questions efficiently
        unique_questions = self.data['question'].unique()
        question_embeddings = self.model.encode(unique_questions, convert_to_tensor=True)
        
        # 2. Compute similarity matrix [num_questions, num_docs]
        # util.cos_sim returns tensor of size [len(query_embeddings), len(corpus_embeddings)]
        similarity_matrix = util.cos_sim(question_embeddings, self.corpus_embeddings)
        
        # 3. Create mapping for fast lookup
        q_to_idx = {q: i for i, q in enumerate(unique_questions)}
        
        # 4. Assign scores to dataframe
        # Iterating might be slow but simpler than vectorizing the indexing
        input_scores = []
        for index, row in self.data.iterrows():
            q_idx = q_to_idx[row['question']]
            doc_content = row['long_answers']
            if doc_content in self.doc_map:
                d_idx = self.doc_map[doc_content]
                score = similarity_matrix[q_idx][d_idx].item()
            else:
                score = 0.0
            input_scores.append(score)
            
        self.data['sbert_score'] = input_scores

    def __get_sbert_mrr_score(self, rankIndex):
        if rankIndex == 0:
            return 0
        else:
            return 1/rankIndex
        
    def __calculate_sbert_ndcg(self, rankIndex):
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
    
    def __calculate_all_batch(self):
        # Combined batch calculation for Efficiency
        
        # 1. Encode all unique questions
        unique_questions = self.data['question'].unique()
        question_embeddings = self.model.encode(unique_questions, convert_to_tensor=True)
        
        # 2. Compute similarity matrix [num_questions, num_docs]
        # util.cos_sim returns tensor of size [len(query_embeddings), len(corpus_embeddings)]
        similarity_matrix = util.cos_sim(question_embeddings, self.corpus_embeddings)
        
        # 3. Create mapping for fast lookup
        q_to_idx = {q: i for i, q in enumerate(unique_questions)}
        
        # Initialize columns
        self.data['sbert_rank'] = 0
        self.data['sbert_mrr'] = 0.0
        self.data['sbert_ndcg'] = 0.0
        self.data['sbert_score'] = 0.0
        
        grouped = self.data.groupby('question')
        
        # For assignment, we can iterate rows or groups.
        # Group iteration allows rank calculation.
        
        for question, group in grouped:
             q_idx = q_to_idx[question]
             sims = similarity_matrix[q_idx] # Tensor [num_docs]
             
             # Metric Calculation Section
             # Find gold answer index
             gold_rows = group[group['relevance_label'] == 1]
             if not gold_rows.empty:
                 gold_ans = gold_rows.iloc[0]['long_answers']
                 if gold_ans in self.doc_map:
                     gold_doc_idx = self.doc_map[gold_ans]
                     gold_score = sims[gold_doc_idx]
                     
                     # Rank: how many docs have higher score?
                     rank = torch.sum(sims > gold_score).item() + 1
                     
                     if rank > self.k:
                         rank_val = 0
                     else:
                         rank_val = rank
                     
                     # MRR
                     mrr = 0 if rank_val == 0 else 1/rank_val
                     
                     # NDCG
                     ndcg = self.__calculate_sbert_ndcg(rank_val)
                     
                     self.data.loc[group.index, 'sbert_rank'] = rank_val
                     self.data.loc[group.index, 'sbert_mrr'] = mrr
                     self.data.loc[group.index, 'sbert_ndcg'] = ndcg
             
             # Feature Score Assignment
             # Iterate rows in group to assign specific (Q, D) score
             # Vectorized logic:
             # Get doc indices for all rows in group
             # But doc_map logic...
             # Fallback to iteration for score assignment as strings need looking up
             
             for idx, row in group.iterrows():
                 doc_content = row['long_answers']
                 if doc_content in self.doc_map:
                     d_idx = self.doc_map[doc_content]
                     score = sims[d_idx].item()
                     self.data.at[idx, 'sbert_score'] = score
    
    def __sBertRank(self):
        pass
    
    def __sBertMRR(self):
        pass
    
    def __sBertNDCG(self):
        pass

    def __calculate_sbert_scores_batch(self):
        pass # Included in __calculate_all_batch
        
    def getFullScoringData(self):
        self.__calculate_all_batch()
        return self.data
        
    