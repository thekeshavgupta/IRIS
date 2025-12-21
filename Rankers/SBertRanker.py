from sentence_transformers import util, SentenceTransformer
import math
class SBertRanker():
    def __init__(self, data, corpus, top_k_rank=20):
        self.k = top_k_rank
        self.data = data
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus = corpus
        self.corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
    
    def get_relevant_score(self, query):
        goldAnswer = self.data[(self.data['question'] == query) & (self.data['relevance_label'] == 1)]['long_answers'].values[0]
        query_embeddings = self.model.encode(query, convert_to_tensor=True)
        goldAnswerCorpusIndex = self.corpus.index(goldAnswer)
        hits = util.semantic_search(query_embeddings=query_embeddings, corpus_embeddings=self.corpus_embeddings, top_k=self.k)[0]
        for hit in range(len(hits)):
            if hits[hit]['corpus_id'] == goldAnswerCorpusIndex:
                return hit+1
        return 0
        
    def get_sbert_mrr_score(self, rankIndex):
        if rankIndex == 0:
            return 0
        else:
            return 1/rankIndex
        
    def calculate_sbert_ndcg(self, rankIndex):
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
    
    def sBertRank(self):
        self.data['sbert_rank'] = self.data['question'].apply(self.get_relevant_score)
    
    def sBertMRR(self):
        self.data['sbert_mrr'] = self.data['sbert_rank'].apply(self.get_sbert_mrr_score)
    
    def sBertNDCG(self):
        self.data['sbert_ndcg'] = self.data['sbert_rank'].apply(self.calculate_sbert_ndcg)
        
    def getFullScoringData(self):
        self.sBertRank()
        self.sBertMRR()
        self.sBertNDCG()
        return self.data
        
    