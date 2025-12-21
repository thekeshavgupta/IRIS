from rank_bm25 import BM25Okapi
import math
class BM25Ranker():
    
    def __init__(self, data, corpus, top_k_rank=20):
        self.data =data
        self.corpus = corpus
        self.k = top_k_rank
    def __get_relevant_rank(self, text, goldAnswer = ""):
        goldAnswer = self.data[(self.data['question'] == text) & (self.data['relevance_label'] == 1)]['long_answers'].values[0]
        ranker = BM25Okapi(self.corpus)
        query = text
        output = ranker.get_scores(query.split())
        indi = self.corpus.index(goldAnswer.split())
        # print(indi)
        relevance_score = output[indi]
        # print(relevance_score)
        sorted_output = sorted(output, reverse=True)
        sorted_relevant_output = sorted_output[:self.k]
        # print(sorted_relevant_output)
        if(relevance_score in sorted_relevant_output):
            return sorted_relevant_output.index(relevance_score)+1
        else:
            return 0
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
    
    def __getBMRank(self):
        self.data['bm25Rank'] = self.data['question'].apply(self.__get_relevant_rank)

    def __getBMMRRScore(self):
        self.data['bm25_mrr'] = self.data['bm25Rank'].apply(self.__get_bm25_mrr)
    
    def __getBMNDCGScore(self):
        self.data['bm25_ndcg'] = self.data['bm25Rank'].apply(self.__calculate_bm25_ndcg)
    
    def getFullScoringData(self):
        self.__getBMRank()
        self.__getBMMRRScore()
        self.__getBMNDCGScore()
        return self.data