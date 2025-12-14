from rank_bm25 import BM25Okapi
class BM25Ranker():
    
    def __init__(self, data, corpus):
        self.data =data
        self.corpus = corpus
    def get_relevant_rank(self, text, k=20, goldAnswer = ""):
        goldAnswer = self.data[(self.data['question'] == text) & (self.data['relevance_label'] == 1)]['long_answers'].values[0]
        ranker = BM25Okapi(self.corpus)
        query = text
        output = ranker.get_scores(query.split())
        indi = self.corpus.index(goldAnswer.split())
        # print(indi)
        relevance_score = output[indi]
        # print(relevance_score)
        sorted_output = sorted(output, reverse=True)
        sorted_relevant_output = sorted_output[:k]
        # print(sorted_relevant_output)
        if(relevance_score in sorted_relevant_output):
            return sorted_relevant_output.index(relevance_score)+1
        else:
            return 0
    def get_bm25_mrr(self, score):
        if score == 0:
            return 0
        else:
            return 1/score
    def getBMRank(self):
        self.data['bm25Rank'] = self.data['question'].apply(self.get_relevant_rank, k=20)
        return self.data
    def getBMMRRScore(self):
        self.data['bm25_mrr'] = self.data['bm25Rank'].apply(self.get_bm25_mrr)
        return self.data 
    