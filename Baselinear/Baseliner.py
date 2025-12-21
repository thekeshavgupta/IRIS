import pandas as pd
import numpy as np

from Rankers.BM25Ranker import BM25Ranker
from Rankers.SBertRanker import SBertRanker


class Baseliner():
    def __init__(self, data):
        self.data = data
        self.rowCount= data.shape[0]
        self.colCount = data.shape[1]
    
    def __prepareTrainDevTestDataset(self):
        train_data = self.data.iloc[:int(0.7*self.rowCount),:]
        dev_data = self.data.iloc[int(0.7*self.rowCount):int(0.8*self.rowCount),:]
        test_data = self.data.iloc[int(0.8*self.rowCount):,:]
        
        return [train_data, dev_data, test_data]
    
    # Implementing the method for generating the relevance scoring dataset basically groundTruth labels
    def __generate_relevance_scoring_data(self, df):
        output_df = pd.DataFrame()
        for index, data in df.iterrows():
            output_df = pd.concat([output_df, pd.DataFrame(data=[{
                'question': data['question'],
                'long_answers': data['long_answers'],
                'intent': data['intent'],
                'relevance_label': 1
                }])])
            indices = list(np.random.randint(df.first_valid_index(), df.last_valid_index(), size=51))
            if index in indices:
                indices.remove(index)
            for ind in indices[:50]:    
                output_df = pd.concat([output_df, pd.DataFrame(data=[{
                    'question': data['question'],
                    'long_answers': df.iloc[ind,1],
                    'intent': data['intent'],
                    'relevance_label': 0
                    }])])
        return output_df
    
    
    def prepareBaselines(self, useBM25 = False, useSBert = False, prepareTrainBaseliner = False, prepareDevBaseliner = False, prepareTestBaseliner = False):
        [train_data, dev_data, test_data] = self.__prepareTrainDevTestDataset()
        
        if prepareTrainBaseliner:
            train_data = self.__generate_relevance_scoring_data(train_data[:100])
            
            doc_corpus = list(set(list(train_data.iloc[:,1].values)))
            tokenised_corpus = [ans.split() for ans in doc_corpus]
            
            if useBM25:
                bm25Ranker = BM25Ranker(train_data, tokenised_corpus)
                train_data = bm25Ranker.getFullScoringData()
            
            if useSBert:
                sbertRanker = SBertRanker(train_data, doc_corpus)
                train_data = sbertRanker.getFullScoringData()
                
        if prepareDevBaseliner or prepareTestBaseliner:
            raise NotImplementedError("Functionality not supported currently.")
        
        print("#"*100)
        print("Baseline Preparation Completed!!")
        print("#"*100)
        
        return [train_data, dev_data, test_data]