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
        data_rows = []
        # Pre-convert useful columns to list to avoid repeated DF access
        # But iterrows is fine-ish if logic is complex. 
        # Better: use indices.
        
        # Original logic:
        # For each row (positive):
        #   Add positive sample.
        #   Pick 50 random negatives from OTHER rows.
        
        # Optimization:
        indices_list = df.index.tolist()
        df_reset = df.reset_index(drop=True)
        total_rows = len(df_reset)
        
        for index, data in df_reset.iterrows():
            # Positive sample
            data_rows.append({
                'question': data['question'],
                'long_answers': data['long_answers'],
                'intent': data['intent'],
                'relevance_label': 1
            })
            
            # Negative samples
            # Random 50 indices not equal to current index
            # This logic assumes df indices are 0..N-1 after reset_index
            
            # Efficient random sampling
            # We want 50 integers from [0, total_rows) excluding index
            # choice is faster if we just sample 51 and filter, or sample from range.
            
            neg_indices = np.random.choice(total_rows, size=55, replace=False) # sample a few more to be safe
            count = 0
            for ind in neg_indices:
                if ind == index:
                    continue
                data_rows.append({
                    'question': data['question'],
                    'long_answers': df_reset.at[ind, 'long_answers'],
                    'intent': data['intent'],
                    'relevance_label': 0
                })
                count += 1
                if count >= 50:
                    break
                    
        output_df = pd.DataFrame(data_rows)
        return output_df
    
    
    def prepareBaselines(self, useBM25 = False, useSBert = False, prepareTrainBaseliner = False, prepareDevBaseliner = False, prepareTestBaseliner = False, training_subcount = 100):
        [train_data, dev_data, test_data] = self.__prepareTrainDevTestDataset()
        
        if prepareTrainBaseliner:
            train_data = self.__generate_relevance_scoring_data(train_data[:training_subcount])
            
            doc_corpus = list(set(list(train_data.iloc[:,1].values)))
            tokenised_corpus = [ans.split() for ans in doc_corpus]
            
            if useBM25:
                bm25Ranker = BM25Ranker(train_data, tokenised_corpus)
                train_data = bm25Ranker.getFullScoringData()
            
            if useSBert:
                sbertRanker = SBertRanker(train_data, doc_corpus)
                train_data = sbertRanker.getFullScoringData()
                
        if prepareDevBaseliner:
            # Generate relevance data for dev set
            # Assuming we want to use all dev data or a subset? Let's use all for now as training_subcount is for training.
            # But generating 50 negatives for ALL dev data might be heavy if dev is large.
            # Let's limit it similar to train if needed, or just run it. 
            # Given dataset size, let's limit to training_subcount for consistency or handle all if reasonable.
            # Let's handle all but safely.
            dev_data = self.__generate_relevance_scoring_data(dev_data)
            
            doc_corpus_dev = list(set(list(dev_data.iloc[:,1].values)))
            tokenised_corpus_dev = [ans.split() for ans in doc_corpus_dev]
            
            if useBM25:
                bm25Ranker = BM25Ranker(dev_data, tokenised_corpus_dev)
                dev_data = bm25Ranker.getFullScoringData()
            
            if useSBert:
                sbertRanker = SBertRanker(dev_data, doc_corpus_dev)
                dev_data = sbertRanker.getFullScoringData()

        if prepareTestBaseliner:
            test_data = self.__generate_relevance_scoring_data(test_data)
            
            doc_corpus_test = list(set(list(test_data.iloc[:,1].values)))
            tokenised_corpus_test = [ans.split() for ans in doc_corpus_test]
            
            if useBM25:
                bm25Ranker = BM25Ranker(test_data, tokenised_corpus_test)
                test_data = bm25Ranker.getFullScoringData()
            
            if useSBert:
                sbertRanker = SBertRanker(test_data, doc_corpus_test)
                test_data = sbertRanker.getFullScoringData()
        
        print("#"*100)
        print("Baseline Preparation Completed!!")
        print("#"*100)
        
        return [train_data, dev_data, test_data]