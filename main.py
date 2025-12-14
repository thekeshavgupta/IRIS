from DataLoader import DataLoader
from BM25Ranker import BM25Ranker
import pandas as pd
import numpy as np

# Implementing the method for generating the relevance scoring dataset
def generate_relevance_scoring_data(df: pd.DataFrame):
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

def main():
    data_loader = DataLoader("Natural-Questions-Filtered.csv")
    data_loader.prepare_filtered_data("output.csv")
    
    data = pd.read_csv('output.csv')
    data.drop(axis=1, columns=['Unnamed: 0'], inplace=True)
    
    rowCount= data.shape[0]
    colCount = data.shape[1]
    
    train_data = data.iloc[:int(0.7*rowCount),:3]
    dev_data = data.iloc[int(0.7*rowCount):int(0.8*rowCount),:3]
    test_data = data.iloc[int(0.8*rowCount):,:3]
    
    relevance_train_data = generate_relevance_scoring_data(train_data[:100])
    
    doc_corpus = list(set(list(relevance_train_data.iloc[:,1].values)))
    tokenised_corpus = [ans.split() for ans in doc_corpus]
    
    bm25Ranker = BM25Ranker(relevance_train_data, tokenised_corpus)
    bm25Ranker.getBMRank()
    relevance_train_data = bm25Ranker.getBMMRRScore()
    # print(relevance_train_data.head())

    

if __name__ == "__main__":
    main()