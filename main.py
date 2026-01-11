from DataLoaders.DataLoader import DataLoader
from Baseliner.Baseliner import Baseliner
from Encoder.IntentEncoderTrainer import IntentEncoderTrainer
from Scorer.Scorer import Scorer
import torch
import pandas as pd

def main():
    # Fetching the dataset
    # data_loader = DataLoader(dataset_path="Natural-Questions-Filtered.csv")
    # clean_data = data_loader.load_and_clean_data()
    
    # # Preparing the baselines
    # baseliner = Baseliner(clean_data)
    # train_data, dev_data, test_data  = baseliner.prepareBaselines(useBM25=True, useSBert=True, prepareTrainBaseliner=True, training_subcount=100)
    # train_data.to_csv('relevance_training_data_with_baselines.csv')
    # test_data.to_csv('relevance_test_data.csv')
    
    # Training the Intent Encoder Network for intent based classification
    intentEncoderTrainer = IntentEncoderTrainer('relevance_training_data_with_baselines.csv')
    intentEncoderTrainer.train(384, 256)
    
    scorer = Scorer(intentEncoderTrainer, 'relevance_training_data_with_baselines.csv')
    scorer.train(epochs=10)
    test_data = pd.read_csv('relevance_test_data.csv')
    test_subset = test_data.loc[:100, ['question', 'long_answers', 'intent']]
    outputs = scorer.predict(test_subset)
    refined_output = torch.sigmoid(outputs).detach().numpy()
    test_data.loc[:100, 'predicted_relevance_score'] = refined_output
    test_data.to_csv('relevance_test_data_with_predictions.csv')
    
if __name__ == "__main__":
    main()