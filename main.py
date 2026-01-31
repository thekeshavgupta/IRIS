from DataLoaders.DataLoader import DataLoader
from Baseliner.Baseliner import Baseliner
from Encoder.IntentEncoderTrainer import IntentEncoderTrainer
from Scorer.Scorer import Scorer
import torch
import pandas as pd

def main():
    # Fetching the dataset
    data_loader = DataLoader(dataset_path="Natural-Questions-Filtered.csv")
    clean_data = data_loader.load_and_clean_data()
    
    # Preparing the baselines
    # baseliner = Baseliner(clean_data)
    # train_data, dev_data, test_data  = baseliner.prepareBaselines(useBM25=True, useSBert=True, prepareTrainBaseliner=True, prepareTestBaseliner=True, training_subcount=500)
    # train_data.to_csv('relevance_training_data_with_baselines.csv')
    # test_data.to_csv('relevance_test_data.csv')
    
    # Training/Loading the Intent Encoder Network
    intentEncoderTrainer = IntentEncoderTrainer('relevance_training_data_with_baselines.csv')
    intent_model_path = 'intent_encoder_model.pt'
    intent_classes_path = 'intent_classes.pt'
    scorer_model_path = 'scorer_model.pt'
    
    import os
    if os.path.exists(intent_model_path) and os.path.exists(intent_classes_path):
        print("Loading existing Intent Encoder...")
        intentEncoderTrainer.load(intent_model_path, intent_classes_path)
    else:
        print("Training Intent Encoder...")
        intentEncoderTrainer.train(384, 256)
        intentEncoderTrainer.save(intent_model_path, intent_classes_path)

    # Training/Loading the Scorer for final relevance prediction
    scorer = Scorer(intentEncoderTrainer, 'relevance_training_data_with_baselines.csv')
    if os.path.exists(scorer_model_path):
        print("Loading existing Scorer...")
        scorer.load(scorer_model_path)
    else:
        print("Training Scorer...")
        scorer.train(epochs=100)
        scorer.save(scorer_model_path)
    
    # Final Prediction
    test_data_for_prediction = pd.read_csv('relevance_test_data.csv').sample(2000) # sampling to speed up prediction and evaluation for now 
    predictions = scorer.predict(test_data_for_prediction)
    test_data = pd.read_csv('relevance_test_data.csv') # Reload test_data for the next steps
    if len(test_data) > 4000:
        test_data = test_data.sample(4000, random_state=42)
    test_subset = test_data[['question', 'long_answers', 'intent', 'bm25_score', 'sbert_score']]
    outputs = scorer.predict(test_subset)
    refined_output = torch.sigmoid(outputs).detach().numpy().flatten()
    test_data['predicted_relevance_score'] = refined_output
    test_data.to_csv('relevance_test_data_with_predictions.csv')
    
if __name__ == "__main__":
    main()