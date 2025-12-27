from DataLoaders.DataLoader import DataLoader
from Baseliner.Baseliner import Baseliner
from Encoder.IntentEncoderTrainer import IntentEncoderTrainer

def main():
    # Fetching the dataset
    data_loader = DataLoader(dataset_path="Natural-Questions-Filtered.csv")
    clean_data = data_loader.load_and_clean_data()
    
    # # Preparing the baselines
    baseliner = Baseliner(clean_data)
    train_data = baseliner.prepareBaselines(useBM25=True, useSBert=True, prepareTrainBaseliner=True, training_subcount=500)[0]
    train_data.to_csv('relevance_training_data_with_baselines.csv')
    
    # Training the Intent Encoder Network for intent based classification
    intentEncoderTrainer = IntentEncoderTrainer('relevance_training_data_with_baselines.csv')
    intentEncoderTrainer.train(384, 256)
    
if __name__ == "__main__":
    main()