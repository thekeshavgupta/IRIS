from DataLoaders.DataLoader import DataLoader
from Baselinear.Baseliner import Baseliner

def main():
    # Fetching the dataset
    data_loader = DataLoader(dataset_path="Natural-Questions-Filtered.csv")
    clean_data = data_loader.load_and_clean_data()
    
    # Preparing the baselines
    baseliner = Baseliner(clean_data)
    train_data = baseliner.prepareBaselines(useBM25=True, useSBert=True, prepareTrainBaseliner=True)[0]
    train_data.to_csv('relevance_training_data_with_baselines.csv')

if __name__ == "__main__":
    main()