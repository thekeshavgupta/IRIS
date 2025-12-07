import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    def preprocess_text(self, text, use_stopwords=True):
        stopwordsList = set(["the", "is", "in", "and", "to", "a", "of", "that", "it", "on", "for", "as", "with", "was", "at", "by", "an"])
        lemmatiser = WordNetLemmatizer()
        output = []
        for word in text.split():
            if not use_stopwords or word.lower() not in stopwordsList:
                word = word.lower()
                word = word.strip('.?!,;:"()[]{}')
                word = lemmatiser.lemmatize(word)
                output.append(word)
        return ' '.join(output)
    
    def classify_intent(self, query):
        if("what is" in query or "define" in query or "meaning of" in query):
            return 'definition'
        elif("how many" in query or "list of" in query or "types of" in query or "type of" in query or "list out" in query):
            return "list"
        elif("how to" in query and "how to train a dragon" not in query):
            return "procedural"
        elif("difference between" in query or "compare" in query or "comparison of" in query or "versus" in query or " vs " in query):
            return "comparison"
        elif("who" in query):
            return "entity"
        else:
            return "factual"
    def clean_data(self, raw_data):
        raw_data.drop_duplicates(inplace=True)
        raw_data = raw_data[['question', 'long_answers']]
        raw_data['question'] = raw_data['question'].apply(self.preprocess_text, use_stopwords = False)      
        raw_data['long_answers'] = raw_data['long_answers'].apply(self.preprocess_text, use_stopwords = False) 
        raw_data['intent'] = raw_data['question'].apply(self.classify_intent)
        return raw_data
    
    def load_and_clean_data(self):
        raw_data = pd.read_csv(self.dataset_path)
        cleaned_data = self.clean_data(raw_data)
        return cleaned_data
    
    def prepare_filtered_data(self, outputPath):
        data = self.load_and_clean_data()
        data.to_csv(outputPath)
        print(f"Cleaned data saved to {outputPath}")