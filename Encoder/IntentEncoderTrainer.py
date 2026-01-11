import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from .IntentEncoder import IntentEncoderNetwork
import torch
import torch.nn as nn

class IntentEncoderTrainer():
    def __init__(self, datasetPath: str):
        self.dataset = pd.read_csv(datasetPath)
        self.dataset = self.dataset[['question', 'intent']]
        self.dataset.drop_duplicates(inplace=True)
        self.transformermodel = SentenceTransformer('all-MiniLM-L6-v2')
        
    def __prepareDataset(self):
        le = LabelEncoder()
        self.intent_labels = torch.from_numpy(le.fit_transform(self.dataset['intent'].tolist()))
        self.intent_classes = le.classes_
        x_embedding_data = self.transformermodel.encode(self.dataset['question'].tolist(), convert_to_tensor=True)
        self.x_embedding_data = x_embedding_data.clone().requires_grad_(True)
    
    def train(self, inputSize = 0, hiddenSize=0, epochs = 100, learning_rate=0.001):
        self.__prepareDataset()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.intent_labels = self.intent_labels .to(device)
        self.x_embedding_data = self.x_embedding_data.to(device)
        
        self.nnModel = IntentEncoderNetwork(input_size=inputSize, hidden_size=hiddenSize, output_size=len(self.intent_classes)).to(device)
        
        learning_rate = learning_rate
        num_epochs = epochs
        loss = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.nnModel.parameters(), lr = learning_rate)

        for epoch in range(num_epochs):
            output = self.nnModel(self.x_embedding_data)
            lossValue = loss(output['intent_logits'], self.intent_labels)
            lossValue.backward()
            optimiser.step()
            optimiser.zero_grad()
            print(f'Epoch {epoch+1}, Loss: {lossValue.item()}')
        self.trained_final_output_params = output
    def predict(self, input: pd.DataFrame | pd.Series):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if(type(input) is pd.DataFrame):
            self.test_embed_output = self.transformermodel.encode(input['question'].to_list(), convert_to_tensor=True).clone().requires_grad_(True).to(device)
        else:
            self.test_embed_output = self.transformermodel.encode(input['question'], convert_to_tensor=True).clone().requires_grad_(True).to(device)
        
        self.test_output = self.nnModel(self.test_embed_output.unsqueeze_(dim=0))
        return self.test_output