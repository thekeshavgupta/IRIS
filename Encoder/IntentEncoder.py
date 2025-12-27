import torch
import torch.nn as nn

class IntentEncoderNetwork(nn.Module):
    def __init__(self, input_size = 0, hidden_size=0, output_size=0):
        super().__init__()
        self.query_intent_embed_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.intent_classifier_layer = nn.Linear(hidden_size, output_size)
        self.weighted_intent_layer = nn.Embedding(output_size, hidden_size)
    def forward(self, query_embedd):
        query_intent_embeddings = self.query_intent_embed_layer(query_embedd)
        query_intent_embeddings = self.relu(query_intent_embeddings)
        intent_logits = self.intent_classifier_layer(query_intent_embeddings)
        intent_logit_prob = torch.softmax(intent_logits, dim=1)
        weighted_intents = torch.matmul(intent_logit_prob, self.weighted_intent_layer.weight)
        return {
            'query_intent_embed': query_intent_embeddings,
            'intent_logits': intent_logits,
            'intent_prob': intent_logit_prob,
            'weighted_intent_sum': weighted_intents
        }