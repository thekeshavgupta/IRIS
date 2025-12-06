# IRIS
Intent Reactive Intelligent Search


## Phases which were followed for the project development
### Phase-0 - In Progress
- **Finalise the scope of the project**
    - In this project we are focussing on E-commerce search relevance
- **Finalise the dataset to be used with proper queries and relevance information**
    - TBU
### Phase-1 - Not Started
- **Data loading**
    - Preparing scripts for loading the data
- **Intent Creation for training data**
    - For each queries, prepare a intent taxonomy
- **Contrastive Learning**
    - Preparing a training triplet
### Phase-2 - Not Started
- **Learning certain existing search ranking methods**
    - BM25
    - BERT-CLS Scoring
    - Sentence-BERT cosine similarity
    - *Store their MRR/NDCG*
- **Preparing the baselines**
    - For better comparison, it is good to prepare these baselines
### Phase-3 - Not Started
- Intent encoder
    - It outputs query intent which is a vector
- Contextual search encoder
    - It outputs embeddings of query and documents
- Intent Document Alignment Layer
    - It makes use of FFN layer which uses query intent, document intent and weighted intent vector and comparing it with baselines after using proper ranking loss
### Phase-4 - Not Started
- Some UI touch and final demo



