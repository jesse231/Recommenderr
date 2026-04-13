import pandas as pd

# Load Ratings
ratings = pd.read_csv('./ml-1m/ratings.csv', sep='::', engine='python', 
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# Treat all interactions as positive reward (any rating from 1-5) to optimize for engagement
interactions = ratings[['UserID', 'MovieID']].copy()

# Format as head, relation, tail
interactions['Head'] = 'user_' + interactions['UserID'].astype(str)
interactions['Relation'] = 'user.interact.movie'
interactions['Tail'] = 'movie_' + interactions['MovieID'].astype(str)

# load knowledge graph
kg = pd.read_csv('kg.txt', sep='\t', names=['Head', 'Relation', 'Tail'])

kg['Head'] = kg['Head'].astype(str)
kg['Tail'] = 'entity_' + kg['Tail'].astype(str)

interaction_t = interactions[['Head', 'Relation', 'Tail']]

graph = pd.concat([interaction_t, kg], ignore_index=True)

print(ratings)
print(kg)

# add reverge edges
graph = pd.concat([graph, graph.rename(columns={'Head': 'Tail', 'Tail': 'Head'})], ignore_index=True)
# self loops
graph = pd.concat([graph, pd.DataFrame({'Head': graph['Head'], 'Relation': 'self_loop', 'Tail': graph['Head']})], ignore_index=True)

# Convert strings to ids for neural network
entities = pd.unique(graph[['Head', 'Tail']].values.ravel('K'))
entity_to_id = {entity: i for i, entity in enumerate(sorted(entities))}

relations = pd.unique(graph['Relation'])
relation_to_id = {relation: i for i, relation in enumerate(sorted(relations))}

graph['Head_ID'] = graph['Head'].map(entity_to_id)
graph['Tail_ID'] = graph['Tail'].map(entity_to_id)
graph['Relation_ID'] = graph['Relation'].map(relation_to_id)

numerical_graph = graph[['Head_ID', 'Relation_ID', 'Tail_ID']].values

# Write the mapped IDs to a tab-separated file for PyKEEN
graph[['Head_ID', 'Relation_ID', 'Tail_ID']].to_csv('train_triplets.txt', sep='\t', header=False, index=False)

print(graph)

import torch
import numpy as np
from pykeen.pipeline import pipeline

# Pretraining pipeline
result = pipeline(
    training='train_triplets.txt', # Path to numerical triplets
    testing='train_triplets.txt',  # Dummy testing to get it to run
    model='TransE',                # The translation model used in PGPR
    model_kwargs=dict(
        embedding_dim=100,         # Size specified in the PGPR paper
        scoring_fct_norm=1,        
    ),
    training_kwargs=dict(
        num_epochs=50,             # Paper suggests 50 epochs for training
        batch_size=64,             # Batch size from the paper
    ),
    optimizer='Adam',              # Adam optimizer from paper
    optimizer_kwargs=dict(
        lr=0.0001,                 # Learning rate from paper
    ),
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
)

# 2. Extract the trained weights
# These vectors are what the RL agent will use as its 'map'
entity_embeddings = result.model.entity_representations[0](indices=None).detach().cpu().numpy()
relation_embeddings = result.model.relation_representations[0](indices=None).detach().cpu().numpy()

# write
np.save('entity_embeddings.npy', entity_embeddings)
np.save('relation_embeddings.npy', relation_embeddings)
