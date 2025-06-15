import torch

from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted", "I Have Played Music"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences, convert_to_tensor=True)
print(embeddings)

"""
Structure of embeddings
embeddings -> (embedding sentence 1, embedding sentence 2)
sentence 1 -> (384, ) vector
"""
print( len(list(embeddings)) )
print( len(list(embeddings[0])) )


query = "A person playing music"
query_embedding = model.encode(query, convert_to_tensor=True)

## Compute Similarities (Cosine)
from torch.nn.functional import cosine_similarity

# Compute cosine similarity between the query and each sentence embedding
cosine_scores = cosine_similarity(query_embedding, embeddings)

## Step 3: Rank Results
top_result = torch.topk(cosine_scores, k=1)

# Print the most similar sentence
best_match_idx = top_result.indices[0].item()
print("Best match:", sentences[best_match_idx])

## Optional: Return All Ranked Matches
scores = cosine_scores.squeeze().tolist()
ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

for sent, score in ranked:
    print(f"{score:.4f} — {sent}")