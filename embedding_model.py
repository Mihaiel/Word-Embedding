import random

class EmbeddingModel:
    def __init__(self, vocab_size, embedding_dim=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embeddings = []
        for _ in range(vocab_size):
            vector = []
            for _ in range(embedding_dim):
                vector.append(random.uniform(-0.5, 0.5))
            self.embeddings.append(vector)

    def get_vector(self, word_id):
        return self.embeddings[word_id]