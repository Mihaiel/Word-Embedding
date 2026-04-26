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

    def train_on_pair(self, center_word, context_word, learning_rate=0.01):
        center_vector = self.embeddings[center_word]
        context_vector = self.embeddings[context_word]

        for i in range(self.embedding_dim):
            center_value = center_vector[i]
            context_value = context_vector[i]

            center_vector[i] = center_value + learning_rate * (context_value - center_value)
            context_vector[i] = context_value + learning_rate * (center_value - context_value)

    # Same arithmetic shape as train_on_pair but with a flipped sign so the
    # two vectors move slightly further apart instead of closer together
    # Called on randomly sampled "negative" word ids to counteract the global
    def train_on_negative(self, center_word, negative_word, learning_rate=0.01):
        center_vector = self.embeddings[center_word]
        negative_vector = self.embeddings[negative_word]

        for i in range(self.embedding_dim):
            center_value = center_vector[i]
            negative_value = negative_vector[i]

            center_vector[i] = center_value - learning_rate * (negative_value - center_value)
            negative_vector[i] = negative_value - learning_rate * (center_value - negative_value)