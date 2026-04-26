import random

from text_preprocessor import TextPreprocessor
from training_data import TrainingDataGenerator
from embedding_model import EmbeddingModel

def main():

    # Part 1 - Preprocessing, vocabulary construction and encoding sentences (see text_preprocessor.py for more)
    with open("corpus.txt") as file:
        text = file.read()

    preprocessor = TextPreprocessor()
    tokenized_sentences, word_to_id, id_to_word, encoded_sentences = preprocessor.preprocess(text)

    print("First 3 sentences:")
    for sentence in tokenized_sentences[:3]:
        print(sentence)

    print("\nVocabulary size:")
    print(len(word_to_id))

    print("\nFirst 10 words in vocabulary:")
    for word in list(word_to_id.keys())[:10]:
        print(word, "->", word_to_id[word])

    print("\nFirst encoded sentence:")
    print(encoded_sentences[0])

    # Part 2 - Training data (see training_data.py for more)
    generator = TrainingDataGenerator()
    pairs = generator.generate_pairs(encoded_sentences, window_size=1)

    print("\nFirst 10 training pairs:")
    for pair in pairs[:10]:
        print(pair, "->", id_to_word[pair[0]], id_to_word[pair[1]])

    # Part 3 - Embedding model
    model = EmbeddingModel(vocab_size=len(word_to_id), embedding_dim=5)

    print("Vector for 'king':")
    print(model.get_vector(word_to_id["king"]))

    print("\nVector for 'queen':")
    print(model.get_vector(word_to_id["queen"]))

    # Part 4 - Simplified Learning from pairs
    # For every observed (center, context) pair we run the attraction update
    # once, then sample NEGATIVES_PER_PAIR random words and run the sign-flipped
    # repulsion update against each of them. The attraction-only variant is
    # recovered by setting NEGATIVES_PER_PAIR = 0.

    # With NEGATIVES_PER_PAIR > 0 the norms no longer collapse but they
    # also do not stay bounded.
    EPOCHS = 100
    LEARNING_RATE = 0.001
    NEGATIVES_PER_PAIR = 1

    vocab_size = len(word_to_id)

    def sample_negative(center_id):
        # Uniform random sampler that avoids returning the center itself.
        while True:
            neg = random.randrange(vocab_size)
            if neg != center_id:
                return neg

    print(
        f"\nTraining for {EPOCHS} epochs "
        f"(learning rate = {LEARNING_RATE}, "
        f"negatives per pair = {NEGATIVES_PER_PAIR})..."
    )
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(pairs)
        for center, context in pairs:
            model.train_on_pair(center, context, learning_rate=LEARNING_RATE)
            for _ in range(NEGATIVES_PER_PAIR):
                negative = sample_negative(center)
                model.train_on_negative(center, negative, learning_rate=LEARNING_RATE)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{EPOCHS} done")

    print("\nVector for 'king' after training:")
    print(model.get_vector(word_to_id["king"]))

    print("\nVector for 'queen' after training:")
    print(model.get_vector(word_to_id["queen"]))


if __name__ == "__main__":
    main()
