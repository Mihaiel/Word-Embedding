import random

from text_preprocessor import TextPreprocessor
from training_data import TrainingDataGenerator
from embedding_model import EmbeddingModel

def main():

    # Part 1 - Preprocessing, vocabulary construction and encoding sentences
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

    # Part 2 - Training data
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
    EPOCHS = 100
    LEARNING_RATE = 0.01

    print(f"\nTraining for {EPOCHS} epochs (learning rate = {LEARNING_RATE})...")
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(pairs)
        for center, context in pairs:
            model.train_on_pair(center, context, learning_rate=LEARNING_RATE)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{EPOCHS} done")

    print("\nVector for 'king' after training:")
    print(model.get_vector(word_to_id["king"]))

    print("\nVector for 'queen' after training:")
    print(model.get_vector(word_to_id["queen"]))


if __name__ == "__main__":
    main()
