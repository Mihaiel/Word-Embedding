from text_preprocessor import TextPreprocessor


def main():
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


if __name__ == "__main__":
    main()
