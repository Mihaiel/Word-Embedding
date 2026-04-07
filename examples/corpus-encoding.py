import re


# Make all text lowercase and remove most punctuation
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z.!?\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Split the text into sentences
def split_sentences(text):
    sentences = re.split(r"[.!?]+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Split one sentence into words
def tokenize(sentence):
    return sentence.split()

# Build a vocabulary by assigning each word a number
def build_vocab(tokenized_sentences):
    word_to_id = {}
    id_to_word = {}
    current_id = 0

    for sentence in tokenized_sentences:
        for word in sentence:
            if word not in word_to_id:
                word_to_id[word] = current_id
                id_to_word[current_id] = word
                current_id += 1

    return word_to_id, id_to_word

# Turn the same words into the same numbers
def encode_sentences(tokenized_sentences, word_to_id):
    encoded_sentences = []

    for sentence in tokenized_sentences:
        encoded_sentence = []
        for word in sentence:
            encoded_sentence.append(word_to_id[word])
        encoded_sentences.append(encoded_sentence)

    return encoded_sentences

# Preprocess the corpus
def preprocess(text):
    clean_text = self.normalize_text(text)
    sentences = self.split_sentences(clean_text)

    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentences.append(self.tokenize(sentence))

    word_to_id, id_to_word = self.build_vocab(tokenized_sentences)
    encoded_sentences = self.encode_sentences(tokenized_sentences, word_to_id)

    return tokenized_sentences, word_to_id, id_to_word, encoded_sentences

with open("corpus.txt") as file:
    text = file.read()

clean_text = normalize_text(text)
print("Cleaned text (characters): ")
print(clean_text[:64])

sentences = split_sentences(clean_text)
print("\nSentences: ")
print(sentences[:2])

tokenized_sentences = []
for sentence in sentences:
    tokenized_sentences.append(tokenize(sentence))
print("\nTokenized sentences: ")
print(tokenized_sentences[:2])

word_to_id, id_to_word = build_vocab(tokenized_sentences)
print("\nWord to ID: ")
print(list(word_to_id.items())[:10])

print("\nEncoding the whole sentence: ")
encoded_sentences = encode_sentences(tokenized_sentences, word_to_id)
print(encoded_sentences[:2])