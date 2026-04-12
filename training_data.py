# This class generatas the word context-pairs from encoded sentences
# E.g.: When we have the sentence [The, king, rules, the, kingdom] -> encoded_sentence = [0,1,2,0,3]
# -> this functions takes the encoded_sentences data and creates context-pairs
# -> (0,1) corresponds to (the, king)
# -> (1,0) corresponds to (king, the)
# -> (1,2) corresponds to (king, rules)
# -> (2,1) corresponds to (rules, king) etc...

class TrainingDataGenerator:
    def generate_pairs(self, encoded_sentences, window_size=1):
        pairs = []

        for sentence in encoded_sentences:
            for i in range(len(sentence)):
                center_word = sentence[i]

                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)

                for j in range(start, end):
                    if i != j:
                        context_word = sentence[j]
                        pairs.append((center_word, context_word))

        return pairs