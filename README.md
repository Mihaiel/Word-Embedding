# Simple Word Embedding
A minimal, from-scratch implementation of a word embedding system.
This project intentionally avoids popular libraries to provide a deeper, hands-on understanding of how embedding models work internally.

The project currently focuses on the first preprocessing steps that are usually needed before training a model:

- cleaning text
- splitting text into sentences
- tokenizing words
- building a vocabulary
- converting words into numbers

## Project Files

- `main.py` starts the preprocessing
- `text_preprocessor.py` contains the `TextPreprocessor` class
- `corpus.txt` is the input text

## Run

```bash
python3 main.py
```

## Note

This project is inspired by educational material about word embeddings and is meant as a simple learning project, not as a full machine learning implementation yet.
