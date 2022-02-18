from common import *
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer

if __name__ == "__main__":
    train_pt, train_eng = load_texts_gpt2("../datasets/eng-pt/train.txt")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_eng)
    train_eng = tokenizer.texts_to_sequences(train_eng)

    length_vector = []
    for sentence in train_eng:
        length_vector.append(len(sentence))
    plt.hist(length_vector, bins = 30)
    plt.xlabel("sentence length")
    plt.ylabel("sentence count")
    plt.show()
