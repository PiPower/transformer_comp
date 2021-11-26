from tensorflow.keras.preprocessing.text import Tokenizer
from base_model import Transformer
from transformer_layers import auto_reg_loss, accuracy_function
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
def load_texts(path, word_limit = None):
  out_pt = []
  out_eng = []
  with open(path) as file:
    for i, line in enumerate(file):
      pt_sent, eng_sent = line.split("<ENG>")
      out_pt.append("<BOS> " + pt_sent + " <EOS>")
      out_eng.append( eng_sent)
      if  word_limit is not None and i == word_limit:
        break
    return out_pt, out_eng

def preprocess_data(eng, pt, tokenizer_eng, tokenizer_pt, batch_size = 32):
  eng = tokenizer_eng.texts_to_sequences(eng)
  pt = tokenizer_pt.texts_to_sequences(pt)

  eng = pad_sequences(eng, padding='post')
  pt = pad_sequences(pt, padding='post')

  dataset = tf.data.Dataset.from_tensor_slices((eng,pt))
  dataset = dataset.batch(batch_size, True)
  return dataset

def train_base(train_dataset, eng_word_count, pt_word_count, optimizer = "Adam", **kwargs ):
  num_layers = 4
  d_model = 128
  dff = 512
  num_heads = 8
  dropout_rate = 0.1

  transformer_model = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                  input_vocab_size=eng_word_count, target_vocab_size=pt_word_count,
                                  pe_input=1000, pe_target=1000, rate=dropout_rate)

  transformer_model.compile(optimizer, loss=auto_reg_loss, metrics=[accuracy_function], run_eagerly = True)
  history = transformer_model.fit(train_dataset, **kwargs)


train_pt, train_eng = load_texts("../datasets/eng-pt/train.txt", 70)

#train_pt = train_pt[49:100]
#train_eng = train_eng[49:100]

tokenizer_eng = Tokenizer()
tokenizer_pt = Tokenizer()

tokenizer_eng.fit_on_texts(train_eng)
tokenizer_pt.fit_on_texts(train_pt)

train_dataset = preprocess_data(train_eng,train_pt, tokenizer_eng, tokenizer_pt, 1)

eng_word_count = len(tokenizer_eng.word_index)+1
pt_word_count = len(tokenizer_pt.word_index)+1

train_base(train_dataset, eng_word_count, pt_word_count, epochs = 1)

