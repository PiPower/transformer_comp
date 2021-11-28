import json
import os
from custom_schedule import CustomSchedule
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from base_model import Transformer
from transformer_layers import auto_reg_loss, accuracy_function
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import sys

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

def preprocess_data(eng, pt, tokenizer_eng, tokenizer_pt, batch_size = 32, pad = True, max_len= None, to_tuple = False):
  eng = tokenizer_eng.texts_to_sequences(eng)
  pt = tokenizer_pt.texts_to_sequences(pt)

  if pad:
    eng = pad_sequences(eng, padding='post', maxlen=max_len)
    pt = pad_sequences(pt, padding='post', maxlen=max_len)

  dataset = tf.data.Dataset.from_tensor_slices((eng,pt))
  dataset = dataset.batch(batch_size, True)
  if to_tuple:
    out = []
    for x, y in dataset:
      out.append((x,y))
      return out

  return dataset

def train_base(train_dataset, eng_word_count, pt_word_count,optimizer = "Adam",  num_layers = 4,
  d_model = 128, dff = 512, num_heads = 8, dropout_rate = 0.1,**kwargs ):


  transformer_model = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                  input_vocab_size=eng_word_count, target_vocab_size=pt_word_count,
                                  pe_input=1000, pe_target=1000, rate=dropout_rate)


  transformer_model.compile(optimizer, loss=auto_reg_loss, metrics=[accuracy_function], run_eagerly = True)
  history = transformer_model.fit(train_dataset, **kwargs)

  return history, transformer_model

def evaluate_sentence_prediction(predicted, real):
  xd = 's'
  real = list(real[0])
  difference = abs( len(real) -  len(predicted))
  #matching size of real and predicted
  if len(real) > len(predicted):
    for _ in range( difference):
      predicted.append(0)

  if len(real) < len(predicted):
    for _ in range(difference):
      real.append(0)

  total_number = -1
  correct = -1
  for i in range(len(real)):
    if real[i] == predicted[i] and real[i] == 0:
      break

    if real[i] == predicted[i] and real[i] != 0 :
      correct += 1

    total_number += 1
  return correct, total_number

def test(tokenizer_pt, model, test_dataset, test_count, max_length = 10):
  begin_char = tokenizer_pt.word_index["bos"]
  completed_predictions = []
  for j, sentence in enumerate(test_dataset):
    input_sentece, target_sentence = sentence
    tar = [begin_char]
    if j % 5 == 0:
      print("it is {}th sentence".format(j))

    for i in range(max_length):
        feed = np.asarray([tar ] )
        prediction = model([input_sentece,feed])
        result = prediction.numpy()[0][0]
        tar.append(result)
        if result == tokenizer_pt.word_index["eos"]:
          break
    completed_predictions.append((tar, target_sentence.numpy() ))
    if j > test_count:
      break

  total_correct, total_predicted = 0, 0

  for pred, real in completed_predictions:
    a,b = evaluate_sentence_prediction(pred, real)
    total_correct+= a
    total_predicted += b
  return total_correct/total_predicted

if __name__ == "__main__":
  train_pt, train_eng = load_texts("../datasets/eng-pt/train.txt", 50_000)
  test_pt, test_eng = load_texts("../datasets/eng-pt/test.txt", 1000)

  tokenizer_eng = Tokenizer()
  tokenizer_pt = Tokenizer()

  tokenizer_eng.fit_on_texts(train_eng)
  tokenizer_pt.fit_on_texts(train_pt)

  train_dataset = preprocess_data(train_eng,train_pt, tokenizer_eng, tokenizer_pt, 64)
  test_dataset = preprocess_data(test_eng,test_pt, tokenizer_eng, tokenizer_pt, 1, to_tuple= True)

  eng_word_count = len(tokenizer_eng.word_index)+1
  pt_word_count = len(tokenizer_pt.word_index)+1
  eos_val = tokenizer_pt.word_index["eos"]
  model_histories = {}

  for name in sys.argv:
      if name == "adam":
        print("training on adam")
        learning_rate = CustomSchedule(128)
        optimizer_adam = tf.keras.optimizers.Adam(learning_rate)
        history, model  = train_base(train_dataset, eng_word_count, pt_word_count,optimizer = optimizer_adam, epochs = 30)
        accuracy = test(tokenizer_pt, model, test_dataset, 500, 60)
        model_histories["adam"] = history.history
        model_histories["adam"]["test accuracy"] = accuracy

      elif name == "rmsprop":
        print("training on rmsprop")
        learning_rate = CustomSchedule(128)
        optimizer_rmsprop = tf.keras.optimizers.RMSprop(learning_rate)
        history, model  = train_base(train_dataset, eng_word_count, pt_word_count, optimizer = optimizer_rmsprop,epochs = 30)
        accuracy = test(tokenizer_pt, model, test_dataset, 500, 60)
        model_histories["rmsprop"] = history.history
        model_histories["rmsprop"]["test accuracy"] = accuracy

      elif name == "sgd":
        print("training on sgd")
        learning_rate = CustomSchedule(128)
        optimizer_sgd = tf.keras.optimizers.SGD(learning_rate)
        history, model  = train_base(train_dataset, eng_word_count, pt_word_count, optimizer = optimizer_sgd,epochs = 30)
        accuracy = test(tokenizer_pt, model, test_dataset, 500, 60)
        model_histories["sgd"] = history.history
        model_histories["sgd"]["test accuracy"] = accuracy

      elif name == "nadam":
        print("training on nadam")
        learning_rate = CustomSchedule(128)
        optimizer_nadam = tf.keras.optimizers.Nadam()
        history, model  = train_base(train_dataset, eng_word_count, pt_word_count,optimizer = optimizer_nadam, epochs=30)
        accuracy = test(tokenizer_pt, model, test_dataset, 500, 60)
        model_histories["nadam"] = history.history
        model_histories["nadam"]["test accuracy"] = accuracy

      else:
        print(name + " no option found")

  try:
    os.mkdir("../trainning_results")
  except FileExistsError:
    pass

  if  not os.path.isfile("../trainning_results/base_model_results.json"):
    fp = open("../trainning_results/base_model_results.json", "w")
  else:
    fp = open("../trainning_results/base_model_results.json", "r+")
    old_json = json.load(fp)
    for key, value in old_json.items():
        if not model_histories.get(key, False):
          model_histories[key] = value
  fp.seek(0)
  json.dump(model_histories, fp, indent=True, ensure_ascii= False)

