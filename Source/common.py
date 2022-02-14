import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


def preprocess_data(net_input, pt, tokenizer_eng, tokenizer_pt, batch_size = 32, max_len= None, to_tuple = False):
  net_input = tokenizer_eng.texts_to_sequences(net_input)
  pt = tokenizer_pt.texts_to_sequences(pt)

  net_input = pad_sequences(net_input, padding='post', maxlen=max_len)
  pt = pad_sequences(pt, padding='post', maxlen=max_len)

  dataset = tf.data.Dataset.from_tensor_slices((net_input,pt))

  if to_tuple:
    dataset = dataset.batch(1, True)
    out = []
    for x, y in dataset:
      out.append((x,y))
      return out

  dataset = dataset.batch(batch_size, True)
  return dataset

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

def load_texts_gpt2(path, word_limit = None):
  out_pt = []
  out_eng = []
  with open(path) as file:
    for i, line in enumerate(file):
      pt_sent, eng_sent = line.split("<ENG>")
      out_pt.append("<BOS> " + pt_sent + " <EOS>")
      out_eng.append( eng_sent + " <BOS> " + pt_sent + " <EOS>" )
      if  word_limit is not None and i == word_limit:
        break
    return out_pt, out_eng

def evaluate_sentence_prediction(predicted, real):
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
