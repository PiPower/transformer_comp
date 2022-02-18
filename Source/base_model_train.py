import json
import os
from custom_schedule import CustomSchedule
from tensorflow.keras.preprocessing.text import Tokenizer
from base_model import Transformer
from transformer_layers import auto_reg_loss, accuracy_function
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import sys
from common import *

def train_base(train_dataset, eng_word_count, pt_word_count,optimizer = "Adam",  num_layers = 4,
  d_model = 128, dff = 512, num_heads = 8, dropout_rate = 0.1,**kwargs ):


  transformer_model = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                  input_vocab_size=eng_word_count, target_vocab_size=pt_word_count,
                                  pe_input=1000, pe_target=1000, rate=dropout_rate)

  transformer_model.compile(optimizer, loss=auto_reg_loss, metrics=[accuracy_function], run_eagerly = True)
  history = transformer_model.fit(train_dataset, **kwargs)

  return history, transformer_model


if __name__ == "__main__":
  train_pt, train_eng = load_texts("../datasets/eng-pt/train.txt")
  test_pt, test_eng = load_texts("../datasets/eng-pt/test.txt")

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

  json_file_path = "../trainning_results/base_model_results.json"

  for name in ["adam"]:
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

  if not os.path.isfile(json_file_path):
    fp = open(json_file_path, "w")
  else:
    fp = open(json_file_path, "r+")
    old_json = json.load(fp)
    for key, value in old_json.items():
        if not model_histories.get(key, False):
          model_histories[key] = value
  fp.seek(0)
  json.dump(model_histories, fp, indent=True, ensure_ascii= False)

