import json
import os
from custom_schedule import CustomSchedule
from tensorflow.keras.preprocessing.text import Tokenizer
from gpt2_model import GPT2
from transformer_layers import auto_reg_loss, accuracy_function
import sys
from common import *

def train_gpt(train_dataset, eng_word_count ,pt_word_count,sep_token, optimizer = "Adam",  num_layers = 8,
  d_model = 256, dff = 512, num_heads = 8, dropout_rate = 0.1,**kwargs ):


  transformer_model = GPT2(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                          input_vocab_size=eng_word_count, target_vocab_size=pt_word_count,
                           sep_token=sep_token, pe_input=1000, pe_target=1000, rate=dropout_rate)

  transformer_model.compile(optimizer, loss=auto_reg_loss, metrics=[accuracy_function], run_eagerly = True)
  history = transformer_model.fit(train_dataset, **kwargs)

  return history, transformer_model

if __name__ == "__main__":
  train_count = None
  test_count = None
  train_pt, train_eng = load_texts_gpt2("../datasets/eng-pt/train.txt", train_count)
  test_pt, test_eng = load_texts("../datasets/eng-pt/test.txt", test_count)

  tokenizer_eng = Tokenizer(oov_token = 'oov')
  tokenizer_pt = Tokenizer(oov_token = 'oov')

  tokenizer_eng.fit_on_texts(train_eng)
  tokenizer_pt.fit_on_texts(train_pt)

  sep_token = tokenizer_pt.word_index['bos']

  train_dataset = preprocess_data(train_eng,train_eng, tokenizer_eng,tokenizer_pt, batch_size= 64, max_len=125)
  test_dataset = preprocess_data(test_eng,test_pt, tokenizer_eng, tokenizer_pt, batch_size=1, to_tuple= True)

  eng_word_count = len(tokenizer_eng.word_index)+1
  pt_word_count = len(tokenizer_pt.word_index)+1
  eos_val = tokenizer_pt.word_index["eos"]
  model_histories = {}

  json_file_path = "../trainning_results/gpt_model_results.json"

  for name in sys.argv[1:]:
      if name == "adam":
        print("training on adam")
        learning_rate = CustomSchedule(256)
        optimizer_adam = tf.keras.optimizers.Adam(learning_rate)
        history, model  = train_gpt(train_dataset, eng_word_count, pt_word_count, sep_token, optimizer = optimizer_adam, epochs = 30)
        accuracy = test(tokenizer_pt, model, test_dataset, 500, 60, mode = "gpt2", tokenizer_eng=tokenizer_eng)
        model_histories["adam"] = history.history
        model_histories["adam"]["test accuracy"] = accuracy

      elif name == "rmsprop":
        print("training on rmsprop")
        learning_rate = CustomSchedule(256)
        optimizer_rmsprop = tf.keras.optimizers.RMSprop(learning_rate)
        history, model  = train_gpt(train_dataset, eng_word_count, pt_word_count, sep_token, optimizer = optimizer_rmsprop, epochs = 30)
        accuracy = test(tokenizer_pt, model, test_dataset, 500, 60, mode = "gpt2", tokenizer_eng=tokenizer_eng)
        model_histories["rmsprop"] = history.history
        model_histories["rmsprop"]["test accuracy"] = accuracy

      elif name == "sgd":
        print("training on sgd")
        learning_rate = CustomSchedule(256)
        optimizer_sgd = tf.keras.optimizers.SGD(learning_rate)
        history, model  = train_gpt(train_dataset, eng_word_count, pt_word_count, sep_token, optimizer = optimizer_sgd, epochs = 30)
        accuracy = test(tokenizer_pt, model, test_dataset, 500, 60, mode = "gpt2", tokenizer_eng=tokenizer_eng)
        model_histories["sgd"] = history.history
        model_histories["sgd"]["test accuracy"] = accuracy

      elif name == "nadam":
        print("training on nadam")
        learning_rate = CustomSchedule(256)
        optimizer_nadam = tf.keras.optimizers.Nadam()
        history, model  = train_gpt(train_dataset, eng_word_count, pt_word_count, sep_token, optimizer = optimizer_nadam, epochs = 30)
        accuracy = test(tokenizer_pt, model, test_dataset, 500, 60, mode = "gpt2", tokenizer_eng=tokenizer_eng)
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