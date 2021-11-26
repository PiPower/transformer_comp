import tensorflow_datasets as tfds

def to_list(dataset):
    out = []
    for pt_examples, en_examples in dataset.batch(1):
        for (pt, en) in zip(pt_examples.numpy(), en_examples.numpy()):

            sentence_pt = pt.decode("utf-8")
            sentence_pt = sentence_pt.replace("\\","").replace("`","").replace("''","")
            sentence_eng = en.decode("utf-8")
            sentence_eng = sentence_eng.replace("\\", "").replace("`", "").replace("''","")

            out.append((sentence_pt, sentence_eng))
    return out

def save_in_file(word_list, path):
    with open(path, "w") as file:
        for pair in word_list:
            to_save = pair[0] + "<ENG>" + pair[1] + "\n"
            file.write(to_save)


examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)

train_examples, val_examples, test_examples = examples['train'], examples['validation'], examples['test']

train_dataset = to_list(train_examples)
save_in_file(train_dataset,"../datasets/eng-pt/train.txt" )

val_dataset = to_list(val_examples)
save_in_file(val_dataset,"../datasets/eng-pt/val.txt" )

test_dataset = to_list(test_examples)
save_in_file(test_dataset,"../datasets/eng-pt/test.txt" )

