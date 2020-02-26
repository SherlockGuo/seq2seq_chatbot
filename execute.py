#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = execute.py
__author__ = 'guo_h'
__mtime__ = 2020/02/25 
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓     ┏┓
            ┏┛┻━━━━━┛┻━┓
            ┃    ☃    ┃
            ┃  ┳┛  ┗┳  ┃
            ┃    ┻     ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━┓
              ┃  神兽保佑 ┣┓
              ┃　永无BUG！┏┛
              ┗┓┓┏━━━┓┓┏━┛
               ┃┫┫   ┃┫┫
               ┗┻┛   ┗┻┛
"""
import os
import sys
import numpy as np
import tensorflow as tf
import io
import pysnooper
import time
import seq2seqModel

from sklearn.model_selection import train_test_split
from getConfig import get_config

CONFIG = get_config()
vocab_input_size = CONFIG['enc_vocab_size']
vocab_target_size = CONFIG['dec_vocab_size']
embedding_dim = CONFIG['embedding_dim']
units = CONFIG['layer_size']
batch_size = CONFIG['batch_size']
max_length_inp = CONFIG['max_length']
max_length_tar = CONFIG['max_length']


def preprocess_sentence(w):
    w = '<start>' + w + '<end>'
    return w


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=CONFIG['enc_vocab_size'], oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(path, num_examples):
    target_lang, input_lang = create_dataset(path=path, num_examples=num_examples)
    input_tensor, input_lang_tokenizer = tokenize(input_lang)
    target_tensor, target_lang_tokenizer = tokenize(target_lang)

    return input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer


input_tensor, target_tensor, input_lang, target_lang = load_dataset(CONFIG['seq_data'], CONFIG['max_train_data_size'])

max_length_target, max_length_input = max_length(input_tensor), max_length(target_tensor)


@pysnooper.snoop()
def train():
    print("Preparing data in %s" % CONFIG['train_data'])
    steps_per_epoch = len(input_tensor) // CONFIG['batch_size']
    print(steps_per_epoch)
    enc_hidden = seq2seqModel.encoder.initialize_hidden_state()
    checkpoint_dir = CONFIG['model_data']
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        print("reload pretrained model")
        seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(len(input_tensor))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    checkpoint_dir = CONFIG['model_data']
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    start_time = time.time()

    while True:
        start_time_epoch = time.time()
        total_loss = 0

        for (batch, (inp, target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = seq2seqModel.train_step(inp, target, target_lang, enc_hidden)
            total_loss += batch_loss
            print(batch_loss.numpy())

        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch

        current_steps = +steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps
        print('训练总步数: {}, 每步耗时: {}, 最新每步耗时: {}, 最新每步loss值 {:.4f}'
              .format(current_steps, step_time_total, step_time_epoch, step_loss.numpy()))

        seq2seqModel.checkpoint.save(file_prefix=checkpoint_prefix)

        sys.stdout.flush()


def predict(sentence):
    sentence = preprocess_sentence(sentence)
    inputs = [input_lang.word_index.get(i, 3) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = seq2seqModel.encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_lang.word_index['start']], 0)

    for t in range(max_length_tar):
        predictions, dec_hidden, attention_weights = seq2seqModel.decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        if target_lang.index_word[predicted_id] == 'end':
            break
        result += target_lang.index_word[predicted_id] + ' '

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


if __name__ == '__main__':
    print('Mode: %s' % CONFIG['mode'])
    if CONFIG['mode'] == 'train':
        train()
    elif CONFIG['mode'] == 'serve':
        print('Serve Usage : >> python3 app.py')
