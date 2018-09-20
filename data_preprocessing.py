
# coding: utf-8



import tensorflow as tf
import numpy as np
import data_helpers
import pickle

pos_text, neg_text = data_helpers.load_text_data()
text = pos_text + neg_text

indexed_sequences = data_helpers.build_indexed_seqs_from_strs(text)

print(text[0])

with open("./data/indexed_sequences.pkl", "wb") as out_file:
    pickle.dump(indexed_sequences, out_file)

