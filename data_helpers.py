
# coding: utf-8



import numpy as np
import math


vocab_full = " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"              "\\^_abcdefghijklmnopqrstuvwxyz{|}"
vocab = " $%'()+,-./0123456789:;=?abcdefghijklmnopqrstuvwxyz"
# indices start from 1 
# o's represents padded values
token_to_index_dic = {x:(i + 1) for i, x in enumerate(vocab)}

positive_data_file_dir = "./data/rt-polaritydata/rt-polarity.pos"
negative_data_file_dir = "./data/rt-polaritydata/rt-polarity.neg"

def load_text_data():
    with open(positive_data_file_dir, "rt", encoding="utf-8") as in_file:
        pos_text = in_file.readlines()
    with open(negative_data_file_dir, "rt", encoding="utf_8") as in_file:
        neg_text = in_file.readlines()
    return pos_text, neg_text

def build_coded_seqs_from_strs(strs):
#     vocab = " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
#             "\\^_abcdefghijklmnopqrstuvwxyz{|}"
#     token_to_code_dic = {x:i for i, x in enumerate(vocab)}
    max_len = max([len(str) for str in strs])
    sequences = np.zeros([len(strs), max_len, len(vocab)])
    for index, str in enumerate(strs):
        temp_codes_holder = []
        for token in str:
            if token in list(token_to_code_dic.keys()):
                code = token_to_code_dic[token]
                temp_codes_holder.append(code)
        for offset, code in enumerate(temp_codes_holder):
                sequences[index, offset, code] = 1.0
    return sequences

def build_indexed_seqs_from_strs(strs, preset_max_len=None):
#     vocab = " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
#             "\\^_abcdefghijklmnopqrstuvwxyz{|}"
#     token_to_code_dic = {x:(i + 1) for i, x in enumerate(vocab)}
    sequences = []
    for str in strs:
        sequence = []
        for token in str:
            if token in list(token_to_code_dic.keys()):
                sequence.append(token_to_code_dic[token])
        sequences.append(sequence)
    if preset_max_len is not None:
        max_seq_len = preset_max_len
    else:
        max_seq_len = max([len(sequence) for sequence in sequences])
    for i in range(len(sequences)):
        if max_seq_len > len(sequences[i]):
            sequences[i] += [0] * (max_seq_len - len(sequences[i]))
    sequences = np.array(sequences)
    return sequences 

def is_code(code):
    is_code_flag = True
    if np.sum(code) != 1.0:
        is_code_flag = False
    for element in code:
        if element not in [0.0, 1.0]:
            is_code_flag = False
    return is_code_flag    

def get_actual_lens(sequences):
    sequences = np.sum(sequences, axis=2)
    lens = np.sum(sequences, axis=1)
    lens = lens.astype(int)
    return lens    

def is_valid_seq(sequence, actual_len):
    for code in sequence[:actual_len]:
        if not is_code(code):
            return False
    for code in sequence[actual_len:]:
        if is_code(code):
            return False
    return True
    
# outmost interface that calls the above three functions
# to check the integrity of coded sequences
def are_valid_seqs(sequences):
    lens = get_actual_lens(sequences)
    for sequence, len in zip(sequences, lens):
        if not is_valid_seq(sequence, len):
            return False
    return True

# partition data into training set and development set according to dev_sample_percentage
def partition_data(data, dev_sample_percentage):
    dev_sample_start = int(float(len(data)) * (1 - dev_sample_percentage))
    return data[:dev_sample_start], data[dev_sample_start:]

# shuffles data, returns a shuffled np.array
def shuffle_data(data):
    data = np.array(data)
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    return data[shuffle_indices]

# batch generator
def batch_generator(data, batch_size):
    num_of_batches = math.ceil(float(len(data)) / float(batch_size))
    for i in range(num_of_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(data))
        yield i, data[start_index : end_index]

def from_str_to_list_of_indices(str):
    list_of_indices = []
    for token in str:
        if token in list(token_to_code_dic.keys()):
            list_of_indices.append(token_to_code_dic[token])
    return list_of_indices

def from_list_of_indices_to_array(list_of_indices, max_len):
    data_array = np.zeros([1, max_len])
    for i, index in enumerate(list_of_indices):
        data_array[0, i] = index
    return data_array

def percentage_of_token(data_arr, token):
    total_length = 0
    token_count = 0    
    for row in data_arr:
        for entry in row:
            if entry != 0:
                total_length += 1
            if entry == token_to_index_dic[token]:
                token_count += 1
    percentage = float(token_count) / float(total_length)
    return percentage

def from_scores_to_str(scores, length):
    str = ""
    indices = np.argmax(scores, axis=1)
    for i in range(length):
        str += vocab[indices[i]]
    return str