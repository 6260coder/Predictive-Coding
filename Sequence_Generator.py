
# coding: utf-8



import numpy as np
import tensorflow as tf
import Predictive_Coding_Model
import data_helpers

class SequenceGenerator:
    def __init__(self, 
                 max_seq_len=300, 
                 rnn_cell=tf.contrib.rnn.GRUCell, 
                 rnn_cell_size=128, 
                 rnn_layers=2, 
                 optimizer=tf.train.RMSPropOptimizer(0.002), 
                 gradient_clipping=2.0, 
                 embedding_dim=100, 
                 ):
        
        # create predictive coding model
        self.max_seq_len = max_seq_len
        self.params = dict()
        self.params["rnn_cell"] = rnn_cell
        self.params["rnn_cell_size"] = rnn_cell_size
        self.params["rnn_layers"] = rnn_layers
        self.params["optimizer"] = optimizer
        self.params["gradient_clipping"] = gradient_clipping
        self.params["embedding_dim"] = embedding_dim
        
        vocab = data_helpers.vocab
        vocab_codes = np.zeros([len(vocab) + 1, len(vocab)])        
        for i in range(len(vocab)):
            vocab_codes[i + 1, i] = 1.0
            
        self.params["vocab_size"] = len(vocab)
        self.params["vocab_codes"] = vocab_codes
        
        self.model = Predictive_Coding_Model.PredictiveCodingModel(max_seq_len=max_seq_len, 
                                                                   params=self.params, 
                                                                   with_initial_state=True)
    def load_pretrained_model(self, dir="./checkpoints/prev/"):
        
        # load previously trained model
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)
        saver.restore(self.sess, tf.train.latest_checkpoint(dir))
        print("model loaded")
        
    def softmax(self, x):
        exps = np.exp(x)
        exps_sum = np.sum(exps)
        for i in range(x.shape[0]):
            x[i] = exps[i] / exps_sum
        return x
    
    def sample_next_token(self, initial_state, current_token):
        data = np.zeros([1, self.max_seq_len])
        data[0, 0] = current_token
        feed_dict = {self.model.initial:initial_state, self.model.data:data}
        scores, state = self.sess.run([self.model.scores[0][0], 
                                  self.model.state], 
                                 feed_dict)
        sampled_token = np.random.choice(len(data_helpers.vocab), p=self.softmax(scores)) + 1
    #     sampled_token = np.argmax(scores) + 1
        return sampled_token, state
    
    def convert_str_to_padded_array(self, str):
        padded_arr = np.zeros([1, self.max_seq_len])
        for i in range(len(str)):
            padded_arr[0][i] = data_helpers.token_to_index_dic[str[i]]
        return padded_arr
    
    def sample_sequence(self, head, length):
        assert (1 <= len(head) <= length) and (len(head) < length), "Length problem."
        sequence_in_str = head
        blank_state = (np.zeros((1, self.params["rnn_cell_size"])),) * 2
        current_token = data_helpers.token_to_index_dic[head[-1]]
        
        # if head is one token long, then initial_state is blank_state
        # if head is more than one token long then initial_state is the state
        # after going through head[:-1]
        if len(head) == 1:
            initial_state = blank_state
        else:
            head = head[:-1]
            head = self.convert_str_to_padded_array(head)
            feed_dict = {}
            feed_dict[self.model.data] = head
            feed_dict[self.model.initial] = blank_state
            state = self.sess.run(self.model.state, feed_dict=feed_dict)
            initial_state = state
        
        # centrual loop that takes an initial_state and a current_token and sample the
        # next_token accordingly
        while len(sequence_in_str) < length:
            next_token, state = self.sample_next_token(initial_state, current_token)
            initial_state = state
            current_token = next_token
            next_token = data_helpers.vocab[next_token - 1]
            sequence_in_str += next_token
        return sequence_in_str
            

