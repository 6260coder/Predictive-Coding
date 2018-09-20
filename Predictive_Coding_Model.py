
# coding: utf-8



import tensorflow as tf


class PredictiveCodingModel:
    def __init__(self, max_seq_len, params, with_initial_state=False):
        self.max_seq_len = max_seq_len
        self.data = self.data()
        self.params = params
        self.with_initial_state = with_initial_state
        self.initial = self.initial()
        self.sequences = self.sequences()
        self.labels = self.labels()
        self.mask = self.mask()
        self.lengths = self.lengths()
        # state: a tuple of 2 tensors with shape [batch_size, cell_size]
        self.scores, self.state = self.scores_and_state()
        self.loss = self.loss()
        self.accuracy = self.accuracy()
        self.pseudo_perplexity = self.pseudo_perplexity()
        self.optimizer = self.optimizer()
        
    def data(self):
        data = tf.placeholder(dtype=tf.float32, shape=[None, self.max_seq_len])
        return data        
        
    def initial(self):
        if self.with_initial_state == False:
            return None
        else:
            initial_rnn_1 = tf.placeholder(dtype=tf.float32, 
                                           shape=[None, self.params["rnn_cell_size"]])

            initial_rnn_2 = tf.placeholder(dtype=tf.float32, 
                                           shape=[None, self.params["rnn_cell_size"]])
            initial = (initial_rnn_1, initial_rnn_2)
            return initial        
        
    def sequences(self):
        max_len = int(self.data.shape[1])
        indices = tf.slice(self.data, (0, 0), (-1, max_len - 1))
        indices = tf.cast(indices, tf.int32)
        embedding_vocab = tf.Variable(tf.truncated_normal([self.params["vocab_size"] + 1, self.params["embedding_dim"]], stddev=0.01), 
                                      dtype=tf.float32)
        sequences = tf.nn.embedding_lookup(embedding_vocab, indices)
        return sequences
    
    def labels(self):
        indices = tf.slice(self.data, (0, 1), (-1, -1))
        indices = tf.cast(indices, tf.int32)
        coded_labels = tf.nn.embedding_lookup(self.params["vocab_codes"], indices)
        coded_labels = tf.cast(coded_labels, tf.float32)
        return coded_labels
    
    def mask(self):
        # one if valid, zero if padded
        mask = tf.reduce_max(self.labels, axis=2)
#         mask = tf.cast(mask, tf.float32)
        return mask
    
    def lengths(self):
        lengths = tf.reduce_sum(self.mask, axis=1)
        lengths = tf.cast(lengths, tf.int32)
        return lengths
    
    def scores_and_state(self):
        num_units = [self.params["rnn_cell_size"]] * self.params["rnn_layers"]
#         num_units = [128, 128, 128]
        stacked_rnn_cells = tf.nn.rnn_cell.MultiRNNCell([self.params["rnn_cell"](n) for n in num_units])
        # debugging code ----------------------------------------------------
        print(stacked_rnn_cells.state_size)
        # -------------------------------------------------------------------
#         rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * self.params["rnn_layers"])
        outputs, state = tf.nn.dynamic_rnn(cell=stacked_rnn_cells, 
                                           inputs=self.sequences, 
                                           dtype=tf.float32, 
                                           sequence_length=self.lengths, 
                                           initial_state=self.initial)
    
        # --debugging code --
#         self.outputs = outputs
#         self.state = state
        # --
        num_class = int(self.labels.shape[2])
        max_seq_len = int(self.labels.shape[1])
        outputs_flattened = tf.reshape(outputs, [-1, self.params["rnn_cell_size"]])
        Weights = tf.Variable(tf.truncated_normal([self.params["rnn_cell_size"], num_class], 
                                                  stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[num_class]))
        scores_flat = tf.nn.xw_plus_b(outputs_flattened, Weights, bias)
        scores = tf.reshape(scores_flat, [-1, max_seq_len, num_class])
        # debugging code ----------------------------------------------------
        self.state_shape = tf.shape(state)
        # -------------------------------------------------------------------
        return scores, state
    
    def average_out(self, tensor):
        # make padded part all zeros
        tensor *= self.mask
        lengths = tf.cast(self.lengths, tf.float32)
        # average out within each sequence
        tensor = tf.reduce_sum(tensor, axis=1) / lengths
        # average out across sequences
        tensor = tf.reduce_mean(tensor)
        return tensor
        
    def loss(self):
        scores_clipped = tf.clip_by_value(self.scores, 1e-10, 1.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores_clipped, 
                                                                   labels=self.labels)
        loss = self.average_out(cross_entropy)
        return loss
    
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self.scores, 2), 
                                       tf.argmax(self.labels, 2))
        correct_predictions = tf.cast(correct_predictions, tf.float32)
        accuracy = self.average_out(correct_predictions)
        return accuracy
    
    def pseudo_perplexity(self):
        pseudo_perplexity = tf.multiply(self.scores, self.labels)
        pseudo_perplexity = tf.reduce_max(pseudo_perplexity, axis=2)
        pseudo_perplexity = tf.clip_by_value(pseudo_perplexity, 1e-10, 1.0)
        pseudo_perplexity = self.average_out(pseudo_perplexity)
        pseudo_perplexity = tf.exp(pseudo_perplexity)
        return pseudo_perplexity
        
    def optimizer(self):
        gradients = self.params["optimizer"].compute_gradients(self.loss)
        if self.params["gradient_clipping"]:
            limit = self.params["gradient_clipping"]
            gradients = [(tf.clip_by_value(grad, -limit, limit), var)                          if grad is not None                         else (None, var)                          for grad, var in gradients]
        optimizer = self.params["optimizer"].apply_gradients(gradients)
        return optimizer

