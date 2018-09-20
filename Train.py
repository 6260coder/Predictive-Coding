
# coding: utf-8

import tensorflow as tf
import numpy as np
import Predictive_Coding_Model
import pickle
import data_helpers

# prepare data
with open("./data/indexed_sequences.pkl", "rb") as in_file:
    indexed_data = pickle.load(in_file)
print(indexed_data.shape)

# dataset = data_helpers.shuffle_data(indexed_data)
dataset = indexed_data
train_set, dev_set = data_helpers.partition_data(dataset, 0.1)

# create model
max_seq_len = indexed_data.shape[1]

vocab = data_helpers.vocab
vocab_codes = np.zeros([len(vocab) + 1, len(vocab)])
for i in range(len(vocab)):
    vocab_codes[i + 1, i] = 1.0

params = dict()
params["rnn_cell"] = tf.contrib.rnn.GRUCell
params["rnn_cell_size"] = 128
params["rnn_layers"] = 2
params["optimizer"] = tf.train.RMSPropOptimizer(0.002)
params["gradient_clipping"] = 2.0
params["embedding_dim"] = 100
params["vocab_size"] = len(vocab)
params["vocab_codes"] = vocab_codes

model = Predictive_Coding_Model.PredictiveCodingModel(max_seq_len, params)

# prepare sess and load previously trained model
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
ckpt = tf.train.get_checkpoint_state("./checkpoints/prev/")
epoch = 0
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch = int(ckpt.model_checkpoint_path.rsplit("-", 1)[1])
    print("model loaded")

# train the model
num_epochs = 10000
batch_size = 100
lowest_train_loss = 9999.0
for epoch in range(num_epochs):
    print("epoch {}:".format(epoch))
    generator = data_helpers.batch_generator(train_set, batch_size)
    train_loss = 0.0
    train_acc = 0.0
    num_batches = 0
    for i, batch in generator:
        feed_dict = {model.data:batch}
        _, loss, acc = sess.run([model.optimizer, model.loss, model.accuracy], 
                                feed_dict)
        num_batches = i + 1
        train_loss += loss
        train_acc += acc
        if i % 10 == 0:
            print("-{}".format(i), end="-")
    train_loss /= num_batches
    train_acc /= num_batches
    print("")
    print("training_loss: {}, training_accuracy: {}".format(train_loss, train_acc))
#             print("-{}-loss: {}, accuracy: {}".format(i,loss, acc))
    if train_loss < lowest_train_loss:
        saver.save(sess, "./checkpoints/the_model", global_step=epoch)
        lowest_train_loss = train_loss
        print("model saved")
    print("evaluating...")
    feed_dict = {model.data:dev_set}
    loss, acc, perp, scores, lengths = sess.run([model.loss, model.accuracy, model.pseudo_perplexity, model.scores, model.lengths], 
                                                feed_dict)
    print("loss: {}, accuracy: {}, perplexity: {}".format(loss, acc, perp))
    predicted_str = data_helpers.from_scores_to_str(scores[0], lengths[0])
    print(predicted_str)
    print("")

