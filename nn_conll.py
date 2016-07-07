import tensorflow as tf
import time
import os
import collections
import sys
import numpy as np

def  _read_words(filename):
    f = tf.gfile.GFile(filename, "r")
    word = []
    chunk = []
    for line in f.readlines():
        if len(line)>3:
            pair = line.strip().split("\t")
            word.append(pair[0])
            chunk.append(pair[1])
        else:
            pass
    return word, chunk

def _build_vocab(wordlist):
    counter = collections.Counter(wordlist)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def _file_to_word_ids(wordlist, word_to_id):
    lists = []
    for word in wordlist:
        if word in word_to_id:
            lists.append([word_to_id[word]])
        else:
            lists.append([len(word_to_id)+1])
    return lists

def _chunk_change(chunk_list):
    chunk_data = [[0 for i in range(3)] for j in range(len(chunk_list))]
    for i in range(len(chunk_list)):
        if chunk_list[i] == "B":
            chunk_data[i][0] = 1
        elif chunk_list[i] == "I":
            chunk_data[i][1] = 1
        else:
            chunk_data[i][2] = 1
    return chunk_data


if __name__ == "__main__":
    start_time = time.time()
    print("start time" + str(start_time))
    print("---read start---")
    data_path = sys.argv[1]
    train_path = os.path.join(data_path, "kadai-train.txt")
    test_path = os.path.join(data_path, "kadai-test.txt")
    train_wordlist, train_chunk_list = _read_words(train_path)
    test_wordlist, test_chunk_list = _read_words(test_path)
    word_to_id = _build_vocab(train_wordlist)
    train_data = np.array(_file_to_word_ids(train_wordlist,word_to_id), dtype=np.float32)
    test_data = np.array(_file_to_word_ids(test_wordlist,word_to_id), dtype=np.float32)
    train_chunk = np.array(_chunk_change(train_chunk_list),dtype=np.float32)
    test_chunk =np.array(_chunk_change(test_chunk_list),dtype=np.float32)
    vocab = len(test_data)
    embed_size = 10

    print("---read finish---")
    
    x = tf.placeholder(tf.int32, [None, 1])
    Wembed = tf.Variable(tf.random_uniform([vocab, embed_size], -1.0, 1.0))
    word_vec = tf.nn.embedding_lookup(Wembed, x)
    
    W = tf.Variable(tf.zeros([embed_size, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.nn.softmax(tf.matmul(tf.reshape(word_vec, [-1, embed_size]), W) + b)
    y_ = tf.placeholder(tf.float32, [None, 3])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    print ("---start train---")
    for i in range(40):
        batch_xs = train_data
        batch_ys = train_chunk
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
        if (i+1)/10 == int((i+1)/10):
            print("count "+str(i+1))

    print ("---train finish---")
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    print("accuracy: ",end="")
    print(sess.run(accuracy, feed_dict={x: test_data, y_: test_chunk}))
    
    end_time = time.time()
    print ("time: " + str(end_time))
    print ("calc time: " + str(end_time - start_time))

