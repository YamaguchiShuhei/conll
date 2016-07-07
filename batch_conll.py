import tensorflow as tf
import numpy as np
import collections
import time
import sys
import os
import random

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
         word.append("<eos>")
         chunk.append("<eos>")
   return word, chunk

def _build_vocab(wordlist):
   counter = collections.Counter(wordlist)
   count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
   words, _ = list(zip(*count_pairs))
   word_to_id = dict(zip(words, range(len(words))))
   return word_to_id

def _chunk_change(chunk):
   chunk_data = [0 for i in range(4)]
   if chunk == "B":
      chunk_data[0] = 1
   elif chunk == "I":
      chunk_data[1] = 1
   elif chunk == "O":
      chunk_data[2] = 1
   else:
      chunk_data[3] = 1
   return chunk_data                  

def _sentence_list(word_list, chunk_list, word_to_id):
   sentence_list = []
   sent_chunk = []
   sentence = []
   chunk = []
   for i in range(len(word_list)):
      if word_list[i] != "<eos>":
         if word_list[i] in word_to_id:
            sentence.append([word_to_id[word_list[i]]])
            chunk.append(_chunk_change(chunk_list[i]))
         else:
            sentence.append([len(word_to_id)+1])
            chunk.append(_chunk_change(chunk_list[i]))
      else:
         for k in range(78):
            if len(sentence) < 78:
               sentence.append([word_to_id["<eos>"]])
               chunk.append(_chunk_change("<eos>"))
            else:
               break   
         sentence_list.append(sentence)
         sent_chunk.append(chunk)
         sentence = []
         chunk = []
   return sentence_list, sent_chunk

def _file_to_word_ids(wordlist, word_to_id):
   lists = []
   for word in wordlist:
      if word in word_to_id:
         lists.append([word_to_id[word]])
      else:
         lists.append([len(word_to_id)+1])
   return lists

def _batch_random(sentence_list, sentence_chunk, batch_size = 1):
   random_list = []
   for i in range(len(sentence_list)):
      random_list.append(i)
   random.shuffle(random_list)
   return (np.array([sentence_list[ random_list[x] ] for x in range(batch_size)], dtype=np.int32),
           np.array([sentence_chunk[ random_list[x] ] for x in range(batch_size)]))


if __name__ == "__main__":
   random.seed(0)
   start_time = time.time()
   print("---read start---")
   data_path = sys.argv[1]
   train_path = os.path.join(data_path, "kadai-train.txt")
   test_path = os.path.join(data_path, "kadai-test.txt")
   train_word_list, train_chunk_list = _read_words(train_path)
   test_word_list, test_chunk_list = _read_words(test_path)
   word_to_id = _build_vocab(train_word_list)

   train_sent_list, train_sent_chunk = _sentence_list(train_word_list, train_chunk_list, word_to_id)
   test_sent_list, test_sent_chunk = _sentence_list(test_word_list, test_chunk_list, word_to_id)

   train_data = np.array(train_sent_list, dtype=np.int32)
   test_data = np.array(test_sent_list, dtype=np.int32)
   train_chunk = np.array(train_sent_chunk, dtype=np.float32)
   test_chunk = np.array(test_sent_chunk, dtype=np.float32)

   vocab = 0
   for i in test_data:
      for k in i:
         vocab += 1
   embed_size = 10

   print("---read finish---")

   batch_size = 10
   #x = tf.placeholder(tf.int32, [None, 1])
   x = tf.placeholder(tf.int32, [batch_size, 78, 1])
   #x = tf.placeholder(tf.int32, [1, 78])
   #x = tf.placeholder(tf.int32, [batch_size, 78])
   Wembed = tf.Variable(tf.random_uniform([vocab, embed_size], -1.0, 1.0))
   word_vec = tf.nn.embedding_lookup(Wembed, x)

   W = tf.Variable(tf.random_uniform([embed_size, 4], -1.0, 1.0))
   b = tf.Variable(tf.random_uniform([4], -1.0, 1.0))
   y = tf.nn.softmax(tf.matmul(tf.reshape(word_vec, [-1, embed_size]), W) + b)
   y_ = tf.placeholder(tf.float32, [batch_size, 78, 4])
   y_reshape = tf.reshape(y_, [-1, 4])
   cross_entropy = -tf.reduce_sum(y_reshape*tf.log(y))

   train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)
   print ("---start train---")
   for i in range(2000):
      batch_xs, batch_ys = _batch_random(train_data, train_chunk, batch_size)
      sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
      if (i+1)/10 == int((i+1)/10):
         print("count "+str(i+1), end = " ")
         if (i+1)/100 == int((i+1)/100):
            print()
            print(sess.run(cross_entropy, feed_dict={x:train_data[i:i+batch_size] , y_:train_chunk[i:i+batch_size] }))
   print ("---train finish---")
   correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_reshape,1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
   
   print("accuracy: ",end="")

   sum_acc = 0
   for i in range(0,len(test_data),batch_size):
      if len(test_data[i:i+batch_size]) == batch_size:
         sum_acc += sess.run(accuracy, feed_dict={x: test_data[i:i+batch_size], y_: test_chunk[i:i+batch_size]})
   print(sum_acc/int(len(test_data)/10))
   
   end_time = time.time()
   print ("calc time: " + str(end_time - start_time))
    
