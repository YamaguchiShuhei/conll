import tensorflow as tf
import numpy as np
import collections
import time
import os
import random
import sys


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
   sent_len = []
   for i in range(len(word_list)):
      if word_list[i] != "<eos>":
         if word_list[i] in word_to_id:
            sentence.append([word_to_id[word_list[i]]])
            chunk.append(_chunk_change(chunk_list[i]))
         else:
            sentence.append([len(word_to_id)+1])
            chunk.append(_chunk_change(chunk_list[i]))
      else:
         sent_len.append(len(sentence))
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
   sent_len = np.array(sent_len)
   return sentence_list, sent_chunk, sent_len

def _batch_random(sentence_list, sentence_chunk, sent_len, batch_size = 1):
   random_list = []
   for i in range(len(sentence_list)):
      random_list.append(i)
   random.shuffle(random_list)
   return (np.array([sentence_list[ random_list[x] ] for x in range(batch_size)], dtype=np.int32),
           np.array([sentence_chunk[ random_list[x] ] for x in range(batch_size)]),
           np.array([sent_len[ random_list[x] ] for x in range(batch_size)], dtype=np.int32))


if __name__ == "__main__":
   random.seed(0)
   tf.set_random_seed(0)

   batch_size = 10

   start_time = time.time()
   print("---read start---")
   data_path = "/home/yamaguchi.13093/conll2000"
   train_path = os.path.join(data_path, "kadai-train.txt")
   test_path = os.path.join(data_path, "kadai-test.txt")
   train_word_list, train_chunk_list = _read_words(train_path)
   test_word_list, test_chunk_list = _read_words(test_path)
   word_to_id = _build_vocab(train_word_list)

   train_sent_list, train_sent_chunk, train_sent_len = _sentence_list(train_word_list, train_chunk_list, word_to_id)
   test_sent_list, test_sent_chunk, test_sent_len = _sentence_list(test_word_list, test_chunk_list, word_to_id)

   train_data = np.array(train_sent_list, dtype=np.int32)
   test_data = np.array(test_sent_list, dtype=np.int32)
   train_chunk = np.array(train_sent_chunk, dtype=np.float32)
   test_chunk = np.array(test_sent_chunk, dtype=np.float32)

   vocab_train = max(map(max,train_data))[0]
   vocab_test = max(map(max,test_data))[0]
   vocab = max(vocab_train, vocab_test) + 1

   devel_size = 1
   devel_data = train_data[-batch_size*devel_size:]
   devel_chunk = train_chunk[-batch_size*devel_size:]
   devel_sent_len = train_sent_len[-batch_size*devel_size:]
   train_data = train_data[:-batch_size*devel_size]
   train_chunk = train_chunk[:-batch_size*devel_size]
   train_sent_len = train_sent_len[:-batch_size*devel_size]

   embed_size = 100

   print("---read finish---")

   hidden_size = 150

   triangle = []
   for i in range(78+1):
      triangle.append([1 for x in range(i)] + [0 for x in range(78-i)])
   l_look = tf.constant(np.array(triangle, dtype=np.float32))

   x = tf.placeholder(tf.int32, [batch_size, 78, 1])
   Wembed = tf.Variable(tf.random_uniform([vocab, embed_size], -1.0, 1.0))
   word_vec = tf.nn.embedding_lookup(Wembed, x)
   keep_prob_wv = tf.placeholder(tf.float32)
   word_vec_drop = tf.nn.dropout(word_vec, keep_prob_wv)
   word_vec_reshape = tf.reshape(word_vec_drop, [batch_size, 78, embed_size])

   lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias = 0.0)
   inputs = [word_vec_reshape[:,time_step,:] for time_step in range(78)]
   l = tf.placeholder(tf.int32, [batch_size])
   outputs, final_state = tf.nn.rnn(lstm_cell, inputs, dtype=tf.float32, sequence_length=l)
   output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
   keep_prob_out = tf.placeholder(tf.float32)
   output_drop = tf.nn.dropout(output, keep_prob_out)

   W = tf.Variable(tf.random_uniform([hidden_size, 4], -1.0, 1.0))
   b = tf.Variable(tf.random_uniform([4], -1.0, 1.0))
   y = tf.nn.softmax(tf.matmul(output_drop, W) + b)
   yreshape = tf.reshape(y, [batch_size, 78, 4])

   W_hist = tf.histogram_summary("weights", W)
   b_hist = tf.histogram_summary("biases", b)

   y_ = tf.placeholder(tf.float32, [batch_size, 78, 4])

   cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_*tf.log(yreshape), reduction_indices=[2]) * tf.nn.embedding_lookup(l_look, l) / tf.cast(tf.reduce_sum(l), dtype=tf.float32))
   global_step = tf.Variable(0, trainable=False)
   learning_rate = tf.train.exponential_decay(0.01, global_step, 2000, 0.9, staircase=True)
   #loss = cross_entropy + 0.00001 * tf.nn.l2_loss(W)
   loss = cross_entropy
   train_step = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)

   cross_summary = tf.scalar_summary("cross_entropy", cross_entropy)

   correct_prediction = tf.equal(tf.argmax(yreshape,2), tf.argmax(y_,2))
   accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float") * tf.nn.embedding_lookup(l_look,l)) / tf.cast(tf.reduce_sum(l), "float")
   accuracy_summary = tf.scalar_summary("accuracy", accuracy)

   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)
   merged = tf.merge_all_summaries()
   train_writer = tf.train.SummaryWriter('data/train_drop', graph=sess.graph)
   test_writer = tf.train.SummaryWriter('data/devel_drop')

   print ("---start train---")
   for i in range(100000): 
      batch_xs, batch_ys, batch_len = _batch_random(train_data, train_chunk, train_sent_len, batch_size)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, l:batch_len, keep_prob_wv:0.8,keep_prob_out:0.5})

      result = sess.run(merged, feed_dict={x:train_data[0:batch_size], y_:train_chunk[0:batch_size], l:train_sent_len[0:batch_size], keep_prob_wv:1.0, keep_prob_out:1.0})
      
      train_writer.add_summary(result, i)

            
      if (i+1)/10 == int((i+1)/10):
         print("count "+str(i+1), end = " ")
         sys.stdout.flush()
         if (i+1)/100 == int((i+1)/100):
            print()
            print(sess.run((cross_entropy, accuracy), feed_dict={x:train_data[0:batch_size] ,
                                                                 y_:train_chunk[0:batch_size],
                                                                 l:train_sent_len[0:batch_size],
                                                                 keep_prob_wv:1.0,keep_prob_out:1.0}))
            devel_cross_entropy = 0.0
            devel_accuracy = 0.0
            for j in range(devel_size):
               devel_cross_entropy += sess.run(cross_entropy,
                                               feed_dict={x:devel_data[batch_size*j:batch_size*(j+1)] ,
                                                          y_:devel_chunk[batch_size*j:batch_size*(j+1)],
                                                          l:devel_sent_len[batch_size*j:batch_size*(j+1)],
                                                          keep_prob_wv:1.0,keep_prob_out:1.0})
               devel_accuracy += sess.run(accuracy,
                                               feed_dict={x:devel_data[batch_size*j:batch_size*(j+1)] ,
                                                          y_:devel_chunk[batch_size*j:batch_size*(j+1)],
                                                          l:devel_sent_len[batch_size*j:batch_size*(j+1)],
                                                          keep_prob_wv:1.0,keep_prob_out:1.0})
            devel_cross_entropy = devel_cross_entropy / devel_size
            devel_accuracy = devel_accuracy / devel_size
            print(devel_cross_entropy, devel_accuracy)
            result = sess.run(merged, feed_dict={x:devel_data[0:batch_size], y_:devel_chunk[0:batch_size], l:devel_sent_len[0:batch_size], keep_prob_wv:1.0,keep_prob_out:1.0})
            test_writer.add_summary(result, i)
            

   print ("---train finish---")
   exc = tf.placeholder(tf.int32, [batch_size])
   exc_acc = tf.reduce_sum(tf.cast(correct_prediction, "float") * tf.nn.embedding_lookup(l_look,l) * tf.nn.embedding_lookup(l_look,exc)) / tf.cast((tf.reduce_sum(l*exc) / 78), "float")
   
   print("accuracy: ",end="")
   sum_acc = 0
   ylist = []
   y_list = []
   for i in range(0,len(test_data),batch_size):
      if len(test_data[i:i+batch_size]) == batch_size:
         sum_acc += sess.run(accuracy, feed_dict={x: test_data[i:i+batch_size],
                                                  y_: test_chunk[i:i+batch_size],
                                                  l: test_sent_len[i:i+batch_size],
                                                  keep_prob_wv:1.0, keep_prob_out:1.0})
         ylist.extend(sess.run(tf.argmax(yreshape,2), feed_dict={x: test_data[i:i+batch_size], y_: test_chunk[i:i+batch_size], l: test_sent_len[i:i+batch_size], keep_prob_wv:1.0, keep_prob_out:1.0}))
         y_list.extend(sess.run(tf.argmax(y_,2),feed_dict={x: test_data[i:i+batch_size], y_: test_chunk[i:i+batch_size], l: test_sent_len[i:i+batch_size], keep_prob_wv:1.0, keep_prob_out:1.0}))
         
      else:
         amari = (len(test_data) % batch_size)
         sum_acc += sess.run(exc_acc, feed_dict={x: test_data[-batch_size:],
                                                 y_: test_chunk[-batch_size:],
                                                 l: test_sent_len[-batch_size:],
                                                 keep_prob_wv:1.0, keep_prob_out:1.0,
                                                 exc: np.array([0 for x in range(batch_size - amari)] + [78 for x in range(amari)], dtype=np.int32)})
   print(sum_acc/(int(len(test_data)/batch_size)+amari))
   
   end_time = time.time()
   print ("calc time: " + str(end_time - start_time))

   b_think = 0
   for i in range(len(ylist)):
      for k in range(test_sent_len[i]):
         if ylist[i][k] == 0:
            b_think += 1
   b_truth = 0
   for i in range(len(y_list)):
      for k in range(test_sent_len[i]):
         if y_list[i][k] == 0:
            b_truth += 1
   
   correct = 0
   for i in range(len(ylist)):
      for k in range(test_sent_len[i]):
         if ylist[i][k] == y_list[i][k] == 0 and (test_sent_len[i]-1) > k:
            for l in range(test_sent_len[i]-k-1):
               if ylist[i][k+l+1] != y_list[i][k+l+1]:
                  break
               elif ylist[i][k+l+1] == y_list[i][k+l+1] == 0:
                  correct += 1
                  break
               elif ylist[i][k+l+1] == y_list[i][k+l+1] == 2:
                  correct += 1
                  break
         elif ylist[i][k] == y_list[i][k] == 0 and k == (test_sent_len[i]-1):
            correct += 1


   tekigou = correct / b_think
   saigen = correct / b_truth
   f_chi = 2 * tekigou * saigen / (tekigou+saigen)

   print("think " +str(b_think), "truth " +str(b_truth), "correct " +str(correct))
   print("tekigou " +str(tekigou), "saigen " +str(saigen))
   print("f " +str(f_chi))
