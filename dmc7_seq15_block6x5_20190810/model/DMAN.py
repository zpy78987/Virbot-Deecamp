import pandas as pd
import numpy as np
from basic_layer.NN_adam import NN
from util.batcher import batcher
from basic_layer.VRNN import VRNN
from basic_layer.LSTMs import bi_lstm_layer as lstm_layer
from basic_layer.Self_Attention import Self_Attention
from data.datahandle import input_prepare
from data.output_handel import output_handle
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import numpy.linalg as npl

class DMAN(NN):
    def __init__(self, config):    #定义超参数，从配置文件调入其他参数
        super(DMAN, self).__init__(config)
        self.seqlen = 12
        self.block_len = 6
        self.block_num = 5
        self.history_num = 25
        self.all_block_len = self.block_len*self.block_num
        self.test_loss = [9999] * 20
        if config != None:
            self.edim = config['edim']
            self.label_dim = config['label_dim']
            self.epoch = config['nepoch']
            self.fmy_output_path = config['fmy_output_path']
            self.model_save_path = config['model_save_path']
            self.batch_size = config['batch_size']
            self.gpu_num = config['gpu_num']
            self.model_name = config['model_name']
        else:
            self.edim = None
            self.label_dim = None
            self.epoch = None
            self.fmy_output_path = None
            self.model_save_path = None
            self.batch_size = None
            self.gpu_num = None
            self.model_name = None

    def set_placeholder(self):  #定义网络输入和输出的形状格式
        self.inputs = tf.placeholder(
            tf.float32,
            [None, None, self.all_block_len, self.edim],  # [batch_sie, sequence_len, 15, 29]
            name="inputs"
        )

        self.labels = tf.placeholder(
            tf.float32,
            [None, None, self.label_dim],   # [bathc_size, sequence_len, label_dim]
            name="labels"
        )

        shape = tf.shape(self.inputs)
        batch_size = shape[0]
        sequence_len = shape[1]
        print('label3')
        print(shape)
        
        self.reshape_inputs = tf.reshape(self.inputs, [batch_size, sequence_len, -1, self.block_len, self.edim]) # [batch_sie, sequence_len, 5, 3, edim]

        return self.reshape_inputs, self.labels

    def lstm_layer1(self, inputs):    #定义底层LSTM(下采样层），return_sequence = ?
        """
        input: [batch_sie, sequence_len, 5, 3, 29]
        output: [batch_sie, sequence_len, 5, 3, 64]
        """
        with tf.variable_scope('layer1'):
            shape = tf.shape(inputs)    # inputs: [batch_sie, sequence_len, 5, 3, 29]
            num1, num2, num3 = shape[0], shape[1], shape[2]

            inputs = tf.reshape(inputs, [num1*num2*num3, -1, 29])
            lstm_output, _ = lstm_layer(inputs, 64, num1*num2*num3)
            lstm_output = tf.reshape(lstm_output, [num1, num2, num3, -1, 64])
            lstm_output = lstm_output[:,:,:,-1,:]  #bs*seq*5*64
            lstm_output = tf.expand_dims(lstm_output,axis=3)

        return lstm_output          # outputs: [batch_sie, sequence_len, 5, 3, 64]
    
    def attn_layer1(self, inputs):  #定义底层attention，对三帧的特征进行归纳。
        """
        input: [batch_sie, sequence_len, 5, 3, 64]
        output: [batch_sie, sequence_len, 5, 64]
        """
        with tf.variable_scope('layer1'):
            shape = inputs.get_shape().as_list()
            batch_size, sequence_len, split_num, dim = shape[0], shape[1], shape[2], shape[4]
            output = tf.layers.dense(inputs, units=dim, activation='tanh')          # output: [batch_sie, sequence_len, 5, 3, 64]
            output = tf.nn.softmax(tf.layers.dense(output, units=1, activation=None))  # output: [batch_sie, sequence_len, 5, 3, 1]
            output = tf.matmul(output, inputs, transpose_a=True)                    # output: [batch_sie, sequence_len, 5, 1, 64]
            output = tf.squeeze(output, squeeze_dims=3)                             # output: [batch_sie, sequence_len, 5, 64]
            
        return output

    def lstm_layer2(self, inputs):     #语义层
        """
        input: [batch_sie, sequence_len, 5, 64]
        ouput: [batch_sie, sequence_len, 5, 128]
        """
        # print(inputs.get_shape())
        with tf.variable_scope('layer2'):
            shape = tf.shape(inputs)
            dim1, dim2, dim3 = shape[0], shape[1], shape[2]

            inputs = tf.reshape(inputs, [dim1*dim2, dim3, 64])    # [batch_sie * sequence_len, 5, 64]
            lstm_output, _ = lstm_layer(inputs, 128, dim1*dim2)
            lstm_output = tf.reshape(lstm_output, [dim1, dim2, dim3, 128])
            
        return lstm_output

    def attn_layer2(self, inputs):
        """
        input: [batch_sie, sequence_len, 5, 128]
        ouput: [batch_sie, sequence_len, 128]
        """
        with tf.variable_scope('layer2'):
            shape = inputs.get_shape().as_list()
            dim1, dim2, dim3, dim4 = shape[0], shape[1], shape[2], shape[3]
            output = tf.layers.dense(inputs, units=dim4, activation='tanh')         # output: [batch_sie, sequence_len, 5, 128]
            output = tf.nn.softmax(tf.layers.dense(output, units=1, activation=None))   # output: [batch_sie, sequence_len, 5, 1]
            output = tf.matmul(output, inputs, transpose_a=True)                   # output: [batch_sie, sequence_len, 1, 128]
            output = tf.squeeze(output, squeeze_dims=2)                             # output: [batch_sie, sequence_len, 128]

        return output

    def lstm_layer3(self, inputs): # 字与字层面的连接
        """
        input: [batch_sie, sequence_len, 256]
        output: [batch_sie, sequence_len, 256]
        """
        with tf.variable_scope('layer3'):
            shape = tf.shape(inputs)
            batch_size = shape[0]

            lstm_output, _ = lstm_layer(inputs, 256, batch_size)   

        return lstm_output
    
    def mlp_layer(self, inputs):      #（sequence loss层）
        """
        input: [batch_sie, sequence_len, 256]
        output: [batch_sie, sequence_len, 30]
        """
        with tf.variable_scope('top_layer'):
            output = tf.layers.dense(inputs, units=128, activation='tanh')  # [batch_sie, sequence_len, 128]
            output = tf.layers.dense(output, units=30, activation=None)     # [batch_sie, sequence_len, 30]
        
        return output

    def build_model(self):
        inputs, labels = self.set_placeholder()

        # ====== build your own model ======

        lstm_layer1_output = self.lstm_layer1(inputs)               # output: [batch_sie, sequence_len, 5, 3, 64]
        attn_layer1_output = self.attn_layer1(lstm_layer1_output)   # output: [batch_sie, sequence_len, 5, 64]
        lstm_layer2_output = self.lstm_layer2(attn_layer1_output)   # output: [batch_sie, sequence_len, 5, 128]
        attn_layer2_output = self.attn_layer2(lstm_layer2_output)   # output: [batch_sie, sequence_len, 128]
        lstm_layer3_output = self.lstm_layer3(attn_layer2_output)   # output: [batch_sie, sequence_len, 256]
        model_output = self.mlp_layer(lstm_layer3_output)
        self.model_output = model_output

        print('flag5')
        print(self.model_output.get_shape(),labels.get_shape())

        self.rse = tf.norm(self.model_output - labels)/tf.norm(labels)
        self.loss = tf.losses.mean_squared_error(self.model_output , labels)

        # ====== build your own model ======


        self.params = tf.trainable_variables()
        self.optimize, self.islfGrad = super(DMAN, self).optimize_normal(
            self.loss, self.params)


    def train(self, sess, train_data, test_data, saver):   #训练过程就是减少loss，更新网络的权重

        # aa = np.append(np.zeros([1,17,29]),np.zeros([1,13,29]),axis=1)

        # 分割数据
        bt = batcher(train_data, batch_size=self.batch_size,seq_len=self.seqlen,all_block_len=self.all_block_len,history_num = self.history_num)
        print("-------------begin train------------------")
        min_loss = 9999
        cnt = 0
        for t_round in range(self.epoch):
            loss = []
            while(1):
                if not bt.has_next():
                    break
                
                batch_input, batch_label = bt.next_batch()      # [batch_size, dequnce_len, 29]
                
                # print('flag2')
                # print(np.shape(batch_input))
                # print(np.shape(batch_label))

                batch_input = input_prepare(batch_input, self.all_block_len)    # [batch_size, sequence_len,15, 29]
                
                # print('flag2.1')
                # print(np.shape(batch_input))
                # print(np.shape(batch_label))

                feed_dict = {
                    self.inputs: batch_input,
                    self.labels: batch_label
                }

                # print('flag4')
                # print(np.shape(batch_label))
                # # print(batch_input)
                # tmpp=tf.shape(self.labels)
                # print(self.labels.get_shape())
                # print(tmpp[0],tmpp[1],tmpp[2])
                # batch_input = np.array(batch_input)
                # print(type(batch_input),type(batch_label))
                # # print(batch_input)
                crt_loss, rse, optimize = sess.run([self.loss, self.rse, self.optimize], feed_dict=feed_dict)
                loss.append(crt_loss)
                # print(lstm.shape)

            mean_loss = np.mean(loss)
            rse = np.mean(rse)
            print("\nEpoch{}\ttain-l2loss: {:.6f}, rse: {:.6f}".format(t_round, mean_loss, rse))
            if min_loss > mean_loss:
                min_loss = mean_loss
                self.save_model(sess, self.model_save_path, self.model_name, saver)
            print("testing....")
            if t_round%20==0:
                self.test(sess, test_data, saver)
            cnt += 1


    def test(self, sess, test_data, saver):     # fm_y = sess.run([self.model_output], feed_dict=feed_dict) 就是model.predict，其他是评估 
        data_inputs = test_data.inputs
        data_labels = test_data.labels
        data_names = test_data.names

        # print('falg10: ',data_labels)
        if data_labels==0:
            # print('flag11')
            sh = np.shape(data_inputs)
            data_labels = np.zeros([sh[0],sh[1],30])
            # print(np.shape(data_labels))

        for i in range(len(data_inputs)):
            sequence_len = len(data_inputs[i])
            batch_input = np.expand_dims(data_inputs[i], 0)     # [1, sequence_len(3600), 29]
            batch_label = np.expand_dims(data_labels[i], 0)
            name = data_names[i]

            print('flag8.1',np.array(batch_input).shape)

            batch_input = np.concatenate((np.zeros([1,self.history_num,29]),batch_input,np.zeros([1,self.all_block_len-self.history_num-1,29])),axis=1)
            sequence_len = sequence_len

            inputs = [input_prepare(batch_input[:, j:j+self.seqlen+self.all_block_len-1, :], self.all_block_len)[0] for j in range(sequence_len-self.seqlen+1)]  # [3571, 30, 15, 29]
            labels = [batch_label[:, j:j+self.seqlen, :][0] for j in range(sequence_len-self.seqlen+1)] # [3571, 30, 30]
            print(np.array(batch_input).shape)
            print(np.array(inputs).shape)
            print(np.array(labels).shape)
            feed_dict = {
                self.inputs: inputs,                   # [3571, 30, 15, 29]
                self.labels: labels                   # [3571, 30, 30]
            }
            fm_y = sess.run([self.model_output], feed_dict=feed_dict)   # [1, 3571, 30, 30]
            fm_y = output_handle(fm_y[0])
            los = npl.norm(fm_y-data_labels[i])/npl.norm(data_labels[i])
            print("test {}-loss: {:.6f}".format(name, los))
            super(DMAN, self).save_fmy(sess, self.fmy_output_path, fm_y, name, self.gpu_num, self.model_name)


    def testonly(self, sess, test_data, saver):   #只predict，不评估
        data_inputs = test_data.inputs
        data_names = test_data.names

        for i in range(len(data_inputs)):
            sequence_len = len(data_inputs[i])
            batch_input = np.expand_dims(data_inputs[i], 0)     # [1, sequence_len(3600), 29]
            name = data_names[i]

            print('flag8.1',np.array(batch_input).shape)

            batch_input = np.concatenate((np.zeros([1,self.history_num,29]),batch_input,np.zeros([1,self.all_block_len-self.history_num-1,29])),axis=1)
            sequence_len = sequence_len

            inputs = [input_prepare(batch_input[:, j:j+self.seqlen+self.all_block_len-1, :], self.all_block_len)[0] for j in range(sequence_len-self.seqlen+1)]  # [3571, 30, 15, 29]
            
            print(np.array(batch_input).shape)
            print(np.array(inputs).shape)
            feed_dict = {
                self.inputs: inputs#,                   # [3571, 30, 15, 29]
                #self.labels: labels                   # [3571, 30, 30]
            }
            fm_y = sess.run([self.model_output], feed_dict=feed_dict)   # [1, 3571, 30, 30]
            fm_y = output_handle(fm_y[0])
            # los = npl.norm(fm_y-data_labels[i])/npl.norm(data_labels[i])
            print("test {}-loss: {:.6f}".format(name, -1.0))
            super(DMAN, self).save_fmy(sess, self.fmy_output_path, fm_y, name, self.gpu_num, self.model_name)

