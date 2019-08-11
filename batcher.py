import numpy as np
from dotmap import DotMap
import copy
import random


class batcher(object):
    '''
    seq2seqatt batcher.
    '''

    def __init__(
        self,
        data,
        seq_len,
        all_block_len,
        history_num,
        batch_size=50,
        random=True
    ):
        self.datas = []
        self.batch_size = batch_size
        self.datas_length = None   # 3600 / 30 = 120
        self.file_num = None    # 437
        self.cur_data = 0       # max = 120-1
        self.cur_batch = 0      # max = 437-1

        od = np.random.permutation(433)
        data_inputs = np.array(data.inputs)#[od[:100],:,:]        # (437, 3600, 29)
        data_labels = np.array(data.labels)#[od[:100],:,:]        # (437, 3600, 30)
        length = len(data_inputs[0])               # 437
        
        shp = list(data_inputs.shape)
        
        # data_inputs = np.concatenate((np.zeros([shp[0],17,shp[2]]),data_inputs, np.zeros([shp[0],12,shp[2]])), axis=1)
        print('flag6',np.shape(data_inputs))

    
        for i in range((shp[1] - seq_len - all_block_len)//seq_len-1):
            rint = np.random.randint(seq_len)
            od = np.random.permutation(shp[0])
            batch_input = data_inputs[od, i*seq_len + rint : i*seq_len + seq_len + all_block_len -1 + rint, :]
            batch_label = data_labels[od, i*seq_len + history_num + rint : i*seq_len + seq_len + history_num + rint, :]
            # batch_data = np.concatenate((batch_input, batch_label), axis=2)
            # np.random.shuffle(batch_data)
            # batch_input = batch_data[:, :, :29]
            # batch_label = batch_data[:, :, 29:]
            self.datas.append({'input': batch_input, 'label': batch_label})

        
        self.datas_length = len(self.datas)
        self.file_num = len(data_inputs) 
        # print(len(self.datas))      # 3600 / 30 = 120
        # print(np.array(self.datas[0]['input']).shape)   # (437, 30, 29)
        # print(np.array(self.datas[0]['label']).shape)   # (437, 30, 30)
        # print(np.array(self.datas[1]['input']).shape)   # (437, 60, 29)
        # print(np.array(self.datas[1]['label']).shape)   # (437, 60, 30)

    def has_next(self):
        if self.cur_data >= self.datas_length:
            self.cur_batch = 0
            self.cur_data = 0
            return False
        else:
            return True

    def next_batch(self):
        if not self.has_next():
            print("use has_next() to identify if has next batch before use next_batch() to get next batch!!!")
            exit()

        inputs = self.datas[self.cur_data]['input']
        labels = self.datas[self.cur_data]['label']
        if self.cur_batch + self.batch_size < self.file_num:
            data_input = inputs[self.cur_batch: self.cur_batch+self.batch_size, :, :]
            data_label = labels[self.cur_batch: self.cur_batch+self.batch_size, :, :]
            self.cur_batch += self.batch_size
            return data_input, data_label
        else:
            # print("cur_batch: ", self.cur_batch)
            # print("labels_len: ", len(labels))
            data_input = inputs[self.cur_batch: , :, :]
            data_label = labels[self.cur_batch: , :, :]
            self.cur_batch = 0
            self.cur_data += 1
            return data_input, data_label


