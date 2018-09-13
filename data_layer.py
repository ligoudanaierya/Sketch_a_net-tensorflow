import numpy as np
import h5py
from scipy.ndimage.interpolation import rotate
import scipy.io as sio
import os
import logging

class DataLayer:
    
    NUM_CLASSES = 250
    NUM_ITEMS_PER_CLASS = 80
    NUM_TRAIN_PER_CLASS = 54
    NUM_TEST_PER_CLASS = 26
    NUM_VALIDATION_COPIES =10

    INPUT_SIZE = 256
    OUTPUT_SIZE = 225
    NUM_CHANNELS = 6

    def __init__(self, file_path, batch_size=135):
        
        '''
        Initialize the variables and loads the data file
        '''
        
        # Initialize the cursor
        self.train_cursor = 0
        self.test_cursor = 0
        self.train_batch_size = batch_size
        self.test_batch_size = batch_size // 10

        #Initialize the idx

        a_train = np.tile(np.arange(self.NUM_TRAIN_PER_CLASS),self.NUM_CLASSES).reshape(self.NUM_CLASSES, self.NUM_TRAIN_PER_CLASS)
        b_train = np.tile(np.arange(self.NUM_CLASSES)*self.NUM_ITEMS_PER_CLASS, self.NUM_TRAIN_PER_CLASS).reshape(self.NUM_TRAIN_PER_CLASS, self.NUM_CLASSES).T
        self.train_idxs = (a_train + b_train).reshape(-1)

        a_test = np.tile(np.arange(self.NUM_TEST_PER_CLASS), self.NUM_CLASSES).reshape(self.NUM_CLASSES, self.NUM_TEST_PER_CLASS)
        b_test = np.tile(np.arange(self.NUM_CLASSES)*self.NUM_ITEMS_PER_CLASS, self.NUM_TEST_PER_CLASS).reshape(self.NUM_TEST_PER_CLASS, self.NUM_CLASSES).T
        self.test_idx = (a_test + b_test+ self.NUM_TRAIN_PER_CLASS).reshape(-1)
        
           
       #load the .mat file containing the data 
        data = h5py.File(file_path)
        self.dataset_images = data['imdb']['images']['data']
        self.dataset_labels = data['imdb']['images']['labels']
        print("Dataset Loaded")
       
    def get_images_shape(self):
        return(self.batch_size, self.OUTPUT_SIZE, self.OUTPUT_SIZE, self.NUM_CHANNELS)
    
    def next_batch_train(self, batch_size=None):
        
        if batch_size is None :
            batch_size = self.train_batch_size
        output_size = self.OUTPUT_SIZE
        input_size = self.INPUT_SIZE
        
        #create an array of indicies to retrive
        idx = self.train_idx[self.train_cursor:self.train_cursor+batch_size]
        if self.train_cursor+batch_size >= self.train_idx.size:
            idx = np.append(idx, self.train_idx[:self.train_cursor+batch_size-self.train_idx.size])
        #retrive the dataset
        laebls = self.dataset_labels[idx, :].reshape(-1)
        images_raw = self.dataset_images[idx,:,:,:].swapaxes(1, 3)
        #data Argumentation
        images = np.zeros((batch_size, output_size, output_size, images_raw.shape[3]))
        x = np.random.randint(input_size - output_size, size=batch_size)
        y = np.random.randint(input_size - output_size, size=batch_size)
        flip = np.random.rand(batch_size)>0.45
        degs = (np.random.rand(batch_size)>0.45)*(np.random.randint(11, size=batch_size) - 5.0)
        
        for i in range(batch_size):
            images[i,:,:,:] = images_raw[i, x[i]:x[i]+output_size, y[i]:y[i]+output_size, :]
            if flip[i]:
                images[i,:, :, :] = np.fliplr(images[i, :, :, :]) 
            if degs[i] != 0 :
                images[i, :, :, :] = rotate(images[i,:, :, :], degs[i], cval=255.0, reshape=False)
        #move the cursor
        self.train_cursor = (self.train_cursor + batch_size) % (self.NUM_TRAIN_PER_CLASS * self.NUM_CLASSES)
        return (255 - images, labels - 1)
    def next_batch_test(self, batch_size=None):
        if batch_size is None:
            batch = self.test_batch_size
        output_size = self.OUTPUT_SIZE
        input_size = self.INPUT_SIZE

        idx = self.test_idx[self.test_cursor:self.test_cursor+batch_size]
        if self.test_cursor + batch_size >= self.test_idx.size:
            idx = np.append(idx, self.test_idx[:self.test_cursor+batch_size - self.test_idx.size])
        print(idx)
        labels = self.dataset_labels[idx,:].reshape(-1)
        #images = self.dataset_images[idx,:,0:output_size,0:output_size].swapaxes(1,3)
        
        #labels = np.tile(self.dataset_labels[idx,:].reshape(-1), 10)#because of the test images' argumentation
        images_raw = self.dataset_images[idx, :, :, :].swapaxes(1, 3)
        images = np.concatenate((images_raw[:, 0:output_size, 0:output_size,:],
                                images_raw[:, input_size-output_size:input_size+1, input_size-output_size:input_size+1,:],
                                images_raw[:, input_size-output_size:input_size+1, 0:output_size, :],
                                images_raw[:, 0:output_size, input_size - output_size:input_size+1, :],
                                images_raw[:, (input_size-output_size+1)//2:input_size-(input_size-output_size+1)//2 + 1, (input_size-output_size+1)//2:input_size-(input_size-output_size+1)//2 + 1,:]), axis=0)
        images = np.concatenate((images, np.fliplr(images)), axis=0)

        #move the test_cursor
        self.test_cursor = (self.test_cursor+batch_size) % (self.NUM_TEST_PER_CLASS * self.NUM_CLASSES)

        return (255.0-images, labels-1)
def load_pretrained_model(file_path):
    if file_path is None or not os.path.isfile(file_path):
        print('Pretrained Model is Not Avaliable')
        return None, None
    data = sio.loadmat(file_path)
    weights = {}
    biases = {}
    conv_idxs = [0, 3, 6, 8, 10, 13, 16, 19]
    for i, idx in enumerate(conv_idxs):
        weights['conv'+str(i+1)] = data['net']['layers'][0][0][0][idx]['filters'][0][0]
        biases['conv' + str(i+1)] = data['net']['layers'][0][0][0][idx]['biases'][0][0].reshape(-1)
        
    print ('Pretrained Model Loaded!')
    return (weights, biases)       
    











