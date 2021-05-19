from image_generation import gen_images, dataset
from tensorflow import keras
# import multiprocessing as mp

WIDTH = 288
HEIGHT = 288
OUT_DOWNSCALING = 2

BATCHES_PER_IMAGE = 10
NUMBERS_PER_BATCH = 6


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, batch_size=32, batches_per_epoch=32, validation=False):
        'Initialization'
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.validation = validation
        self.current_batch = 0

        # self.pool = mp.Pool()

        self.gen_data(asyn=False)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # # Generate data
        # X, y = self.__data_generation(list_IDs_temp)

        return (self.x[index*self.batch_size:(index+1)*self.batch_size],
                self.y[index*self.batch_size:(index+1)*self.batch_size])

    def on_epoch_end(self):
        'Generate dataset after each epoch'
        self.current_batch += 1
        if (self.current_batch % 4) == 0:
            self.gen_data()

    def apply_data(self, result):
        # (self.x, self.y) = result
        print('Data changed')

    def gen_data(self, asyn=False):
        if self.validation:
            (x, y) = dataset[1]
        else:
            (x, y) = dataset[0]

        if asyn and False:
            self.pool.apply_async(gen_images, args=(self.batch_size*self.batches_per_epoch, HEIGHT, WIDTH,
                                                    OUT_DOWNSCALING, BATCHES_PER_IMAGE, NUMBERS_PER_BATCH, x, y), callback=self.apply_data)
        else:
            self.x = self.y = None
            (self.x, self.y) = gen_images(self.batch_size*self.batches_per_epoch, HEIGHT, WIDTH,
                                          OUT_DOWNSCALING, BATCHES_PER_IMAGE, NUMBERS_PER_BATCH, x, y)

    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     y = np.empty((self.batch_size), dtype=int)

    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i,] = np.load('data/' + ID + '.npy')

    #         # Store class
    #         y[i] = self.labels[ID]

    #     return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
