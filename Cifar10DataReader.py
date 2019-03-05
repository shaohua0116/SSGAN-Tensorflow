import pickle
import numpy as np
import os


class Cifar10DataReader():
    def __init__(self, cifar_folder, onehot=True):
        self.cifar_folder = cifar_folder
        self.onehot = onehot
        self.data_index = 1
        self.read_next = True
        self.data_label_train = None
        self.data_label_test = None
        self.batch_index = 0

    def unpickle(self, f):
        fo = open(f, 'rb')
        d = pickle.load(fo, encoding='bytes')
        fo.close()
        return d

    def next_train_data(self, batch_size=100):
        assert 10000 % batch_size == 0, "10000%batch_size!=0"
        rdata = None
        rlabel = None
        if self.read_next:
            f = os.path.join(self.cifar_folder, "data_batch_%s" % (self.data_index))
            print('read: %s' % f)
            dic_train = self.unpickle(f)
            self.data_label_train = list(zip(dic_train[b'data'], dic_train[b'labels']))  # label 0~9
            np.random.shuffle(self.data_label_train)

            self.read_next = False
            if self.data_index == 5:
                self.data_index = 1
            else:
                self.data_index += 1

        if self.batch_index < len(self.data_label_train) // batch_size:
            # print self.batch_index
            datum = self.data_label_train[self.batch_index * batch_size:(self.batch_index + 1) * batch_size]
            self.batch_index += 1
            rdata, rlabel = self._decode(datum, self.onehot)
        else:
            self.batch_index = 0
            self.read_next = True
            return self.next_train_data(batch_size=batch_size)

        return rdata, rlabel

    def _decode(self, datum, onehot):
        rdata = list();
        rlabel = list()
        if onehot:
            for d, l in datum:
                rdata.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
                hot = np.zeros(10)
                hot[int(l)] = 1
                rlabel.append(hot)
        else:
            for d, l in datum:
                rdata.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
                rlabel.append(int(l))
        return rdata, rlabel

    def next_test_data(self, batch_size=100):
        if self.data_label_test is None:
            f = os.path.join(self.cifar_folder, "test_batch")
            print('read: %s' % f)
            dic_test = self.unpickle(f)
            data = dic_test[b'data']
            labels = dic_test[b'labels']  # 0~9
            self.data_label_test = list(zip(data, labels))

        np.random.shuffle(self.data_label_test)
        datum = self.data_label_test[0:batch_size]

        return self._decode(datum, self.onehot)


if __name__ == "__main__":
    dr = Cifar10DataReader(cifar_folder=r"/home/yc/PycharmProjects/SSGAN-Tensorflow/datasets/cifar10/cifar-10-batches-py")

    import matplotlib.pyplot as plt
    d, l = dr.next_test_data()
    print(np.shape(d), np.shape(l))
    plt.imshow(d[0])
    plt.show()
    for i in range(600):
        d, l = dr.next_train_data(batch_size=100)
        print(np.shape(d), np.shape(l))
