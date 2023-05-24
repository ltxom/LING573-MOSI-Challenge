import pickle

with open('./pre-processed-data/train.data', 'rb') as f:
    train = pickle.load(f)

with open('./pre-processed-data/test.data', 'rb') as f:
    test = pickle.load(f)

with open('./pre-processed-data/dev.data', 'rb') as f:
    dev = pickle.load(f)


def save_10_percent(data, filename):
    temp = []
    for i in range(len(data) // 10):
        temp.append(data[i])
    with open(filename, 'wb') as w:
        pickle.dump(temp, w)


save_10_percent(train, './pre-processed-data/train0.1.data')
save_10_percent(test, './pre-processed-data/test0.1.data')
save_10_percent(dev, './pre-processed-data/dev0.1.data')