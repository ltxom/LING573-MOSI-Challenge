import pickle
import sys
import numpy as np
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dot, Activation
from tensorflow.keras.models import load_model
from keras import backend as K


def shrink_emotion_data():
    with open('pre-processed-data/sentiment/avg_train_X.data', 'rb') as w:
        train_X = pickle.load(w)
    with open('pre-processed-data/emotion/train_y.data', 'rb') as w:
        train_y = pickle.load(w)
    class_dict = {}
    for i in range(len(train_y)):
        max_indices = np.argmax(train_y[i])
        if max_indices not in class_dict:
            class_dict[max_indices] = 0
        class_dict[max_indices] += 1
    index = 0
    new_train_X = []
    new_train_y = []
    counter = 0
    while index < len(train_y):
        max_indices = np.where(train_y[index] == np.max(train_y[index]))[0]
        if len(max_indices) == 1 and max_indices[0] == 0 and counter < 8000:
            counter += 1
        else:
            new_train_X.append(train_X[index])
            new_train_y.append(train_y[index])
        index += 1
    new_class_dict = {}
    for i in range(len(new_train_y)):
        max_indices = np.argmax(new_train_y[i])
        if max_indices not in new_class_dict:
            new_class_dict[max_indices] = 0
        new_class_dict[max_indices] += 1

    with open('pre-processed-data/emotion/shrink_train_x.data', 'wb') as w:
        pickle.dump(new_train_X, w)
    with open('pre-processed-data/emotion/shrink_train_y.data', 'wb') as w:
        pickle.dump(new_train_y, w)


def save_emotion_labels():
    with open('./pre-processed-data/train.data', 'rb') as f:
        train = pickle.load(f)

    with open('./pre-processed-data/dev.data', 'rb') as f:
        dev = pickle.load(f)

    with open('./pre-processed-data/test.data', 'rb') as f:
        test = pickle.load(f)

    def get_Xy(dataset):
        X = []
        y = []
        for t in dataset:
            X.append(t[0])
            y.append(t[1][0][1:])
        return X, y

    (train_X, train_y) = get_Xy(train)
    (test_X, test_y) = get_Xy(test)
    (dev_X, dev_y) = get_Xy(dev)

    with open('pre-processed-data/emotion/train_y.data', 'wb') as w:
        pickle.dump(train_y, w)
    with open('pre-processed-data/emotion/test_y.data', 'wb') as w:
        pickle.dump(test_y, w)
    with open('pre-processed-data/emotion/dev_y.data', 'wb') as w:
        pickle.dump(dev_y, w)


def save_sentiment_labels():
    with open('./pre-processed-data/train.data', 'rb') as f:
        train = pickle.load(f)

    with open('./pre-processed-data/dev.data', 'rb') as f:
        dev = pickle.load(f)

    with open('./pre-processed-data/test.data', 'rb') as f:
        test = pickle.load(f)

    def get_Xy(dataset):
        X = []
        y = []
        for t in dataset:
            X.append(t[0])
            y.append(t[1][0][0] / 6.0 + 0.5)
        return X, y

    (train_X, train_y) = get_Xy(train)
    (test_X, test_y) = get_Xy(test)
    (dev_X, dev_y) = get_Xy(dev)

    with open('pre-processed-data/sentiment/train_y.data', 'wb') as w:
        pickle.dump(train_y, w)
    with open('pre-processed-data/sentiment/test_y.data', 'wb') as w:
        pickle.dump(test_y, w)
    with open('pre-processed-data/sentiment/dev_y.data', 'wb') as w:
        pickle.dump(dev_y, w)


def save_padded_data():
    with open('./pre-processed-data/train.data', 'rb') as f:
        train = pickle.load(f)

    with open('./pre-processed-data/dev.data', 'rb') as f:
        dev = pickle.load(f)

    with open('./pre-processed-data/test.data', 'rb') as f:
        test = pickle.load(f)

    # Sentiment Task: Get (x, y)
    def get_Xy(dataset):
        X = []
        y = []
        for t in dataset:
            X.append(t[0])
            y.append(t[1][0][0])
        return X, y

    (train_X, train_y) = get_Xy(train)
    (test_X, test_y) = get_Xy(test)
    (dev_X, dev_y) = get_Xy(dev)

    # Get maximum shape of each modality
    def get_max_shape(a, b, c):
        max_shape = 0  # Initialize with zeros
        for t in [a, b, c]:
            for modality in t:
                text_shape = modality[0].shape[0]
                max_shape = max(max_shape, text_shape)
        return max_shape

    # Prepare the data
    def prepare_data(X, max_shape):
        prepared_data = []
        for modality in X:
            text_shape = modality[0].shape[0]
            acoustic_shape = modality[1].shape[0]
            visual_shape = modality[2].shape[0]
            pad_text = np.pad(modality[0], ((0, max_shape - text_shape), (0, 0)))
            pad_acoustic = np.pad(modality[1], ((0, max_shape - acoustic_shape), (0, 0)))
            pad_visual = np.pad(modality[2], ((0, max_shape - visual_shape), (0, 0)))
            prepared_data.append((pad_text, pad_acoustic, pad_visual))
        return prepared_data

    max_shape = get_max_shape(train_X, test_X, dev_X)
    pad_train_X = prepare_data(train_X, max_shape)
    pad_dev_X = prepare_data(dev_X, max_shape)
    pad_test_X = prepare_data(test_X, max_shape)
    with open('pre-processed-data/sentiment/padded_train_X.data', 'wb') as w:
        pickle.dump(pad_train_X, w)
    with open('pre-processed-data/sentiment/padded_test_X.data', 'wb') as w:
        pickle.dump(pad_test_X, w)
    with open('pre-processed-data/sentiment/padded_dev_X.data', 'wb') as w:
        pickle.dump(pad_dev_X, w)


def save_ave_data():
    with open('./pre-processed-data/train.data', 'rb') as f:
        train = pickle.load(f)

    with open('./pre-processed-data/dev.data', 'rb') as f:
        dev = pickle.load(f)

    with open('./pre-processed-data/test.data', 'rb') as f:
        test = pickle.load(f)

    # Sentiment Task: Get (x, y)
    def get_Xy(dataset):
        X = []
        y = []
        for t in dataset:
            X.append(t[0])
            y.append(t[1][0][0])
        return X, y

    (train_X, train_y) = get_Xy(train)
    (test_X, test_y) = get_Xy(test)
    (dev_X, dev_y) = get_Xy(dev)

    def get_ave(data):
        result = []
        for t in data:
            result.append((np.mean(t[0], axis=0).reshape((1, 300)), np.mean(t[1], axis=0).reshape((1, 713)),
                           np.mean(t[2], axis=0).reshape((1, 74))))
        return result

    with open('pre-processed-data/sentiment/avg_train_X.data', 'wb') as w:
        pickle.dump(get_ave(train_X), w)
    with open('pre-processed-data/sentiment/avg_test_X.data', 'wb') as w:
        pickle.dump(get_ave(test_X), w)
    with open('pre-processed-data/sentiment/avg_dev_X.data', 'wb') as w:
        pickle.dump(get_ave(dev_X), w)


def train_SVR():
    with open('pre-processed-data/sentiment/avg_train_X.data', 'rb') as w:
        train_X = pickle.load(w)
    with open('pre-processed-data/sentiment/train_y.data', 'rb') as w:
        train_y = pickle.load(w)
    text = np.array([x[0].reshape((300,)) for x in train_X])
    m = SVR(kernel='linear', verbose=1)
    m.fit(text, train_y)
    with open('trained_model/SVR.pkl', 'wb') as f:
        pickle.dump(m, f)


def train_SVR_emotion():
    with open('pre-processed-data/sentiment/avg_train_X.data', 'rb') as w:
        train_X = pickle.load(w)
    with open('pre-processed-data/emotion/train_y.data', 'rb') as w:
        train_y = pickle.load(w)
    text = np.array([x[0].reshape((300,)) for x in train_X])
    y = [np.argmax(x, axis=0) / 5 + 0.0001 for x in train_y]
    m = SVR(kernel='rbf', verbose=1)
    m.fit(text, y)
    with open('trained_model/SVR_emotion.pkl', 'wb') as f:
        pickle.dump(m, f)


def train_sentiment_model():
    with open('pre-processed-data/sentiment/avg_train_X.data', 'rb') as w:
        train_X = pickle.load(w)
    with open('pre-processed-data/sentiment/avg_dev_X.data', 'rb') as w:
        dev_X = pickle.load(w)
    with open('pre-processed-data/sentiment/train_y.data', 'rb') as w:
        train_y = pickle.load(w)
    with open('pre-processed-data/sentiment/dev_y.data', 'rb') as w:
        dev_y = pickle.load(w)

    def multimodal_sentiment_model(text_shape, acoustic_shape, visual_shape):
        text_input = Input(shape=text_shape, name='text_input')
        text_lstm = LSTM(256, return_sequences=True)(text_input)

        acoustic_input = Input(shape=acoustic_shape, name='acoustic_input')
        acoustic_lstm = LSTM(256, return_sequences=True)(acoustic_input)

        visual_input = Input(shape=visual_shape, name='visual_input')
        visual_lstm = LSTM(256, return_sequences=True)(visual_input)

        attention_text_acoustic = Dot(axes=[2, 2])([text_lstm, acoustic_lstm])
        attention_text_visual = Dot(axes=[2, 2])([text_lstm, visual_lstm])
        attention_weights = Concatenate(axis=2)([attention_text_acoustic, attention_text_visual])
        attention_weights = Activation('softmax')(attention_weights)

        attended_text = Dot(axes=[1, 1])([attention_weights, text_lstm])
        attended_acoustic = Dot(axes=[1, 1])([attention_weights, acoustic_lstm])
        attended_visual = Dot(axes=[1, 1])([attention_weights, visual_lstm])

        multimodal_features = Concatenate()([attended_text, attended_acoustic, attended_visual])

        dense1 = Dense(256, activation='relu')(multimodal_features)
        dense2 = Dense(128, activation='relu')(dense1)
        output = Dense(1, activation='tanh')(dense2)

        model = Model(inputs=[text_input, acoustic_input, visual_input], outputs=output)
        return model

    # Train the model
    def train_model(train_X, train_y, dev_X, dev_y, batch_size=128, epochs=20):
        text_shape = (train_X[0][0].shape[0], train_X[0][0].shape[1],)
        acoustic_shape = (train_X[0][0].shape[0], train_X[1][0].shape[1],)
        visual_shape = (train_X[0][0].shape[0], train_X[2][0].shape[1],)
        model = multimodal_sentiment_model(text_shape, acoustic_shape, visual_shape)
        model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)
        callbacks = tf.keras.callbacks.CallbackList(
            None,
            add_history=True,
            add_progbar=True,
            model=model,
            epochs=epochs,
            verbose=1,
            steps=len(train_X[0]) // batch_size
        )
        callbacks.on_train_begin()
        for epoch in range(epochs):
            model.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            print('Epoch', epoch + 1)
            for i in range(0, len(train_X[0]), batch_size):
                callbacks.on_train_batch_begin(i // batch_size - 1)
                offset = batch_size
                if i + batch_size >= len(train_X[0]):
                    offset = len(train_X[0]) - i
                batch_X = [train_X[0][i:i + offset], train_X[1][i:i + offset], train_X[2][i:i + offset]]
                batch_y = train_y[i:i + offset]
                logs = model.train_on_batch(batch_X, batch_y, reset_metrics=False, return_dict=True)
                callbacks.on_train_batch_end(i // batch_size, logs)

            validation_logs = model.evaluate(dev_X, dev_y, callbacks=callbacks, return_dict=True)
            logs.update({'val_' + name: v for name, v in validation_logs.items()})
            callbacks.on_epoch_end(epoch, logs)
            model.save('trained_model/model_E' + str(epoch) + '.h5')

    # change list of tuples to list of list
    text = np.array([x[0] for x in train_X])
    acoustic = np.array([x[1] for x in train_X])
    visual = np.array([x[2] for x in train_X])

    text_dev = np.array([x[0] for x in dev_X])
    acoustic_dev = np.array([x[1] for x in dev_X])
    visual_dev = np.array([x[2] for x in dev_X])

    train_model([text, acoustic, visual], np.array(train_y), [text_dev, acoustic_dev, visual_dev], np.array(dev_y))


def to_one_hot(y):
    result = []
    for a in y:
        t = np.zeros((6,))
        for max_index in np.where(a == np.max(a))[0]:
            if a[max_index] == 0:
                break
            t[max_index] = 1
        result.append(t)
    return result


def train_emotion_model():
    with open('pre-processed-data/sentiment/avg_train_X.data', 'rb') as w:
        train_X = pickle.load(w)
    with open('pre-processed-data/sentiment/avg_dev_X.data', 'rb') as w:
        dev_X = pickle.load(w)
    with open('pre-processed-data/emotion/train_y.data', 'rb') as w:
        train_y = pickle.load(w)
    with open('pre-processed-data/emotion/dev_y.data', 'rb') as w:
        dev_y = pickle.load(w)

    train_y = to_one_hot(train_y)
    dev_y = to_one_hot(dev_y)

    def multimodal_emotion_model(text_shape, acoustic_shape, visual_shape):
        text_input = Input(shape=text_shape, name='text_input')
        text_lstm = LSTM(256, return_sequences=True)(text_input)

        acoustic_input = Input(shape=acoustic_shape, name='acoustic_input')
        acoustic_lstm = LSTM(256, return_sequences=True)(acoustic_input)

        visual_input = Input(shape=visual_shape, name='visual_input')
        visual_lstm = LSTM(256, return_sequences=True)(visual_input)

        attention_text_acoustic = Dot(axes=[2, 2])([text_lstm, acoustic_lstm])
        attention_text_visual = Dot(axes=[2, 2])([text_lstm, visual_lstm])
        attention_weights = Concatenate(axis=2)([attention_text_acoustic, attention_text_visual])
        attention_weights = Activation('softmax')(attention_weights)

        attended_text = Dot(axes=[1, 1])([attention_weights, text_lstm])
        attended_acoustic = Dot(axes=[1, 1])([attention_weights, acoustic_lstm])
        attended_visual = Dot(axes=[1, 1])([attention_weights, visual_lstm])

        multimodal_features = Concatenate()([attended_text, attended_acoustic, attended_visual])

        dense1 = Dense(256, activation='relu')(text_lstm)
        dense2 = Dense(128, activation='relu')(dense1)
        output = Dense(6, activation='tanh')(dense2)
        normalized_output = tf.keras.layers.Lambda(lambda x: x[:, 1:2, :])(output)
        model = Model(inputs=[text_input, acoustic_input, visual_input], outputs=output)
        return model

    # Train the model
    def train_model(train_X, train_y, dev_X, dev_y, batch_size=128, epochs=10):
        text_shape = (train_X[0][0].shape[0], train_X[0][0].shape[1],)
        acoustic_shape = (train_X[0][0].shape[0], train_X[1][0].shape[1],)
        visual_shape = (train_X[0][0].shape[0], train_X[2][0].shape[1],)
        model = multimodal_emotion_model(text_shape, acoustic_shape, visual_shape)
        model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=False)

        callbacks = tf.keras.callbacks.CallbackList(
            None,
            add_history=True,
            add_progbar=True,
            model=model,
            epochs=epochs,
            verbose=1,
            steps=len(train_X[0]) // batch_size
        )
        callbacks.on_train_begin()
        for epoch in range(epochs):
            model.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            print('Epoch', epoch + 1)
            for i in range(0, len(train_X[0]), batch_size):
                callbacks.on_train_batch_begin(i // batch_size - 1)
                offset = batch_size
                if i + batch_size >= len(train_X[0]):
                    offset = len(train_X[0]) - i
                batch_X = [train_X[0][i:i + offset], train_X[1][i:i + offset], train_X[2][i:i + offset]]
                batch_y = train_y[i:i + offset]
                logs = model.train_on_batch(batch_X, batch_y, reset_metrics=False, return_dict=True)
                callbacks.on_train_batch_end(i // batch_size, logs)

            validation_logs = model.evaluate(dev_X, dev_y, callbacks=callbacks, return_dict=True)
            logs.update({'val_' + name: v for name, v in validation_logs.items()})
            callbacks.on_epoch_end(epoch, logs)
            model.save('trained_model/model_emotion_E' + str(epoch) + '.h5')

    # change list of tuples to list of list
    text = np.array([x[0] for x in train_X])
    acoustic = np.array([x[1] for x in train_X])
    visual = np.array([x[2] for x in train_X])

    text_dev = np.array([x[0] for x in dev_X])
    acoustic_dev = np.array([x[1] for x in dev_X])
    visual_dev = np.array([x[2] for x in dev_X])

    train_model([text, acoustic, visual], np.array(train_y), [text_dev, acoustic_dev, visual_dev], np.array(dev_y))


def coeff_determination(y_true, y_pred):
    SS_res = np.sum(K.square(y_true - y_pred))
    SS_tot = np.sum(K.square(y_true - np.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def get_accuracy_2(y_real, y_predict):
    true_pos = 0
    for i in range(len(y_real)):
        if y_real[i] < 0.5 and y_predict[i] < 0.5:
            true_pos += 1
        elif y_real[i] > 0.5 and y_predict[i] > 0.5:
            true_pos += 1
    return true_pos / len(y_real)


def get_range_1_to_7(num):
    return round(float(num) / (1 / 7.0))


def get_accuracy_7(y_real, y_predict):
    true_pos = 0
    for i in range(len(y_real)):
        if get_range_1_to_7(y_real[i]) == get_range_1_to_7(y_predict[i]):
            true_pos += 1
    return true_pos / len(y_real)


def get_mae(y_real, y_predict):
    mae = tf.keras.losses.MeanAbsoluteError()
    return mae(y_real * 6 - 3, y_predict * 6 - 3).numpy()


def get_classification_lists(y_real, y_predict):
    true_list = []
    predict_list = []
    for i in range(len(y_real)):
        if y_real[i] < 0.5:
            true_list.append(0)
        if y_predict[i] < 0.5:
            predict_list.append(0)
        if y_real[i] >= 0.5:
            true_list.append(1)
        if y_predict[i] >= 0.5:
            predict_list.append(1)
    return true_list, predict_list


def get_f1(y_real, y_predict):
    true_list, predict_list = get_classification_lists(y_real, y_predict)
    return f1_score(true_list, predict_list, average='weighted')


def evaluation(model_path, test_X, test_y):
    text_dev = np.array([x[0] for x in test_X])
    acoustic_dev = np.array([x[1] for x in test_X])
    visual_dev = np.array([x[2] for x in test_X])
    model = load_model(model_path)
    prediction = model.predict([text_dev, acoustic_dev, visual_dev])
    pred_y = []
    for i in range(len(test_y)):
        pred_y.append(prediction[i][0][0])
    pred_y = np.asarray(pred_y)
    test_y = np.asarray(test_y)
    print("Attention Model F1: " + str(get_f1(test_y, pred_y)))
    print("Attention Model MAE: " + str(get_mae(test_y, pred_y)))
    print("Attention Model accuracy (binary): " + str(get_accuracy_2(test_y, pred_y)))
    print(
        "Attention Model accuracy (7 class classification): " + str(get_accuracy_7(test_y, pred_y)))
    print("Attention Model R^2: " + str(coeff_determination(test_y, pred_y)))


def acc_emo(real, predict):
    _, predict_list = get_emo_classification_lists(real, predict)
    return np.sum(predict_list) / len(real)


def get_emo_classification_lists(real, predict):
    true_list = []
    predict_list = []
    prediction_index = np.argmax(predict, axis=1)
    for i in range(len(real)):
        max_v = np.max(real[i])
        real_indices = np.where(real[i] == max_v)[0]
        if prediction_index[i] in real_indices:
            true_list.append(1)
            predict_list.append(1)
        else:
            true_list.append(1)
            predict_list.append(0)
    return true_list, predict_list


def evaluation_emotion(model_path, test_X, test_y):
    text_dev = np.array([x[0] for x in test_X])
    acoustic_dev = np.array([x[1] for x in test_X])
    visual_dev = np.array([x[2] for x in test_X])
    model = load_model(model_path)
    prediction = model.predict([text_dev, acoustic_dev, visual_dev])
    pred_y = []
    for i in range(len(test_y)):
        pred_y.append(prediction[i][0][:])
    pred_y = np.asarray(pred_y) * 5
    test_y = [np.argmax(x) for x in test_y]
    # tp = 0
    # for i in range(len(pred_y)):
    #     if int(test_y[i]) == int(pred_y[i]):
    #         tp += 1
    # print(tp / len(pred_y))
    print("Attention Model Emotion Accuracy: " + str(acc_emo(test_y, pred_y)))


def evaluate_SVR():
    with open('pre-processed-data/sentiment/avg_test_X.data', 'rb') as w:
        test_X = pickle.load(w)
    with open('pre-processed-data/sentiment/test_y.data', 'rb') as w:
        test_y = pickle.load(w)

    text = np.array([x[0].reshape((300,)) for x in test_X])
    with open('trained_model/SVR.pkl', 'rb') as f:
        model = pickle.load(f)
        pred_y = model.predict(text)
        pred_y = np.asarray(pred_y)
        test_y = np.asarray(test_y)
        print("SVR F1: " + str(get_f1(test_y, pred_y)))
        print("SVR MAE: " + str(get_mae(test_y, pred_y)))
        print("SVR accuracy (binary): " + str(get_accuracy_2(test_y, pred_y)))
        print(
            "SVR accuracy (7 class classification): " + str(get_accuracy_7(test_y, pred_y)))
        print("SVR R^2: " + str(coeff_determination(test_y, pred_y)))


def evaluate_SVR_emotion():
    with open('pre-processed-data/sentiment/avg_test_X.data', 'rb') as w:
        test_X = pickle.load(w)
    with open('pre-processed-data/emotion/test_y.data', 'rb') as w:
        test_y = pickle.load(w)

    text = np.array([x[0].reshape((300,)) for x in test_X])
    with open('trained_model/SVR_emotion.pkl', 'rb') as f:
        model = pickle.load(f)
        pred_y = model.predict(text)
        pred_y = np.floor(np.asarray(pred_y) * 5)
        test_y = np.asarray(test_y)
        tp = 0
        zero_cnt = 0
        for i in range(len(test_y)):
            max_v = np.max(test_y[i])
            real_indices = np.where(test_y[i] == max_v)[0]
            if pred_y[i] in real_indices:
                tp += 1
            if 0 in real_indices:
                zero_cnt += 1
        print('Baseline: ' + str(zero_cnt / len(pred_y)))
        print('SVR ACC: ' + str(tp / len(pred_y)))


def evaluate_multimodal_sentiment():
    with open('pre-processed-data/sentiment/avg_test_X.data', 'rb') as w:
        test_X = pickle.load(w)
    with open('pre-processed-data/sentiment/test_y.data', 'rb') as w:
        test_y = pickle.load(w)
    for i in range(19, 20):
        print('Epoch ' + str(i))
        evaluation('trained_model/model_E' + str(i) + '.h5', test_X, test_y)


def evaluate_multimodal_emotion():
    with open('pre-processed-data/sentiment/avg_test_X.data', 'rb') as w:
        test_X = pickle.load(w)
    with open('pre-processed-data/emotion/test_y.data', 'rb') as w:
        test_y = pickle.load(w)
    for i in range(0, 7):
        print('Epoch ' + str(i))
        evaluation_emotion('trained_model/model_emotion_E' + str(i) + '.h5', test_X, test_y)


# save_emotion_labels()
# save_ave_data()
# save_sentiment_labels()
# save_sentiment_labels()
# shrink_emotion_data()

# train_sentiment_model()
# train_emotion_model()

# train_SVR()
# train_SVR_emotion()

evaluate_SVR_emotion()
# evaluate_SVR()
# evaluate_multimodal_sentiment()
# evaluate_multimodal_emotion()