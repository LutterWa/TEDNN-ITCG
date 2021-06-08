import os
import sys
import scipy.io
import numpy as np
import tensorflow.keras as keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_data(path):
    m = scipy.io.loadmat(path)
    data_raw = np.array(m['data_raw'])
    x_raw = data_raw[:, :-1]
    y_raw = data_raw[:, -1:]
    print('data load DONE! Train data size=', len(data_raw))
    return x_raw, y_raw


def init_network(learn_rate=0.001, num=0):
    keras_model = keras.Sequential()
    keras_model.add(keras.layers.Dense(units=100, activation="relu", name="dense1_model{}".format(num),
                                       input_shape=[4]))
    keras_model.add(keras.layers.Dense(units=100, activation="relu", name="dense2_model{}".format(num)))
    keras_model.add(keras.layers.Dense(units=100, activation="relu", name="dense3_model{}".format(num)))
    keras_model.add(keras.layers.Dense(units=1, name="dense4_model{}".format(num)))
    keras_model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learn_rate))

    return keras_model


def train_network(path, num, trans_set_size=None):
    x_raw, y_raw = load_data(path)
    if trans_set_size is not None:
        x_raw = x_raw[0:trans_set_size, :]
        y_raw = y_raw[0:trans_set_size, :]

    model = init_network(num=num)
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
    model.fit(x_raw, y_raw, batch_size=50, epochs=30, validation_split=0.02, callbacks=[early_stopping],
              use_multiprocessing=True)
    model.save("./flight_model{}.h5".format(num))


def test_network(path, num):
    # 加载数据
    x_raw, y_raw = load_data(path)
    x_val = x_raw[-int(len(x_raw) * 0.02):, :]
    y_val = y_raw[-int(len(y_raw) * 0.02):, :]

    model = keras.models.load_model("./flight_model{}.h5".format(num))  # 加载模型

    scipy.io.savemat("dnn_test{}.mat".format(num), {'x': x_val, 'y_hat': model.predict(x_val), 'y': y_val})


def transfer_ensemble(path, trans_set_size, learn_rate, batch_size, epochs):
    """
    迁移-集成学习
    :param path: 迁移数据集路径
    :param trans_set_size: 迁移数据集大小
    :param learn_rate: 迁移学习率
    :param batch_size: 迁移样本尺寸
    :param epochs: 迁移样本使用次数
    :return:
    """
    model_pre = {"input": [], "output": []}  # 预训练模型
    for _ in range(5):
        model = keras.models.load_model("./flight_model{}.h5".format(_))  # 加载各模型
        for layer in model.layers[:-1]:  # 冻结参数
            layer.trainable = False
        model_pre['input'].append(model.input)
        model_pre['output'].append(model.output)

    model = keras.layers.concatenate(model_pre['output'])
    output = keras.layers.Dense(units=1,
                                activation="relu",
                                use_bias=False,
                                kernel_initializer=keras.initializers.Constant(value=1 / 5))(model)
    model_en = keras.Model(inputs=model_pre['input'], outputs=output)

    model_en.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learn_rate))

    x_raw, y_raw = load_data(path)
    x_trans = x_raw[0:trans_set_size, :]
    y_trans = y_raw[0:trans_set_size, :]
    print("迁移学习样本数={}".format(len(x_trans)))

    model_en.fit(x=[x_trans, x_trans, x_trans, x_trans, x_trans],
                 y=y_trans,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2)
    model_en.save('./flight_TrEN.h5')

    return model_en


def transfer_greedy(path, trans_set_size, learn_rate, batch_size, epochs):
    """
    迁移-贪心学习
    :param path: 迁移数据集路径
    :param trans_set_size: 迁移数据集大小
    :param learn_rate: 迁移学习率
    :param batch_size: 迁移样本尺寸
    :param epochs: 迁移样本使用次数
    :return:
    """
    x_raw, y_raw = load_data(path)
    x_trans = x_raw[0:trans_set_size, :]
    y_trans = y_raw[0:trans_set_size, :]
    print("迁移学习样本数={}".format(len(x_trans)))

    def MSE(y, t):
        return np.mean((y - t) ** 2)

    model_pre = None
    mse_min = 100
    for _ in range(5):
        model = keras.models.load_model("./flight_model{}.h5".format(_))  # 加载各模型
        y_hat = model.predict(x_trans)
        mse = MSE(y_trans, y_hat)
        if mse < mse_min:
            mse_min = mse
            model_pre = model

    for layer in model_pre.layers[:-1]:  # 冻结参数
        layer.trainable = False

    model_pre.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learn_rate))

    model_pre.fit(x=[x_trans, x_trans, x_trans, x_trans, x_trans],
                  y=y_trans,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2)
    model_pre.save('./flight_Tr.h5')

    return model_pre


if __name__ == '__main__':
    pre_train = False
    test_TrEN = True

    if pre_train is True:
        if test_TrEN is True:
            p = "flight_data_concat_TrEN.mat"
            num = "TrEN"
            # train_network(path=p, num=num, trans_set_size=4651)
            x_test, y_test = load_data(path=p)
            x_test = x_test[100000:200000, :]
            y_test = y_test[100000:200000, :]

            model = keras.models.load_model("./flight_model{}.h5".format(num))  # 加载模型
            scipy.io.savemat("dnn_test{}.mat".format(num), {'x': x_test, 'y_hat': model.predict(x_test), 'y': y_test})
        else:
            i = int(sys.argv[1])
            train_network(path="flight_data_concat_{}.mat".format(i + 1), num=i)
    else:
        p = "flight_data_concat_TrEN.mat"
        # transfer_ensemble(path=p, trans_set_size=4651, learn_rate=0.0001, batch_size=1, epochs=10)
        x_test, y_test = load_data(path=p)
        x_test = x_test[100000:200000, :]
        y_test = y_test[100000:200000, :]
        y_en = []
        for i in range(20):
            m_en = transfer_greedy(path=p, trans_set_size=4651, learn_rate=0.0001, batch_size=1, epochs=i + 1)
            y_en.append(m_en.predict(x_test))

        y_hat = []
        # for i in range(5):
        #     m_pre = keras.models.load_model('flight_model{}.h5'.format(i, i))
        #     y_hat.append(m_pre.predict(x_test))
        # y_hat = np.squeeze(y_hat, axis=2).T

        dict_En = {'x': x_test, 'y_hat': y_hat, 'y': y_test, 'y_en': np.squeeze(y_en).T}
        dataEn = r'Tr_test.mat'
        scipy.io.savemat(dataEn, dict_En)
