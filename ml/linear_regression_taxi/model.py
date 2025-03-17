import pandas as pd
import numpy as np
from tensorflow import keras
from visualization import make_plots

def _model_info(feature_names, label_name, model_output):
    """
    輸出模型資訊，包括特徵權重、偏置和模型方程式。

    Args:
        feature_names (list of str): 特徵名稱的列表。
        label_name (str): 標籤名稱。
        model_output (tuple): 模型輸出的元組 (weights, bias)。

    Returns:
        str: 格式化的模型資訊。
    """
    weights = model_output[0]
    bias = model_output[1]

    nl = "\n"
    header = "-" * 80
    banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header

    info = ""
    equation = label_name + " = "

    for index, feature in enumerate(feature_names):
        info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
        equation = equation + "{:.3f} * {} + ".format(weights[index][0], feature)

    info = info + "Bias: {:.3f}\n".format(bias[0])
    equation = equation + "{:.3f}\n".format(bias[0])

    return banner + nl + info + nl + equation

def _build_model(my_learning_rate, num_features):
    """
    建立並編譯一個簡單的線性迴歸模型。

    Args:
        my_learning_rate (float): 學習率，用於調整模型更新權重的速度。
        num_features (int): 特徵數量，即模型的輸入維度。

    Returns:
        keras.Model: 已編譯的 Keras 模型，包含一個 Dense 層和 MSE 損失函數。
    """
    inputs = keras.Input(shape=(num_features,))
    outputs = keras.layers.Dense(units=1)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[keras.metrics.RootMeanSquaredError()])

    return model


def _train_model(model, df, features, label, epochs, batch_size):
    """
    使用指定的資料訓練模型。

    Args:
        model (keras.Model): 線性迴歸模型。
        df (pd.DataFrame): 包含特徵和標籤的資料集。
        features (np.ndarray): 特徵矩陣（Numpy 格式）。
        label (np.ndarray): 標籤向量（Numpy 格式）。
        epochs (int): 訓練迭代次數。
        batch_size (int): 每次更新權重時使用的樣本數。

    Returns:
        tuple: 包含以下四項的元組：
            - trained_weight (np.ndarray): 訓練後的模型權重。
            - trained_bias (np.ndarray): 訓練後的模型偏置。
            - epochs (list): 每個訓練迭代的數字列表。
            - rmse (pd.Series): 每個 epoch 的均方根誤差 (RMSE)。
    """
    history = model.fit(x=features,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs)

    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    epochs = history.epoch

    hist = pd.DataFrame(history.history)

    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


def _run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):
    """
    執行完整的線性迴歸實驗，包括模型建立、訓練及結果視覺化。

    Args:
        df (pd.DataFrame): 包含特徵與標籤的資料集。
        feature_names (list): 特徵名稱列表。
        label_name (str): 標籤名稱。
        learning_rate (float): 模型的學習率。
        epochs (int): 訓練迭代次數。
        batch_size (int): 每次更新權重時使用的樣本數。

    Returns:
        keras.Model: 訓練後的 Keras 模型。

    Notes:
        此函式包含以下步驟：
        1. 使用指定的參數建立線性迴歸模型。
        2. 使用資料進行模型訓練。
        3. 輸出模型資訊（包括權重、偏置與方程式）。
        4. 視覺化模型的損失曲線與數據擬合效果。
    """
    print('INFO: starting training experiment with features={} and label={}\n'.format(feature_names, label_name))

    num_features = len(feature_names)

    features = df.loc[:, feature_names].values
    label = df[label_name].values

    model = _build_model(learning_rate, num_features)
    model_output = _train_model(model, df, features, label, epochs, batch_size)

    print('\nSUCCESS: training experiment complete\n')
    print('{}'.format(_model_info(feature_names, label_name, model_output)))
    make_plots(df, feature_names, label_name, model_output, learning_rate, epochs, batch_size)

    return model

def execute_training_experiment(training_df, features, label, learning_rate = 0.001, epochs = 20, batch_size = 50):
    """
    設定模型的超參數並執行線性回歸訓練實驗。

    Args:
        training_df (DataFrame): 訓練數據集，包含特徵列和標籤列。
        features (list): 特徵名稱列表，用於模型的輸入。
        label (str): 標籤名稱，模型需要預測的目標值。
        learning_rate (float, optional): 學習率，控制每次權重更新的步伐大小。默認為 0.001。
        epochs (int, optional): 訓練的迭代次數，即模型將看到完整數據集的次數。默認為 20。
        batch_size (int, optional): 每次訓練步驟中處理的數據樣本數量。默認為 50。

    Returns:
        keras.Model: 訓練完成的 Keras 模型。
    """
    # 執行訓練實驗
    print("INFO: 開始執行訓練實驗...")
    model = _run_experiment(training_df, features, label, learning_rate, epochs, batch_size)
    print("INFO: 訓練實驗完成！")
    return model