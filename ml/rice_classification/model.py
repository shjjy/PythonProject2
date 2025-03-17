import dataclasses
import numpy as np
import pandas as pd

from tensorflow import keras
from visualization import *

@dataclasses.dataclass()
class ExperimentSettings:
    """
    訓練模型所需的超參數與輸入特徵

    屬性:
        learning_rate (float): 學習率 (learning rate)
        number_epochs (int): 訓練的總回合數 (epochs)
        batch_size (int): 每次訓練批次的大小 (batch size)
        classification_threshold (float): 二元分類的閾值 (threshold)
        input_features (list[str]): 模型輸入的特徵名稱 (feature names)
    """

    learning_rate: float
    number_epochs: int
    batch_size: int
    classification_threshold: float
    input_features: list[str]


@dataclasses.dataclass()
class Experiment:
    """
    儲存訓練設定與結果，包括模型、歷史數據等資訊

    屬性:
        name (str): 實驗名稱 (experiment name)
        settings (ExperimentSettings): 本次訓練的設定 (hyperparameters)
        model (keras.Model): 訓練好的 Keras 模型
        epochs (np.ndarray): 記錄訓練過的回合數 (epochs)
        metrics_history (keras.callbacks.History): 記錄訓練過程的歷史指標 (history)
    """

    name: str
    settings: ExperimentSettings
    model: keras.Model
    epochs: np.ndarray
    metrics_history: pd.DataFrame  # 轉換為 DataFrame 方便分析

    def get_final_metric_value(self, metric_name: str) -> float:
        """
        取得指定指標 (metric) 在最後一個訓練回合的數值

        參數:
            metric_name (str): 指標名稱，例如 'loss'、'accuracy'

        回傳:
            float: 指標在最後一個回合的值

        錯誤處理:
            若指標名稱不存在於 `metrics_history`，則拋出錯誤
        """
        if metric_name not in self.metrics_history:
            raise ValueError(
                f'未知的指標 {metric_name}，可用指標: {list(self.metrics_history.columns)}'
            )
        return self.metrics_history[metric_name].iloc[-1]  # 取得最後一個 epoch 的數值


def create_model(
    settings: ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    """
    建立並編譯 (compile) 一個簡單的二元分類模型

    參數:
        settings (ExperimentSettings): 訓練設定 (包含輸入特徵)
        metrics (list[keras.metrics.Metric]): 監測的評估指標，例如 ['accuracy']

    回傳:
        keras.Model: 建立並編譯好的 Keras 模型
    """

    # 建立模型輸入層，每個特徵都作為一個獨立的 Input
    model_inputs = [
        keras.Input(name=feature, shape=(1,))  # 單一數值輸入，shape=(1,)
        for feature in settings.input_features
    ]

    # 使用 Concatenate 層將多個輸入合併成一個向量
    concatenated_inputs = keras.layers.Concatenate()(model_inputs)

    # 建立全連接層 (Dense layer)，激活函數使用 sigmoid
    dense = keras.layers.Dense(
        units=1, input_shape=(1,), name='dense_layer', activation=keras.activations.sigmoid
    )

    # 建立模型輸出層
    model_output = dense(concatenated_inputs)

    # 建立 Keras 模型
    model = keras.Model(inputs=model_inputs, outputs=model_output)

    # 編譯模型，使用 RMSprop 優化器與 Binary Crossentropy 作為損失函數
    model.compile(
        optimizer=keras.optimizers.RMSprop(settings.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ExperimentSettings,
) -> Experiment:
    """
    訓練模型 (feed dataset into the model)

    參數:
        experiment_name (str): 本次訓練的名稱
        model (keras.Model): 已建立的 Keras 模型
        dataset (pd.DataFrame): 訓練用的資料集 (features)
        labels (np.ndarray): 對應的標籤 (labels)，
        settings (ExperimentSettings): 訓練設定 (包含 batch_size, epochs)

    回傳:
        Experiment: 記錄訓練過程與結果的 Experiment 物件。
    """

    # 將 dataset 轉換為字典格式，每個 feature 對應一個 NumPy 陣列
    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    # 訓練模型 (fit)
    history = model.fit(
        x=features,  # 訓練特徵
        y=labels,  # 訓練標籤
        batch_size=settings.batch_size,  # 設定 batch 大小
        epochs=settings.number_epochs,  # 訓練回合數
    )

    # 回傳 Experiment 物件，儲存訓練過程的歷史記錄
    return Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=np.array(history.epoch),
        metrics_history=pd.DataFrame(history.history),
    )

def train_experiment(
    experiment_name: str,
    train_features: pd.DataFrame,
    train_labels: np.array,
    settings: ExperimentSettings,
) -> Experiment:
    """建立並訓練模型

    Args:
        experiment_name (str): 實驗名稱
        train_features (pd.DataFrame): 訓練特徵數據
        train_labels (np.array): 訓練標籤
        settings (ExperimentSettings): 訓練設定參數

    Returns:
        Experiment: 訓練完成的實驗物件
    """

    # 定義評估指標
    metrics = [
        keras.metrics.BinaryAccuracy(
            name="accuracy", threshold=settings.classification_threshold
        ),
        keras.metrics.Precision(
            name="precision", thresholds=settings.classification_threshold
        ),
        keras.metrics.Recall(
            name="recall", thresholds=settings.classification_threshold
        ),
        keras.metrics.AUC(num_thresholds=100, name="auc"),
    ]

    # 建立模型
    model = create_model(settings, metrics)

    # 訓練模型
    experiment = train_model(
        experiment_name, model, train_features, train_labels, settings
    )

    # 繪製訓練過程的指標變化
    plot_experiment_metrics(experiment, ["accuracy", "precision", "recall"])
    plot_experiment_metrics(experiment, ["auc"])

    return experiment