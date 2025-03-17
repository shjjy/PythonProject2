import keras

from ml.rice_classification.model import *
from print_stats import DatasetStats
from visualization import *
from data_process import *
from evaluation import *

def main():
    pd.options.display.max_rows = 10  # 顯示最多 10 行
    pd.options.display.float_format = "{:.1f}".format  # 浮點數只顯示一位小數

    # 讀檔
    print("讀取稻米數據集...")
    rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

    rice_dataset = rice_dataset_raw[[
        'Area',
        'Perimeter',
        'Major_Axis_Length',
        'Minor_Axis_Length',
        'Eccentricity',
        'Convex_Area',
        'Extent',
        'Class',
    ]]

    # 顯示數據集的統計信息
    print("顯示數據集統計信息...")
    stats = DatasetStats(rice_dataset)
    stats.print_stat()
    stats.print_length()
    stats.print_area()
    stats.print_stdev()

    # 繪製散點圖
    print("繪製 2D, 3D 散點圖...")
    create_scatter_plot(rice_dataset)
    create_scatter_plot_3d(rice_dataset)

    # 數據正規化
    print("數據正規化...")
    normalized_data = normalize_dataset(rice_dataset)
    print(normalized_data.head())

    # 設定隨機種子，確保每次運行時結果一致
    print("設定隨機種子...")
    keras.utils.set_random_seed(42)

    normalized_dataset = add_class_bool_column(normalized_data)

    # 拆分數據集
    print("拆分訓練集、驗證集和測試集...")
    train_data, validation_data, test_data = split_dataset(normalized_dataset)

    label_columns = ['Class', 'Class_Bool']

    train_features = train_data.drop(columns=label_columns)
    train_labels = train_data['Class_Bool'].to_numpy()
    validation_features = validation_data.drop(columns=label_columns)
    validation_labels = validation_data['Class_Bool'].to_numpy()
    test_features = test_data.drop(columns=label_columns)
    test_labels = test_data['Class_Bool'].to_numpy()

    input_features = [
        'Eccentricity',
        'Major_Axis_Length',
        'Area',
    ]

    settings = ExperimentSettings(
        learning_rate=0.001,
        number_epochs=60,
        batch_size=100,
        classification_threshold=0.35,
        input_features=input_features,
    )

    metrics = [
        keras.metrics.BinaryAccuracy(
            name='accuracy', threshold=settings.classification_threshold
        ),
        keras.metrics.Precision(
            name='precision', thresholds=settings.classification_threshold
        ),
        keras.metrics.Recall(
            name='recall', thresholds=settings.classification_threshold
        ),
        keras.metrics.AUC(num_thresholds=100, name='auc'),
    ]

    # 建立模型的拓撲結構（Topography）
    print("建立模型...")
    model = create_model(settings, metrics)

    # 在訓練集上訓練模型
    print("訓練模型...")
    experiment = train_model(
        'baseline', model, train_features, train_labels, settings
    )

    # 繪製指標（metrics）與訓練輪數（epochs）之間的變化圖
    print("繪製訓練指標變化圖...")
    plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
    plot_experiment_metrics(experiment, ['auc'])

    # 評估測試數據
    print("評估測試數據...")
    test_metrics = evaluate_experiment(experiment, test_features, test_labels)

    # 比較訓練集與測試集的表現
    print("比較訓練集與測試集的表現...")
    compare_train_test(experiment, test_metrics)

    all_input_features = [
      'Eccentricity',
      'Major_Axis_Length',
      'Minor_Axis_Length',
      'Area',
      'Convex_Area',
      'Perimeter',
      'Extent',
    ]

    # 設定訓練參數
    settings_all_features = ExperimentSettings(
        learning_rate=0.001,
        number_epochs=60,
        batch_size=100,
        classification_threshold=0.5,
        input_features=all_input_features,
    )

    # 執行訓練
    print("使用所有特徵進行訓練...")
    experiment_all_features = train_experiment(
        "all features", train_features, train_labels, settings_all_features
    )

    test_metrics_all_features = evaluate_experiment(
        experiment_all_features, test_features, test_labels
    )

    # 比較訓練集與測試集的表現
    print("比較所有特徵的訓練集與測試集的表現...")
    compare_train_test(experiment_all_features, test_metrics_all_features)

    # 比較兩個實驗
    print("比較兩個實驗的結果...")
    compare_experiment([experiment, experiment_all_features],
                       ['accuracy', 'auc'],
                       test_features, test_labels)

if __name__ == "__main__":
    main()