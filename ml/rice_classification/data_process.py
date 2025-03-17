def normalize_dataset(rice_dataset):
    """
    標準化數值型資料集，將每個數值型特徵轉換

    參數:
        rice_dataset (pandas.DataFrame): 要標準化的資料集，應包含數值型的資料與類別標籤

    回傳:
        pandas.DataFrame: 標準化後的資料集，數值型特徵已標準化並保留原來的 'Class' 欄位
    """
    feature_mean = rice_dataset.mean(numeric_only=True)
    feature_std = rice_dataset.std(numeric_only=True)

    numerical_features = rice_dataset.select_dtypes('number').columns

    normalized_dataset = (rice_dataset[numerical_features] - feature_mean) / feature_std

    normalized_dataset['Class'] = rice_dataset['Class']

    return normalized_dataset


def add_class_bool_column(normalized_data, class_name='Cammeo'):
    """
    將 'Class' 欄位中的指定值轉換為布林值 True，其他值為 False，
    然後將布林值轉換為整數 (True -> 1, False -> 0)，並新增 'Class_Bool' 欄位。

    參數:
        normalized_data (pandas.DataFrame): 要處理的資料集，應包含 'Class' 欄位。
        class_name (str): 用來比對 'Class' 欄位的目標類別名稱，預設為 'Cammeo'。

    回傳:
        pandas.DataFrame: 新增了 'Class_Bool' 欄位的資料集。
    """
    # 將 'Class' 欄位中的指定類別轉換為 True，其他類別為 False
    normalized_data['Class_Bool'] = (
            normalized_data['Class'] == class_name
    ).astype(int)  # 將布林值轉換為整數

    # 顯示隨機抽取的 10 行資料，以檢查 'Class_Bool' 欄位是否已成功新增
    print(normalized_data.sample(10))

    return normalized_data


def split_dataset(normalized_dataset, train_ratio=0.8, validation_ratio=0.1, random_state=100):
    """
    將資料集按指定比例隨機分割為訓練集、驗證集和測試集。

    參數:
        normalized_dataset (pandas.DataFrame): 要分割的資料集。
        train_ratio (float): 訓練集所佔比例，預設為 0.8。
        validation_ratio (float): 驗證集所佔比例，預設為 0.1。
        random_state (int): 隨機種子，確保每次執行結果一致，預設為 100。

    回傳:
        tuple: 包含訓練集、驗證集和測試集的三個資料集。
    """

    # 計算資料集的長度
    number_samples = len(normalized_dataset)

    # 計算 80% 和 90% 百分位的索引
    index_80th = round(number_samples * train_ratio)
    index_90th = index_80th + round(number_samples * validation_ratio)

    # 隨機打亂資料集
    shuffled_dataset = normalized_dataset.sample(frac=1, random_state=random_state)

    # 分割資料集為訓練集、驗證集和測試集
    train_data = shuffled_dataset.iloc[0:index_80th]
    validation_data = shuffled_dataset.iloc[index_80th:index_90th]
    test_data = shuffled_dataset.iloc[index_90th:]

    return train_data, validation_data, test_data