import pandas as pd
import numpy as np

def _format_currency(x):
  """
      將數字格式化，保留兩位小數。

      參數:
          x (float): 要格式化的數字。

      回傳:
          str: 格式化後的字串，例如 "123.45"。
      """
  return "${:.2f}".format(x)

def _build_batch(df, batch_size):
  """
     從資料集中隨機選取一個批次，並將其設置索引。

     參數:
         df (pandas.DataFrame): 要從中抽取資料的資料集。
         batch_size (int): 批次的大小。

     回傳:
         pandas.DataFrame: 隨機選取的批次資料。
     """
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  """
      使用模型進行批次預測，並回傳包含預測結果、實際值及損失的資料框。

      參數:
          model (object): 用來預測的模型，必須有 predict_on_batch 方法
          df (pandas.DataFrame): 資料集，用來選取特徵和標籤
          features (list): 特徵的欄位名稱列表
          label (str): 標籤欄位名稱（即實際的 fare）
          batch_size (int, 可選): 每次預測的批次大小，預設為 50

      回傳:
          pandas.DataFrame: 包含預測結果、實際結果及 L1 損失的資料框
      """
  batch = _build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x=batch.loc[:, features].values)

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(_format_currency(predicted))
    data["OBSERVED_FARE"].append(_format_currency(observed))
    data["L1_LOSS"].append(_format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return