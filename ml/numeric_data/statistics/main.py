import pandas as pd

def main():
    pd.options.display.max_rows = 10
    pd.options.display.float_format = "{:.1f}".format
    pd.set_option('display.max_columns', None)

    training_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

    print(training_df.describe())

    print("""
    以下欄位可能包含離群值：

      * total_rooms（總房間數）
      * total_bedrooms（總臥室數）
      * population（人口數）
      * households（家庭數）
      * 可能還包括 median_income（中位數收入）
    
    這些欄位的數據特性如下：
    
      * 標準差幾乎與平均值相當，表示數據變異性較大
      * 75% 分位數與最大值之間的差距，遠大於最小值與 25% 分位數之間的差距，顯示資料分布可能有長尾現象""")

if __name__ == "__main__":
    main()