from print_stats import DatasetStats
from visualization import create_pairplot
from model import execute_training_experiment
from prediction import *

def main():
    # 讀檔
    chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

    # 資料儲存到 Pandas DataFrame，做基本資料列印
    training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
    print('Read dataset completed successfully.')
    stats = DatasetStats(training_df)
    stats.print_rows_count()
    stats.print_head(10)
    stats.print_stat()
    stats.print_max_value('FARE')
    stats.print_mean_value('TRIP_MILES')
    stats.print_num_unique_values('COMPANY')
    stats.print_most_freq_value('PAYMENT_TYPE')
    stats.print_missing_data()
    stats.print_corr_with_column('FARE')

    # 欲分析的變數
    columns = ["FARE", "TRIP_MILES", "TRIP_SECONDS"]

    # 呼叫 create_pairplot，傳入 DataFrame 與變數名稱
    create_pairplot(training_df, columns)

    # 不同條件執行訓練實驗
    model_1 = execute_training_experiment(training_df, ['TRIP_MILES'], 'FARE')
    model_2 = execute_training_experiment(training_df, ['TRIP_MILES'], 'FARE', 1)
    model_3 = execute_training_experiment(training_df, ['TRIP_MILES'], 'FARE', 0.0001)
    model_4 = execute_training_experiment(training_df, ['TRIP_MILES'], 'FARE', 0.001, 20, 500)

    training_df_copy = training_df.copy()
    training_df_copy['TRIP_MINUTES'] = training_df_copy['TRIP_SECONDS'] / 60

    model_5 = execute_training_experiment(training_df_copy, ['TRIP_MILES', 'TRIP_MINUTES'], 'FARE')

    output = predict_fare(model_5, training_df_copy, ['TRIP_MILES', 'TRIP_MINUTES'], 'FARE')
    show_predictions(output)

if __name__ == "__main__":
    main()