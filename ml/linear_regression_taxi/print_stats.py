class DatasetStats:
    def __init__(self, training_df):
        self.training_df = training_df

    def print_rows_count(self):
        print('Total number of rows: {0}\n\n'.format(len(self.training_df.index)))

    def print_head(self, num_rows):
        print(f"First {num_rows} rows of the dataset:")
        print(self.training_df.head(num_rows))

    def print_stat(self):
        print(self.training_df.describe(include='all'))

    def print_max_value(self, column_name):
        max_value = self.training_df[column_name].max()
        print(f"What is the maximum value of {column_name}? \tAnswer: ${max_value:.2f}")

    def print_mean_value(self, column_name):
        mean_value = self.training_df[column_name].mean()
        print(f"What is the mean value of {column_name}? \tAnswer: {mean_value:.4f}")

    def print_num_unique_values(self, column_name):
        num_unique_values = self.training_df[column_name].nunique()
        print(f"How many unique values are in {column_name}? \t\tAnswer: {num_unique_values}")

    def print_most_freq_value(self, column_name):
        most_freq_value = self.training_df[column_name].value_counts().idxmax()
        print(f"What is the most frequent value in {column_name}? \tAnswer: {most_freq_value}")

    def print_missing_data(self):
        missing_values = self.training_df.isnull().sum().sum()
        print("Are any features missing data? \t\t\t\tAnswer:", "No" if missing_values == 0 else "Yes")

    # 計算與指定欄位的相關性並找出最強與最弱的相關特徵
    def print_corr_with_column(self, column_name):
        # 計算相關性矩陣
        corr_matrix = self.training_df.corr(numeric_only=True)

        # 找出與指定欄位的相關性
        correlations_with_column = corr_matrix[column_name]

        # 去除自己與自己之間的相關性
        correlations_with_column = correlations_with_column.drop(column_name)

        # 找到最強和最弱的相關性
        max_corr_feature = correlations_with_column.idxmax()
        min_corr_feature = correlations_with_column.idxmin()

        # 顯示結果
        answer = (
            "\n"
            f"與 {column_name} \n"
            f"相關性係數最高的是：{max_corr_feature}\n"
            f"相關性係數最低的是：{min_corr_feature}"
        )

        print(answer)