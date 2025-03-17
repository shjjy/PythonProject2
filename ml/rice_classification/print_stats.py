class DatasetStats:
    def __init__(self, rice_dataset):
        self.rice_dataset = rice_dataset

    def print_stat(self):
        print(self.rice_dataset.describe())

    def print_length(self):
        print(
            f'最短米粒長度 {self.rice_dataset.Major_Axis_Length.min():.1f}px, '
            f'最長米粒長度 {self.rice_dataset.Major_Axis_Length.max():.1f}px.'
        )
    def print_area(self):
        print(
            f'最小米粒的面積 {self.rice_dataset.Area.min()}px, '
            f'最大米粒的面積 {self.rice_dataset.Area.max()}px.'
        )
    def print_stdev(self):
        print(
            f'最大米粒周長 {self.rice_dataset.Perimeter.max():.1f} px, '
            f'與平均周長 {self.rice_dataset.Perimeter.mean():.1f} px 相比, '
            f'差了 {(self.rice_dataset.Perimeter.max() - self.rice_dataset.Perimeter.mean()) / self.rice_dataset.Perimeter.std():.1f} 倍標準差, '
            f'一倍標準差為 {self.rice_dataset.Perimeter.std():.1f} px, '
            f'計算過程：({self.rice_dataset.Perimeter.max():.1f} - {self.rice_dataset.Perimeter.mean():.1f}) / {self.rice_dataset.Perimeter.std():.1f} = '
            f'{(self.rice_dataset.Perimeter.max() - self.rice_dataset.Perimeter.mean()) / self.rice_dataset.Perimeter.std():.1f}'
        )