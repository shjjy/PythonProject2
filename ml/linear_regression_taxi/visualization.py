import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_correlation(x, y):
    """
    計算兩個變數的相關性係數
    """
    return x.corr(y)

def create_pairplot(df, columns):
    """
    繪製 pairplot 並在子圖上顯示相關性係數

    參數:
    df: pandas DataFrame, 資料來源
    columns: list of str, 欲繪製的變數名稱
    """
    # 畫出 pairplot
    pairplot = sns.pairplot(df,
                            x_vars=columns,
                            y_vars=columns,
                            kind="reg",
                            plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'color': 'blue'}})

    # 在每個子圖上顯示相關性係數
    for i in range(len(columns)):
        for j in range(len(columns)):
            if i != j:  # 排除對角線圖
                x_column = columns[j]
                y_column = columns[i]

                # 計算相關性係數
                corr = calculate_correlation(df[x_column], df[y_column])

                # 在子圖上註解顯示相關性係數
                pairplot.axes[i, j].annotate(f'Corr: {corr:.2f}',
                                             (0.5, 0.9),  # 設定註解顯示位置
                                             xycoords='axes fraction',  # 使用相對座標系統
                                             ha='center',
                                             fontsize=10,  # 設定字體大小
                                             color='white',  # 改成白色字體
                                             weight='bold',  # 設定字體粗細
                                             bbox=dict(facecolor='black', edgecolor='none',
                                                       boxstyle='round,pad=0.5'))  # 加上黑色背景框

    plt.show()


def make_plots(df, feature_names, label_name, model_output, learning_rate, epochs_count, batch_size, sample_size=200):
    """
    繪製訓練資料和模型的可視化，包括損失曲線和模型預測，並顯示超參數。

    Args:
        df (pd.DataFrame): 包含數據的 DataFrame。
        feature_names (list of str): 特徵名稱的列表。
        label_name (str): 標籤名稱。
        model_output (tuple): 模型輸出的元組 (weights, bias, epochs, rmse)。
        learning_rate (float): 模型的學習率。
        epochs_count (int): 訓練的迭代次數。
        batch_size (int): 每次訓練步驟中處理的數據樣本數量。
        sample_size (int, optional): 隨機取樣的數量。預設為 200。

    Returns:
        None
    """
    random_sample = df.sample(n=sample_size).copy()
    random_sample.reset_index(drop=True, inplace=True)
    weights, bias, epochs, rmse = model_output

    is_2d_plot = len(feature_names) == 1
    model_plot_type = "scatter" if is_2d_plot else "surface"
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Loss Curve", "Model Plot"),
                        specs=[[{"type": "scatter"}, {"type": model_plot_type}]])

    _plot_data(random_sample, feature_names, label_name, fig)
    _plot_model(random_sample, feature_names, weights, bias, fig)
    _plot_loss_curve(epochs, rmse, fig)

    fig.add_annotation(
        x=0.5, y=1.05,
        xref="paper", yref="paper",
        text=f"Learning Rate: {learning_rate}, Epochs: {epochs_count}, Batch Size: {batch_size}",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="center"
    )

    fig.add_annotation(
        x=0.95, y=-0.11,
        xref="paper", yref="paper",
        text="Red line : represents the model's predicted result.",
        showarrow=False,
        font=dict(size=10, color="red"),
        align="left"
    )

    fig.add_annotation(
        x=0.95, y=-0.13,  # Adjusting the second annotation below the previous one
        xref="paper", yref="paper",
        text="Blue dots : Actual data points from the training dataset.",
        showarrow=False,
        font=dict(size=10, color="blue"),
        align="left"
    )

    fig.show()

def _plot_loss_curve(epochs, rmse, fig):
    """
    繪製訓練過程的損失曲線。

    Args:
        epochs (list of int): 訓練過程中的 epoch 數值。
        rmse (list of float): 每個 epoch 的均方根誤差 (RMSE)。
        fig (plotly.graph_objects.Figure): 用於更新的 Plotly 圖表對象。

    Returns:
        None
    """
    curve = px.line(x=epochs, y=rmse)
    curve.update_traces(line_color='#ff0000', line_width=3)

    fig.append_trace(curve.data[0], row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min() * 0.8, rmse.max()])

def _plot_data(df, features, label, fig):
    """
    繪製原始資料的散點圖或三維散點圖。

    Args:
        df (pd.DataFrame): 包含數據的 DataFrame。
        features (list of str): 特徵名稱的列表。
        label (str): 標籤名稱。
        fig (plotly.graph_objects.Figure): 用於更新的 Plotly 圖表對象。

    Returns:
        None
    """
    if len(features) == 1:
        scatter = px.scatter(df, x=features[0], y=label)
    else:
        scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)

    fig.append_trace(scatter.data[0], row=1, col=2)
    if len(features) == 1:
        fig.update_xaxes(title_text=features[0], row=1, col=2)
        fig.update_yaxes(title_text=label, row=1, col=2)
    else:
        fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))


def _plot_model(df, features, weights, bias, fig):
    """
    繪製模型預測的線條或平面。

    Args:
        df (pd.DataFrame): 包含數據的 DataFrame。
        features (list of str): 特徵名稱的列表。
        weights (list of list): 模型的權重。
        bias (list): 模型的偏置。
        fig (plotly.graph_objects.Figure): 用於更新的 Plotly 圖表對象。

    Returns:
        None
    """
    df['FARE_PREDICTED'] = bias[0]

    for index, feature in enumerate(features):
        df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

    if len(features) == 1:
        model = px.line(df, x=features[0], y='FARE_PREDICTED')
        model.update_traces(line_color='#ff0000', line_width=3)
    else:
        z_name, y_name = "FARE_PREDICTED", features[1]
        z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
        y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
        x = []
        for i in range(len(y)):
            x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

        plane = pd.DataFrame({'x': x, 'y': y, 'z': [z] * 3})

        light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
        model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
                                          colorscale=light_yellow))

    fig.add_trace(model.data[0], row=1, col=2)