import plotly.express as px
from matplotlib import pyplot as plt
from model import Experiment

def create_scatter_plot(rice_dataset):
    for x_axis_data, y_axis_data in [
        ('Area', 'Eccentricity'),
        ('Convex_Area', 'Perimeter'),
        ('Major_Axis_Length', 'Minor_Axis_Length'),
        ('Perimeter', 'Extent'),
        ('Eccentricity', 'Major_Axis_Length'),
    ]:
        px.scatter(rice_dataset, x=x_axis_data, y=y_axis_data, color='Class').show()

def create_scatter_plot_3d(
        rice_dataset, x_axis_data='Eccentricity', y_axis_data='Area', z_axis_data='Major_Axis_Length', color='Class'):
    px.scatter_3d(
        rice_dataset,
        x=x_axis_data,
        y=y_axis_data,
        z=z_axis_data,
        color=color,
    ).show()

def plot_experiment_metrics(experiment: Experiment, metrics: list[str]):
  """Plot a curve of one or more metrics for different epochs."""
  plt.figure(figsize=(12, 8))

  for metric in metrics:
    plt.plot(
        experiment.epochs, experiment.metrics_history[metric], label=metric
    )

  plt.xlabel("Epoch")
  plt.ylabel("Metric value")
  plt.grid()
  plt.legend()
  plt.show()

