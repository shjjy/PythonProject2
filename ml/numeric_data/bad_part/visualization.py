from matplotlib import pyplot as plt

def plot_the_dataset(training_df, feature, label, number_of_points_to_plot):
    """Plot N random points of the dataset."""
    # Label the axes.
    plt.xlabel(feature)
    plt.ylabel(label)

    # Create a scatter plot from n random points of the dataset.
    random_examples = training_df.sample(n=number_of_points_to_plot)
    plt.scatter(random_examples[feature], random_examples[label])

    # Render the scatter plot.
    plt.show()

def plot_a_contiguous_portion_of_dataset(training_df, feature, label, start, end, day):
    """Plot the data points from start to end."""
    # Label the axes.
    plt.xlabel(feature + " Day " + str(day))
    plt.ylabel(label)

    # Create a scatter plot.
    plt.scatter(training_df[feature][start:end], training_df[label][start:end])

    # Render the scatter plot.
    plt.show()