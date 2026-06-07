import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(df, target):
    plt.figure(figsize=(6,4))

    sns.countplot(x=target, data=df)

    plt.title(f"{target} Distribution")
    plt.show()


def plot_histogram(df, column):
    plt.figure(figsize=(8,4))

    sns.histplot(df[column], kde=True)

    plt.title(column)
    plt.show()