import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_class_distribution(generator):
    labels = list(generator.class_indices.keys())
    count = np.zeros(len(labels))

    for _, y in generator:
        for row in y:
            count[np.argmax(row)] += 1
        if sum(count) >= generator.samples:
            break

    df = pd.DataFrame({
        'Classe': labels,
        'Numero': count.astype(int)
    })

    plt.figure(figsize=(16, 6))
    ax = sns.barplot(data=df, x='Classe', y='Numero')

    plt.title("Distribuzione delle Classi")
    plt.xlabel("Classi")
    plt.ylabel("Numero di campioni")

    ax.tick_params(axis='x', which='major', pad=15)

    xticklabels = ax.get_xticklabels()
    for i, label in enumerate(xticklabels):
        label.set_fontsize(8)  # riduci la dimensione del font
        if i % 2 == 0:
            label.set_y(-0.05)  # sposta un po' sopra l'asse
        else:
            label.set_y(-0.40)  # sposta molto più sotto (più distanziato)

    ax.set_xticklabels(xticklabels, rotation=30, ha='right')  # ruota le etichette per migliorare la leggibilità

    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_labels, normalize=False, show_cbar=True):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    num_classes = len(class_labels)
    fig_width = max(12, num_classes * 0.6)
    fig_height = max(8, num_classes * 0.5)

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=show_cbar,
        linewidths=0.4,
        linecolor='gray',
        annot_kws={"size": 7}
    )

    plt.title("Confusion Matrix", fontsize=18)
    plt.ylabel("True label", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)

    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_labels, rotation=0, va="center", fontsize=9)

    plt.tight_layout()
    plt.show()