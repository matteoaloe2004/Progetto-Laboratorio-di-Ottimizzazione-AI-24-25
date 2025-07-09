import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def plot_training(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

def evaluate_model(model, val_generator):
    val_generator.reset()
    y_pred = model.predict(val_generator, steps=len(val_generator), verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())

    print(classification_report(y_true, y_pred_classes, target_names=class_labels))
    cm = confusion_matrix(y_true, y_pred_classes)
    return cm, class_labels
