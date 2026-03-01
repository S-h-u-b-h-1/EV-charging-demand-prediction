# Evaluation script
import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_test, preds):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, preds, alpha=0.3)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()
