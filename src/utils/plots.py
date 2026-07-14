import matplotlib.pyplot as plt

def plot_predictions(
    real,
    predicted,
    save_path
):
    plt.figure(figsize=(12,6))
    plt.plot(real, label="Real")
    plt.plot(predicted, label="Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()