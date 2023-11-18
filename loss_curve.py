import json
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt


def plot_loss_curve(
    results_dir: Union[str, Path], key: str = "mae", plot_train: bool = False
):
    """plot loss curves """
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)
    with open(results_dir / "history_val.json", "r") as f:
        val = json.load(f)

    p = plt.plot(val[key], label="Validation", color="red", linewidth=2)
    if plot_train:
        # plot the training trace in the same color
        with open(results_dir / "history_train.json", "r") as f:
            train = json.load(f)

        c = p[0].get_color()
        plt.plot(train[key],  label="Train", c="green", linewidth=2)
        # plt.legend("training")

    plt.xlabel("epochs")
    plt.ylabel(key)

    min_val_index = val[key].index(min(val[key]))
    min_val = min(val[key])
    print(min_val_index, min_val)
    plt.axvline(x=min_val_index, c="black", linestyle=':', label="Lowest validation loss", linewidth=2)
    plt.annotate(f"{min_val_index}, {min_val:.4f}", (min_val_index, min_val), textcoords="offset points", xytext=(0, 10))
    plt.legend(loc="upper right")
    plt.savefig("Loss.eps", format='eps')
    plt.show()
    plt.close()

    return train, val
# plot_loss_curve("temp_bulk",plot_train=True)
