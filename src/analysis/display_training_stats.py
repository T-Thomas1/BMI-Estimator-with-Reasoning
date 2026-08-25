import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
from pydantic import BaseModel
import pandas as pd
from typing import Optional


class DisplayTrainingStats(BaseModel):
    state_file_path: str
    display: Optional[bool] = True
    title: Optional[str] = None
    df: Optional[pd.DataFrame] = None
    legend_fontsize: int = 30

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._prepare_data()

    def _prepare_data(self):
        # Initialize a list to hold all entries
        data = []

        # Define a regex pattern to remove np.float32 and np.float64
        pattern = re.compile(r"np\.float(?:32|64)\(([^)]+)\)")

        # Function to replace np.float32(...) and np.float64(...) with the numeric value
        def replace_np_float(match):
            return match.group(1)

        # Read and parse each line in the file
        with open(self.state_file_path, "r") as file:
            for line_num, line in enumerate(file, start=1):
                # Strip any leading/trailing whitespace
                line = line.strip()
                if line:  # Ensure the line is not empty
                    try:
                        # Remove np.float32(...) and np.float64(...) using regex
                        cleaned_line = pattern.sub(r"\1", line)
                        # Safely evaluate the dictionary string
                        entry = ast.literal_eval(cleaned_line)
                        data.append(entry)
                    except (SyntaxError, ValueError) as e:
                        print(f"Error parsing line {line_num}: {line}\n{e}")

        # Check if data was parsed successfully
        if not data:
            raise ValueError("No data was parsed. Please check the state.txt file format.")

        # Create a DataFrame from the list of dictionaries
        self.df = pd.DataFrame(data).iloc[2:]

    def training_mae_graph(self, ax1=None, ax2=None):
        # fig, ax1 = plt.subplots(figsize=(12, 6))
        if ax2 is None:
            ax2 = ax1.twinx()
        # Reset all spines visibility first
        for spine in ax2.spines.values():
            spine.set_visible(False)
        # Only show the right spine
        ax2.spines["right"].set_visible(True)

        train_data = self.df[self.df["mode"] == "Train"]
        val_data = self.df[self.df["mode"] == "Val"]
        # Plot Train MAE on the primary y-axis
        color_train = "tab:green"
        ax1.set_xlabel("Epoch", fontsize=20)
        ax1.set_ylabel("Train MAE", color=color_train)
        ax1.plot(
            train_data["epoch"],
            train_data["MAE"],
            color=color_train,
            marker="o",
            label="Train MAE",
        )
        ax1.tick_params(axis="y", labelcolor=color_train)

        color_val = "tab:orange"
        ax2.set_ylabel("Val MAE", color=color_val)
        ax2.plot(
            val_data["epoch"], val_data["MAE"], color=color_val, marker="s", label="Val MAE"
        )
        ax2.tick_params(axis="y", labelcolor=color_val)

        # Set x-axis ticks to each epoch
        epochs = sorted(self.df["epoch"].unique())
        ax1.set_xticks(epochs)

        # Add a title
        # plt.title(f"{self.title}: MAE")

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper right",
            fontsize=self.legend_fontsize,
        )

        # Show grid
        ax1.grid(True)

        # Adjust layout to prevent overlap
        # fig.tight_layout()

        # Display the plot
        if self.display:
            plt.show()

        return ax1, ax2

    def training_mape_graph(self, ax1=None, ax2=None):
        # fig, ax1 = plt.subplots(figsize=(12, 6))
        if ax2 is None:
            ax2 = ax1.twinx()
        # Reset all spines visibility first
        for spine in ax2.spines.values():
            spine.set_visible(False)
        # Only show the right spine
        ax2.spines["right"].set_visible(True)

        train_data = self.df[self.df["mode"] == "Train"]
        val_data = self.df[self.df["mode"] == "Val"]
        # Plot Train MAPE on the primary y-axis
        color_train = "tab:purple"
        ax1.set_xlabel("Epoch", fontsize=20)
        ax1.set_ylabel("Train MAPE", color=color_train)
        ax1.plot(
            train_data["epoch"],
            train_data["MAPE"],
            color=color_train,
            marker="o",
            label="Train MAPE",
        )
        ax1.tick_params(axis="y", labelcolor=color_train)

        color_val = "tab:cyan"
        ax2.set_ylabel("Val MAPE", color=color_val)
        ax2.plot(
            val_data["epoch"],
            val_data["MAPE"],
            color=color_val,
            marker="s",
            label="Val MAPE",
        )
        ax2.tick_params(axis="y", labelcolor=color_val)

        # Set x-axis ticks to each epoch
        epochs = sorted(self.df["epoch"].unique())
        ax1.set_xticks(epochs)

        # Add a title
        # plt.title(f"{self.title}: MAPE")

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper right",
            fontsize=self.legend_fontsize,
        )

        # Show grid
        ax1.grid(True)

        # Adjust layout to prevent overlap
        # fig.tight_layout()

        # Display the plot
        if self.display:
            plt.show()

        return ax1, ax2

    def training_loss_graph(self, ax1=None, ax2=None, legend_center=False):
        if ax2 is None:
            ax2 = ax1.twinx()
        # Reset all spines visibility first
        for spine in ax2.spines.values():
            spine.set_visible(False)
        # Only show the right spine
        ax2.spines["right"].set_visible(True)

        train_data = self.df[self.df["mode"] == "Train"]
        val_data = self.df[self.df["mode"] == "Val"]

        # Set Seaborn style for better aesthetics
        sns.set(style="whitegrid")

        # Initialize the matplotlib figure and primary axis

        # Plot Train Loss on the primary y-axis
        color_train = "tab:blue"
        ax1.set_xlabel("Epoch", fontsize=20)
        ax1.set_ylabel("Train Loss", color=color_train)
        ax1.plot(
            train_data["epoch"],
            train_data["loss"],
            color=color_train,
            marker="o",
            label="Train Loss",
        )
        ax1.tick_params(axis="y", labelcolor=color_train)

        color_val = "tab:red"
        ax2.set_ylabel("Val Loss", color=color_val)
        ax2.plot(
            val_data["epoch"],
            val_data["loss"],
            color=color_val,
            marker="s",
            label="Val Loss",
        )
        ax2.tick_params(axis="y", labelcolor=color_val)

        # Set x-axis ticks to each epoch
        epochs = sorted(self.df["epoch"].unique())
        ax1.set_xticks(epochs)

        # Add a title
        # plt.title("Train vs. Validation Loss per Epoch")
        # plt.title(f"{self.title}: Loss (MSE)")

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="center right" if legend_center else "upper right",
            fontsize=self.legend_fontsize,
        )

        # Show grid
        ax1.grid(True)

        # Adjust layout to prevent overlap
        # fig.tight_layout()

        # Display the plot
        if self.display:
            plt.show()

        return ax1, ax2