import matplotlib.pyplot as plt
from os import environ
from typing import Optional, Any
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

if "PHOTOS_DIR" not in environ:
    raise ValueError("Please set the 'PHOTOS_DIR' environment variable.")

PHOTO_FOLDER: str = environ.get("PHOTOS_DIR")  # type: ignore


class DatasetVisualiser:
    """Ipywidget slider to visualize the dataset."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        custom_photo_visualizer: Optional[callable] = None,
        extra_information_printer: Optional[callable] = None,
        index_col: Optional[str] = None,
        show_weight: Optional[bool] = False,
        photo_path_col: str = "photo_path",
    ) -> None:
        # assert "photo" in dataset.columns, "Dataset should contain 'photo' column."

        self.photo_path_col = photo_path_col
        self.full_dataset = dataset
        self.index_col = index_col
        self.show_weight = show_weight
        if custom_photo_visualizer is not None:
            self.show_photo = custom_photo_visualizer
            # Check that the custom photo visualizer is a callable and accepts either instance or photo_id
            assert callable(
                custom_photo_visualizer
            ), "Custom photo visualizer should be a callable."
            assert (
                "instance" in custom_photo_visualizer.__code__.co_varnames
                or "photo_id" in custom_photo_visualizer.__code__.co_varnames
            ), "Custom photo visualizer should accept either 'instance' or 'photo_id' as input."
        else:
            self.show_photo = self.default_photo_visualizer

        if extra_information_printer is not None:
            self.extra_information_printer = extra_information_printer
            # Check that the custom extra information printer is a callable and accepts instance
            assert callable(
                extra_information_printer
            ), "Custom extra information printer should be a callable."
            assert (
                "instance" in extra_information_printer.__code__.co_varnames
            ), "Custom extra information printer should accept 'instance' as input."
        else:
            self.extra_information_printer = None

        if self.index_col is not None:
            # Create the index slider
            unique_indices = self.full_dataset[self.index_col].unique()
            self.index_slider = widgets.Dropdown(
                options=unique_indices,
                description=f"{self.index_col}:",
            )
            self.index_slider.observe(self.update_filtered_dataset, names="value")
            # Initialize filtered dataset
            self.filtered_dataset = self.full_dataset[
                self.full_dataset[self.index_col] == unique_indices[0]
            ]
        else:
            self.filtered_dataset = self.full_dataset

        # Create the image index slider
        self.image_index_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.filtered_dataset) - 1,
            step=1,
            description="Image Index:",
            continuous_update=False,
            orientation="horizontal",
        )
        self.image_index_slider.observe(self.render, names="value")

        # Create an Output widget to display the image
        self.image_output = widgets.Output()

        # Display the widgets
        if self.index_col is not None:
            display(self.index_slider, self.image_index_slider, self.image_output)
        else:
            display(self.image_index_slider, self.image_output)

        # Initial render
        self.render()

    def update_filtered_dataset(self, change):
        selected_index = change["new"]
        self.filtered_dataset = self.full_dataset[
            self.full_dataset[self.index_col] == selected_index
        ].reset_index(drop=True)
        # Update the image index slider
        self.image_index_slider.max = len(self.filtered_dataset) - 1
        self.image_index_slider.value = 0  # Reset slider to 0
        self.render()

    def render(self, change=None):
        idx = self.image_index_slider.value
        instance = self.filtered_dataset.iloc[idx]
        with self.image_output:
            self.image_output.clear_output()
            self.show_photo(instance)

    def default_photo_visualizer(self, instance):
        # Default implementation to show photo
        img_path = instance[self.photo_path_col]

        if self.extra_information_printer is not None:
            self.extra_information_printer(instance)
        if self.show_weight:
            weight = instance["weight"]
            print(f"Weight: {weight}")
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()