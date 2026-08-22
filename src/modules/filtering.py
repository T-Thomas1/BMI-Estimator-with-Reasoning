from pydantic import (
    BaseModel,
    model_validator,
    ConfigDict
)
from typing import List, Optional, Self
import pandas as pd
from matplotlib_venn import venn3
from matplotlib import pyplot as plt
from os import environ, listdir
import sqlite3
import json

class FilteringModule(BaseModel):
    data: pd.DataFrame
    bounding_box_data: Optional[pd.DataFrame] = None
    posture_cluster_data: Optional[pd.DataFrame] = None

    bounding_box_confidence_threshold: float
    bounding_box_ratio_threshold: float
    outlying_posture_clusters: List[int]

    posture_cluster_col : str = "posture_cluster"
    bounding_box_ratio_col : str = "bbox_area_ratio"
    bounding_box_confidence_col : str = "bbox_confidence"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def check_input_data(self) -> Self:
        """
        Check if:
            - Bounding box data is provided for all images (confidence and area ratio)
            - Posture clusters are provided for all images
        """        
        
        starting_length = len(self.data)

        if self.posture_cluster_col not in self.data.columns:
            # Join posture cluster data into self.data
            if self.posture_cluster_col not in self.posture_cluster_data.columns:
                raise ValueError(f"Posture cluster data not provided in data or posture_cluster_data (Make sure that the column name is {self.posture_cluster_col})")
            self.data = self.data.join(self.posture_cluster_data.set_index("image_id")[self.posture_cluster_col])

        if self.bounding_box_ratio_col not in self.data.columns:
            # Join bounding box data into self.data
            if self.bounding_box_ratio_col not in self.bounding_box_data.columns:
                raise ValueError(f"Bounding box data not provided in data or bounding_box_data (Make sure that the column name is {self.bounding_box_ratio_col})")
            self.data = self.data.join(self.bounding_box_data.set_index("image_id")[self.bounding_box_ratio_col])

        if self.bounding_box_confidence_col not in self.data.columns:
            # Join bounding box data into self.data
            if self.bounding_box_confidence_col not in self.bounding_box_data.columns:
                raise ValueError(f"Bounding box data not provided in data or bounding_box_data (Make sure that the column name is {self.bounding_box_confidence_col})")
        
            self.data = self.data.join(self.bounding_box_data.set_index("image_id")[self.bounding_box_confidence_col])

        required_columns = [self.posture_cluster_col, self.bounding_box_ratio_col, self.bounding_box_confidence_col]
        for col in required_columns:
            if self.data[col].isna().sum() > 0:
                raise ValueError(f"Missing values in column {col}")


        assert len(self.data) == starting_length, "Data was modified during validation!"

        return self
    
    @property
    def too_small(self) -> pd.DataFrame:
        return self.data[self.data[self.bounding_box_ratio_col] < self.bounding_box_ratio_threshold]

    @property
    def bad_posture(self) -> pd.DataFrame:
        return self.data[self.data[self.posture_cluster_col].isin(self.outlying_posture_clusters)]
    
    @property
    def low_confidence(self) -> pd.DataFrame:
        return self.data[self.data[self.bounding_box_confidence_col] < self.bounding_box_confidence_threshold]

    def show_filter_overlap(self) -> None:
        plt.figure(figsize=(8, 8))
        venn3([set(self.too_small.index),
               set(self.bad_posture.index),
               set(self.low_confidence.index)],
               ('Too small', 'Bad posture', 'Low confidence'))
        plt.title("Filter Steps")
        plt.show()
        return

    @property
    def filtered_data(self) -> pd.DataFrame:
        filtered_data = self.data[~self.data.index.isin(self.too_small.index) & ~self.data.index.isin(self.bad_posture.index) & ~self.data.index.isin(self.low_confidence.index)]

        assert len(filtered_data[filtered_data.index.isin(self.too_small.index)]) == 0
        assert len(filtered_data[filtered_data.index.isin(self.low_confidence.index)]) == 0
        assert len(filtered_data[filtered_data.index.isin(self.bad_posture.index)]) == 0
        
        print(f"{len(filtered_data)} images left after filtering")
        return filtered_data

    @property
    def __hash__(self) -> int:
        """Generate hash based on filtering parameters and data length"""
        return hash((self.bounding_box_confidence_threshold, 
                self.bounding_box_ratio_threshold, 
                ",".join(map(str, self.outlying_posture_clusters)), 
                len(self.data)))
                
    def save_filtered_data(self, path: str = environ["CUSTOM_DATA_DIR"]) -> str:
        """Save filtered data to sqlite database with hash as filename"""
        filename = f"filtered_data_{self.__hash__}.db"
        if filename in listdir(path):
            raise FileExistsError(f"File {filename} already exists in {path}")

        con = sqlite3.connect(f"{path}/{filename}")
        self.filtered_data.to_sql("weigh_ins", con=con, index=False)
        con.close()
        print(f"Filtered data saved to {path}/{filename}")

        # Write hash as json key to a file that logs the settings. But only the settings that contributed to the hash. Create a new file if it doesn't exist
        # Otherwise, append to the existing file
        with open(f"{path}/filtering_settings.json", "a") as f:
            settings = { 
                "bounding_box_confidence_threshold": self.bounding_box_confidence_threshold,
                "bounding_box_ratio_threshold": self.bounding_box_ratio_threshold,
                "outlying_posture_clusters": self.outlying_posture_clusters
            }
            f.write(json.dumps({self.__hash__: settings}) + "\n")


        return f"{path}/{filename}"