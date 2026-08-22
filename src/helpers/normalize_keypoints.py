import pandas as pd
# Normalize set of keypoints to bounding boxes so they can be directly compared.

def normalize_keypoints(keypoints_df: pd.DataFrame, bounding_boxes_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize keypoints to bounding boxes
    keypoints_df = keypoints_df.copy()
    bounding_boxes_df = bounding_boxes_df.copy()
    
    # Normalize keypoints to bounding boxes
    joined_df = keypoints_df.join(bounding_boxes_df.add_prefix("bounding_box_"))

    # List of keypoint x and y columns
    x_columns = [col for col in joined_df.columns if col.endswith('-x')]
    y_columns = [col for col in joined_df.columns if col.endswith('-y')]

    # Normalize the x coordinates
    joined_df[x_columns] = joined_df[x_columns].subtract(joined_df['bounding_box_x1'], axis=0).div(joined_df['bounding_box_width'], axis=0)

    # Normalize the y coordinates
    joined_df[y_columns] = joined_df[y_columns].subtract(joined_df['bounding_box_y1'], axis=0).div(joined_df['bounding_box_height'], axis=0)

    return joined_df