import pandas as pd
import numpy as np

def split_dataframe(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Splits the dataframe into train, validation, and test partitions and adds a 'partition' column.

    Args:
        df (pd.DataFrame): The original dataframe.
        train_size (float): Proportion of the data to include in the train split.
        val_size (float): Proportion of the data to include in the validation split.
        test_size (float): Proportion of the data to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: The dataframe with an added 'partition' column.
    """
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1."

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate split indices
    train_end = int(train_size * len(df))
    val_end = train_end + int(val_size * len(df))

    # Initialize partition column
    df['partition'] = 'test'  # Default partition

    # Assign partitions
    df.iloc[:train_end, df.columns.get_loc('partition')] = 'train'
    df.iloc[train_end:val_end, df.columns.get_loc('partition')] = 'val'

    return df

def split_dataframe_without_user_overlap(
        df: pd.DataFrame,
        train_size: float = 0.75,
        val_size: float = 0.10,
        random_state: int = 42
) -> pd.DataFrame:
    df = df.copy()
    total_photos = len(df)
    target_train_photos = int(train_size * total_photos)
    target_val_photos = int(val_size * total_photos)
    target_test_photos = total_photos - target_train_photos - target_val_photos

    # Group users by photo count
    user_counts = df.groupby('user_id').size().reset_index(name='photo_count')

    # Shuffle users
    np.random.seed(42)
    user_counts = user_counts.sample(frac=1, random_state=42)

    # Initialize counters and assignment
    assigned_train = assigned_val = assigned_test = 0
    user_to_partition = {}

    for _, row in user_counts.iterrows():
        user = row['user_id']
        count = row['photo_count']

        # Compute remaining "space" in each partition
        train_gap = target_train_photos - assigned_train
        val_gap = target_val_photos - assigned_val
        test_gap = target_test_photos - assigned_test
        
        # Pick the partition that best keeps the balance
        # (e.g., the one with the largest gap)
        # This is a simple heuristic; you can refine it as needed.
        best_partition = max(
            [('train', train_gap), ('val', val_gap), ('test', test_gap)],
            key=lambda x: x[1]
        )[0]

        user_to_partition[user] = best_partition
        
        if best_partition == 'train':
            assigned_train += count
        elif best_partition == 'val':
            assigned_val += count
        else:
            assigned_test += count

    df['partition'] = df['user_id'].map(user_to_partition)
    return df

def get_test_data(df: pd.DataFrame, without_user_overlap: bool) -> pd.DataFrame:
    """
    Returns the test partition of the dataframe.

    Args:
        df (pd.DataFrame): The dataframe with a 'partition' column.

    Returns:
        pd.DataFrame: The test partition of the dataframe.
    """
    if without_user_overlap:
        df = split_dataframe_without_user_overlap(df)
    else:
        df = split_dataframe(df)
    return df[df['partition'] == 'test']

def split_visual_bmi_dataframe(
    df: pd.DataFrame,
    train_image_count: int = 4000,
    val_image_count: int = 950,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Splits the Visual-BMI dataframe into train, validation, and test partitions based on fixed numbers of images.
    This strategy is the same as that used in the original paper.

    Args:
        df (pd.DataFrame): The original dataframe from the Visual-BMI dataset.
        train_image_count (int): The number of images to include in the train split.
        val_image_count (int): The number of images to include in the validation split.
        random_state (int): Random seed for reproducibility of the shuffle.

    Returns:
        pd.DataFrame: The dataframe with an added 'partition' column ('train', 'val', or 'test').
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if train_image_count <= 0:
        raise ValueError("train_image_count must be positive.")
    if val_image_count < 0:
        raise ValueError("val_image_count must be non-negative.")

    total_images = len(df)
    if train_image_count + val_image_count > total_images:
        raise ValueError(
            f"Sum of train_image_count ({train_image_count}) and val_image_count ({val_image_count}) "
            f"({train_image_count + val_image_count}) cannot be greater than the total number of images ({total_images})."
        )

    df_copy = df.copy()

    # Shuffle the dataframe
    df_shuffled = df_copy.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Initialize partition column
    df_shuffled['partition'] = 'test'

    # Assign train partition
    train_end_idx = train_image_count
    df_shuffled.iloc[:train_end_idx, df_shuffled.columns.get_loc('partition')] = 'train'

    # Assign val partition from the remainder
    if val_image_count > 0:
        val_end_idx = train_end_idx + val_image_count
        df_shuffled.iloc[train_end_idx:val_end_idx, df_shuffled.columns.get_loc('partition')] = 'val'
        test_image_count = total_images - train_image_count - val_image_count
    else:
        test_image_count = total_images - train_image_count
    
    print(f"Splitting Visual-BMI dataset: {train_image_count} for train, "
          f"{val_image_count} for validation, {test_image_count} for test.")

    return df_shuffled
