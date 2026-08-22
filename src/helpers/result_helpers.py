from typing import Optional, TypedDict
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def setup_test_results(test_set: pd.DataFrame) -> None:
    """
    Requirements:
    - test_set must have columns for height in centimeters and weight in pounds.
    - test_set must have a column for the predicted BMI.
    Outputs:
    - Adds columns for predicted weight in kilograms and pounds.
    - Adds the actual weight in kilograms.

    """
    test_set["predicted_weight_kg"] = test_set.apply(lambda row: calculate_weight_KG_from_bmi(row["height_cm"], row["output"]), axis=1)
    test_set["predicted_weight_lbs"] = test_set["predicted_weight_kg"] * 2.20462
    test_set["weight_kg"] = test_set["weight"] / 2.20462
    return test_set

def calculate_weight_KG_from_bmi(height_cm: float, bmi: float) -> float:
    """
    Calculate weight in kilograms given height in centimeters and BMI.
    
    Parameters:
    height_cm (float): Height in centimeters
    bmi (float): Body Mass Index (BMI)
    
    Returns:
    float: Weight in kilograms
    """
    # Convert height from cm to meters
    height_m = height_cm / 100
    # Calculate weight using the BMI formula
    weight_kg = bmi * (height_m ** 2)
    return weight_kg

def show_image_with_results(instance: Optional[pd.Series] = None) -> None:
    """
    Shows an image with the predicted and actual weight, height, and BMI.
    """
    # Create a figure with a specific layout
    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 3], hspace=0.1)
    
    # First subplot for the image
    ax_image = fig.add_subplot(gs[0, 0])
    photo_path = instance.get("photo_path")
    
    img = plt.imread(photo_path)
    ax_image.imshow(img)
    ax_image.axis('off')  # Hide the axes
    
    # Second subplot for the statistics
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_stats.axis('off')  # Hide the axes

    # Extract and format the statistics
    actual_weight_lbs = instance.get("weight", "N/A")
    actual_weight_kg = instance.get("weight_kg", "N/A")
    predicted_weight_lbs = instance.get("predicted_weight_lbs", "N/A")
    predicted_weight_kg = instance.get("predicted_weight_kg", "N/A")
    height_cm = instance.get("height_cm", "N/A")
    height_in = instance.get("height_in", "N/A")
    actual_bmi = instance.get("bmi", "N/A")
    predicted_bmi = instance.get("output", "N/A")

    stats_lines = [
    f"Weight: {int(actual_weight_lbs)} lbs / {int(actual_weight_kg)} kg",
    f"Predicted Weight: {int(predicted_weight_lbs)} lbs / {int(predicted_weight_kg)} kg",
    f"Height: {int(height_in)} in / {int(height_cm)} cm",
    f"BMI: {actual_bmi:.1f}",
    f"Predicted BMI: {predicted_bmi:.1f}"
    ]

    # Combine the lines into a single string with line breaks
    stats_text = "\n".join(stats_lines)

    # Display the statistics text
    ax_stats.text(0.5, 0.5, stats_text, 
                 horizontalalignment='center', 
                 verticalalignment='center', 
                 fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.show()

class TestResults(TypedDict):
    mae_lbs: float
    mae_kg: float
    mape: float

def test_set_performance(test_set: pd.DataFrame) -> TestResults:
    assert "predicted_weight_lbs" in test_set.columns, "No predicted weight in test set"
    assert "predicted_weight_kg" in test_set.columns, "No predicted weight in test set"
    assert "weight" in test_set.columns, "No weight in test set"
    assert "weight_kg" in test_set.columns, "No weight in test set"

    mae_lbs = mean_absolute_error(test_set["weight"], test_set["predicted_weight_lbs"])
    mape_lbs = mean_absolute_percentage_error(test_set["weight"], test_set["predicted_weight_lbs"])
    mae_kg = mean_absolute_error(test_set["weight_kg"], test_set["predicted_weight_kg"])
    mape_kg = mean_absolute_percentage_error(test_set["weight_kg"], test_set["predicted_weight_kg"])

    assert round(mape_kg, 3) == round(mape_lbs, 3), "MAPE in lbs and kg should be the same"
    return {"mae_lbs": mae_lbs, "mae_kg": mae_kg, "mape": mape_kg}