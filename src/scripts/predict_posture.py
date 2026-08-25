"""
POSTURE PREDICTION - Extremely Simple Usage

This script takes RAW keypoint data and predicts posture clusters.
Only clusters 2 and 3 are considered GOOD postures.

Usage:
    python predict_posture.py

Or import and use:
    from predict_posture import predict_posture
    result = predict_posture(keypoint_data, bounding_box_data)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def normalize_keypoints(keypoints_df, bounding_boxes_df):
    """Normalize raw keypoint coordinates using bounding boxes."""
    # Join with bounding boxes
    if "image_id" in bounding_boxes_df.columns:
        bounding_boxes_df = bounding_boxes_df.set_index("image_id")

    joined_df = keypoints_df.join(bounding_boxes_df.add_prefix("bounding_box_"))

    # Get x and y columns
    x_columns = [col for col in joined_df.columns if col.endswith("-x")]
    y_columns = [col for col in joined_df.columns if col.endswith("-y")]

    # Normalize coordinates
    normalized_df = joined_df.copy()
    normalized_df[x_columns] = (
        normalized_df[x_columns]
        .subtract(joined_df["bounding_box_x1"], axis=0)
        .div(joined_df["bounding_box_width"], axis=0)
    )
    normalized_df[y_columns] = (
        normalized_df[y_columns]
        .subtract(joined_df["bounding_box_y1"], axis=0)
        .div(joined_df["bounding_box_height"], axis=0)
    )

    return normalized_df[x_columns + y_columns]


def load_pretrained_models(model_dir="trained_models/keypoint_models"):
    """Load the pretrained posture clustering models."""
    model_path = Path(model_dir)

    with open(model_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_path / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(model_path / "kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)

    return scaler, pca, kmeans


def predict_posture(
    keypoints_df, bounding_boxes_df, model_dir="trained_models/keypoint_models"
):
    """
    Predict posture from RAW keypoint data.

    Args:
        keypoints_df: DataFrame with raw keypoint coordinates (x,y for each keypoint)
        bounding_boxes_df: DataFrame with bounding box info (x1, y1, width, height)
        model_dir: Path to trained models

    Returns:
        dict: {
            'clusters': array of cluster predictions,
            'good_posture': boolean array (True for clusters 1&2),
            'confidence': prediction confidence scores
        }
    """
    # Load models
    scaler, pca, kmeans = load_pretrained_models(model_dir)

    # Normalize keypoints
    normalized_features = normalize_keypoints(keypoints_df, bounding_boxes_df)

    # Apply ML pipeline
    X_scaled = scaler.transform(normalized_features.values)
    X_pca = pca.transform(X_scaled)

    # Predict clusters
    clusters = kmeans.predict(X_pca)

    # Calculate distances to cluster centers (confidence)
    distances = kmeans.transform(X_pca)
    confidence = 1 / (1 + distances.min(axis=1))  # Higher = more confident

    # Determine good posture (only clusters 2 and 3 are good)
    good_posture = np.isin(clusters, [2, 3])

    return {"clusters": clusters, "good_posture": good_posture, "confidence": confidence}


def example_usage():
    """Example showing how to use the posture prediction."""
    print("🏃 POSTURE PREDICTION EXAMPLE")
    print("=" * 50)

    try:
        # Load sample data
        from helpers.load_waybetter_db import load_keypoints_db, load_bounding_boxes_db

        print("📊 Loading sample data...")
        keypoints_df = load_keypoints_db().head(10)  # First 10 samples
        bounding_boxes_df = load_bounding_boxes_db()

        print(f"   - {len(keypoints_df)} keypoint samples")
        print(f"   - {len(bounding_boxes_df)} bounding boxes")

        # Predict postures
        print("\n🤖 Predicting postures...")
        results = predict_posture(keypoints_df, bounding_boxes_df)

        # Display results
        print("\n📋 RESULTS:")
        print(f"   Clusters: {results['clusters']}")
        print(f"   Good posture: {results['good_posture']}")
        print(f"   Confidence: {results['confidence']:.3f}")

        # Summary
        n_good = results["good_posture"].sum()
        n_total = len(results["clusters"])

        print(f"\n✅ SUMMARY:")
        print(f"   Good postures: {n_good}/{n_total} ({n_good/n_total*100:.1f}%)")
        print(f"   Cluster breakdown: {np.bincount(results['clusters'])}")

        # Individual results
        print(f"\n🔍 INDIVIDUAL RESULTS:")
        for i, (cluster, good, conf) in enumerate(
            zip(results["clusters"], results["good_posture"], results["confidence"])
        ):
            status = "✅ GOOD" if good else "❌ BAD"
            print(f"   Sample {i+1}: Cluster {cluster} | {status} | Confidence: {conf:.3f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have trained models in 'trained_models/keypoint_models/'")


if __name__ == "__main__":
    example_usage()