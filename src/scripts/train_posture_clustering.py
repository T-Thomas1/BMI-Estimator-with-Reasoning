"""
POSTURE CLUSTERING TRAINING

This script trains new posture clustering models and provides interactive
helpers to identify good vs bad clusters.

Usage:
    python train_posture_clustering.py

Or import in notebook:
    from train_posture_clustering import train_clustering, InteractiveClusterExplorer

    # Train models
    models = train_clustering(data)

    # Explore clusters interactively
    explorer = InteractiveClusterExplorer(models, data)
    explorer.show()
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Try to import interactive widgets (optional)
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("⚠️  ipywidgets not available. Interactive features will be limited.")


def prepare_keypoint_data():
    """Load and normalize keypoint data for training."""
    from helpers.load_waybetter_db import load_keypoints_db, load_bounding_boxes_db

    print("📊 Loading training data...")
    keypoints_df = load_keypoints_db()
    bounding_boxes_df = load_bounding_boxes_db().set_index("image_id")

    # Join and normalize
    joined_df = keypoints_df.join(bounding_boxes_df.add_prefix("bounding_box_"))

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

    # Extract features
    features = normalized_df[x_columns + y_columns]

    print(f"   ✅ Loaded {len(features)} samples with {features.shape[1]} features")
    return features, normalized_df


def train_clustering(features=None, n_clusters=4, pca_variance=0.95, random_state=42):
    """
    Train posture clustering models.

    Args:
        features: DataFrame with normalized keypoint features (if None, loads from DB)
        n_clusters: Number of clusters (default 4)
        pca_variance: Fraction of variance to retain in PCA
        random_state: Random state for reproducibility

    Returns:
        dict: Trained models and results
    """
    if features is None:
        features, _ = prepare_keypoint_data()

    print(f"🤖 Training clustering models...")
    print(f"   - Features: {features.shape}")
    print(f"   - Clusters: {n_clusters}")
    print(f"   - PCA variance: {pca_variance}")

    # Step 1: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)  # Use .values to get numpy array
    print(f"   ✅ Scaled features")

    # Step 2: Apply PCA
    pca = PCA(n_components=pca_variance, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    print(
        f"   ✅ PCA: {features.shape[1]} → {X_pca.shape[1]} components ({pca.explained_variance_ratio_.sum():.1%} variance)"
    )

    # Step 3: Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state)
    clusters = kmeans.fit_predict(X_pca)
    print(f"   ✅ K-means clustering complete")

    # Create results
    results = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "features": features,
        "clusters": clusters,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
    }

    # Show cluster distribution
    cluster_counts = np.bincount(clusters)
    print(f"\n📊 Cluster distribution:")
    for i, count in enumerate(cluster_counts):
        print(f"   Cluster {i}: {count} samples ({count/len(clusters)*100:.1f}%)")

    return results


def save_trained_models(results, model_dir="trained_models/keypoint_models"):
    """Save the trained models to disk."""
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # Save models
    with open(model_path / "scaler.pkl", "wb") as f:
        pickle.dump(results["scaler"], f)
    with open(model_path / "pca.pkl", "wb") as f:
        pickle.dump(results["pca"], f)
    with open(model_path / "kmeans.pkl", "wb") as f:
        pickle.dump(results["kmeans"], f)

    print(f"💾 Models saved to: {model_path}")


def plot_elbow_curve(features=None, max_k=10):
    """Plot elbow curve to help choose optimal number of clusters."""
    if features is None:
        features, _ = prepare_keypoint_data()

    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)  # Use .values to get numpy array
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Calculate WCSS for different K values
    wcss = []
    K_range = range(1, max_k + 1)

    print("📈 Calculating elbow curve...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
        kmeans.fit(X_pca)
        wcss.append(kmeans.inertia_)
        print(f"   K={k}: WCSS={kmeans.inertia_:.0f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, wcss, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title("Elbow Method for Optimal K")
    plt.grid(True, alpha=0.3)

    # Highlight recommended range
    plt.axvspan(3, 5, alpha=0.2, color="green", label="Recommended range")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return wcss


class InteractiveClusterExplorer:
    """Interactive widget for exploring clusters in notebooks."""

    def __init__(self, results, normalized_df=None):
        """
        Initialize the interactive cluster explorer.

        Args:
            results: Results from train_clustering()
            normalized_df: Full normalized dataframe with metadata
        """
        self.results = results

        if normalized_df is None:
            _, self.normalized_df = prepare_keypoint_data()
        else:
            self.normalized_df = normalized_df

        # Add cluster labels
        self.normalized_df["cluster"] = results["clusters"]

        if not WIDGETS_AVAILABLE:
            print(
                "❌ Interactive widgets not available. Use simple_cluster_exploration() instead."
            )
            return

        self._setup_widgets()

    def _setup_widgets(self):
        """Setup interactive widgets."""
        n_clusters = self.results["kmeans"].n_clusters

        # Widgets
        self.k_slider = widgets.IntSlider(
            value=min(3, n_clusters - 1),
            min=0,
            max=n_clusters - 1,
            description="Cluster:",
            continuous_update=False,
        )

        self.sample_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,  # Will be updated
            description="Sample:",
            continuous_update=False,
        )

        self.output_area = widgets.Output()

        # Connect events
        self.k_slider.observe(self._on_cluster_change, "value")
        self.sample_slider.observe(self._on_sample_change, "value")

        # Initial setup
        self._update_sample_range()

    def _on_cluster_change(self, change):
        """Handle cluster selection change."""
        self._update_sample_range()
        self._on_sample_change(None)

    def _update_sample_range(self):
        """Update sample slider range based on selected cluster."""
        cluster_id = self.k_slider.value
        cluster_samples = self.normalized_df[self.normalized_df["cluster"] == cluster_id]

        max_samples = len(cluster_samples) - 1
        self.sample_slider.max = max(0, max_samples)
        self.sample_slider.value = 0

    def _on_sample_change(self, change):
        """Handle sample selection change."""
        with self.output_area:
            clear_output(wait=True)

            cluster_id = self.k_slider.value
            sample_idx = self.sample_slider.value

            # Get cluster data
            cluster_samples = self.normalized_df[
                self.normalized_df["cluster"] == cluster_id
            ]

            if len(cluster_samples) == 0:
                print(f"❌ No samples in cluster {cluster_id}")
                return

            # Get specific sample
            if sample_idx >= len(cluster_samples):
                print(f"❌ Sample index {sample_idx} out of range")
                return

            sample = cluster_samples.iloc[sample_idx]

            # Display info
            print(
                f"🎯 CLUSTER {cluster_id} - Sample {sample_idx + 1}/{len(cluster_samples)}"
            )
            print(f"📊 Total samples in cluster: {len(cluster_samples)}")
            print(
                f"📈 Cluster distribution: {dict(self.normalized_df['cluster'].value_counts().sort_index())}"
            )

            # Try to display image if possible
            try:
                self._display_sample_image(sample)
            except Exception as e:
                print(f"⚠️  Could not display image: {e}")

    def _display_sample_image(self, sample):
        """Display sample with keypoints if image data available."""
        if "photo_path" in sample:
            # This would need the image display function from the original notebook
            print(f"🖼️  Image: {sample.get('photo_path', 'N/A')}")
        else:
            print("📍 Keypoint coordinates available (no image path)")

    def show(self):
        """Display the interactive explorer."""
        if not WIDGETS_AVAILABLE:
            print("❌ Widgets not available. Use simple_cluster_exploration() instead.")
            return

        ui = widgets.VBox(
            [
                widgets.HTML("<h3>🔍 Interactive Cluster Explorer</h3>"),
                widgets.HTML(
                    "<p>Explore different clusters to identify good vs bad postures:</p>"
                ),
                self.k_slider,
                self.sample_slider,
                self.output_area,
            ]
        )

        display(ui)

        # Trigger initial display
        self._on_sample_change(None)


def simple_cluster_exploration(results, n_samples_per_cluster=3):
    """Simple non-interactive cluster exploration."""
    clusters = results["clusters"]
    n_clusters = results["kmeans"].n_clusters

    print("🔍 CLUSTER EXPLORATION")
    print("=" * 50)

    for cluster_id in range(n_clusters):
        cluster_mask = clusters == cluster_id
        cluster_count = cluster_mask.sum()

        print(f"\n📊 CLUSTER {cluster_id}:")
        print(f"   Samples: {cluster_count} ({cluster_count/len(clusters)*100:.1f}%)")

        if cluster_count > 0:
            # Show some statistics
            cluster_indices = np.where(cluster_mask)[0]
            sample_indices = cluster_indices[:n_samples_per_cluster]

            print(f"   Sample indices: {sample_indices}")
            print(f"   💡 Use these indices to manually inspect images")

            # Calculate cluster center distance (confidence measure)
            cluster_center = results["kmeans"].cluster_centers_[cluster_id]
            X_pca_cluster = results["X_pca"][cluster_mask]
            distances = np.linalg.norm(X_pca_cluster - cluster_center, axis=1)

            print(f"   Avg distance to center: {distances.mean():.3f}")
            print(
                f"   Most representative samples: {cluster_indices[np.argsort(distances)][:3]}"
            )


def main():
    """Main training workflow."""
    print("🏋️ POSTURE CLUSTERING TRAINING")
    print("=" * 50)

    # Step 1: Show elbow curve
    print("\n1️⃣ Finding optimal number of clusters...")
    try:
        wcss = plot_elbow_curve()
    except Exception as e:
        print(f"⚠️  Could not plot elbow curve: {e}")

    # Step 2: Train clustering
    print("\n2️⃣ Training clustering models...")
    results = train_clustering(n_clusters=4)  # Default to 4 clusters

    # Step 3: Simple exploration
    print("\n3️⃣ Exploring clusters...")
    simple_cluster_exploration(results)

    # Step 4: Save models
    print("\n4️⃣ Saving models...")
    save_trained_models(results)

    print("\n✅ TRAINING COMPLETE!")
    print("💡 Next steps:")
    print("   1. Review cluster exploration above")
    print("   2. Identify which clusters represent GOOD postures")
    print("   3. Good clusters are currently set to [2, 3] in predict_posture.py")
    print("   4. Use trained models with predict_posture.py")


if __name__ == "__main__":
    main()