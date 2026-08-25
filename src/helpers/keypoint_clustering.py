"""
Keypoint clustering utilities for posture analysis.

This module provides functions to save and load trained PCA and K-means models
for keypoint-based posture clustering, allowing reuse during inference.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple, Union, Optional

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class KeypointClusteringPipeline:
    """
    A complete pipeline for keypoint-based posture clustering.

    This class encapsulates the preprocessing (scaling), dimensionality reduction (PCA),
    and clustering (K-means) steps for keypoint analysis.
    """

    def __init__(
        self,
        pca_variance_ratio: float = 0.95,
        n_clusters: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize the clustering pipeline.

        Args:
            pca_variance_ratio: Fraction of variance to retain in PCA
            n_clusters: Number of clusters for K-means (if None, must be set later)
            random_state: Random state for reproducibility
        """
        self.pca_variance_ratio = pca_variance_ratio
        self.n_clusters = n_clusters
        self.random_state = random_state

        # Initialize models
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_variance_ratio, random_state=random_state)
        self.kmeans = None

        # Training state
        self.is_fitted = False

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], n_clusters: Optional[int] = None
    ) -> "KeypointClusteringPipeline":
        """
        Fit the complete pipeline on training data.

        Args:
            X: Training data (features should be keypoint x,y coordinates)
            n_clusters: Number of clusters (overrides constructor value if provided)

        Returns:
            self: Fitted pipeline
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters

        if self.n_clusters is None:
            raise ValueError(
                "n_clusters must be specified either in constructor or fit method"
            )

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Step 1: Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Step 2: Apply PCA
        X_pca = self.pca.fit_transform(X_scaled)

        # Step 3: Fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, init="k-means++", random_state=self.random_state
        )
        self.kmeans.fit(X_pca)

        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            X: New data with same structure as training data

        Returns:
            cluster_labels: Predicted cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Apply preprocessing pipeline
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        # Predict clusters
        return self.kmeans.predict(X_pca)

    def fit_predict(
        self, X: Union[pd.DataFrame, np.ndarray], n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit the pipeline and predict on the same data.

        Args:
            X: Training data
            n_clusters: Number of clusters

        Returns:
            cluster_labels: Predicted cluster labels for training data
        """
        self.fit(X, n_clusters)
        return self.predict(X)

    def save(self, model_dir: Union[str, Path]) -> None:
        """
        Save the fitted pipeline to disk.

        Args:
            model_dir: Directory to save models
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save individual components
        with open(model_path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(model_path / "pca.pkl", "wb") as f:
            pickle.dump(self.pca, f)
        with open(model_path / "kmeans.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)

        # Save pipeline metadata
        metadata = {
            "pca_variance_ratio": self.pca_variance_ratio,
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "pca_components": self.pca.n_components_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.sum(),
        }

        with open(model_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print(f"Pipeline saved to: {model_path}")
        print(f"PCA components: {self.pca.n_components_}")
        print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        print(f"Number of clusters: {self.n_clusters}")

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "KeypointClusteringPipeline":
        """
        Load a saved pipeline from disk.

        Args:
            model_dir: Directory containing saved models

        Returns:
            Loaded pipeline
        """
        model_path = Path(model_dir)

        # Load metadata
        with open(model_path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Create pipeline instance
        pipeline = cls(
            pca_variance_ratio=metadata["pca_variance_ratio"],
            n_clusters=metadata["n_clusters"],
            random_state=metadata["random_state"],
        )

        # Load fitted models
        with open(model_path / "scaler.pkl", "rb") as f:
            pipeline.scaler = pickle.load(f)
        with open(model_path / "pca.pkl", "rb") as f:
            pipeline.pca = pickle.load(f)
        with open(model_path / "kmeans.pkl", "rb") as f:
            pipeline.kmeans = pickle.load(f)

        pipeline.is_fitted = True

        return pipeline

    def get_cluster_centers_original_space(self) -> np.ndarray:
        """
        Get cluster centers transformed back to original feature space.

        Returns:
            cluster_centers: Cluster centers in original feature space
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before accessing cluster centers")

        # Get centers in PCA space
        centers_pca = self.kmeans.cluster_centers_

        # Transform back to scaled space
        centers_scaled = self.pca.inverse_transform(centers_pca)

        # Transform back to original space
        centers_original = self.scaler.inverse_transform(centers_scaled)

        return centers_original


# Simple utility functions
def load_models(model_dir: Union[str, Path]) -> Tuple[StandardScaler, PCA, KMeans]:
    """
    Load the three pickle files directly.

    Args:
        model_dir: Directory containing saved models

    Returns:
        tuple: (scaler, pca, kmeans) - the loaded models
    """
    model_path = Path(model_dir)

    with open(model_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_path / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(model_path / "kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)

    return scaler, pca, kmeans


def predict_clusters(
    keypoint_features: Union[pd.DataFrame, np.ndarray],
    scaler: StandardScaler,
    pca: PCA,
    kmeans: KMeans,
) -> np.ndarray:
    """
    Predict clusters for keypoint data.

    Args:
        keypoint_features: DataFrame/array with keypoint x,y coordinates
        scaler: Fitted StandardScaler
        pca: Fitted PCA model
        kmeans: Fitted KMeans model

    Returns:
        cluster_labels: Array of cluster predictions
    """
    # Convert to numpy array if needed
    if isinstance(keypoint_features, pd.DataFrame):
        keypoint_features = keypoint_features.values

    # Apply preprocessing pipeline
    X_scaled = scaler.transform(keypoint_features)
    X_pca = pca.transform(X_scaled)

    # Predict clusters
    cluster_labels = kmeans.predict(X_pca)

    return cluster_labels