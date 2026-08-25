from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.helpers.visualizer import DatasetVisualiser

class LongitudinalExploration(BaseModel):
    df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)
        self._prepare_data()

    def _prepare_data(self):
        # Ensure 'taken_at' is datetime
        self.df["taken_at"] = pd.to_datetime(self.df["taken_at"])

        # Number of photos per user
        self.user_id_counts = self.df["user_id"].value_counts().reset_index()
        self.user_id_counts.columns = ['user_id', 'photo_count']

        # Weight variance per user
        self.weight_variance = self.df.groupby('user_id')['weight'].var().reset_index()
        self.weight_variance.columns = ['user_id', 'weight_variance']
        self.weight_variance = self.weight_variance.merge(self.user_id_counts, on='user_id')

        # Weight change per user
        weight_change = self.df.sort_values("taken_at").groupby('user_id')['weight'].agg(['first', 'last'])
        weight_change["weight_change"] = weight_change["last"] - weight_change["first"]
        weight_change['absolute_weight_change'] = weight_change['weight_change'].abs()
        self.weight_change = weight_change.merge(self.user_id_counts, on='user_id').reset_index()

        # Merge computed metrics back into the original DataFrame
        self.df = self.df.merge(
            self.weight_change[['user_id', 'weight_change', 'absolute_weight_change']], on='user_id'
        ).merge(
            self.weight_variance[['user_id', 'weight_variance']], on='user_id'
        )

    def plot_photo_counts_histogram(self):
        # Define bins and plot histogram
        bins = list(range(1, 20)) + [20, float('inf')]
        plt.figure(figsize=(12, 6))
        sns.histplot(self.user_id_counts, x="photo_count", bins=bins, discrete=True)
        plt.xlabel("Number of Photos")
        plt.ylabel("Number of Users")
        plt.xticks(list(range(1, 20)) + [20], labels=list(range(1, 20)) + ['20+'])
        plt.xlim(0, 21)
        plt.tight_layout()
        plt.show()

    def plot_weight_variance_vs_photo_count(self):
        # Scatter plot of weight variance vs. photo count
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=self.weight_variance, x='photo_count', y='weight_variance', alpha=0.5)
        plt.xlabel("Number of Photos")
        plt.ylabel("Weight Variance")
        plt.show()

    def plot_weight_change_vs_photo_count(self):
        # Scatter plot of weight change vs. photo count for users with more than one photo
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=self.weight_change[self.weight_change["photo_count"] > 1],
            x='photo_count',
            y='weight_change',
            alpha=0.3
        )
        plt.xlabel("Number of Photos")
        plt.ylabel("Weight Change (lbs)")
        plt.tight_layout()
        plt.show()

    def plot_weight_change_boxplot(self):
        # Create bins and plot boxplot of weight change vs. photo bins
        bins = [0, 5, 10, 15, 20, float('inf')]
        labels = ['1-5', '6-10', '11-15', '16-20', '20+']
        self.weight_change['photo_bin'] = pd.cut(self.weight_variance['photo_count'], bins=bins, labels=labels)
        sns.boxplot(data=self.weight_change, x='photo_bin', y='weight_change')
        plt.xlabel("Number of Photos")
        plt.ylabel("Weight Change (lbs)")
        plt.show()

    def show_top_variance_users(self, top_n=100):
        # Get top N users by weight variance
        top_variance_users = self.weight_variance.sort_values('weight_variance', ascending=False).head(top_n)["user_id"]
        print(f"Top {top_n} users by weight variance:")

        DatasetVisualiser(
            self.df[self.df['user_id'].isin(top_variance_users)],
            show_weight=True,
            index_col="user_id"
        )