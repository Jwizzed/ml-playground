import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


COLORS = ["black", "red"]


class NumericalAnalysis:
    """A class to represent the Numerical Analysis of the dataframe."""

    def __init__(self, df, numeric_columns, tab):
        self.df = df
        self.numeric_columns = numeric_columns
        self.tab = tab

    def display_numerical_analysis(self):
        """Display the numerical analysis of the dataframe."""
        with self.tab:
            if self.numeric_columns:
                st.subheader("Numerical Columns")
                st.write(self.numeric_columns)

                with st.expander("View Numerical Data Distribution"):
                    num_plots = len(self.numeric_columns)
                    num_cols = min(num_plots,
                                   4)
                    num_rows = num_plots // num_cols + (
                            num_plots % num_cols > 0)
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(
                        5 * num_cols,
                        4 * num_rows))

                    axes = axes.flatten() if num_plots > 1 else np.array(
                        [axes])

                    for idx, col in enumerate(self.numeric_columns):
                        sns.histplot(self.df[col], bins='auto', kde=True,
                                     ax=axes[idx], color=COLORS[1],
                                     edgecolor=COLORS[0], alpha=0.7)
                        axes[idx].set_title(f'Distribution for {col}',
                                            fontsize=12)
                        axes[idx].set_xlabel(col,
                                             fontsize=10)
                        axes[idx].set_ylabel('Count',
                                             fontsize=10)
                        axes[idx].grid(True, which='both', linestyle='--',
                                       linewidth=0.5,
                                       alpha=0.5)
                        axes[idx].tick_params(axis='x',
                                              labelrotation=45)

                    for idx in range(num_plots,
                                     len(axes)):
                        fig.delaxes(axes[idx])

                    fig.tight_layout(
                        pad=2.0)
                    st.pyplot(fig)
                    plt.close(fig)

                with st.expander("Outlier Detection"):
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(
                        5 * num_cols, 4 * num_rows))
                    axes = axes.flatten() if num_plots > 1 else np.array(
                        [axes])

                    for idx, col in enumerate(self.numeric_columns):
                        bp = sns.boxplot(x=self.df[col], ax=axes[idx],
                                         color=COLORS[1],
                                         patch_artist=True)

                        for box in bp.artists:
                            box.set_alpha(0.5)

                        axes[idx].set_title(f'Boxplot for {col}')

                    for idx in range(num_plots, len(axes)):
                        fig.delaxes(axes[idx])

                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with st.expander("Feature Correlation Matrix"):

                    corr = self.df[self.numeric_columns].corr()

                    mask = np.triu(np.ones_like(corr, dtype=bool))

                    fig, axes = plt.subplots(figsize=(5 * num_cols,
                                                      4 * num_rows))

                    cmap = LinearSegmentedColormap.from_list(
                        'custom_diverging', COLORS,
                        N=256)

                    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                                center=0,
                                annot=True,
                                square=True, linewidths=.5,
                                cbar_kws={"shrink": .5},
                                fmt=".2f",
                                annot_kws={
                                    "size": 10})

                    plt.xticks(rotation=45, ha='right',
                               fontsize=10)
                    plt.yticks(fontsize=10)
                    plt.title('Feature Correlation Matrix',
                              fontsize=14)

                    st.pyplot(fig)
                    plt.close(fig)

                with st.expander("PCA Analysis"):
                    scaler = StandardScaler()
                    scaled_df = scaler.fit_transform(self.df[self.numeric_columns])
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_df)

                    if 'target' in self.df.columns and self.df[
                        'target'].dtype == 'object':
                        categories = self.df['target'].unique()

                        fig, ax = plt.subplots(figsize=(5 * num_cols,
                                                        4 * num_rows))
                        for category in categories:
                            ix = np.where(self.df['target'] == category)[0]
                            ax.scatter(pca_result[ix, 0],
                                       pca_result[ix, 1],
                                       label=category, s=100)
                        ax.legend()

                    else:
                        fig, ax = plt.subplots(figsize=(7, 7))
                        ax.scatter(pca_result[:, 0], pca_result[:, 1])

                    ax.set_title(
                        'PCA Result (First Two Principal Components)',
                        fontsize=14)
                    ax.set_xlabel('PC1', fontsize=12)
                    ax.set_ylabel('PC2', fontsize=12)
                    ax.grid(True)
                    if 'target' in self.df.columns and self.df[
                        'target'].dtype == 'object':
                        ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.write(
                        f'Explained variance ratio for PC1: {pca.explained_variance_ratio_[0]:.4f}')
                    st.write(
                        f'Explained variance ratio for PC2: {pca.explained_variance_ratio_[1]:.4f}')
                    st.write(
                        f'Total explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}')

                skew_values = self.df[self.numeric_columns].skew()
                skew_types = skew_values.apply(
                    lambda x: "Right-Skewed" if x > 1 else (
                        "Left-Skewed" if x < -1 else "Normal"))
                skew_df = pd.DataFrame(
                    {"Skewness": skew_values, "Type": skew_types})
                with st.expander("View Skewness Details"):
                    st.write(skew_df)
