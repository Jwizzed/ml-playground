import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_echarts import st_echarts
from wordcloud import WordCloud


COLORS = ["black", "red"]


def classify_columns(df):
    numeric_columns = []
    categorical_columns = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        else:
            temp_series = pd.to_numeric(df[col], errors='coerce')
            if not temp_series.isna().all():
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)

    return numeric_columns, categorical_columns


def main():
    st.sidebar.title("ML Playground")
    st.sidebar.header("Navigation")
    options = ["Exploratory Data Analysis", "Machine Learning",
               "Others"]

    choice = st.sidebar.selectbox("Go to", options)

    st.sidebar.header("About")
    st.sidebar.info("""
        ML Playground is an interactive tool designed to simplify and enhance data management tasks. It provides users with the ability to:

        - Perform Exploratory Data Analysis (EDA)
        - Establish Machine Learning Baselines
        - And much more!

        Dive into the world of data with ease and discover insights through an intuitive interface.
    """)

    if choice == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis (EDA)")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Dataframe")
            st.dataframe(df)
            numeric_columns, categorical_columns = classify_columns(df)

            tab1, tab2, tab3, tab4 = st.tabs(
                ["Summary Statistics", "Numerical Analysis",
                 "Categorical Analysis"])

            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("Missing values:",
                             df.isnull().sum().rename("Number of null"))
                with col2:
                    st.write("Data types:", df.dtypes.rename("Dtype"))
                with col3:
                    st.write("Describe:", df.describe())

                if df.duplicated().sum() > 0:
                    st.write("There are duplicated rows in the dataset.")
                else:
                    st.write("No duplicated rows in the dataset.")

                constant_columns = [col for col in df.columns if
                                    df[col].nunique() == 1]
                if constant_columns:
                    st.write(f"Constant columns: {constant_columns}")
                else:
                    st.write("No constant columns in the dataset.")

            with tab2:
                if numeric_columns:
                    st.subheader("Numerical Columns")
                    st.write(numeric_columns)

                    with st.expander("View Numerical Data Distribution"):
                        num_plots = len(numeric_columns)
                        num_cols = min(num_plots,
                                       4)
                        num_rows = num_plots // num_cols + (
                                num_plots % num_cols > 0)
                        fig, axes = plt.subplots(num_rows, num_cols, figsize=(
                            5 * num_cols,
                            4 * num_rows))

                        axes = axes.flatten() if num_plots > 1 else np.array(
                            [axes])

                        for idx, col in enumerate(numeric_columns):
                            sns.histplot(df[col], bins='auto', kde=True,
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

                        for idx, col in enumerate(numeric_columns):
                            bp = sns.boxplot(x=df[col], ax=axes[idx],
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

                        corr = df[numeric_columns].corr()

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
                        scaled_df = scaler.fit_transform(df[numeric_columns])
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(scaled_df)

                        if 'target' in df.columns and df[
                            'target'].dtype == 'object':
                            categories = df['target'].unique()

                            fig, ax = plt.subplots(figsize=(5 * num_cols,
                                                            4 * num_rows))
                            for category in categories:
                                ix = np.where(df['target'] == category)[0]
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
                        if 'target' in df.columns and df[
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

                    skew_values = df[numeric_columns].skew()
                    skew_types = skew_values.apply(
                        lambda x: "Right-Skewed" if x > 1 else (
                            "Left-Skewed" if x < -1 else "Normal"))
                    skew_df = pd.DataFrame(
                        {"Skewness": skew_values, "Type": skew_types})
                    with st.expander("View Skewness Details"):
                        st.write(skew_df)

            with tab3:
                if categorical_columns:
                    st.subheader("Categorical Columns")
                    st.write(categorical_columns)

                    with st.expander("Value Counts and Proportions"):
                        for categorical_column in categorical_columns:
                            value_counts = df[
                                categorical_column].value_counts()
                            chart_options = {
                                "tooltip": {"trigger": "item",
                                            "formatter": "{a} <br/>{b}: {c} ({d}%)"},
                                "legend": {
                                    "data": value_counts.index.tolist()},
                                "series": [{
                                    "name": categorical_column,
                                    "type": "pie",
                                    "radius": "50%",
                                    "data": [{"value": count, "name": label}
                                             for
                                             label, count in
                                             value_counts.items()]
                                }]
                            }

                            st.subheader(f"Pie Chart for {categorical_column}")
                            st_echarts(options=chart_options, height="400px")

                    with st.expander("View Categorical Data Distribution"):
                        for col in categorical_columns:
                            value_counts = df[col].value_counts()

                            # Generate a color palette for the bars
                            color_palette = sns.color_palette("husl",
                                                              len(value_counts))

                            # Define chart options
                            chart_options = {
                                "tooltip": {"trigger": "axis",
                                            "axisPointer": {"type": "shadow"}},
                                "xAxis": {"type": "category",
                                          "data": value_counts.index.tolist(),
                                          "axisLabel": {"rotate": 45}},
                                "yAxis": {"type": "value"},
                                "series": [
                                    {
                                        "name": "Count",
                                        "type": "bar",
                                        "data": [
                                            {"value": count, "name": label,
                                             "itemStyle": {
                                                 "color": color_palette[i]}}
                                            for i, (label, count) in
                                            enumerate(value_counts.items())],
                                        "label": {"show": True,
                                                  "position": "top"}
                                    }
                                ],
                            }

                            st.subheader(f"Bar Chart for {col}")
                            st_echarts(options=chart_options, height="400px")

                    if len(categorical_columns) > 1:
                        with st.expander(
                                "Cross-Tabulation Between Categorical Features"):
                            cat_col_pairs = [(categorical_columns[i],
                                              categorical_columns[j]) for i in
                                             range(len(categorical_columns))
                                             for j in range(i + 1,
                                                            len(categorical_columns))]
                            for col1, col2 in cat_col_pairs:
                                cross_tab = pd.crosstab(index=df[col1],
                                                        columns=df[col2],
                                                        normalize='index').mul(
                                    100).round(2)
                                st.write(
                                    f"**Cross-Tabulation: {col1} vs {col2}**")
                                st.dataframe(cross_tab)

                    with st.expander("Word Clouds for Text Data"):
                        for col in categorical_columns:
                            if df[col].dtype == 'object':
                                text = ' '.join(df[col].dropna())
                                wordcloud = WordCloud(width=800, height=400,
                                                      background_color='white').generate(
                                    text)
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis("off")
                                ax.set_title(f"Word Cloud for {col}")
                                st.pyplot(fig)
                                plt.close(fig)


    elif choice == "Machine Learning":
        st.write("Machine Learning section to be implemented.")

    elif choice == "Others":
        st.write("Other functionalities to be implemented.")


if __name__ == "__main__":
    main()
