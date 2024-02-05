"""The module is uncleaned AT ALL."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

COLORS = ["black", "red"]

st.set_page_config(
    page_title="My App with Pink Theme",
    page_icon=":tada:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def generate_echart(categories, values, color_scheme):
    options = {
        "tooltip": {
            "trigger": 'item',
            "formatter": '{a} <br/>{b} : {c} ({d}%)'
        },
        "legend": {
            "left": 'center',
            "top": 'bottom',
            "data": categories
        },
        "toolbox": {
            "show": True,
            "feature": {
                "mark": {"show": True},
                "dataView": {"show": True, "readOnly": False},
                "restore": {"show": True},
                "saveAsImage": {"show": True}
            }
        },
        "series": [
            {
                "name": 'Radius Mode',
                "type": 'pie',
                "radius": [20, 140],
                "center": ['50%', '50%'],
                "roseType": 'radius',
                "itemStyle": {
                    "borderRadius": 5
                },
                "label": {
                    "show": False
                },
                "emphasis": {
                    "label": {
                        "show": True
                    }
                },
                "data": [
                    {"value": v, "name": c,
                     "itemStyle": {"color": color_scheme[i]}}
                    for i, (c, v) in enumerate(zip(categories, values))
                ]
            }
        ]
    }
    return options


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
    options = ["Main", "Exploratory Data Analysis", "Machine Learning",
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

    if choice == "Main":
        st.title("Welcome to ML Playground")
        st.image("./assets/images/muay.png")
        st.subheader(
            "Navigation -> Exploratory Data Analysis -> Upload csv file.")


    elif choice == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis (EDA)")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Dataframe")
            st.dataframe(df)
            numeric_columns, categorical_columns = classify_columns(df)

            tab1, tab2, tab3, tab4 = st.tabs(
                ["Summary Statistics", "Numerical Analysis",
                 "Categorical Analysis", "Visualization"])

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

                    with st.expander("View Distribution Curve Fits"):
                        fig, axes = plt.subplots(num_rows, num_cols,
                                                 figsize=(
                                                     5 * num_cols,
                                                     4 * num_rows))
                        axes = axes.flatten() if num_plots > 1 else [axes]

                        for idx, col in enumerate(numeric_columns):
                            sns.histplot(df[col], kde=True, stat="density",
                                         linewidth=0, ax=axes[idx],
                                         color="#DF2E38")
                            axes[idx].set_title(
                                f'Distribution with KDE for {col}')
                        plt.tight_layout()
                        st.pyplot(fig)

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

                    with st.expander("Feature Correlation Matrix"):

                        corr = df[numeric_columns].corr()

                        mask = np.triu(np.ones_like(corr, dtype=bool))

                        f, ax = plt.subplots(figsize=(5 * num_cols,
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

                        st.pyplot(f)

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
                    red_palette = sns.color_palette([
                        "#ff0000",
                        "#e60000",
                        "#cc0000",
                        "#b30000",
                        "#990000",
                        "#800000",
                        "#660000",
                        "#4d0000",
                        "#330000",
                        "#1a0000"
                    ])

                    with st.expander("Value Counts and Proportions"):
                        for col in categorical_columns:
                            value_counts = df[col].value_counts()
                            proportions = (
                                                      value_counts / value_counts.sum()) * 100

                            data = pd.DataFrame({'Labels': value_counts.index,
                                                 'Counts': value_counts.values,
                                                 'Proportion (%)': proportions.values})
                            data.sort_values('Counts', ascending=True,
                                             inplace=True)

                            fig, ax = plt.subplots(figsize=(8,
                                                            len(data) * 0.4))
                            bars = ax.barh(data['Labels'], data['Counts'],
                                           color=red_palette)


                            for bar, proportion in zip(bars,
                                                       data['Proportion (%)']):
                                ax.text(bar.get_width(),
                                        bar.get_y() + bar.get_height() / 2,
                                        f'{proportion:.1f}%',
                                        va='center', ha='left', fontsize=8)

                            ax.set_xlabel('Counts')
                            ax.set_title(f'Distribution for {col}')

                            st.pyplot(fig)

                    with st.expander("View Categorical Data Distribution"):
                        num_plots = len(categorical_columns)
                        num_cols = min(num_plots, 2)
                        num_rows = num_plots // num_cols + (
                                num_plots % num_cols > 0)

                        fig, axes = plt.subplots(num_rows, num_cols, figsize=(
                            10 * num_cols, 6 * num_rows))
                        axes = axes.flatten() if num_plots > 1 else np.array(
                            [axes])

                        for idx, col in enumerate(categorical_columns):
                            sns.countplot(x=df[col], ax=axes[idx], palette=red_palette)
                            axes[idx].set_title(f"Count Plot for {col}")
                            axes[idx].tick_params(axis='x', rotation=45)

                        for idx in range(num_plots, len(axes)):
                            fig.delaxes(axes[idx])

                        fig.tight_layout(pad=3.0)
                        st.pyplot(fig)

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
                                plt.figure(figsize=(10, 5))
                                plt.imshow(wordcloud, interpolation='bilinear')
                                plt.axis("off")
                                plt.title(f"Word Cloud for {col}")
                                st.pyplot(plt)

            # with tab4:
            #     if not df.empty and numeric_columns:
            #         values = [df[col].sum() or df[col].mean() for col in
            #                   numeric_columns]
            #         categories = numeric_columns
            #         color_scheme = sns.color_palette('husl',
            #                                          len(categories)).as_hex()
            #
            #         chart_options = generate_echart(categories, values,
            #                                         color_scheme)
            #
            #         st_echarts(options=chart_options, height="400px")


    elif choice == "Machine Learning":
        st.write("Machine Learning section to be implemented.")

    elif choice == "Others":
        st.write("Other functionalities to be implemented.")


if __name__ == "__main__":
    main()
