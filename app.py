import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_echarts import st_echarts
import matplotlib.pyplot as plt


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


def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ("Main", "EDA", "ML", "Others"))

    if choice == "Main":
        st.title("Welcome to ML Playground")
        st.image("./assets/images/muay.png")
        st.write("What is ML Playground? ML Playground is a tool designed to simplify data management tasks like")
        st.write("- Exploratory Data Analysis")
        st.write("- Machine Learning Baseline")


    elif choice == "EDA":
        st.title("Exploratory Data Analysis (EDA)")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Dataframe")
            st.dataframe(df)

            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Missing values:", df.isnull().sum())
            with col2:
                st.write("Data types:", df.dtypes)
            with col3:
                st.write("Describe:", df.describe())

            st.subheader("Specify Column Types")
            all_columns = df.columns.tolist()
            numeric_columns = st.multiselect("Select Numerical Columns",
                                             all_columns, default=None)
            categorical_columns = [col for col in all_columns if
                                   col not in numeric_columns]

            if numeric_columns:
                st.write("Selected Numerical Columns:", numeric_columns)
                num_plots = len(numeric_columns)
                num_cols = 4
                num_rows = num_plots // num_cols + (num_plots % num_cols > 0)

                fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6 * num_rows))
                axes = axes.flatten()

                for idx, col in enumerate(numeric_columns):
                    sns.histplot(df[col], bins='auto', kde=True, ax=axes[idx])
                    axes[idx].set_title(f'Histogram for {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')

                for idx in range(num_plots, num_rows * num_cols):
                    fig.delaxes(axes[idx])

                fig.tight_layout()
                st.pyplot(fig)

                skew_values = df[numeric_columns].skew()
                st.write("Skewness of the numerical columns:", skew_values)

            if categorical_columns:
                st.write("Selected Categorical Columns:", categorical_columns)
                num_plots = len(categorical_columns)
                num_cols = 2
                num_rows = num_plots // num_cols + (num_plots % num_cols > 0)

                fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6 * num_rows))
                axes = axes.flatten()

                for idx, col in enumerate(categorical_columns):
                    value_counts = df[col].value_counts()
                    sns.barplot(x=value_counts.index, y=value_counts,
                                ax=axes[idx])
                    axes[idx].set_title(f"Value Counts for {col}")
                    axes[idx].tick_params(axis='x',
                                          rotation=45)
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Counts')

                for idx in range(num_plots, num_rows * num_cols):
                    fig.delaxes(axes[idx])

                fig.tight_layout(pad=3.0)
                st.pyplot(fig)

            st.subheader("Visualization")
            if not df.empty and numeric_columns:
                values = [df[col].sum() or df[col].mean() for col in
                          numeric_columns]
                categories = numeric_columns
                color_scheme = sns.color_palette('husl',
                                                 len(categories)).as_hex()

                chart_options = generate_echart(categories, values,
                                                color_scheme)

                st_echarts(options=chart_options, height="400px")


    elif choice == "ML":
        st.write("Machine Learning section to be implemented.")

    elif choice == "Others":
        st.write("Other functionalities to be implemented.")


if __name__ == "__main__":
    main()
