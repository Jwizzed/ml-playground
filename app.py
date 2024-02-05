import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_echarts import st_echarts


def generate_echart(data, categories, values, color_scheme):
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
        st.title("Welcome to the Streamlib Web App")
        st.image("./assets/images/muay.png")
        st.write("Some description text...")



    elif choice == "EDA":
        st.title("Exploratory Data Analysis (EDA)")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Dataframe")
            st.dataframe(df)
            st.subheader("Summary Statistics")
            st.write(df.describe())
            st.subheader("Visualization")

            if not df.empty:
                categories = df.iloc[:, 0].tolist()
                values = df.iloc[:, 1].tolist()

                color_scheme = sns.color_palette('husl',
                                                 len(categories)).as_hex()

                chart_options = generate_echart(df, categories,
                                                values,
                                                color_scheme)

                st_echarts(options=chart_options, height="400px")

    elif choice == "ML":
        st.write("Machine Learning section to be implemented.")

    elif choice == "Others":
        st.write("Other functionalities to be implemented.")


if __name__ == "__main__":
    main()
