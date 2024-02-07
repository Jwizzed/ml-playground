import streamlit as st
from eda_section.eda import EDAApp
from ml_section.ml import MachineLearningApp
import pandas as pd


def info():
    """Display the information about the application."""
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
    return choice


def main():
    """Main function to run the application."""
    choice = info()

    if choice == "Exploratory Data Analysis":
        uploaded_file = st.file_uploader("Upload your CSV file for EDA", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            EDAApp(df).run()
        else:
            st.warning("Please upload a CSV file to continue.")

    elif choice == "Machine Learning":
        uploaded_file = st.file_uploader(
            "Upload your CSV file for Machine Learning", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            MachineLearningApp(df=df).run()
        else:
            st.warning("Please upload a CSV file to continue.")

    elif choice == "Others":
        st.write("Other functionalities to be implemented.")


if __name__ == "__main__":
    main()
