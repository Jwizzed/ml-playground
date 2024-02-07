import pandas as pd
import streamlit as st

from eda_section.categorical import CategoricalAnalysis
from eda_section.numerical import NumericalAnalysis
from eda_section.statistics import SummaryStatistics

COLORS = ["black", "red"]


class EDAApp:
    """A class to represent the Exploratory Data Analysis (EDA) app."""

    def __init__(self):
        self.df = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.tab1, self.tab2, self.tab3 = None, None, None

    def run(self):
        """Run the EDA app."""
        st.title("Exploratory Data Analysis (EDA)")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            self.df = pd.read_csv(uploaded_file)
            st.subheader("Dataframe")
            st.dataframe(self.df)
            self.numeric_columns, self.categorical_columns = self.classify_columns()

            self.tab1, self.tab2, self.tab3 = st.tabs(
                ["Summary Statistics", "Numerical Analysis",
                 "Categorical Analysis"])

            summary_statistic = SummaryStatistics(self.df, self.tab1)
            summary_statistic.display_summary_statistics()

            numerical_analysis = NumericalAnalysis(self.df, self.numeric_columns, self.tab2)
            numerical_analysis.display_numerical_analysis()

            categorical_analysis = CategoricalAnalysis(self.df, self.categorical_columns, self.tab3)
            categorical_analysis.display_categorical_analysis()

    def classify_columns(self):
        """Classify the columns of the dataframe into numerical and categorical columns."""
        numeric_columns = self.df.select_dtypes(
            include=['float64', 'int64']).columns.tolist()
        categorical_columns = self.df.select_dtypes(
            include=['object']).columns.tolist()
        return numeric_columns, categorical_columns
