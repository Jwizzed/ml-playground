import streamlit as st


class SummaryStatistics:
    """A class to represent the Summary Statistics of the dataframe."""

    def __init__(self, df, tab):
        self.df = df
        self.tab = tab

    def display_summary_statistics(self):
        """Display the summary statistics of the dataframe."""
        with self.tab:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Missing values:",
                         self.df.isnull().sum().rename("Number of null"))
            with col2:
                st.write("Data types:", self.df.dtypes.rename("Dtype"))
            with col3:
                st.write("Describe:", self.df.describe())

            if self.df.duplicated().sum() > 0:
                st.write("There are duplicated rows in the dataset.")
            else:
                st.write("No duplicated rows in the dataset.")

            constant_columns = [col for col in self.df.columns if
                                self.df[col].nunique() == 1]
            if constant_columns:
                st.write(f"Constant columns: {constant_columns}")
            else:
                st.write("No constant columns in the dataset.")
