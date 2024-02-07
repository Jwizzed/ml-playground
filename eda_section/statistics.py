import streamlit as st


class SummaryStatistics:
    """A class to represent the Summary Statistics of the dataframe."""

    def __init__(self, df, tab):
        self.df = df
        self.tab = tab
        self.col1, self.col2, self.col3 = None, None, None

    def display_summary_statistics(self):
        """Display the summary statistics of the dataframe."""
        with self.tab:
            self.col1, self.col2, self.col3 = st.columns(3)
            self.display_missing_values()
            self.display_data_types()
            self.display_descriptive_statistics()
            self.display_duplicated_rows()
            self.display_constant_columns()

    def display_missing_values(self):
        """Display the number of missing values."""
        with self.col1:
            st.write("Missing values:", self.df.isnull().sum().rename("Number of null"))

    def display_data_types(self):
        """Display the data types of columns."""
        with self.col2:
            st.write("Data types:", self.df.dtypes.rename("Dtype"))

    def display_descriptive_statistics(self):
        """Display descriptive statistics of numerical columns."""
        with self.col3:
            st.write("Describe:", self.df.describe())

    def display_duplicated_rows(self):
        """Display whether there are duplicated rows."""
        if self.df.duplicated().sum() > 0:
            st.write("There are duplicated rows in the dataset.")
        else:
            st.write("No duplicated rows in the dataset.")

    def display_constant_columns(self):
        """Display constant columns in the dataset."""
        constant_columns = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if constant_columns:
            st.write(f"Constant columns: {constant_columns}")
        else:
            st.write("No constant columns in the dataset.")
