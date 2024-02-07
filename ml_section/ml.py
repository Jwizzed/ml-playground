import streamlit as st
from pycaret.classification import setup as setup_class, \
    compare_models as compare_models_class, pull as pull_class
from pycaret.regression import setup as setup_reg, \
    compare_models as compare_models_reg, pull as pull_reg


class MachineLearningApp:
    """Class to display the Machine Learning section of the application."""

    def __init__(self, df):
        self.df = df

    def run(self):
        """Method to run the Machine Learning section."""
        st.title("Machine Learning")

        tab1, tab2 = st.tabs(["Classification", "Regression"])

        with tab1:
            self.run_classification()

        with tab2:
            self.run_regression()

    def run_classification(self):
        """Run the classification section of the application."""
        chosen_target = st.selectbox('Select Your Target for Classification',
                                     self.df.columns, key='class_target')

        if st.button('Train Classification Model', key='class_train'):
            progress_bar = st.progress(0)
            st.write("Running setup for classification... Please wait.")
            progress_bar.progress(10)

            setup_class(data=self.df, target=chosen_target, html=False,
                        verbose=False)
            progress_bar.progress(50)
            st.write("Setup complete.")

            st.markdown("### Classification Experiment Settings")
            setup_df = pull_class()
            st.dataframe(setup_df.iloc[:, :2])

            st.write("Comparing classification models... Please wait. (Might take a few minutes)")
            progress_bar.progress(
                75)

            best_model = compare_models_class()
            progress_bar.progress(
                100)
            st.write("Model comparison complete.")

            st.markdown("### Best Classification Model")
            compare_df = pull_class()
            st.dataframe(compare_df)

            st.write("Best Model:")
            st.write(best_model)
            progress_bar.empty()

    def run_regression(self):
        """Run the regression section of the application."""
        chosen_target = st.selectbox('Select Your Target for Regression',
                                     self.df.columns, key='reg_target')

        if st.button('Train Regression Model', key='reg_train'):
            progress_bar = st.progress(0)
            st.write("Running setup for regression... Please wait.")
            progress_bar.progress(10)

            setup_reg(data=self.df, target=chosen_target, html=False,
                      verbose=False)
            progress_bar.progress(50)
            st.write("Setup complete.")

            st.markdown("### Regression Experiment Settings")
            setup_df = pull_reg()
            st.dataframe(setup_df.iloc[:, :2])

            st.write("Comparing regression models... Please wait. (Might take a few minutes)")
            progress_bar.progress(75)

            best_model = compare_models_reg()
            progress_bar.progress(100)
            st.write("Model comparison complete.")

            st.markdown("### Best Regression Model")
            compare_df = pull_reg()
            st.dataframe(compare_df)

            st.write("Best Model:")
            st.write(best_model)
            progress_bar.empty()
