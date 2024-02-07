import streamlit as st
from eda_section.eda import EDAApp


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
        EDAApp().run()

    elif choice == "Machine Learning":
        st.write("Machine Learning section to be implemented.")

    elif choice == "Others":
        st.write("Other functionalities to be implemented.")


if __name__ == "__main__":
    main()
