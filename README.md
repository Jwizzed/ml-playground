# ML Playground App
The ML Playground app is an interactive web application built with Streamlit, designed to perform exploratory data analysis (EDA), machine learning (ML), and statistical analysis on uploaded datasets. This application allows users to explore their data through visualizations, statistical summaries, and machine learning models with ease.

# Features
- Exploratory Data Analysis (EDA): Visualize and summarize your datasets to understand the distribution, correlation, and patterns within your data.
- Statistical Analysis: Generate descriptive statistics, including measures of central tendency, dispersion, and shape of the dataset's distribution.
- Machine Learning: Train and compare multiple machine learning models to identify the best performer based on your dataset.
- Categorical Analysis: Analyze and visualize categorical data, including frequency counts and cross-tabulations.
- Numerical Analysis: Dive deep into numerical data with distribution plots, outlier detection, and principal component analysis (PCA).

# Prerequisites
Before you begin, ensure you have Python installed on your system. This app requires Python 3.6 or newer.

# Installation
1. Clone the repository:
```
git clone https://github.com/Jwizzed/ml-playground.git
cd ml-playground
```
2. Create and activate a virtual environment:
On macOS and Linux:
```
python3 -m venv env
source env/bin/activate
```
On Windows:
```
python -m venv env
.\env\Scripts\activate
```
3. Install the required packages:
```
pip install -r requirements.txt
```
4. Run the app:
```
streamlit run app.py
```
# Usage
After launching the app, navigate through the tabs to access different functionalities:
- Upload your dataset in CSV format to begin analysis.
- Select the type of analysis you wish to perform from the provided options.
- Configure model parameters and visualize results directly within the app.
For detailed instructions on using each feature, refer to the in-app guidance provided in each section.

# Contributing
Contributions to the ML Playground app are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

# Note
This app is deployed on a free service, so it may take some time to load initially. Please be patient and wait for the app to load.