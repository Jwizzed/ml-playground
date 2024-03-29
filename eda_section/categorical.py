import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_echarts import st_echarts
from wordcloud import WordCloud


class CategoricalAnalysis:
    """A class to represent the Categorical Analysis of the dataframe."""

    def __init__(self, df, categorical_columns, tab):
        self.df = df
        self.categorical_columns = categorical_columns
        self.tab = tab

    def display_categorical_analysis(self):
        """Display the categorical analysis of the dataframe."""
        with self.tab:
            if self.categorical_columns:
                st.subheader("Categorical Columns")

                progress_text = st.empty()
                progress_placeholder = st.empty()

                progress_text.text("Progress: 0%, Starting.. (This should take around 30 seconds to complete)")
                progress_placeholder.progress(0)

                st.write(self.categorical_columns)

                progress_placeholder.progress(25)
                progress_text.text("Progress: 25%, Plotting distribution...")

                self.display_categorical_data_distribution()
                progress_placeholder.progress(50)
                progress_text.text("Progress: 50%, Ploting cross-tabulation...")

                self.display_cross_tabulation()
                progress_placeholder.progress(75)
                progress_text.text("Progress: 75%, Plotting word clouds...")

                self.display_word_clouds()
                progress_placeholder.progress(100)
                progress_text.text("Progress: 100%, Finalizing...")

                progress_placeholder.empty()
                progress_text.empty()

    def display_categorical_data_distribution(self):
        """Display the data distribution of categorical columns."""
        with st.expander("View Categorical Data Distribution"):
            for col in self.categorical_columns:
                value_counts = self.df[col].value_counts()

                color_palette = sns.color_palette("husl", len(value_counts))

                chart_options = {
                    "tooltip": {"trigger": "axis",
                                "axisPointer": {"type": "shadow"}},
                    "xAxis": {"type": "category",
                              "data": value_counts.index.tolist(),
                              "axisLabel": {"rotate": 45}},
                    "yAxis": {"type": "value"},
                    "series": [
                        {
                            "name": "Count",
                            "type": "bar",
                            "data": [
                                {"value": count, "name": label,
                                 "itemStyle": {"color": color_palette[i]}}
                                for i, (label, count) in
                                enumerate(value_counts.items())],
                            "label": {"show": True, "position": "top"}
                        }
                    ],
                }

                st.subheader(f"Bar Chart for {col}")
                st_echarts(options=chart_options, height="400px")

    def display_cross_tabulation(self):
        """Display cross-tabulation between categorical features."""
        if len(self.categorical_columns) > 1:
            with st.expander("Cross-Tabulation Between Categorical Features"):
                cat_col_pairs = [
                    (self.categorical_columns[i], self.categorical_columns[j])
                    for i in
                    range(len(self.categorical_columns))
                    for j in range(i + 1, len(self.categorical_columns))]
                for col1, col2 in cat_col_pairs:
                    cross_tab = pd.crosstab(index=self.df[col1],
                                            columns=self.df[col2],
                                            normalize='index').mul(
                        100).round(2)
                    st.write(f"**Cross-Tabulation: {col1} vs {col2}**")
                    st.dataframe(cross_tab)

    def display_word_clouds(self):
        """Display word clouds for text data."""
        with st.expander("Word Clouds for Text Data"):
            for col in self.categorical_columns:
                if self.df[col].dtype == 'object':
                    text = ' '.join(self.df[col].dropna())
                    wordcloud = WordCloud(width=800, height=400,
                                          background_color='white').generate(
                        text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title(f"Word Cloud for {col}")
                    st.pyplot(fig)
                    plt.close(fig)
