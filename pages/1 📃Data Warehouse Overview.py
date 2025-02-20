import streamlit as st

# import the create_columns function from the helper.py file located in the config folder which is one level up


from config.helper import create_columns
from analytics import (
    get_heart_rate_variability_data,
    get_sleep_data,
    visualize_heart_rate_variability_data,
    visualize_sleep_score_data,
)


def my_records():
    st.title("My Records")
    st.write("Here you can view and manage your health records.")

    create_columns(
        lambda: st.subheader("Heart Rate Variability Data"),
        lambda: st.image("./assets/heart-pulse-svgrepo-com.svg", width=100),
    )

    hrv_df = get_heart_rate_variability_data()
    st.write(hrv_df.head(10))

    timeseries, heatmap = visualize_heart_rate_variability_data()

    create_columns(
        lambda: st.plotly_chart(timeseries, key="my_unique_key_for_fig1"),
        lambda: st.plotly_chart(heatmap, key="my_unique_key_for_fig2"),
    )

    st.write("---")
    st.subheader("Sleep Data")
    sleep_score_df = get_sleep_data()
    st.write(sleep_score_df.head(10))

    sleep_hm = visualize_sleep_score_data()
    st.write(sleep_hm)


my_records()
