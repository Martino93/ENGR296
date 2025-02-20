import sqlite3 as sql
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
from datetime import datetime

from config.config import DB_NAME


def get_data(query: str) -> pd.DataFrame:
    conn = sql.connect(DB_NAME)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_heart_rate_variability_data():
    q = "select * from heart_rate_variability_details limit 20;"
    return get_data(q)


def get_sleep_data():
    q = "select * from sleep_score_details limit 20;"
    return get_data(q)


def visualize_heart_rate_variability_data():
    df = get_heart_rate_variability_data()
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Create the plot
    timeseries = go.Figure()

    # Add traces for each variable
    for column in ["rmssd", "coverage", "low_frequency", "high_frequency"]:
        timeseries.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df[column], mode="lines+markers", name=column
            )
        )

    # Update layout
    timeseries.update_layout(
        title="Heart Rate Variability Metrics Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        legend_title="Metrics",
    )

    # Calculate correlation matrix
    corr = df[["rmssd", "coverage", "low_frequency", "high_frequency"]].corr()

    # Create heatmap
    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.index.values,
            y=corr.columns.values,
            colorscale="Viridis",
            showscale=False,
            zmin=-1.0,
            zmax=1.0,
        )
    )

    heatmap.update_layout(
        title="Correlation Heatmap of HRV Metrics",
        xaxis_title="Metrics",
        yaxis_title="Metrics",
    )

    return timeseries, heatmap


def visualize_sleep_score_data():
    df = get_sleep_data()
    # Convert timestamp to datetime
    # df['timestamp'] = pd.to_datetime(df['timestamp'])

    # exclude the first 2 columns
    data = df.iloc[:, 2:]
    data = data.corr()

    # Create the heatmap using Plotly
    fig = ff.create_annotated_heatmap(
        z=data.values,
        x=data.columns.tolist(),
        y=data.index.tolist(),
        annotation_text=data.round(2).values,
        colorscale="Viridis",
        showscale=False,  # This line removes the side color bar
    )

    # Update layout for better appearance
    fig.update_layout(
        title="Correlation Matrix of Sleep Data",
        xaxis_title="Variables",
        yaxis_title="Variables",
    )

    # Show the figure
    return fig


#######################
# MACHINE LEARNING ANALYTICS
#######################


# get heart
def analyze_heart_failure_data():
    q = "select * from heart_training_data_1;"
    df = get_data(q)
