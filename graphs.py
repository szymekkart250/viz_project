import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import *

df_team_data = pd.read_csv('Data/FIFA World Cup 2022 Team Data/team_data.csv')

def update_pcp(columns):
    dimensions = [go.parcoords.Dimension(values=df_team_data[col], label=relevant_team_data_columns[col]) for col in columns]  # Exclude the first dimension ("team")
    first_dimension = go.parcoords.Dimension(values=df_team_data.index, label="Team", tickvals=df_team_data.index, ticktext=df_team_data["team"])
    # Create a trace for parallel coordinates
    trace = go.Parcoords(
    line=dict(color=df_team_data.index, colorscale="Jet", showscale=False),
    dimensions=[first_dimension] + dimensions,
    )
    # Create layout
    layout = go.Layout(
        title="PCP for Team Statistics",
        font=dict(family="Arial", size=12),
    )
    # Create figure
    fig = go.Figure(data=[trace], layout=layout)

    return fig


def update_scatter(param1, param2):

    fig = px.scatter(
        data_frame=df_team_data,
        x= param1,
        y=param2,
        hover_data=['team'],
        size='games',
        color='games',
        title=f'Tylko Legia, Ukochana Legia'
        )
    return fig

