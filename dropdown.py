from dash import Dash, dcc, html
import io
import base64
import plotly.tools as tls
from mplsoccer.pitch import Pitch
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output



app = Dash(__name__)
pitch = Pitch()
fig, ax = pitch.draw()

file_path = 'Data/FIFA World Cup 2022 Team Data/team_data.csv'
df = pd.read_csv(file_path)

# Convert the "team" column to strings
df["team"] = df["team"].astype(str)

# Select the columns you want to include in the parallel coordinates plot
cols = ["team", "goals", "assists", "goals_per_shot", "shots_on_target_pct",
                    "tackles", "interceptions", "blocks", "cards_yellow", "cards_red",
                    "gk_saves"]

app.layout = html.Div([
   dcc.Dropdown(
            id='column-picker',
            options=[{'label': col, 'value': col} for col in df.columns if '90' not in col],
            # value=cols,  # Default value
            multi=True  # Allow multiple selections
            ),
   dcc.Graph(id='pitch')
])


@app.callback(
   Output('pitch', 'figure'),
   Input('column-picker', 'value')
)
def updatePCP(columns):
   # Create dimensions for the selected columns with proper labels
    dimensions = [go.parcoords.Dimension(values=df[col], label=col.replace("_", " ").title()) for col in columns]  # Exclude the first dimension ("team")

    # Explicitly set tickvals and ticktext for the first dimension
    first_dimension = go.parcoords.Dimension(values=df.index, label="Team", tickvals=df.index, ticktext=df["team"])

    # Create a trace for parallel coordinates
    trace = go.Parcoords(
        line=dict(color=df.index, colorscale="Jet", showscale=False),
        dimensions=[first_dimension] + dimensions,
    )

    # Create layout
    layout = go.Layout(
        title="PCP for Team Statistics",
        font=dict(family="Arial", size=12),
    )
    # Create figure
    fig = go.Figure(data=[trace], layout=layout)

    # fig.update_layout(
    #     template="plotly_dark"
    # )

    return fig

if __name__ == '__main__':
   app.run_server(debug=True)