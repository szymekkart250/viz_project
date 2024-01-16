import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px

CHOSEN_COLS = []
MATCH_ID = 3857256

match_ids = {3857256: ['Serbia', 'Switzerland'],
 3869151: ['Argentina', 'Australia'],
 3857257: ['Australia', 'Denmark'],
 3857258: ['Brazil', 'Serbia'],
 3857288: ['Tunisia', 'Australia'],
 3857267: ['Ecuador', 'Senegal'],
 3869321: ['Netherlands', 'Argentina'],
 3857287: ['Uruguay', 'South Korea'],
 3869486: ['Morocco', 'Portugal'],
 3869685: ['Argentina', 'France'],
 3857260: ['Saudi Arabia', 'Mexico'],
 3857264: ['Poland', 'Argentina'],
 3857266: ['France', 'Denmark'],
 3857289: ['Argentina', 'Mexico'],
 3857269: ['Brazil', 'Switzerland'],
 3857294: ['Netherlands', 'Qatar'],
 3869254: ['Portugal', 'Switzerland'],
 3869118: ['England', 'Senegal'],
 3869684: ['Croatia', 'Morocco'],
 3869519: ['Argentina', 'Croatia'],
 3869354: ['England', 'France'],
 3869552: ['France', 'Morocco'],
 3869420: ['Croatia', 'Brazil'],
 3869220: ['Morocco', 'Spain'],
 3869219: ['Japan', 'Croatia'],
 3869253: ['Brazil', 'South Korea'],
 3869152: ['France', 'Poland'],
 3869117: ['Netherlands', 'United States'],
 3857270: ['Portugal', 'Uruguay'],
 3857263: ['Spain', 'Germany'],
 3857259: ['Cameroon', 'Serbia'],
 3857295: ['Japan', 'Costa Rica'],
 3857283: ['Belgium', 'Morocco'],
 3857284: ['Germany', 'Japan'],
 3857282: ['United States', 'Wales'],
 3857286: ['Qatar', 'Ecuador'],
 3857301: ['Qatar', 'Senegal'],
 3857300: ['Argentina', 'Saudi Arabia'],
 3857299: ['South Korea', 'Ghana'],
 3857298: ['Portugal', 'Ghana'],
 3857297: ['Poland', 'Saudi Arabia'],
 3857296: ['Croatia', 'Belgium'],
 3857293: ['Ghana', 'Uruguay'],
 3857292: ['Costa Rica', 'Germany'],
 3857291: ['Spain', 'Costa Rica'],
 3857290: ['Switzerland', 'Cameroon'],
 3857285: ['Senegal', 'Netherlands'],
 3857281: ['Croatia', 'Canada'],
 3857280: ['Cameroon', 'Brazil'],
 3857279: ['France', 'Australia'],
 3857278: ['Iran', 'United States'],
 3857277: ['Morocco', 'Croatia'],
 3857276: ['Canada', 'Morocco'],
 3857275: ['Tunisia', 'France'],
 3857274: ['Netherlands', 'Ecuador'],
 3857273: ['Wales', 'Iran'],
  3857255: ['Japan', 'Spain'],
 3857254: ['Denmark', 'Tunisia']}
# Sample DataFrame, replace this with your actual DataFrame
# Assuming you have a CSV file called 'data.csv' with your data
# Replace 'path_to_file.csv' with the actual path to your CSV file
df = pd.read_csv('Data/FIFA World Cup 2022 Team Data/team_data.csv')

# Iterate over the columns and perform the division where applicable
for column in df.columns:
    if '90' not in column and column not in ['games', 'team']:  # Check if '90' is not in the column name and it's not the 'games' column
        df[column] = df[column] / df['games']  # Perform the division

team1, team2 = match_ids[MATCH_ID]

cols = ['goals', 'assists']

home_data = df[df['team'] == team1][cols].values[0]
away_data = df[df['team'] == team2][cols].values[0]

fig = go.Figure(data=[
    go.Bar(name=team2, x=cols, y=away_data),
    go.Bar(name=team1, x=cols, y=home_data)
])



# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns if '90' not in col],
        value=cols,  # Default value
        multi=True  # Allow multiple selections
    ),
    dcc.Graph(id='bar-chart'),
    html.H1(id='test-text'),
    html.H1(id='test-text2'),
    dcc.Graph(figure={}, id='scatter')
])

# Define the callback to update the bar chart
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('column-dropdown', 'value')]
    )
def update_chart(selected_columns):
    global team1, team2
    home_data = df[df['team'] == team1][selected_columns].values[0]
    away_data = df[df['team'] == team2][selected_columns].values[0]

    # Create an empty figure
    fig = go.Figure()
    fig = go.Figure(data=[
    go.Bar(name=team2, x=selected_columns, y=away_data),
    go.Bar(name=team1, x=selected_columns, y=home_data)
    ])
    print(f'{selected_columns}')
    # Specific row to use for the bar chart values
    # row_to_display = 0
    # Add a bar to the figure for each selected column
    # for selected_column in selected_columns:
    #     fig.add_trace(go.Bar(
    #         x=[selected_column],
    #         y=[df.at[row_to_display, selected_column] / df.at[row_to_display, 'games']],
    #         name=selected_column
    #     ))
    # Update layout if needed (e.g., titles)
    fig.update_layout(
        title='Bar Chart',
        xaxis_title='Column',
        yaxis_title='Value'
    )
    return fig


@app.callback(
#    Output('output-div', 'children'),
   Output('test-text', 'children'),
   Output('test-text2', 'children'),
    Output('scatter', 'figure'),
#    Output('scatter', 'figure'),
   Input('bar-chart', 'clickData')
)
def display_click_data(clickData):
    global team1, team2
    global CHOSEN_COLS
    if clickData is None:
       return f'{clickData}', "Click on a bar to see its details.", {}
    else:
        column = clickData["points"][0]["label"]
        if len(CHOSEN_COLS) < 2:
            CHOSEN_COLS.append(column)
            text = 'choose one more parameter'
            fig = {}

        else:
            # if CHOSEN_COLS[0] == CHOSEN_COLS[1]:
            #     return f'{value}', 'pick another column', str(CHOSEN_COLS)
            CHOSEN_COLS.append(column)
            text = 'Nice'
            CHOSEN_COLS = CHOSEN_COLS[1:]
            if CHOSEN_COLS[0] == CHOSEN_COLS[1]:
                return 'pick another column, they are the same now', None, {}

            fig = px.scatter(
                data_frame=df,
                x= CHOSEN_COLS[0],
                y=CHOSEN_COLS[1],
                hover_data=['team'],
                size='games',
                color='games' 
                )
            # team1_star = {
            #     type:'scatter',
            #     x: df[df['team'] == team1][CHOSEN_COLS[0]].values[0],
            #     y: df[df['team'] == team1][CHOSEN_COLS[1]].values[0],
            #     mode:'markers',
            #     marker:dict(symbol='star', size=10, color='red')
            # }
            star_marker1 = dict(
            type='scatter',
            x=[df[df['team'] == team1][CHOSEN_COLS[0]].values[0]],  # X-coordinate of the star marker
            y=[df[df['team'] == team1][CHOSEN_COLS[1]].values[0]],  # Y-coordinate of the star marker
            mode='markers',
            marker=dict(symbol='star', size=30, color='red'),
            name=team1,
            showlegend=False
            )            
            star_marker2 = dict(
            type='scatter',
            x=[df[df['team'] == team2][CHOSEN_COLS[0]].values[0]],  # X-coordinate of the star marker
            y=[df[df['team'] == team2][CHOSEN_COLS[1]].values[0]],  # Y-coordinate of the star marker
            mode='markers',
            marker=dict(symbol='star', size=30, color='red'),
            name=team2,
            showlegend=False
            )            
            fig.add_trace(star_marker1)
            fig.add_trace(star_marker2)


        return str(CHOSEN_COLS), f'{clickData["points"][0]["label"]}', fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)