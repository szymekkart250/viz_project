import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


app = dash.Dash(__name__)

MATCH_ID = 3857256

CHOSEN_COLS = []

df_test = pd.read_csv('Data/FIFA World Cup 2022 Team Data/team_data.csv')

df_test['passes_long'] = df_test['passes_long'] / df_test['games']
df_test['offsides'] = df_test['offsides'] / df_test['games']
df_test['fouls'] = df_test['fouls'] / df_test['games']
df_test['passes_completed'] = df_test['passes_completed'] / df_test['games']
# Sample dictionary with values assigned to dropdown options
match_ids = {
 3857256: ['Serbia', 'Switzerland'],
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

trans = {
    'xg':'xg_per90',
    'completed_passes':'passes_completed',
    'long_balls': 'passes_long',
    'total_shots': 'shots_per90',
    'offsides':'offsides',
    'fouls':'fouls'
}
# app.layout = html.Div([
    
#     # dcc.Dropdown(
#     #     id='my-dropdown',
#     #     options=[
#     #         {'label': str(value), 'value': option} for option, value in my_dict.items()
#     #     ],
#     #     value='Select a game'  # Default selected value
#     # ),
#     html.Div(id='output-container', children=[
#         dcc.Graph(figure={}, id='test-scatter')
#     ])
# ])
def name_change(name):
    if name == "Korea Republic":
        name = "South Korea"
    if name == "IR Iran":
        name = "Iran"
    return name


team1, team2 = match_ids[MATCH_ID]

home_away_list = pd.read_csv("Data/FIFA World Cup 2022 Match Data/data.csv")
home_away_list['home_team'] = home_away_list['home_team'].apply(name_change)
home_away_list['away_team'] = home_away_list['away_team'].apply(name_change)

params_radar1 = ['xg', 'completed_passes', 'long_balls', 'fouls', 'total_shots', 'offsides']
away_params = ['away_'+col for col in params_radar1]
home_params = ['home_'+col for col in params_radar1]

away_data = home_away_list[(home_away_list['home_team'] == team1) & (home_away_list['away_team'] == team2)][away_params].values[0]
home_data = home_away_list[(home_away_list['home_team'] == team1) & (home_away_list['away_team'] == team2)][home_params].values[0]

params_radar = ['Expected Goals', 'Completed passes', 'Long balls', 'Fouls', 'Shots', 'Offsides']

fig = go.Figure(data=[
    go.Bar(name=team2, x=params_radar, y=away_data,customdata=params_radar1),
    go.Bar(name=team1, x=params_radar, y=home_data, customdata=params_radar1)
])

app.layout = html.Div([
   dcc.Graph(id='my-graph', figure=fig),
   html.Div(id='output-div'),
   html.H1(id='test-text'),
   html.H1(id='test-text2'),
   dcc.Graph(figure={}, id='scatter')
])
# fig.show()

@app.callback(
   Output('output-div', 'children'),
   Output('test-text', 'children'),
   Output('test-text2', 'children'),
   Output('scatter', 'figure'),
   Input('my-graph', 'clickData')
)
def display_click_data(clickData):
    global CHOSEN_COLS
    if clickData is None:
       return f'{clickData}',"Click on a bar to see its details.", None, {}
    else:
        point_number = clickData['points'][0]['pointNumber']
        name = clickData['points'][0]['x']
        value = clickData['points'][0]['customdata']
        if len(CHOSEN_COLS) < 2:
            CHOSEN_COLS.append(value)
            text = 'choose one more parameter'
            fig = {}

        else:
            # if CHOSEN_COLS[0] == CHOSEN_COLS[1]:
            #     return f'{value}', 'pick another column', str(CHOSEN_COLS)
            CHOSEN_COLS.append(value)
            text = 'Nice'
            CHOSEN_COLS = CHOSEN_COLS[1:]
            if CHOSEN_COLS[0] == CHOSEN_COLS[1]:
                return f'{value}', 'pick another column', str(CHOSEN_COLS), {}

            fig = px.scatter(
                data_frame=df_test,
                x= trans[CHOSEN_COLS[0]],
                y=trans[CHOSEN_COLS[1]],
                hover_data=['team'],
                size='games',
                color='games'
                )
        return f'{value}', [text], str(CHOSEN_COLS), fig

if __name__ == '__main__':
    app.run_server(debug=True)
