import base64
from io import BytesIO
from flask import Flask
import dash
from dash import dcc, html, dash_table, no_update, callback
import plotly.express as px
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd 
from utils import * 
import plotly.graph_objects as go
from mplsoccer import Pitch, VerticalPitch, Radar, FontManager, grid
import ast
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import re

highlighted_cols = {}

URL1 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
        'SourceSerifPro-Regular.ttf')
serif_regular = FontManager(URL1)
URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
        'SourceSerifPro-ExtraLight.ttf')
serif_extra_light = FontManager(URL2)
URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/'
        'RubikMonoOne-Regular.ttf')
rubik_regular = FontManager(URL3)
URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
robotto_thin = FontManager(URL4)
URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
        'RobotoSlab%5Bwght%5D.ttf')
robotto_bold = FontManager(URL5)

df = pd.read_csv('events.csv')


def name_change(name):
    if name == "Korea Republic":
        name = "South Korea"
    if name == "IR Iran":
        name = "Iran"
    return name

def make_pass_df(df):
    params = ['match_id','location', 'minute', 'pass_end_location', 'type', 'pass_recipient', 'player', 'pass_body_part', 'second', 'team_id', 'team']
    df = df[params]
    df['team'] = df['team'].apply(name_change)
    df_p = df[df['type'] == 'Pass']
    df_p['location'] = df_p['location'].apply(ast.literal_eval)
    df_p[['x', 'y']] = df_p['location'].tolist()
    df_p['x'] = df_p['x'] * 5/6
    df_p['y'] = df_p['y'] * 1.25
    df_p['pass_end_location'] = df_p['pass_end_location'].apply(ast.literal_eval)
    df_p[['x_end', 'y_end']] = df_p['pass_end_location'].tolist()
    df_p['x_end'] = df_p['x_end'] * 5/6
    df_p['y_end'] = df_p['y_end'] * 1.25
    df_p['outcome'] = df_p['pass_recipient'].apply(lambda x: 1 if not pd.isna(x)  else 0)
    df_p['color'] = df_p['outcome'].apply(lambda x: 'blue' if x == 1 else 'orange')
    return df_p

def make_carry_df(df):
    # df = pd.read_csv('events.csv')
    params = ['match_id','location', 'minute', 'carry_end_location', 'type', 'pass_recipient', 'player', 'pass_body_part', 'second', 'team_id', 'team']
    df = df.copy()[params]
    df['team'] = df['team'].apply(name_change)
    df_p = df[df['type'] == 'Carry']
    df_p['location'] = df_p['location'].apply(ast.literal_eval)
    df_p[['x', 'y']] = df_p['location'].tolist()
    df_p['x'] = df_p['x'] * 5/6
    df_p['y'] = df_p['y'] * 1.25
    df_p['carry_end_location'] = df_p['carry_end_location'].apply(ast.literal_eval)
    df_p[['x_end', 'y_end']] = df_p['carry_end_location'].tolist()
    df_p['x_end'] = df_p['x_end'] * 5/6
    df_p['y_end'] = df_p['y_end'] * 1.25
    df_p['outcome'] = df_p['pass_recipient'].apply(lambda x: 1 if not pd.isna(x)  else 0)
    df_p['color'] = df_p['outcome'].apply(lambda x: 'blue' if x == 1 else 'blue')
    return df_p

def make_shot_df(df):
    # df = pd.read_csv('events.csv')
    params = ['match_id','location', 'minute', 'shot_end_location', 'type', 'shot_outcome', 'player', 'pass_body_part', 'second', 'team_id', 'team', 'pass_recipient','shot_statsbomb_xg']
    df = df.loc[:,params]
    df['team'] = df['team'].apply(name_change)
    df_p = df[df['type'] == 'Shot']
    # return df_p
    df_p['location'] = df_p['location'].apply(ast.literal_eval)
    df_p[['x', 'y']] = df_p['location'].tolist()
    # return df_p
    df_p['x'] = df_p['x'] * 5/6
    df_p['y'] = df_p['y'] * 1.25
    df_p['shot_end_location'] = df_p['shot_end_location'].apply(ast.literal_eval)
    df_p[['x_end', 'y_end']] = df_p['shot_end_location'].apply(lambda x: pd.Series([x[0], x[1]]))
    df_p['x_end'] = df_p['x_end'] * 5/6
    df_p['y_end'] = df_p['y_end'] * 1.25
    df_p['outcome'] = df_p['shot_outcome'].apply(lambda x: 1 if x == 'Goal' else 0)
    df_p['color'] = df_p['outcome'].apply(lambda x: 'blue' if x == 1 else 'orange')
    return df_p

df_pass, df_carry, df_shot = make_pass_df(df), make_carry_df(df), make_shot_df(df)

def make_name_label(row):
    if row['type'] == 'Pass':
        return f'{row["player"]} to {row["pass_recipient"]} ({format_minute(str(row["minute"]),str(row["second"]))})'
    if row['type'] == 'Carry':
        return f'Carry from ({round(row["x"], 2)}, {round(row["y"], 2)}) to ({round(row["x_end"], 2)}, {round(row["y_end"], 2)})'
    if row['type'] == 'Shot':
        return f'shot outcome: {row["shot_outcome"]} in {format_minute(str(row["minute"]),str(row["second"]))}'

def plot_passmap(match_id, player, df):
    temp = df.copy()

    df_3857256 = temp[(temp['match_id'] == match_id) & (temp['player'] == player)]

    print(df_3857256.head(2))
    # Create a figure
    fig = go.Figure()

    if not df_3857256['type'].empty:
        type = df_3857256['type'].iloc[0]
        fig.update_layout(title_text=f'{player} {type} map')
    else:
        fig.update_layout(title_text=f'{player} did not have an event of this kind')
        type = None

    if type is not None:
        trace_blue = go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(color='blue', size=10),
            showlegend=True,
            name=f"Successful {type}"
        )

        trace_red = go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(color='orange', size=10),
            showlegend=True,
            name=f"Unsuccessful {type}"
        )


        # Create figure
        # fig.add_trace(go.Figure(data=[trace_blue, trace_red]))

    # Vertical Pitch Outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="white"), layer='below')

    # Top Penalty Area
    fig.add_shape(type="rect", x0=30, y0=88, x1=70, y1=100, line=dict(color="white"), layer='below')

    # Bottom Penalty Area
    fig.add_shape(type="rect", x0=30, y0=0, x1=70, y1=12, line=dict(color="white"), layer='below')

    # Top 6-yard Box
    fig.add_shape(type="rect", x0=44, y0=94, x1=56, y1=100, line=dict(color="white"), layer='below')

    # Bottom 6-yard Box
    fig.add_shape(type="rect", x0=44, y0=0, x1=56, y1=6, line=dict(color="white"), layer='below')

    # Middle line
    fig.add_shape(type='line', x0=0, y0=50, x1=100, y1=50, line=dict(color='white'), layer='below')

    # Centre Circle
    fig.add_shape(type="circle", xref="x", yref="y", x0=40, y0=45, x1=60, y1=55, line_color="white", layer='below')

    # Centre Spot
    fig.add_trace(go.Scatter(x=[50], y=[50], mode='markers', marker=dict(color='white', size=5), hoverinfo='none', showlegend=False))

    # Top Penalty Spot
    fig.add_trace(go.Scatter(x=[50], y=[89], mode='markers', marker=dict(color='white', size=5), hoverinfo='none', showlegend=False))

    # Bottom Penalty Spot
    fig.add_trace(go.Scatter(x=[50], y=[11], mode='markers', marker=dict(color='white', size=5), hoverinfo='none', showlegend=False))

    # Hide axis      
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False)
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    
    for index, row in df_3857256.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['y'], row['y_end']],  # Swap x and y
            y=[row['x'], row['x_end']],  # Swap x and y
            mode='lines+markers',
            line={'color':row['color']},
            # name=f'{row["pass_recipient"]}',
            marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"),
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=df_3857256['y'],  # Swap x and y
        y=df_3857256['x'],  # Swap x and y
        mode='markers',
        marker=dict(size=5, symbol='circle', color=df_3857256['color']),
        text=df_3857256.apply(make_name_label, axis=1),
        showlegend=False
    ))

    # type = df_3857256.iloc[0]
    # print(type)

    # Adjust the layout
    fig.update_layout(width=490, height=620)
    fig.update_xaxes(range=[-5, 105], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-2, 102], showgrid=False, zeroline=False)
    fig.update_layout(
    margin=dict(l=20, r=20, t=40, b=20),
    paper_bgcolor="LightSteelBlue",
    )

    return fig

def swap(x, y):
    c = None
    c = y.copy()
    y = x
    x = c 
    return x, y

def format_minute(minutes, seconds):
    if len(seconds) == 1:
        time = f"{minutes}:0{seconds}"
    else:
        time = f"{minutes}:{seconds}"
    return time

def icon(text, id):
    widget = html.Div([
        html.I(className="fas fa-info-circle", id=id),
        dbc.Tooltip(
           children=[text],
            target=id,
        ),
    ], style={'float':'right'})

    return widget

init_columns = ['goals', 'assists', 'touches']

# Initialize Flask server
server = Flask(__name__)

plt.switch_backend('Agg')

# Initialize Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/', external_stylesheets=[dbc.themes.BOOTSTRAP, "https://use.fontawesome.com/releases/v5.8.1/css/all.css"])

def name_change(name):
    if name == "Korea Republic":
        name = "South Korea"
    if name == "IR Iran":
        name = "Iran"
    return name

# make pcp plot
df_team_data = pd.read_csv('Data/FIFA World Cup 2022 Team Data/team_data.csv')

def update_pcp(columns):
    dimensions = [go.parcoords.Dimension(values=df_team_data[col], label=relevant_team_data_columns[col]) for col in columns]  # Exclude the first dimension ("team")
    first_dimension = go.parcoords.Dimension(values=df_team_data.index, label="Team", tickvals=df_team_data.index, ticktext=df_team_data["team"])
    # Create a trace for parallel coordinates
    trace = go.Parcoords(
        line=dict(showscale=False, color=df_team_data.index, colorscale="Jet"),
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

# Function to create the PCP with highlighted line
def create_pcp(columns, highlighted_country=None):
    df = df_team_data.copy()
    df['Color'] = df['team'].apply(lambda x: 1 if x == highlighted_country else 0)
    dimensions = [go.parcoords.Dimension(values=df[col], label=relevant_team_data_columns[col]) for col in columns]  # Exclude the first dimension ("team")
    first_dimension = go.parcoords.Dimension(values=df.index, label="Team", tickvals=df.index, ticktext=df["team"])

    fig = go.Figure(
        data=go.Parcoords(
            line= dict(color=df['Color'],colorscale=[[0, 'blue'], [1, 'red']], showscale=False),
            dimensions=[first_dimension] + dimensions
        )
    )

    return fig

def home_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            icon('The graph allows to compare teams over all games in the tournament. Click on the dropdown menu to find your favourite metrics.', id='pcp')
                        ])
                    ]),
                    dbc.Row(
                        dcc.Dropdown(
                        id='column-picker',
                        options=[{'label': relevant_team_data_columns[col], 'value': col} for col in df_team_data.columns if relevant_team_data_columns.get(col) is not None],
                        value=init_columns,  # Default value
                        multi=True  # Allow multiple selections
                    )),
                    dbc.Row([
                        dcc.Graph(id='feature-graph', style={'height': '600px'})
                    ])
            ])), width=6),
            dbc.Col(dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            icon('pick three metrics which will define the scatter plot, first on is x-axis value, second is y-axis and the third is a size of the dot. Colors indicate the ', id='scatter')
                        ])
                    ]),
                    html.Div(
                        dash_table.DataTable(
                            id='data-table',
                            columns=[{'name': col, 'id': col} for col in ['team'] + init_columns],
                            data=df_team_data.to_dict('records'),
                            row_selectable='single',
                            sort_action="native"
                        ),
                        style={'height': '630px', 'overflow-y': 'auto'}  # Set a fixed height and make it scrollable
                    )
            ])), width=6)
        ], style={'padding-top':'2rem'}),
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        icon('pick three metrics which will define the scatter plot, first on is x-axis value, second is y-axis and the third is a size of the dot. Colors indicate the ', id='scatter1'),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(
                                id='scatter1',
                                options=[{'label': relevant_team_data_columns[col], 'value': col} for col in df_team_data.columns if relevant_team_data_columns.get(col) is not None],
                                value=init_columns[0],  # Default value
                            )),
                            dbc.Col(dcc.Dropdown(
                                id='scatter2',
                                options=[{'label': relevant_team_data_columns[col], 'value': col} for col in df_team_data.columns if relevant_team_data_columns.get(col) is not None],
                                value=init_columns[1],  # Default value
                            ))
                        ]),
                        dbc.Row([
                            dcc.Graph(id='scatter', style={'height': '600px'})
                        ])
                    ]),
                    className="mt-3",  # Add margin-top for spacing
                    style={"border": "1px solid #ddd"}  # Add a border for distinction
                )
            ], width=6),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        icon('pick three metrics which will define the scatter plot, first on is x-axis value, second is y-axis and the third is a size of the dot. Colors indicate the ', id='radar'),

                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    id='country-1',
                                    options=[{'label': col, 'value': col} for col in countries_world_cup_2022],
                                    value='Poland'
                                ), 
                                dcc.Dropdown(
                                    id='country-2',
                                    options=[{'label': col, 'value': col} for col in countries_world_cup_2022],
                                    value='Qatar'
                                ) 
                            ]),
                        ]),
                        dbc.Row([
                            dcc.Dropdown(
                                id='radar-picker',
                                options=[{'label': col, 'value': col} for col in df_team_data.columns if relevant_team_data_columns.get(col) is not None],
                                value=init_columns,  # Default value
                                multi=True,  # Allow multiple selections
                                style={'margin-bottom':'1em'}
                        )]),
                        dbc.Row([
                            html.Img(id='radar-chart')
                        ])
                    ], style={'padding-top':'1em'})
                )
            ], width=6, style={'padding-top':'1em'})
        ])
    ])

def matches_layout():
    return dbc.Container([
        dbc.Row([
            ### Dropdown with matches
            dcc.Dropdown(
                id='match-dropdown',
                options=[{'label': f"{value[0]} vs. {value[1]}", 'value': option} for option, value in my_dict.items()],
                value= [option for option, value in my_dict.items()][0],
                searchable=True,
                style={
                    'color': '#36a2cc',
                    'cursor': 'pointer',
                    'text-align': 'center',
                    'font-size': '14px',
                    'align': 'center',
                    'height':'30px',
                    'margin-top':'1em'
                },
            )
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.H2(id='home-name', style={'transform': 'rotate(270deg)', 'margin':'0'})
                    ], width=1, style={'display':'flex', 'flex-flow':'row', 'align-items':'center', 'justify-content':'center'}),
                    dbc.Col([
                        icon('average positions of the players and most frequent connections on the pitch. Arrow size denotes the amount of passes between players.', id='network1'),
                        html.Img(id='passnet-home')
                    ], width=3),
                    dbc.Col([
                        icon('average positions of the players and most frequent connections on the pitch. Arrow size denotes the amount of passes between players.', id='heatmap1'),
                        html.Img(id='heatmap-home')
                    ], width=3),
                    dbc.Col([
                        icon('average positions of the players and most frequent connections on the pitch. Arrow size denotes the amount of passes between players.', id='shots1'),
                        html.Img(id='shotmap-home')
                    ], width=3)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H2(id='away-name', style={'transform': 'rotate(270deg)', 'margin':'0'})
                    ], width=1, style={'display':'flex', 'flex-flow':'row', 'align-items':'center', 'justify-content':'center'}),
                    dbc.Col([
                        icon('average positions of the players and most frequent connections on the pitch. Arrow size denotes the amount of passes between players.', id='network2'),
                        html.Img(id='passnet-away')
                    ], width=3),
                    dbc.Col([
                        icon('average positions of the players and most frequent connections on the pitch. Arrow size denotes the amount of passes between players.', id='heatmap2'),
                        html.Img(id='heatmap-away')
                    ], width=3),
                    dbc.Col([
                        icon('average positions of the players and most frequent connections on the pitch. Arrow size denotes the amount of passes between players.', id='shots2'),
                        html.Img(id='shotmap-away')
                    ], width=3)
                ]),
            ], width=7),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        dbc.RadioItems(id='type-select', 
                            options=[
                            {'label':'Pass', 'value':'Pass'},
                            {'label':'Shot', 'value':'Shot'},
                            {'label':'Carry', 'value':'Carry'}
                            ],
                            # style={'display': 'inline-block'},
                            value='Pass',
                            inline=True
                        ),
                        dcc.Dropdown(
                            id='player-dropdown',
                            value=None,
                            placeholder='Select a player'
                        ),
                        dbc.Row([
                            dcc.Graph(figure={}, id='pass-map')
                        ], justify='center')
                    ], className='text-center')
                )
            ], width=5)
        ], style={'padding-top':'2rem'})
    ])

# Define the layout of the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(dbc.NavLink("Matches", href="/matches")),
        ],
        brand="CWKS",
        brand_href="/",
        color="success",
        dark=True,
    ),
    html.Div(id='page-content')
])

@app.callback(
    Output('feature-graph', 'figure'),
    [Input('column-picker', 'value')]
)
def update_graph(selected_feature):
    global highlighted_cols 
    highlighted_cols = {}
    return update_pcp(selected_feature)

# Define callback to update page content
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    global highlighted_cols
    if pathname == '/':
        return home_layout()
    else:
        highlighted_cols = []
        return matches_layout()
    
@app.callback(
    Output('scatter', 'figure'),
    Input('scatter1', 'value'),
    Input('scatter2', 'value'),
    Input('country-1', 'value'),
    Input('country-2', 'value')
)
def scatter(param1, param2, team1, team2):
    # Create a new color column based on team1 and team2
    df_team_data['color'] = '#808080'  # Default color grey
    df_team_data.loc[df_team_data['team'] == team1, 'color'] = '#0000FF'  # Blue
    df_team_data.loc[df_team_data['team'] == team2, 'color'] = '#FFA500'  # Orange

    # fig = px.scatter(
    #     data_frame=df_team_data,
    #     x=param1,
    #     y=param2,
    #     size=param3,
    #     title='Scatter plot for team statistics'
    # )
    fig = go.Figure()

    # Define the size to stage mapping
    size_to_stage = {
        7: 'Top 4',
        5: 'Quarterfinal',
        4: 'Round of 16',
        3: 'Group Stage'
    }

    # Create traces for each stage
    for size, stage in size_to_stage.items():
        # Filter the DataFrame for each stage
        stage_df = df_team_data[df_team_data['games'] == size]

        # Add a scatter trace for this stage
        fig.add_trace(go.Scatter(
            x=stage_df[param1], 
            y=stage_df[param2], 
            mode='markers',
            marker=dict(size=size * 4, color=stage_df['color']),
            name=stage,
            text=stage_df['team'] + '<br>' + param1 + ': ' + stage_df[param1].astype(str) + '<br>' + param2 + ': ' + stage_df[param2].astype(str),
            hoverinfo='text'
        ))

    # Update layout
    fig.update_layout(
        title='Scatter plot for team statistics',
        xaxis_title=param1,
        yaxis_title=param2
    )

    # Update marker colors
    # fig.update_traces(marker=dict(color=df_team_data['color']))

    return fig

@app.callback(
    Output(component_id='passnet-home', component_property='src'),
    Output(component_id='passnet-away', component_property='src'),
    Output(component_id='heatmap-home', component_property='src'),
    Output(component_id='heatmap-away', component_property='src'),
    Output(component_id='shotmap-home', component_property='src'),
    Output(component_id='shotmap-away', component_property='src'),
    Output(component_id='home-name', component_property='children'),
    Output(component_id='away-name', component_property='children'),
    Input('match-dropdown', 'value')
)
def make_pass_nets(match_num):
    team1, team2 = my_dict[match_num]
    # print(team1, team2)

    def make_pass_network_home(match_id, thresh = 3):
        team1, team2 = my_dict[match_num]
        minute, second = df[(df['match_id'] == match_id) & (df['type'] == "Substitution")][['minute', 'second']].iloc[0].tolist()
        team_name = team1

        df_nonsub = df_pass[(df_pass['minute'] < minute) & (df_pass['match_id'] == match_id) & (df_pass['team'] == team1)]
        df_nonsub['x'], df_nonsub['y'] = df_nonsub['x'] * 6/5, df_nonsub['y'] * .8
        df_nonsub['x_end'], df_nonsub['y_end'] = df_nonsub['x_end'] * 6/5, df_nonsub['y_end'] * .8
        avg_pos = df_nonsub.groupby('player')[['x', 'y']].agg({
        'x':['mean'],
        'y':['mean', 'count']
        })
        avg_pos.columns = ['x', 'y', 'count']
        pass_between = df_nonsub.groupby(['player', 'pass_recipient']).count().reset_index()
        pass_between = pass_between[['player', 'pass_recipient', 'x']]
        pass_between.columns = ['passer', 'recipient', 'count']
        x = pd.merge(pass_between, avg_pos, left_on='passer', right_index=True)
        x = pd.merge(x, avg_pos, left_on='recipient', right_index=True, suffixes=['', '_end'])
        df_plot = x[x['count_x'] > thresh]
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white')
        fig, ax = pitch.draw(figsize=(2, 3), constrained_layout=False, tight_layout=True)

        for index, row in df_plot.iterrows():
            pitch.arrows(1.2 * row['x'], .8 * row['y'], 1.2 * row['x_end'], .8 *row['y_end'], ax=ax, color= 'blue', alpha=.8, headwidth = 6, width=row['count_x']/6)
        
        nodes = pitch.scatter(1.2 * avg_pos.x, .8 * avg_pos.y, s=100, alpha = .4, ax=ax)
        buf = BytesIO()

        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
        
        return fig_bar_matplotlib, team_name
    
    def make_pass_network_away(match_id, thresh = 3):
        team1, team2 = my_dict[match_num]
        minute, second = df[(df['match_id'] == match_id) & (df['type'] == "Substitution")][['minute', 'second']].iloc[0].tolist()
        # team_id = df_pass[(df_pass['team'] == team2)]['team_id'].unique()
        # print(id_to_team[team_id])
        # print(id_to_team[team_to_id[team2]])
        team_name = team2

        df_nonsub = df_pass[(df_pass['minute'] < minute) & (df_pass['match_id'] == match_id) & (df_pass['team'] == team2)]
        df_nonsub['x'], df_nonsub['y'] = df_nonsub['x'] * 6/5, df_nonsub['y'] * 0.8
        df_nonsub['x_end'], df_nonsub['y_end'] = df_nonsub['x_end'] * 6/5, df_nonsub['y_end'] * 0.8
        avg_pos = df_nonsub.groupby('player')[['x', 'y']].agg({
        'x':['mean'],
        'y':['mean', 'count']
        })
        avg_pos.columns = ['x', 'y', 'count']
        pass_between = df_nonsub.groupby(['player', 'pass_recipient']).count().reset_index()
        pass_between = pass_between[['player', 'pass_recipient', 'x']]
        pass_between.columns = ['passer', 'recipient', 'count']
        x = pd.merge(pass_between, avg_pos, left_on='passer', right_index=True)
        x = pd.merge(x, avg_pos, left_on='recipient', right_index=True, suffixes=['', '_end'])
        df_plot = x[x['count_x'] > thresh]
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white')
        fig, ax = pitch.draw(constrained_layout=False, tight_layout=True, figsize=(2, 3))
        
        for index, row in df_plot.iterrows():
            pitch.arrows(1.2 * row['x'], .8 * row['y'], 1.2 * row['x_end'], .8 * row['y_end'], ax=ax, color= 'blue', alpha=.8, headwidth = 6, width=row['count_x']/6)
        nodes = pitch.scatter(1.2 * avg_pos['x'], .8 * avg_pos['y'], s=50, alpha = .4, ax=ax)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
        
        return fig_bar_matplotlib, team_name
    
    def make_heatmaps(match_id):
        team1, team2 = my_dict[match_num]
        match_df_a = df_pass[(df_pass['match_id'] == match_id) & (df_pass['team'] == team1)]
        match_df_b = df_pass[(df_pass['match_id'] == match_id) & (df_pass['team'] == team2)]

        # Heatmap for Team a
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white', line_zorder=2)
        fig, ax = pitch.draw(constrained_layout=False, tight_layout=True, figsize=(2, 3))
        bin_statistic = pitch.bin_statistic(match_df_a['x'] * 6/5 , match_df_a['y'] * 4/5, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_heatmap_1 = f'data:image/png;base64,{fig_data}'

        # Heatmap for Team b
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white', line_zorder=2)
        fig, ax = pitch.draw(constrained_layout=False, tight_layout=True, figsize=(2, 3))
        bin_statistic = pitch.bin_statistic(match_df_b['x'] * 6/5, match_df_b['y'] * 4/5, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_heatmap_2 = f'data:image/png;base64,{fig_data}'
        return fig_heatmap_1, fig_heatmap_2
    
    def make_shot_map(match_num, team):
        # Filter DataFrame for events related to shots from the selected team and match_id
        df_team_shots = df_shot[(df_shot['team'] == team) & (df_shot['match_id'] == match_num)]
        df_team_shots['x'], df_team_shots['y'] = df_team_shots['x'] * 6/5, df_team_shots['y'] * .8

        # Set up the mpl soccer pitch
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white', line_zorder=2)
        fig, ax = pitch.draw(constrained_layout=False, tight_layout=True, figsize=(2, 3))
        pitch.draw(ax=ax)

        # Iterate over shots in the match
        for index, row in df_team_shots.iterrows():
            # Check shot outcome and set color accordingly
            if pd.notna(row['shot_outcome']):
                if row['shot_outcome'] == 'Goal':
                    color = '#ff0000'
                    marker = '*'  # Square for goals
                elif row['shot_outcome'] == 'Saved':
                    color = 'blue'
                    marker = 'o'  # Circle for saved shots
                else:
                    color = 'orange'
                    marker = 'o'  # Default to red circle for other cases

                scatter_team_shots = pitch.scatter(row['x'], row['y'],
                                                s=row['shot_statsbomb_xg']*1000, alpha=0.5,
                                                color=color, marker=marker, ax=ax)
                
        buf = BytesIO()

        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
        # Show the plot
        return fig_bar_matplotlib

    home_fig, home_name = make_pass_network_home(match_num)
    fig_away, away_name = make_pass_network_away(match_num)
    heatmap_home, heatmap_away = make_heatmaps(match_num)
    shot_map1, shot_map2 = make_shot_map(team=team1, match_num=match_num), make_shot_map(team=team2, match_num=match_num)

    return home_fig, fig_away, heatmap_home, heatmap_away, shot_map1, shot_map2, home_name, away_name

@app.callback(
    Output('player-dropdown', 'options'),
    Input('match-dropdown', 'value')
)
def update_item_dropdown(match_id):
    if match_id is None:
        # If no category is selected, return empty options for the item dropdown
        return [], None, {'display': 'none'}
    else:
        items = df_pass[df_pass['match_id'] == match_id][['player', 'team']].values.tolist()
        df_temp = pd.DataFrame(items)
        df_tuples = df_temp.apply(tuple, axis=1)
        unique_tuples = df_tuples.unique()
        items = [list(tup) for tup in unique_tuples]

        # [options.append(x) for x in items if x not in options]        # items = list(set(items))
        return [{'label': f'{item[0]} ({item[1]})', 'value': item[0]} for item in items]

@app.callback(
    Output('pass-map', 'figure'),
    Input('match-dropdown', 'value'),
    Input('player-dropdown', 'value'),
    Input('type-select', 'value')
)
def make_pitch_traces(match_id, player, type):
    print(player, match_id, type)
    if type == 'Pass':
        return plot_passmap(match_id=match_id, player=player, df=df_pass)
    elif type == 'Carry':
        return plot_passmap(match_id=match_id, player=player, df=df_carry)
    if type == 'Shot':
        return plot_passmap(match_id=match_id, player=player, df=df_shot)

@app.callback(
    # Output('highlight-info', 'children'),
    Output('data-table', 'columns'),
    Output('data-table', 'data'),
    [Input('feature-graph', 'restyleData')],
    [Input('column-picker', 'value')]
)
def display_selected_range(selectedData, columns):
    global highlighted_cols
    # # Check if callback is triggered by user interaction
    pattern = r'\d+'

    df = df_team_data.copy()

    if selectedData:

        columns = ['team'] + columns
        string = selectedData[0]
        string = list(string.keys())[0]
        index = int(''.join(re.findall(pattern, string)))

        if selectedData[0][string] == None:
            del highlighted_cols[columns[index]]

            for column_name, constraint in highlighted_cols.items():
                if isinstance(constraint[0], list):  # Check if it's a range constraint
                    min_value1, max_value1 = constraint[0]
                    min_value2, max_value2 = constraint[1]
                    
                    df = df[(df[column_name].between(min_value1, max_value1)) | (df[column_name].between(min_value2, max_value2))]
                else:  # It's a single value constraint
                    value = constraint
                    df = df[(df[column_name] >= value[0]) & (df[column_name] <= value[1])]
            
            return [{'name': relevant_team_data_columns[col], 'id': col} for col in columns], df.to_dict('records')


        values = selectedData[0][string][0]
        if values[0] > values[1]:
            values[0], values[1] = swap(values[0],  values[1])

        # highlighted_cols.append((columns[index], values))
        # print(highlighted_cols)
        

        highlighted_cols[columns[index]] = values
        # print(highlighted_cols)
        # print(df_team_data[(df_team_data[columns[index]] > values[0]) & (df_team_data[columns[index]] < values[1])][columns].to_dict('records'))
        for column_name, constraint in highlighted_cols.items():
            if isinstance(constraint[0], list):  # Check if it's a range constraint
                min_value1, max_value1 = constraint[0]
                min_value2, max_value2 = constraint[1]
                
                df = df[(df[column_name].between(min_value1, max_value1)) | (df[column_name].between(min_value2, max_value2))]
            else:  # It's a single value constraint
                value = constraint
                df = df[(df[column_name] >= value[0]) & (df[column_name] <= value[1])]

        return [{'name': relevant_team_data_columns[col], 'id': col} for col in columns], df.to_dict('records')

    return no_update, no_update

@callback(
    Output('feature-graph', 'figure', allow_duplicate=True),
    Input('column-picker', 'value'),
    Input('data-table', 'selected_rows'),
    State('data-table', 'data'),
    prevent_initial_call = True
)
def update_pc(columns, selected_country, data):
    global highlighted_cols
    highlighted_cols = {}
    country = data[selected_country[0]]['team']
    print(pd.DataFrame(data)[['team'] + columns])
    figure = create_pcp(columns=columns, highlighted_country=country)

    return figure

@app.callback(
    Output('radar-chart', 'src'),
    Input('country-1', 'value'),
    Input('country-2', 'value'),
    Input('radar-picker', 'value')
)
def radar_update(team1, team2, columns):
    print(team1, team2, columns)

    home_data = df_team_data[df_team_data['team'] == team1][columns].values[0]
    away_data = df_team_data[df_team_data['team'] == team2][columns].values[0]

    maxi = [0] * len(columns)
    mini = [0] * len(columns)

    for i,(x1,x2) in enumerate(zip(list(away_data), list(home_data))):
        if x1 >= x2:
            maxi[i] = x1 * 1.1
            mini[i] = x2 * 0.7
        else:
            maxi[i] = x2 * 1.1
            mini[i] = x1 * 0.7
    
    lower_is_better = ['Miscontrol']

    cols = [relevant_team_data_columns[col] for col in columns]

    radar = Radar(cols, mini, maxi,
              lower_is_better=lower_is_better,
              # whether to round any of the labels to integers instead of decimal places
              round_int=[False]*len(columns),
              num_rings=4,  # the number of concentric circles (excluding center circle)
              # if the ring_width is more than the center_circle_radius then
              # the center circle radius will be wider than the width of the concentric circles
              ring_width=1, center_circle_radius=1)

    plt.figure(figsize=(6, 6)) 

    fig, axs = grid(figheight=4, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                    title_space=0, endnote_space=0, grid_key='radar', axis=False)

    # plot radar
    radar.setup_axis(ax=axs['radar'])  # format axis as a radar
    rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='#b8b8b8', edgecolor='#999999')
    radar_output = radar.draw_radar_compare(home_data, away_data, ax=axs['radar'],
                                            kwargs_radar={'facecolor': '#3946bd', 'alpha': 0.6},
                                            kwargs_compare={'facecolor': '#fc941c', 'alpha': 0.6})
    radar_poly, radar_poly2, vertices1, vertices2 = radar_output

    range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=15,
                                        fontproperties=robotto_thin.prop)

    param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=15,
                                        fontproperties=robotto_bold.prop)

    axs['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
                        c='#3946bd', edgecolors='#6d6c6d', marker='o', s=150, zorder=2)

    axs['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
                        c='#fc941c', edgecolors='#6d6c6d', marker='o', s=150, zorder=2)

    # adding the endnote and title text (these axes range from 0-1, i.e. 0, 0 is the bottom left)
    # Note we are slightly offsetting the text from the edges by 0.01 (1%, e.g. 0.99)
    title1_text = axs['title'].text(0.01, 0.4, f'{team1}', fontsize=12, color='#3946bd',
                                    fontproperties=rubik_regular.prop, ha='left', va='center')
    title3_text = axs['title'].text(0.99, 0.4, f'{team2}', fontsize=12,
                                    fontproperties=rubik_regular.prop,
                                    ha='right', va='center', color='#fc941c')

    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
    # Show the plot
    return fig_bar_matplotlib

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)