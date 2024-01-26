import base64
from io import BytesIO
from flask import Flask
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd 
from utils import * 
import plotly.graph_objects as go
from mplsoccer import Pitch
import ast
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

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

def make_pass_df():
    df = pd.read_csv('events.csv')
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
    df_p['color'] = df_p['outcome'].apply(lambda x: 'blue' if x == 1 else 'red')
    return df_p, df

df_pass, df = make_pass_df()
# make pcp plot
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

def home_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        icon('The graph allows to compare teams over all games in the tournament. Click on the dropdown menu to find your favourite metrics.', id='pcp')
                    ])
                ]),
                dbc.Row(dcc.Dropdown(
                    id='column-picker',
                    options=[{'label': relevant_team_data_columns[col], 'value': col} for col in df_team_data.columns if relevant_team_data_columns.get(col) is not None],
                    value=init_columns,  # Default value
                    multi=True  # Allow multiple selections
                )),
            dcc.Graph(id='feature-graph')
            ]),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        icon('pick three metrics which will define the scatter plot, first on is x-axis value, second is y-axis and the third is a size of the dot. Colors indicate the ', id='scatter')
                    ])
                ]),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        id='scatter1',
                        options=[{'label': relevant_team_data_columns[col], 'value': col} for col in df_team_data.columns if relevant_team_data_columns.get(col) is not None],
                        value=init_columns[0],  # Default value
                    )),
                    dbc.Col(dcc.Dropdown(
                        id='scatter3',
                        options=[{'label': relevant_team_data_columns[col], 'value': col} for col in df_team_data.columns if relevant_team_data_columns.get(col) is not None],
                        value=init_columns[2],  # Default value
                    )),
                    dbc.Col(dcc.Dropdown(
                        id='scatter2',
                        options=[{'label': relevant_team_data_columns[col], 'value': col} for col in df_team_data.columns if relevant_team_data_columns.get(col) is not None],
                        value=init_columns[1],  # Default value
                    ))
                ]),
            dcc.Graph(id='scatter')
            ]),

        ])
        , html.H1('ddddd', id='d')

    ])

def matches_layout():
    return html.Div([
        dbc.Row([
            ### Dropdown with matches
            dcc.Dropdown(
                id='match-dropdown',
                options=[{'label': f"{value[0]} vs. {value[1]}", 'value': option} for option, value in my_dict.items()],
                value='Tylko Legia, Ukochana Legia',
                searchable=True,
                style={
                    'color': '#36a2cc',
                    'cursor': 'pointer',
                    'text-align': 'center',
                    'font-size': '14px',
                    'align': 'center',
                    'height':'30px'
                },
            )
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        icon('average positions of the players and most frequent connections on the pitch. Arrow size denotes the amount of passes between players.', id='network1')
                    ])
                ]),
                dbc.Row([
                    ### PASS NETWORK
                    html.Img(id='passnet-home')

                ]),
                dbc.Row([
                    dbc.Col([
                        icon('Rucham ci matke', id='heatmap1')
                    ])
                ]),
                dbc.Row([
                    ### PASS HEATMAP
                    html.Img(id='heatmap-home')
                ]),
                dbc.Row([
                    ### SHOTMAP

                ]),
            ]),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        icon('average positions of the players and most frequent connections on the pitch. Arrow size denotes the amount of passes between players.', id='network2')
                    ])
                ]),
                dbc.Row([
                    ### PASS NETWORK
                    html.Img(id='passnet-away') 

                ]),
                dbc.Row([
                    dbc.Col([
                        icon('average positions', id='heatmap2')
                    ])
                ]),
                dbc.Row([
                    ### PASS HEATMAP
                    html.Img(id='heatmap-away')
                ]),
                dbc.Row([
                    ### SHOTMAP
                ]),
            ]),
            dbc.Col([
                dcc.Dropdown(
                    id='player-dropdown',
                    value=None,
                    placeholder='Select a player'
                ),
                dcc.Graph(figure={}, id='pass-map')
            ])
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
    return update_pcp(selected_feature)

# Define callback to update page content
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return home_layout()
    else:
        return matches_layout()
    
@app.callback(
    Output('scatter', 'figure'),
    Input('scatter1', 'value'),
    Input('scatter2', 'value'),
    Input('scatter3', 'value')
)
def scatter(param1, param2, param3):
    return px.scatter(
        data_frame=df_team_data,
        x=param1,
        y=param2,
        size=param3,
        title='Jazda z kurwami',
        color='games'
    )

@app.callback(
    Output(component_id='passnet-home', component_property='src'),
    Output(component_id='passnet-away', component_property='src'),
    Output(component_id='heatmap-away', component_property='src'),
    Output(component_id='heatmap-home', component_property='src'),
    Input('match-dropdown', 'value')
)
def make_pass_nets(match_num):
    def make_pass_network_home(match_id, thresh = 3):
        minute, second = df[(df['match_id'] == match_id) & (df['type'] == "Substitution")][['minute', 'second']].iloc[0].tolist()
        team_id = df_pass[(df_pass['match_id'] == match_id)]['team_id'].unique()[0]
        team_name = id_to_team[team_id]

        df_nonsub = df_pass[(df_pass['minute'] < minute) & (df_pass['match_id'] == match_id) & (df_pass['team_id'] == team_id)]
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
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white')
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=False, tight_layout=True)
        # ax.invert_yaxis()
        # fig.set_facecolor('#22312b')
        for index, row in df_plot.iterrows():
            pitch.arrows(1.2 * row['x'], .8 * row['y'], 1.2 * row['x_end'], .8 *row['y_end'], ax=ax, color= 'blue', alpha=.8, headwidth = 6, width=row['count_x']/3)
        nodes = pitch.scatter(1.2 * avg_pos.x, .8 * avg_pos.y, s=300, alpha = .4, ax=ax)
        buf = BytesIO()

        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
        
        return fig_bar_matplotlib, team_name
    
    def make_pass_network_away(match_id, thresh = 3):
        minute, second = df[(df['match_id'] == match_id) & (df['type'] == "Substitution")][['minute', 'second']].iloc[0].tolist()
        team_id = df_pass[(df_pass['match_id'] == match_id)]['team_id'].unique()[1]
        team_name = id_to_team[team_id]
        df_nonsub = df_pass[(df_pass['minute'] < minute) & (df_pass['match_id'] == match_id) & (df_pass['team_id'] == team_id)]
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
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white')
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=False, tight_layout=True)
        
        for index, row in df_plot.iterrows():
            pitch.arrows(1.2 * row['x'], .8 * row['y'], 1.2 * row['x_end'], .8 * row['y_end'], ax=ax, color= 'blue', alpha=.8, headwidth = 6, width=row['count_x']/3)
        nodes = pitch.scatter(1.2 * avg_pos['x'], .8 * avg_pos['y'], s=300, alpha = .4, ax=ax)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
        
        return fig_bar_matplotlib, team_name
    
    def make_heatmaps(match_id):
        match_df_a = df_pass[(df_pass['match_id'] == match_id) & (df_pass['team'] == my_dict[match_id][0])]
        match_df_b = df_pass[(df_pass['match_id'] == match_id) & (df_pass['team'] == my_dict[match_id][1])]

        # Heatmap for Team a
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white', line_zorder=2)
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=False, tight_layout=True)
        bin_statistic = pitch.bin_statistic(match_df_a['x'] * 6/5 , match_df_a['y'] * 4/5, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_heatmap_1 = f'data:image/png;base64,{fig_data}'

        # Heatmap for Team b
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white', line_zorder=2)
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=False, tight_layout=True)
        bin_statistic = pitch.bin_statistic(match_df_b['x'] * 6/5, match_df_b['y'] * 4/5, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_heatmap_2 = f'data:image/png;base64,{fig_data}'
        return fig_heatmap_1, fig_heatmap_2

    home_fig, home_name = make_pass_network_home(match_num)
    fig_away, away_name = make_pass_network_away(match_num)
    heatmap_home, heatmap_away = make_heatmaps(match_num)
    return home_fig, fig_away,heatmap_home, heatmap_away

@app.callback(
    Output('player-dropdown', 'options'),
    # Output('player-dropdown', 'value'),
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
)
def plot_passmap(match_id, player):

    df_3857256 = df_pass[(df_pass['match_id'] == match_id) & (df_pass['player'] == player)]
    # Create a figure
    fig = go.Figure()

    trace_blue = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(color='blue', size=10),
        showlegend=True,
        name="Successful Pass"
    )

    trace_red = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(color='red', size=10),
        showlegend=True,
        name="Unsuccessful Pass"
    )


    # Create figure
    fig = go.Figure(data=[trace_blue, trace_red])



    # # Left Penalty Area
    # fig.add_shape(type="rect", x0=0, y0=30, x1=12, y1=70, line=dict(color="white"))

    # # Right Penalty Area
    # fig.add_shape(type="rect", x0=88, y0=30, x1=100, y1=70, line=dict(color="white"))

    # # Left 6-yard Box
    # fig.add_shape(type="rect", x0=0, y0=44, x1=6, y1=56, line=dict(color="white"))

    # # Right 6-yard Box
    # fig.add_shape(type="rect", x0=94, y0=44, x1=100, y1=56, line=dict(color="white"))
    
    # # middle line
    # fig.add_shape(type='line', x0=50, x1=50, y0=0, y1=100,line=dict(color='white'))
    
    # # Add pitch outline and centre line
    # fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, fillcolor='LightSeaGreen', opacity=0, line_color='white')
    
    
    # fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line_color='white')

    # # Prepare the centre circle
    # fig.add_shape(type="circle",
    #     xref="x", yref="y",
    #     x0=45, y0=40, x1=55, y1=60,
    #     line_color="white",
    # )

    # fig.update_layout(
    # yaxis = dict(autorange="reversed")
    # )   

    # # Centre spot
    # fig.add_trace(go.Scatter(x=[50], y=[50], mode='markers', marker=dict(color='white', size=5), hoverinfo='none', showlegend=False))

    # # Left Penalty Spot
    # fig.add_trace(go.Scatter(x=[11], y=[50], mode='markers', marker=dict(color='white', size=5), hoverinfo='none', showlegend=False))

    # # Right Penalty Spot
    # fig.add_trace(go.Scatter(x=[89], y=[50], mode='markers', marker=dict(color='white', size=5), hoverinfo='none', showlegend=False))

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

    # Hide axis labels
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False)
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)

    # Update the Traces for Passes
    for index, row in df_3857256.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['y'], row['y_end']],  # Swap x and y
            y=[row['x'], row['x_end']],  # Swap x and y
            mode='lines+markers',
            line={'color':row['color']},
            name=f'{row["pass_recipient"]}',
            marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"),
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=df_3857256['y'],  # Swap x and y
        y=df_3857256['x'],  # Swap x and y
        mode='markers',
        marker=dict(size=5, symbol='circle', color=df_3857256['color']),
        text=df_3857256.apply(lambda row: f'{row["player"]} to {row["pass_recipient"]} ({format_minute(str(row["minute"]),str(row["second"]))})', axis=1),
        showlegend=False
    ))

    # Adjust the layout
    fig.update_layout(width=530, height=620)
    fig.update_layout(title_text=f'{player} pass map')
    fig.update_xaxes(range=[-5, 105], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-2, 102], showgrid=False, zeroline=False)

    return fig

    # # Hide axis labels
    # fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False)
    # fig.update_layout(xaxis_visible=False, yaxis_visible=False)



    # # fig.update_xaxes(range=[-2, 102], visible=False)
    # # fig.update_yaxes(range=[-2, 100], visible=False)

    # # Add a trace for each row in the DataFrame
    # for index, row in df_3857256.iterrows():
    #     # Connect starting points to ending points with lines
    #     fig.add_trace(go.Scatter(
    #         x=[row['x'], row['x_end']],
    #         y=[row['y'], row['y_end']],
    #         mode='lines+markers',
    #         line= {'color':row['color']},
    #         name= f'{row["pass_recipient"]}',
    #         marker=dict(size=10,symbol= "arrow-bar-up", angleref="previous"),
    #         showlegend=False
    #     ))

    # fig.add_trace(go.Scatter(
    #     x=df_3857256['x'],
    #     y=df_3857256['y'],
    #     mode='markers',
    #     marker=dict(size=5, symbol='circle', color=df_3857256['color']),
    #     text=df_3857256.apply(lambda row: f'{row["player"]} to {row["pass_recipient"]} ({format_minute(str(row["minute"]),str(row["second"]))})', axis=1),
    #     showlegend=False
    # ))

    # fig.add_annotation(
    #     text='&#x27F6;',  # HTML entity for a right arrow symbol
    #     x=0.5,  # x-coordinate of the arrow (center of the plot)
    #     y=1.8,  # y-coordinate of the arrow (above the plot)
    #     xref='paper',  # Use paper coordinates for x
    #     yref='paper',  # Use paper coordinates for y
    #     showarrow=False,
    #     font=dict(size=100),  # Adjust the font size as needed
    # )

    # # fig.update_layout(template="plotly_dark")

    # fig.update_layout(width=600, height=400)
    # fig.update_layout(title_text=f'{player} pass map')
    #     # Set axes properties
    # fig.update_xaxes(range=[-2, 102], showgrid=False, zeroline=False)
    # fig.update_yaxes(range=[-5, 105], showgrid=False, zeroline=False)

    # return fig

@app.callback(
    Output('d', 'children'),
    [Input('pcp', 'relayoutData')]
)
def xxx(selectedData):
    # print(selectedData)
    return str(selectedData) if selectedData is not None else 'Dupa dupa dupa'


# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
