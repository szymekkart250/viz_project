import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import ast
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from mplsoccer import Pitch, Radar, FontManager, grid
import plotly.graph_objects as go
import plotly.express as px
CHOSEN_COLS = []


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

plt.switch_backend('Agg')

app = dash.Dash(__name__)

home_away_list = pd.read_csv("Data/FIFA World Cup 2022 Match Data/data.csv")

my_dict = {3857256: ['Serbia', 'Switzerland'],
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

id_to_team = {786: 'Serbia',
 773: 'Switzerland',
 779: 'Argentina',
 792: 'Australia',
 776: 'Denmark',
 781: 'Brazil',
 777: 'Tunisia',
 3565: 'Ecuador',
 787: 'Senegal',
 941: 'Netherlands',
 783: 'Uruguay',
 791: 'South Korea',
 788: 'Morocco',
 780: 'Portugal',
 771: 'France',
 799: 'Saudi Arabia',
 794: 'Mexico',
 789: 'Poland',
 3566: 'Qatar',
 768: 'England',
 785: 'Croatia',
 772: 'Spain',
 778: 'Japan',
 1839: 'United States',
 770: 'Germany',
 2722: 'Cameroon',
 795: 'Costa Rica',
 782: 'Belgium',
 907: 'Wales',
 4885: 'Ghana',
 1833: 'Canada',
 797: 'Iran'}

team_to_id = {value:key for key,value in id_to_team.items()}

def name_change(name):
    if name == "Korea Republic":
        name = "South Korea"
    if name == "IR Iran":
        name = "Iran"
    return name

def fig_to_uri(in_fig, close_all=True):
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png')
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

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


def format_minute(minutes, seconds):
    if len(seconds) == 1:
        time = f"{minutes}:0{seconds}"
    else:
        time = f"{minutes}:{seconds}"
    return time

df_pass, df = make_pass_df()

df_team_data = pd.read_csv('Data/FIFA World Cup 2022 Team Data/team_data.csv')

# Iterate over the columns and perform the division where applicable
# for column in df.columns:
#     if '90' not in column and column not in ['games', 'team']:  # Check if '90' is not in the column name and it's not the 'games' column
#         df[column] = df[column] / df['games']  # Perform the division

cols = ['goals', 'assists']


home_away_list['home_team'] = home_away_list['home_team'].apply(name_change)
home_away_list['away_team'] = home_away_list['away_team'].apply(name_change)

app.layout = dbc.Container([
            html.Header(
            children=[
                html.H1("MatchMania", style={
                    'margin': '5px 0',
                    'font-family': 'Montserrat',
                    'font-weight': '800',
                    'color': '#fff'
                }),
            ],
            style={
                'background-color': '#3d42cc',
                'color': '#fff',
                'padding': '10px',
                'text-align': 'center'
            },
        ),
        dcc.Dropdown(
            id='match-dropdown',
            options=[{'label': f"{value[0]} vs. {value[1]}", 'value': option} for option, value in my_dict.items()],
            value='Argentina vs. Poland',
            searchable=True,
            style={
                'color': '#36a2cc',
                'cursor': 'pointer',
                'text-align': 'center',
                'font-size': '14px',
                'align': 'center',
                'height':'30px'
            },
        ),
        # html.Header(className= '.header-banner',
        #     # className='header-banner',
        #     children=[
        #         # html.H1('WOR'),
        #         # html.P('Subtitle or description for your dashboard'),
        #         html.Img(src='/assets/cwks.png', style={"width":'50px', 'heigth':'50px'}),

        #         html.H1('MatchMania'),

        #         dcc.Dropdown(
        #         id='match-dropdown',
        #         className='dropdown',
        #         options=[{'label': f"{value[0]} vs. {value[1]}", 'value': option} for option, value in my_dict.items()],
        #         value="Tylko Legia, Ukochana Legia",
        #         placeholder='Select a game',
        #         style={
        #             'width':'300px'
        #         }
        #         )
        #     ], style={'display':'flex'}
        # ),
        dbc.Row([
        html.Div(className='passmap',children=[
            dbc.Col([
                dcc.Dropdown(
                    id='player-dropdown',
                    value=None,
                    placeholder='Select a player',
                    style={
                        'width':'80%'
                    }
                )
            ]
        ),
        dcc.Graph(id='bar-graph-plotly', figure={})])], style={'width': "200%", 'padding': '1%', 'display': 'flex'}),
        # dcc.Graph(id='bar-graph-plotly', figure={})]),
        dbc.Row([
        dbc.Col([
            html.H1("Home team", id='home-name'),
            html.Img(id='example-home', style={'width':'100%', 'heigth':'100%'}) # img element),

        ], width=5),

        dbc.Col([
            html.H1("Away team", id='away-name'),
            html.Img(id='example-away', style={'width':'100%', 'heigth':'100%'}) # img element,

        ], width=5),
        dbc.Col([
            html.H1('Radar Chart'),
            html.Img(id='radar-chart', style={'width':'80%'})
        ])
    ], style={'display': 'flex'}),
    html.Div(id='bar-chart-div', children=[
        dcc.Dropdown(
            id='column-picker',
            options=[{'label': col, 'value': col} for col in df_team_data.columns if '90' not in col],
            value=cols,  # Default value
            multi=True  # Allow multiple selections
            ),
        dbc.Row([
            dbc.Col(dcc.Graph(id='bar-chart'), width=5),
            dbc.Col(dcc.Graph(id='pcp'), width=5)
        ], style={'width':'100%', 'dipsplay':'flex'}),
        dcc.Graph(id='scatter')
    ], style={'width':12})
], fluid=True)


@app.callback(
    Output('player-dropdown', 'options'),
    Output('player-dropdown', 'value'),
    Output('bar-chart-div', 'style'),
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
        return [{'label': f'{item[0]} ({item[1]})', 'value': item[0]} for item in items], None, {'display': 'inline'}


### CREATE PASSMAP FOR A PLAYER
# Create interactivity between dropdown component and graph
@app.callback(
    # Output(component_id='bar-graph-matplotlib', component_property='src'),
    Output('bar-graph-plotly', 'figure'),
    Input('match-dropdown', 'value'),
    Input('player-dropdown', 'value'),
)
def plot_passmap(match_id, player):

    # Build the Plotly figure
    import ast
    
    df_3857256 = df_pass[(df_pass['match_id'] == match_id) & (df_pass['player'] == player)]
    import plotly.graph_objects as go

    # Create a figure
    fig = go.Figure()

    image_path = "/Users/szymonkozak/Documents/TUE/visualization/viz_project/cwks.png"
    fig.add_layout_image(
        source=image_path,
        x=0.4,  # x-coordinate of the image (0 corresponds to the left side of the plot)
        y=.65,  # y-coordinate of the image (1 corresponds to the top side of the plot)
        xref="paper",  # Use paper coordinates for x
        yref="paper",  # Use paper coordinates for y
        sizex=0.3,  # width of the image
        sizey=0.3,  # height of the image
        opacity=1,  # image opacity (1 is fully opaque)
        layer="below"  # place the image above other plot elements
    )

    # fig.update_layout(
    #     plot_bgcolor='#AFE1AF',
    #     width = 1000,
    #     height = 750,
    #     showlegend = False
    # )

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


    # Set axes properties
    fig.update_xaxes(range=[-2, 102], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-2, 102], showgrid=False, zeroline=False)


    # Left Penalty Area
    fig.add_shape(type="rect", x0=0, y0=30, x1=12, y1=70, line=dict(color="white"))

    # Right Penalty Area
    fig.add_shape(type="rect", x0=88, y0=30, x1=100, y1=70, line=dict(color="white"))

    # Left 6-yard Box
    fig.add_shape(type="rect", x0=0, y0=44, x1=6, y1=56, line=dict(color="white"))

    # Right 6-yard Box
    fig.add_shape(type="rect", x0=94, y0=44, x1=100, y1=56, line=dict(color="white"))
    
    # middle line
    fig.add_shape(type='line', x0=50, x1=50, y0=0, y1=100,line=dict(color='white'))
    
    # Add pitch outline and centre line
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, fillcolor='LightSkyBlue', opacity=.2, line_color='white')
    
    
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line_color='white')

    # Prepare the centre circle
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=45, y0=40, x1=55, y1=60,
        line_color="white",
    )

    fig.update_layout(
    yaxis = dict(autorange="reversed")
    )   

    # Centre spot
    fig.add_trace(go.Scatter(x=[50], y=[50], mode='markers', marker=dict(color='white', size=5), hoverinfo='none', showlegend=False))

    # Left Penalty Spot
    fig.add_trace(go.Scatter(x=[11], y=[50], mode='markers', marker=dict(color='white', size=5), hoverinfo='none', showlegend=False))

    # Right Penalty Spot
    fig.add_trace(go.Scatter(x=[89], y=[50], mode='markers', marker=dict(color='white', size=5), hoverinfo='none', showlegend=False))

    # Hide axis labels
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False)
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)



    # fig.update_xaxes(range=[-2, 102], visible=False)
    fig.update_yaxes(range=[-2, 100], visible=False)

    # Add a trace for each row in the DataFrame
    for index, row in df_3857256.iterrows():
        # Connect starting points to ending points with lines
        fig.add_trace(go.Scatter(
            x=[row['x'], row['x_end']],
            y=[row['y'], row['y_end']],
            mode='lines+markers',
            line= {'color':row['color']},
            name= f'{row["pass_recipient"]}',
            marker=dict(size=10,symbol= "arrow-bar-up", angleref="previous"),
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=df_3857256['x'],
        y=df_3857256['y'],
        mode='markers',
        marker=dict(size=5, symbol='circle', color=df_3857256['color']),
        text=df_3857256.apply(lambda row: f'{row["player"]} to {row["pass_recipient"]} ({format_minute(str(row["minute"]),str(row["second"]))})', axis=1),
        showlegend=False
    ))

    fig.add_annotation(
        text='&#x27F6;',  # HTML entity for a right arrow symbol
        x=0.5,  # x-coordinate of the arrow (center of the plot)
        y=1.25,  # y-coordinate of the arrow (above the plot)
        xref='paper',  # Use paper coordinates for x
        yref='paper',  # Use paper coordinates for y
        showarrow=False,
        font=dict(size=100),  # Adjust the font size as needed
    )

    fig.update_layout(template="plotly_dark")

    fig.update_layout(width=800, height=600)
    fig.update_layout(title_text=f'{player} pass map')

    return fig


@app.callback(
    Output(component_id='example-home', component_property='src'),
    Output(component_id='example-away', component_property='src'),
    Output('home-name', 'children'),
    Output('away-name', 'children'),
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
    
    home_fig, home_name = make_pass_network_home(match_num)
    fig_away, away_name = make_pass_network_away(match_num)
    return home_fig, fig_away, home_name, away_name

@app.callback(
        Output(component_id='radar-chart', component_property='src'),
        Input('match-dropdown', 'value')
)
def make_radar_chart(match_id):
    team1, team2 = my_dict[match_id]

    params_radar = ['xg', 'completed_passes', 'long_balls', 'fouls', 'total_shots', 'offsides']
    away_params = ['away_'+col for col in params_radar]
    home_params = ['home_'+col for col in params_radar]
    params_radar = ['Expected Goals', 'Completed passes', 'Long balls', 'Fouls', 'Shots', 'Offsides']




    away_data = home_away_list[(home_away_list['home_team'] == team1) & (home_away_list['away_team'] == team2)][away_params].values[0]
    home_data = home_away_list[(home_away_list['home_team'] == team1) & (home_away_list['away_team'] == team2)][home_params].values[0]

    maxi = [0] * len(params_radar)
    mini = [0] * len(params_radar)
    index = 0

    for i,(x1,x2) in enumerate(zip(list(away_data), list(home_data))):
        if x1 >= x2:
            maxi[i] = x1 * 1.1
            mini[i] = x2 * 0.7
        else:
            maxi[i] = x2 * 1.1
            mini[i] = x1 * 0.7

    lower_is_better = ['Miscontrol']

    radar = Radar(params_radar, mini, maxi,
                lower_is_better=lower_is_better,
                # whether to round any of the labels to integers instead of decimal places
                round_int=[False]*len(params_radar),
                num_rings=4,  # the number of concentric circles (excluding center circle)
                # if the ring_width is more than the center_circle_radius then
                # the center circle radius will be wider than the width of the concentric circles
                ring_width=1, center_circle_radius=1)

    # fig, ax = radar.setup_axis()
    # rings_inner = radar.draw_circles(ax=ax, facecolor='#ffb2b2', edgecolor='#fc5f5f')
    # radar_output = radar.draw_radar_compare(home_data, away_data, ax=ax,
    #                                         kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
    #                                         kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
    # radar_poly, radar_poly2, vertices1, vertices2 = radar_output
    # range_labels = radar.draw_range_labels(ax=ax, fontsize=15,
    #                                     fontproperties=robotto_thin.prop)
    # param_labels = radar.draw_param_labels(ax=ax, fontsize=15,
    #                                     fontproperties=robotto_thin.prop)
    fig, axs = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)

    # plot radar
    radar.setup_axis(ax=axs['radar'])  # format axis as a radar
    rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='#c4c4c4', edgecolor='#fc5f5f')
    radar_output = radar.draw_radar_compare(home_data, away_data, ax=axs['radar'],
                                            kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                            kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
    radar_poly, radar_poly2, vertices1, vertices2 = radar_output
    range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=25,
                                        fontproperties=robotto_bold.prop)
    param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=35,
                                        fontproperties=robotto_bold.prop)
    axs['radar'].scatter(vertices1[:, 0], vertices1[:, 1],
                        c='#00f2c1', edgecolors='#6d6c6d', marker='o', s=150, zorder=2)
    axs['radar'].scatter(vertices2[:, 0], vertices2[:, 1],
                        c='#d80499', edgecolors='#6d6c6d', marker='o', s=150, zorder=2)

    # adding the endnote and title text (these axes range from 0-1, i.e. 0, 0 is the bottom left)
    # Note we are slightly offsetting the text from the edges by 0.01 (1%, e.g. 0.99)
    endnote_text = axs['endnote'].text(0.99, 0.5, 'Inspired By: StatsBomb / Rami Moghadam', fontsize=15,
                                    fontproperties=robotto_thin.prop, ha='right', va='center')
    title1_text = axs['title'].text(0.01, 0.65, f'{team1}', fontsize=40, color='#01c49d',
                                    fontproperties=robotto_bold.prop, ha='left', va='center')
    title3_text = axs['title'].text(0.99, 0.65, f'{team2}', fontsize=40,
                                    fontproperties=robotto_bold.prop,
                                    ha='right', va='center', color='#d80499')
    
    fig.set_facecolor('#ffffff')

    buf = BytesIO()

    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'

    return fig_bar_matplotlib


@app.callback(
    Output('bar-chart', 'figure'),
    [Input('column-picker', 'value')],
    Input('match-dropdown', 'value')
)
def update_bar_chart(selected_columns, match):
    team1, team2 = my_dict[match]
    home_data = df_team_data[df_team_data['team'] == team1][selected_columns].values[0]
    away_data = df_team_data[df_team_data['team'] == team2][selected_columns].values[0]

    # Create an empty figure
    fig = go.Figure()
    fig = go.Figure(data=[
    go.Bar(name=team2, x=selected_columns, y=away_data),
    go.Bar(name=team1, x=selected_columns, y=home_data)
    ])
    print(f'{selected_columns}')
    fig.update_layout(
        title='Bar Chart',
        xaxis_title='Column',
        yaxis_title='Value',
        template="plotly_dark"
    )

    return fig

@app.callback(
    Output('scatter', 'figure'),
    Input('bar-chart', 'clickData'),
    Input('match-dropdown', 'value')
)
def display_click_data(clickData, match):
    global CHOSEN_COLS
    team1, team2 = my_dict[match]


    if clickData is None:
       return {}
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
                return {}

            fig = px.scatter(
                data_frame=df_team_data,
                x= CHOSEN_COLS[0],
                y=CHOSEN_COLS[1],
                hover_data=['team'],
                size='games',
                color='games',
                template="plotly_dark"
                )
            
            star_marker1 = dict(
            type='scatter',
            x=[df_team_data[df_team_data['team'] == team1][CHOSEN_COLS[0]].values[0]],  # X-coordinate of the star marker
            y=[df_team_data[df_team_data['team'] == team1][CHOSEN_COLS[1]].values[0]],  # Y-coordinate of the star marker
            mode='markers',
            marker=dict(symbol='star', size=30, color='red'),
            name=team1,
            showlegend=False
            )
            # for size in [10, 20, 30]:  # Replace with your actual sizes
            #     fig.add_trace(go.Scatter(
            #     x=[None],
            #     y=[None],
            #     mode='markers',
            #     marker=dict(color='black', size=size),  # Use a neutral color like black
            #     name=f'Size: {size}',
            #     showlegend=True
            #     ))

            star_marker2 = dict(
            type='scatter',
            x=[df_team_data[df_team_data['team'] == team2][CHOSEN_COLS[0]].values[0]],  # X-coordinate of the star marker
            y=[df_team_data[df_team_data['team'] == team2][CHOSEN_COLS[1]].values[0]],  # Y-coordinate of the star marker
            mode='markers',
            marker=dict(symbol='star', size=30, color='red'),
            name=team2,
            showlegend=False
            )        

            fig.add_trace(star_marker1)
            fig.add_trace(star_marker2)

        return fig

@app.callback(
    Output('pcp', 'figure'),
    [Input('column-picker', 'value')]
)
def update_pcp(columns):
    print(columns)
    dimensions = [go.parcoords.Dimension(values=df_team_data[col], label=col.replace("_", " ").title()) for col in columns]  # Exclude the first dimension ("team")
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


if __name__ == '__main__':
    app.run_server(debug=True)