import dash
from dash import dcc, html

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.Header(
            children=[
                html.H1("MatchMania", style={
                    'margin': '5px 0',
                    'font-family': 'Montserrat',
                    'font-weight': '700',
                    'color': '#fff'
                }),
            ],
            style={
                'background-color': '#36a2cc',
                'color': '#fff',
                'padding': '10px',
                'text-align': 'center'
            },
        ),
        dcc.Dropdown(
            id='dropdown-options',
            options=[
                {'label': 'Visualization 1', 'value': 'visualization1'},
                {'label': 'Visualization 2', 'value': 'visualization2'},
                {'label': 'Visualization 3', 'value': 'visualization3'},
            ],
            value='visualization1',
            searchable=False,
            style={
                'color': '#36a2cc',
                'cursor': 'pointer',
                'text-align': 'center',
                'font-size': '14px',
                'align': 'center',
            },
        )
    ],
    style={'font-family': 'Montserrat', 'margin': '0', 'padding': '0'},
)

if __name__ == '__main__':
    app.run_server(debug=True)