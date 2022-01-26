
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go # or plotly.express as px  
import pandas as pd
import base64
import io
from pathlib import Path
import sentspace

app = dash.Dash('sentspace')


def load_df_from_directory(corpus: str, directory: str, module: str = 'lexical'):
    dfs = pd.DataFrame()
    print(Path(directory+'/'+module).resolve())
    glob = [*Path(directory+'/'+module).glob('sentence-features_part*.pkl.gz')]
    for path in glob:
        df = pd.read_pickle(path)
        dfs = pd.concat([dfs,df])
    # reassigning the index as a column rather than just an index (sometimes
    # input files have sentence as their index)
    dfs[dfs.index.name] = dfs.index
    dfs['corpus'] = corpus
    dfs['sentence_length'] = dfs['sentence'].apply(lambda x: len(sentspace.Sentence(x, warn=False)))
    return dfs.sample(frac=.1)

df = pd.DataFrame()
df = df.append(load_df_from_directory('brown', 'out/brown_subsampled_grouped_by_length_n=500_stimuli'))
df = df.append(load_df_from_directory('torontoadv', 'out/torontoadv_subsampled_grouped_by_length_n=500_stimuli'))
df = df.append(load_df_from_directory('wsj', 'out/wsj_subsampled_grouped_by_length_n=500_stimuli'))
df = df.append(load_df_from_directory('ud', 'out/ud_subsampled_grouped_by_length_n=500_stimuli'))
df = df.append(load_df_from_directory('c4', 'out/c4_subsampled_grouped_by_length_n=500_stimuli'))
df = df.append(load_df_from_directory('cocaspok1991', 'out/cocaspok1991_subsampled_grouped_by_length_n=500_stimuli'))
df = df.append(load_df_from_directory('cocaspok2012', 'out/cocaspok2012_subsampled_grouped_by_length_n=500_stimuli'))
# df = df.drop(columns=[None])

print('example df:', df.head(5), df.columns)
print('-'*79)


app.layout = html.Div(
    children=[
        html.H3(children='sentspace-vis'),
        html.Div(children='''
            visualize sentences on the backdrop of large benchmarks
        '''),

        html.Div([
                    html.Div([
                        html.P(children='Plot type:'),
                        dcc.Dropdown(id=f'plot-type',
                                options=[{'label': i, 'value': i} for i in ('histogram', 'scatter')],
                                value='histogram',
                                    ), 
                    ], style={'width': '20%', 'float': 'right'}),           
                ]),
        

        html.Div([
                html.Div([
                    html.P(children=f'Histogram variable to plot:',
                        style=dict(height='10px', lineheight='10px')),
                    dcc.Dropdown(
                        id=f'hist-column',
                        options=[{'label': i, 'value': i} for i in df.columns],
                        value='surprisal-3',
                    ),
                    
                    ], style={'width': '20%', 'display': 'inline-block', 'float':'left'}
                ),

                *[
                    html.Div([
                        html.Div([
                            html.P(children=f'Scatterplot {which} axis value:',
                                style=dict(height='10px', lineheight='10px')),
                            dcc.Dropdown(
                                id=f'{which}axis-column',
                                options=[{'label': i, 'value': i} for i in df.columns],
                                value=init_value,
                            ),
                            ], style={'width': '10%', 'display': 'inline-block', 'float':'left'}
                        ),

                    ]) for which, init_value in zip('xyz', ['surprisal-4', 'aoa', 'lexical_decision_RT'])
                ],               

        ], style={'width': '88%',  'display': 'inline-block'}),
        html.Div([

            dcc.Upload(
                id='upload-csv',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files'),
                    ' (allowed filetypes: .tsv, .pkl, .pkl.gz)'
                ]),
                style={
                    'width': '50%',
                    'height': '20px',
                    'lineHeight': '20px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'display': 'inline-block',
                    'float': 'left',
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
        ], style={'width': '88%',  'display': 'inline-block'}),

        html.Div([
            html.Div([
                html.P("sentence length ="),
                dcc.Dropdown(
                                id=f'filter-length',
                                options=[{'label': i, 'value': i} for i in range(6,50)] + [dict(label='None', value=0)],
                                value=None,
                            ),
            ]),
        ]),

        dcc.Graph(id='the-graph',),

])


def parse_df(data, filename) -> pd.DataFrame:
    '''
    '''
    content_type, content = data.split(',')
    decoded = base64.b64decode(content)
    f = io.BytesIO(decoded)
    if filename.endswith('tsv'):
        temp_df = pd.read_csv(f, delimiter='\t')
    else:
        temp_df = pd.read_pickle(f, compression='gzip' if filename.endswith('pkl.gz') else None)
    temp_df['corpus'] = filename
    temp_df[temp_df.index.name] = temp_df.index
    print(temp_df.head(), 'temp_df')
    return temp_df


@app.callback(
    Output('the-graph', 'figure'),
    Input('plot-type', 'value'),
    # dropdowns for axis titles
    Input('hist-column', 'value'),
    # scatter
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('zaxis-column', 'value'),

    Input('filter-length', 'value'),
    # uploading related args
    Input('upload-csv', 'contents'),
    State('upload-csv', 'filename'),
    # Input('year--slider', 'value')
    )
def update_graph(plot_type, 
                 hist_column=None, 
                 x_column=None, y_column=None, z_column=None,
                 filter_length=None,
                 upload_contents=None, filename=None):

    df_ = df
    # print(upload_contents, filename)
    if upload_contents and filename:
        uploaded_df = parse_df(upload_contents, filename)
        uploaded_df['sentence_length'] = uploaded_df.apply(lambda x: len(sentspace.Sentence(x, warn=False)))
        df_ = df.append(uploaded_df)

    if filter_length: 
        df_ = df_[df_['sentence_length'] == int(filter_length)]

    if plot_type == 'histogram':
        fig = px.histogram(df_, x=hist_column, color="corpus",
                        marginal="violin", # 'rug', "box", "violin"
                        #hover_data=df.columns
                        histnorm='probability density',
                        hover_name='sentence',
                        )
        return fig

    else:
        #fig = px.bar(lexical, x="sentence", y="aoa", color="sentence")
        fig = px.scatter_3d(df_, x=x_column, y=y_column, z=z_column,
                        color="corpus", 
                        hover_name="sentence", height=700,
                        symbol='corpus', opacity=.8,
                        )

        fig.update_traces(marker=dict(size=3,
                                      opacity=1,
                                      line=dict(width=.1,
                                                color='DarkSlateGray')
                                      ),
                          selector=dict(mode='markers'))

        return fig


if __name__ == '__main__':
    app.run_server(debug=True)

