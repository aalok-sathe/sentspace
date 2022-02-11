
import functools
from sys import maxsize
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go # or plotly.express as px  
import pandas as pd
import numpy as np
import base64
import io
from pathlib import Path
import dash_bootstrap_components as dbc
import textwrap

import sentspace

app = dash.Dash('sentspace', external_stylesheets=[dbc.themes.LUMEN])


def load_df_from_directory(corpus: str, directory: str, module: str = 'lexical',
                           subsample=True, fmt='pkl.gz'):
    dfs = pd.DataFrame()
    glob = [*Path(directory+'/'+module).resolve().glob(f'sentence-features_part*.{fmt}')]
    if module == 'syntax':
        glob += [*Path(directory+'/'+module).resolve().glob(f'token-features_part*.{fmt}')]
    print(Path(directory+'/'+module).resolve(), glob)
    for path in glob:
        if 'pkl' in fmt: 
            df = pd.read_pickle(path)
        else:
            df = pd.read_csv(path, sep='\t')
        dfs = pd.concat([dfs,df])

    print(dfs.head())

    # reassigning the index as a column rather than just an index (sometimes
    # input files have sentence as their index)
    dfs[dfs.index.name] = dfs.index
    dfs['corpus'] = corpus
    dfs['sentence_length'] = dfs['sentence'].apply(lambda x: len(sentspace.Sentence(x, warn=False)))
    if subsample:
        return dfs.sample(frac=.3)
    return dfs

# @functools.lru_cache(maxsize=5)
def load_benchmarks(module='lexical', subsample=False):
    df = pd.DataFrame()
    df = df.append(load_df_from_directory('gpt_stories', 'out/gpt2stories_sents_target0_subset', 
                                          module=module, subsample=subsample, fmt='tsv'))
    df = df.append(load_df_from_directory('human_stories', 'out/gpt2stories_sents_target1', 
                                          module=module, subsample=subsample))
    # df = df.append(load_df_from_directory('brown', 'out/benchmarks/brown_subsampled_grouped_by_length_n=500_stimuli', 
    #                                       module=module, subsample=subsample))
    # df = df.append(load_df_from_directory('torontoadv', 'out/benchmarks/torontoadv_subsampled_grouped_by_length_n=500_stimuli', 
    #                                       module=module, subsample=subsample))
    # df = df.append(load_df_from_directory('wsj', 'out/benchmarks/wsj_subsampled_grouped_by_length_n=500_stimuli', 
    #                                       module=module, subsample=subsample))
    # df = df.append(load_df_from_directory('ud', 'out/benchmarks/ud_subsampled_grouped_by_length_n=500_stimuli', 
    #                                       module=module, subsample=subsample))
    # df = df.append(load_df_from_directory('c4', 'out/benchmarks/c4_subsampled_grouped_by_length_n=500_stimuli', 
    #                                       module=module, subsample=subsample))
    # df = df.append(load_df_from_directory('cocaspok1991', 'out/benchmarks/cocaspok1991_subsampled_grouped_by_length_n=500_stimuli', 
    #                                       module=module, subsample=subsample))
    # df = df.append(load_df_from_directory('cocaspok2001', 'out/benchmarks/cocaspok2001_subsampled_grouped_by_length_n=500_stimuli', 
    #                                       module=module, subsample=subsample))
    # df = df.append(load_df_from_directory('cocaspok2012', 'out/benchmarks/cocaspok2012_subsampled_grouped_by_length_n=500_stimuli', 
    #                                       module=module, subsample=subsample))
    
    print(f'loaded df for module {module}:', df.head(5), df.columns)
    print('-'*79)

    return df
    # df = df.drop(columns=[None])


################################################################ 
# Top horizontal bar to align at the same height from top edge
################################################################ 
TITLE_DIV = html.Div(children=[
    # TITLE DIV
    html.Div(children=[
        html.H3(children='SentSpace'),
        html.Div(children='''
                visualize sentences on the backdrop of large benchmarks''',
                style=dict(float='left'),
                ),
    ], style=dict(display='inline-block', float='left', width='50%')),
    # SELECT PLOT TYPE DIV
    html.Div(children=[
        html.P(children='Plot type:'),
        dcc.Dropdown(id=f'plot-type',
                        options=[{'label': i, 'value': i} for i in ('histogram', 'scatter')],
                        value='histogram',
                    ), 
        ], style={'width': '20%', 'display': 'inline-block', 'float': 'left', "margin-left": "30px"}), 
    # SELECT MODULE DIV
    html.Div(children=[
        html.P(children='Feature Set:'),
        dcc.Dropdown(id=f'module',
                        options=[{'label': key, 'value': val} for key, val in [('lexical', 'lexical'), ('contextual', 'syntax')]],
                        value='lexical',
                    ), 
        ], style={'width': '20%', 'display': 'inline-block', 'float': 'left', "margin-left": "30px"}), 
], style={'width': '100%', 'hAlign': 'center', "margin-left": "30px"})

################################################################ 
# PLOTTING VARIABLES
################################################################ 
def PLOTTING_DROPDOWNS(df):
    return html.Div(children=[
    # UNWRAPPED LISTCOMP for THREE (3) AXES LABELS
    *[  
        html.Div(children=[
            html.Div(children=[
                html.P(children=f'{which} axis value:',
                    style=dict(height='10px', lineheight='10px')),
                dcc.Dropdown(
                    id=f'{which}axis-column',
                    options=[{'label': str(i), 'value': str(i)} for i in df.columns],
                    value=init_value,
                )
            ], style={'width': '14%', 'textAlign': 'center', 
                        'display': 'inline-block', 'float':'left',
                        "margin-left": "20px"}),
        ])  
        for which, init_value in zip('xyz', [*np.random.choice(df.columns, 3), 'surprisal-4', 'aoa', 'lexical_decision_RT'])
    ],               

    html.Div(children=[
        html.Div(children=[
            html.P(children=f'filter by sentence length:',
                style=dict(height='10px', lineheight='10px')),
            dcc.Dropdown(id=f'filter-length',
                options=[{'label': i, 'value': i} for i in [-1] + [*range(6,50)]],
                value=-1,
            ),
        ], style={'width': '14%', 'textAlign': 'center', 
                    'display': 'inline-block', 'float':'left',
                    "margin-left": "20px"}),
    ])
], id='dropdowns')

# print(PLOTTING_DROPDOWNS)

################################################################ 
# FILE UPLOAD DIV
FILE_UPLOAD = html.Div(children=[
    html.P(''),
    dcc.Upload(
        id='upload-csv',
        children=html.Div(children=[
            'Drag and Drop or ',
            html.A('Select Files'),
            ' (.tsv, .pkl, .pkl.gz)'
        ]),
        style={
            'height': '40px', 'lineHeight': '15px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'textWrap': 'true',
            'borderRadius': '5px', 'textAlign': 'center', 'margin':
            '10px', 'display': 'inline-block', 'float': 'left',
        },
        # Allow multiple files to be uploaded
        multiple=False,
    ),
], style={'width': '35%', 'display': 'inline-block'})

BUTTON_ROW = html.Div(children=[html.Hr(), PLOTTING_DROPDOWNS(load_benchmarks()), FILE_UPLOAD],
                      style={'width': '88%', 'display': 'inline-block'})

app.layout = html.Div(
    children=[
        html.Div(children=[
            TITLE_DIV,
            # BIG OL' HORIZONTAL LINE
            BUTTON_ROW, 
        ]),
        html.Div(children=[dcc.Graph(id='the-graph')])
        
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
    Output('dropdowns', 'children'),
    Input('module', 'value'),
    Input('upload-csv', 'contents'),
)
def populate_options(module='lexical', upload_contents=None):
    #return html.Div(children=
    return PLOTTING_DROPDOWNS(load_benchmarks(module)) #, id='dropdowns')


@app.callback(
    Output('the-graph', 'figure'),
    Input('plot-type', 'value'),
    # dropdowns for axis titles
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('zaxis-column', 'value'),

    Input('filter-length', 'value'),
    # uploading related args
    Input('upload-csv', 'contents'),
    State('upload-csv', 'filename'),
    # Input('year--slider', 'value')
    Input('module', 'value'),
    )
def update_graph(plot_type, 
                 x_column=None, y_column=None, z_column=None,
                 filter_length=-1,
                 upload_contents=None, filename=None,
                 module='lexical'):

    if filter_length > 0: 
        df_ = load_benchmarks(module=module, subsample=False)
    else:
        df_ = load_benchmarks(module=module)
    # print(upload_contents, filename)
    if upload_contents and filename:
        uploaded_df = parse_df(upload_contents, filename)
        uploaded_df['sentence_length'] = uploaded_df['sentence'].apply(lambda x: len(sentspace.Sentence(x, warn=False)))
        df_ = df_.append(uploaded_df)

    df_['sentence'] = df_["sentence"].apply( lambda t: "<br>".join(textwrap.wrap(t)) )

    if filter_length > 0: 
        df_ = df_[df_['sentence_length'] == int(filter_length)]

    if plot_type == 'histogram':
        fig = px.box(df_, y="corpus", x=x_column, color='corpus',
                     points='all',
                     hover_name='sentence', #notched=True,
                     height=400,
                    )
        #fig = px.histogram(df_, x=x_column, color="corpus",
        #                marginal="box", # 'rug', "box", "violin"
        #                #hover_data=df.columns
        #                histnorm='probability density',
        #                hover_name='sentence',
        #                nbins=100,
        #                )
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


def main():
    app.run_server(debug=True, port=8051)

if __name__ == '__main__':
    main()
