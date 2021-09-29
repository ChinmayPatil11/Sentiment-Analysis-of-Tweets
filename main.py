import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import spacy
import re
from get_tweets import get_results
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


app = dash.Dash(__name__)
df = get_results('random')
df = df.drop(columns=['created_at', 'tweet_id', 'cleaned_text', 'length', 'entities'])
filtered_df = pd.DataFrame()

fig = px.bar(x=df['prediction'].value_counts().index, y=df['prediction'].value_counts())
fig.update_layout(xaxis_title='Sentiment', yaxis_title='Number of tweets',title_x=0.5)

app.layout = html.Div(
    children=[
            html.Title('Sentiment Analysis of tweets'),
            html.Div([
            html.H4('Enter a word'),
            dcc.Input(id='input-text', value='', type='text'),
            html.Button(id='submit-button', type='submit', children='Submit'),
            html.H4('Press submit to update the tweets once the graph updates'),
        ]),
        html.H1('Tweets Analysis',style={'textAlign': 'center'}),
        dcc.Graph(id='graph-with-input',
                  figure=fig),
        dash_table.DataTable(id='trend-table',
                             columns=[{'name': i, 'id': i, 'selectable':True} for i in df.columns],
                             data=df.to_dict('records'),
                             style_cell={
                                 'whiteSpace': 'normal',
                                 'height': 'auto',
                             },
                             style_data_conditional=[
                                 {
                                     'if': {
                                         'filter_query': "{prediction} = 'positive'"
                                     },
                                     'backgroundColor': 'green'
                                 },
                                 {
                                     'if': {
                                         'filter_query': "{prediction} = 'negative'"
                                     },
                                     'backgroundColor': 'red'
                                 }
                             ]
                             ),
        html.H4('Entities'),
        html.Div(id='selected-entity')
])


@app.callback(
    Output('graph-with-input','figure'),
    [Input('input-text','value')]
)
def update_figure(query):
    global filtered_df
    filtered_df = get_results(query)
    fig = px.bar(x=filtered_df['prediction'].value_counts().index,
                 y=filtered_df['prediction'].value_counts(), title=query)
    fig.update_layout(xaxis_title='Sentiment', yaxis_title='Number of tweets',title_x=0.5)
    return fig


@app.callback(
    Output('trend-table', 'data'),
    Input('submit-button', 'n_clicks'),
    State('input-text', 'value')
)
def update_table(n_clicks, value):
    print(n_clicks, value)
    table = filtered_df
    table = table.to_dict('records')
    return table


@app.callback(
    Output('selected-entity', 'children'),
    Input('trend-table', 'active_cell')
)
def update_entity_div(active_cell):
    if len(filtered_df.columns) == 0:
        return 'Click on a tweet to find the entities'
    else:
        txt = filtered_df['tweet_text'][active_cell['row']]
        txt = txt.replace('@', '')
        txt = txt.replace('#', '')
        txt = re.sub('https\S+', '', txt)
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(txt)
        entlist = []
        for ent in doc.ents:
            entity = ''
            entity = entity + ent.text + '-' + ent.label_
            entlist.append(entity)
        if len(entlist) == 0:
            return 'No Entities present'
        else:
            return html.Ul([html.Li(i) for i in entlist])


if __name__ == '__main__':
    app.run_server(debug=True)