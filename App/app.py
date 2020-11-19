"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location. There are two callbacks,
one uses the current location to render the appropriate page content, the other
uses the current location to toggle the "active" properties of the navigation
links.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import csv
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
from modelos import predictModel1, loadImage,predictModel2

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

USER_SELECTION=['HOLA']

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Bone Age App", className="display-4"),
        html.Hr(),
        html.P(
            "RSNA Bone Age Predict Age from X-Rays", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Inicio", href="/page-1", id="page-1-link"),
                dbc.NavLink("Análisis de datos",
                            href="/page-2", id="page-2-link"),
                dbc.NavLink("Predicción", href="/page-3", id="page-3-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

imagePicker = html.Div([
    html.Div('Seleccione los modelos que quiere utilizar para predecir:'),
    dcc.Checklist(
        id='checkListModel',
        options=[
            {'label': 'Modelo 1', 'value': 'M1'},
            {'label': 'Modelo 2', 'value': 'M2'}
        ],
        value=['M1', 'M2'],
        labelStyle={'display': 'block'}
    ),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
    html.Div(id='output-selection'),
])

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

###########################################Page No.1############################################################


markdown_titel = " # Acerca del proyecto"
markdown_text = '''
Existen dos procesos biológicos íntimamente relacionados de un individuo los cuales no necesariamente van paralelos a lo largo de la infancia y adolescencia, estos dos procesos son el crecimiento y la maduración. Cada niño madura a distinta velocidad, es decir que la edad no es un buen indicativo para ello (Pérez, 2011). 

La edad ósea (EO) es la mejor forma de expresar la edad biológica de una persona, y es de suma importancia en el campo de la medicina dado que muchos tratamientos y procedimientos deben ser cuidadosamente basados en la edad del individuo. Esto también permite evaluar la maduración ósea que, según medios, es "es un fenómeno biológico a través del cual los seres vivos incrementan su masa adquiriendo progresivamente una maduración morfológica y funcional" (Europapress, 2015).

El principal problema se centra en que, a pesar de tener una radiografía presente y clara, no existe un proceso 100% automatizado que identifique la edad ósea de una persona. El procedimiento se centra siempre en el criterio de un experto y dado que es una tarea manual, existirá variabilidad interindividual. Esto abre lugar a resultados raramente precisos y expuestos a un porcentaje de error humano (Abad. D, 2011).

Las computadoras y el uso de algoritmos complejos, junto con las aplicaciones de inteligencia artificial (como lo es machine learning y deep learning), han presentado resultados prontos y exactos. Esto da lugar a un aumento de eficiencia y confiabilidad en los resultados, por lo que obtener un modelo capaz de predecir la edad ósea de un individuo es sumamente conveniente.
'''

markdown_biblio = '''
* Pérez, R. (2011). Valoración y utilidad de la edad ósea en la práctica clínica. Extraído 05 de septiembre del 2020 de: https://fapap.es/articulo/180/valoracion-y-utilidad-de-la-edad-osea-en-la-practica-clinica
* Europapress. (2015). Una App permite obtener la edad ósea del niño y su predicción de talla adulta. Extraído 6 de septiembre del 2020, de: https://www.europapress.es/economia/red-empresas-00953/noticia-app-permite-obtener-edad-osea-nino-prediccion-talla-adulta-20151214123040.html
* Abad. D, L. J. (2011) Estimación automática de la edad ósea mediante procesado y segmentación de radiografías, Universidad Carlos III de Madrid. Disponible en: https://e-archivo.uc3m.es/bitstream/handle/10016/13728/PFC_DANIEL_ABAD.pdf?sequence=1&isAllowed=y.
'''

markdown_biblio_titel = " # Referencias"

page1 = html.Div([
    dcc.Markdown(
        children=markdown_titel,
        style={
            'textAlign': 'justify',
            'padding': '25px 50px 25px',
        },
    ),
    dcc.Markdown(
        children=markdown_text,
        style={
            'textAlign': 'justify',
            'padding': '25px 50px 50px',
            'fontSize': '14'
        },
    ),
    dcc.Markdown(
        children=markdown_biblio_titel,
        style={
            'textAlign': 'justify',
            'padding': '25px 50px 25px',
        },
    ),
    dcc.Markdown(
        children=markdown_biblio,
        style={
            'textAlign': 'justify',
            'padding': '25px 50px 75px',
            'fontSize': '14'
        },
    )
])

#############################################Page No.2############################################################
import pandas as pd 
df = pd.read_csv('boneage-training-dataset.csv')
df2 = pd.read_csv('dataModelos.csv')

fig = px.bar(
    df.groupby(['boneage']).size().reset_index(name='count')
    , title ="Cantidad de imágenes por cantidad de meses"
    , x = 'boneage'
    , y = 'count'
    , labels = {"boneage": "Edad Ósea (meses)", "count": "Cantidad de imágenes"}
    , color_discrete_sequence = ["#004B8F", "#DB8700"]
    )

fig2 = px.pie(
    df.groupby(['male']).size().reset_index(name='count')
    , values = 'count'
    , title ="Cantidad de imágenes por género"
    , names='male'
    , labels = {"male": "Género", "count": "Cantidad de imágenes"}
    , color_discrete_sequence = ["#004B8F", "#DB8700"]
)

fig3 = go.Figure(data=[
    go.Bar(name='Predicción', x=df2.loc[1:,'Iteracion'], y=df2.loc[:,'predMod1'] , marker_color="#004B8F" ),
    go.Bar(name='Valor real', x=df2.loc[1:,'Iteracion'], y=df2.loc[:,'edadMod1'], marker_color="#DB8700"),
])
fig3.update_xaxes(title_text="Número de iteración")
fig3.update_yaxes(title_text="Edad Ósea (meses)")
fig3.update_layout(title_text='Predicciones modelo #1 (Iteración/ Predicción en meses)')

fig4 = go.Figure(data=[
    go.Bar(name='Predicción', x=df2.loc[1:,'Iteracion'], y=df2.loc[:,'predMod2'],marker_color="#004B8F"),
    go.Bar(name='Valor real', x=df2.loc[1:,'Iteracion'], y=df2.loc[:,'edadMod2'], marker_color="#DB8700")
])
fig4.update_xaxes(title_text="Número de iteración")
fig4.update_yaxes(title_text="Edad Ósea (meses)")
fig4.update_layout(title_text='Predicciones modelo #2 (Iteración/ Predicción en meses)')


diffMod1 = df2['edadMod1'] - df2['predMod1']
diffMod2 = df2['edadMod2'] - df2['predMod2']

fig5 = go.Figure(data=[
    go.Bar(name='Modelo #1', x=df2.loc[1:,'Iteracion'], y=diffMod1 , marker_color="#004B8F"),
    go.Bar(name='Modelo #2', x=df2.loc[1:,'Iteracion'], y=diffMod2 , marker_color="#DB8700")
])
fig5.update_xaxes(title_text="Número de iteración")
fig5.update_yaxes(title_text="Diferencia en la Edad Ósea (meses)")
fig5.update_layout(title_text='Comparación del Error Entre Modelos por Iteración (Iteración/ Error en meses)')


page2 = html.Div([
    dcc.Dropdown(
        id = 'dropdown-to-show_or_hide-element',
        options=[
            {'label': 'Modelo 1', 'value': 'one'},
            {'label': 'Modelo 2', 'value': 'two'},
            {'label': 'Ambos modelos', 'value': 'both'}
        ],
        value = 'one'
    ),
    dcc.Graph(
            id='example-graph-1',
            figure=fig
        ),
    html.Div([
        dcc.Graph(
                id='example-graph-2',
                figure=fig2
            ),
        html.Div([
            dcc.Graph(
                    id='element-to-hide1',
                    figure=fig3
                )
        ], style= {'display': 'block'} ),
        html.Div([
            dcc.Graph(
                id='element-to-hide2',
                figure=fig4
            )], style= {'display': 'block'} ),
        html.Div([
            dcc.Graph(
                id='element-to-hide3',
                figure=fig5
            )], style= {'display': 'block'} )
    ], style={
            'display': 'flex',
            'flexDirection': 'column',
            'width': '100%',
            'flexWrap': 'wrap'
        }),
])

@app.callback(
   Output(component_id='element-to-hide1', component_property='style'),
   [Input(component_id='dropdown-to-show_or_hide-element', component_property='value')])


def show_hide_element(visibility_state):
    if (visibility_state == 'one' or visibility_state == 'both'):
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
   Output(component_id='element-to-hide2', component_property='style'),
   [Input(component_id='dropdown-to-show_or_hide-element', component_property='value')])

def show_hide_element(visibility_state):
    if (visibility_state == 'two' or visibility_state == 'both'):
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
   Output(component_id='element-to-hide3', component_property='style'),
   [Input(component_id='dropdown-to-show_or_hide-element', component_property='value')])

def show_hide_element(visibility_state):
    if visibility_state == 'both':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('output-selection', 'children'),
              [Input('checkListModel', 'value')])
def getUserSelection(selection):
    print(selection)
    global USER_SELECTION
    print(USER_SELECTION)
    USER_SELECTION=selection


###################################################################################################################v
def parse_contents(contents, filename, date):
    # print(contents)
    global USER_SELECTION

    if (len(USER_SELECTION)==2):
        return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, style={'height':'35%', 'width':'35%'}),
        html.Hr(),
        html.Div('Predicciones: '),
        html.Pre("Modelo 1: "+str(predictModel1(loadImage(contents))), style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        }),
        html.Pre("Modelo 2: "+str(predictModel2(loadImage(contents))), style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

    try:
        if (USER_SELECTION[0]=='M1'):
            return html.Div([
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            html.Img(src=contents, style={'height':'35%', 'width':'35%'}),
            html.Hr(),
            html.Div('Predicciones: '),
            html.Pre("Modelo 1: "+str(predictModel1(loadImage(contents))), style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all'
            })
        ])

        if (USER_SELECTION[0]=='M2'):
            return html.Div([
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            html.Img(src=contents, style={'height':'35%', 'width':'35%'}),
            html.Hr(),
            html.Div('Predicciones: '),
            html.Pre("Modelo 2: "+str(predictModel2(loadImage(contents))), style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all'
            })
        ])
    except:
        return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Div('Seleccione que modelo quiere usar!')])
        

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return page1
        #return html.P("Contenido del !")
    elif pathname == "/page-2":
        return page2
    elif pathname == "/page-3":
        return imagePicker
        #return html.P("Contenido de la prediccion")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(port=8888)
