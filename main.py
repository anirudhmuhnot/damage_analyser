from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash
import time
import random
from google.cloud import automl_v1beta1
from base64 import decodestring
from google.oauth2 import service_account
import operator
app = dash.Dash()
app.config.supress_callback_exceptions = True
server = app.server
external_css = [
    'https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0-rc.2/css/materialize.min.css',
    'https://fonts.googleapis.com/icon?family=Material+Icons',
    'https://codepen.io/muhnot/pen/bKzaZr.css',
]

external_js = [
     'https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0-rc.2/js/materialize.min.js',
     'https://codepen.io/muhnot/pen/bKzaZr.js'
]

for my_js in external_js:
  app.scripts.append_script({"external_url": my_js})


for css in external_css:
    app.css.append_css({"external_url": css})



#loaded navbar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(className='navbar-fixed',children=[
        html.Nav(className='animated fadeIn',children=[
            html.Div(className='nav-wrapper grey darken-4',children=[
                dcc.Link(className='brand-logo right hide-on-med-and-down',children=[html.I(className='material-icons left',children=['blur_on']),'Vehicle Damage Analyser'],href='#')
            ])
        ])
    ]),
    html.Div(className='container',children=[
        html.H3("Introduction: "),
        html.H4(className='blue-text',children=['This is a Dash application which uses deep '
                                                                   'learning to '
                                                        'predict '
                                                    'the '
                                               'type of '
                                     'damage in the '
                    'vehicle.']),
        html.H3('Upload File(s): '),
        dcc.Upload(
            id='upload-data',
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
        html.Div(id='output-data-upload',className='row')
    ])
])


def get_prediction(content):
    project_id = 'prefab-root-213517'
    model_id = 'ICN8410956039961963412'
    credentials = service_account.Credentials.from_service_account_file('key.json')
    prediction_client = automl_v1beta1.PredictionServiceClient(credentials=credentials)
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'image': {'image_bytes': content}}
    params = {"score_threshold": '0.0'}
    request = prediction_client.predict(name, payload, params)
    return request  # waits till request is returned

def parse_contents(contents, filename, m):
    image = contents.split(',')[1]
    data = decodestring(image.encode('ascii'))
    with open("data/test/" + filename, "wb") as f:
        f.write(data)
    with open("data/test/" + filename, "rb") as image_file:
        content = image_file.read()
    r = get_prediction(content)
    d = {}
    for response in r.payload:
        d[response.display_name]=response.classification.score
    pred = max(d.items(), key=operator.itemgetter(1))[0]
    return html.Div(className='col s6 m6 l6 animated '+m,children=[
        html.H5(className='grey-text',children=['Filename: '+filename]),
        html.Hr(),
        #set img size
        html.Img(className='responsive-img',src=contents),
        html.H5('Predicted category of vehicle is: '+pred),
        html.Hr()
    ])



@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename')
              ])
def update_output(list_of_contents, list_of_names):
    m=str(random.choice(['fadeIn', 'pulse', 'fadeInLeft', 'fadeInRight',
                        'fadeInUp', 'flipInX', 'rotateInDownLeft',
                        'rotateInUpLeft', 'zoomIn', 'rollIn'
                        ]))
    start_time = time.time()
    children=[]
    counter = 0
    if list_of_contents is not None:
        for c, n in zip(list_of_contents,list_of_names):
            if counter % 2 == 0:
                children.append(html.Div(className='row'))
            children.append(parse_contents(c,n,m))
            counter = counter + 1
        end_time = time.time()
        total =end_time-start_time
        children.append(html.Div(className='row', children=[
            html.Br(),
            html.Div(className='row', children=[
                html.Div(className='col s12 l12 m12', children=[
                    html.H4('Total Time: ' + str(round(total, 2)) + ' second(s)'),
                    html.H4('Avg. Time per prediction: ' + str(round(total / len(
                        list_of_contents), 2)) + ' second(s)')])
            ])
        ]))

    return children

if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0',port=9601)
