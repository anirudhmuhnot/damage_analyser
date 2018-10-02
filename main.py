from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash
from tensorflow.keras import backend as K
import cv2
import time
from base64 import decodestring
import numpy as np
import random
#from tensorflow import keras
import tensorflow as tf


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


def parse_contents(contents, filename, m):
    tf.keras.backend.clear_session()
    lookup = {
        0: 'Broken Windshield',
        1: 'Bumper Damage',
        2: 'Car Accident',
        3: "Flat Tire",
        4: "No Damage(flat_tire)",
        5: "No Damage(accident)",
        6: "No Damage(body)",
        7: "No Damage(windshield)",
    }

    img_width, img_height = 150, 150
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)


    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation='softmax'))
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    #
    # model.compile(loss='categorical_crossentropy',
    #                   optimizer='rmsprop',
    #                   metrics=['accuracy'])
    #
    # model.load_weights('first_try.h5')
    #
    model = tf.keras.applications.Xception(include_top=True, weights=None, input_tensor=None, input_shape=input_shape,
                                           pooling='max', classes=8)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.load_weights('xception1/weights_35epoch_8classes.h5')
    image = contents.split(',')[1]
    data = decodestring(image.encode('ascii'))
    with open("data/test/"+filename, "wb") as f:
        f.write(data)
    img = cv2.imread('data/test/'+filename)
    img = cv2.resize(img, (150, 150))
    img = img.reshape([1, 150, 150, 3])
    pred = model.predict(img)
    index = np.unravel_index(np.argmax(pred, axis=None), pred.shape)
    cat = lookup[index[1]]
    #value = pred[index]
    return html.Div(className='col s6 m6 l6 animated '+m,children=[
        html.H5(className='grey-text',children=['Filename: '+filename]),
        html.Hr(),
        #set img size
        html.Img(className='responsive-img',src=contents),
        html.H5('Predicted category of vehicle is: '+cat),
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
