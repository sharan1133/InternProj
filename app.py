from flask import Flask, render_template, request , send_from_directory
import pickle
import joblib
import pandas as pd
import numpy as np
import dash
import dash_renderer
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import tensorflow as tf
import plotly.express as px
#from tensorflow.keras.models import load_model


app = Flask(__name__)
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/' , external_stylesheets=[dbc.themes.BOOTSTRAP])

#df = pd.read_csv('/home/sharanbalakrishnan/Desktop/InternProj/df2.csv')
df = pd.read_csv('/home/sharanbalakrishnan/Desktop/InternProj/data/data_dash.csv')

dash_app.layout = html.Div(children=[
    html.Title('Mobile Phone Prices and Features Dashboard'),
    html.Link(rel='stylesheet', href='/static/css/style.css'),

    html.H1(children='Mobile Phone Prices and Features'),

    html.Div(children='''
        Select a brand to see the average price of their phones:
    '''),

    dcc.Dropdown(
        id='brand-dropdown',
        options=[{'label': brand, 'value': brand} for brand in df['Brand'].unique()],
        value='Apple'
    ),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='price-graph',
                style={'height': '400px'}
            ),
            width={'size': 4}
        ),

        dbc.Col(
            dcc.Graph(
                id='color-counts',
                figure={
                    'data': [
                        go.Bar(
                            x=df['Color'].value_counts().index,
                            y=df['Color'].value_counts().values
                        )
                    ],
                    'layout': go.Layout(
                        title='Count of Mobile Phone Colors',
                        xaxis={'title': 'Color'},
                        yaxis={'title': 'Count'}
                    )
                },
                style={'height': '400px'}
            ),
            width={'size': 4}
        ),

        dbc.Col(
            dcc.Graph(
                id='brand-counts',
                figure={
                    'data': [
                        go.Bar(
                            x=df['Brand'].value_counts().index,
                            y=df['Brand'].value_counts().values
                        )
                    ],
                    'layout': go.Layout(
                        title='Count of Mobile Phone Brands',
                        xaxis={'title': 'Brand'},
                        yaxis={'title': 'Count'}
                    )
                },
                style={'height': '400px'}
            ),
            width={'size': 4}
        ),
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='ram-counts',
                figure={
                    'data': [
                        go.Bar(
                            x=df['Ram'].value_counts().index,
                            y=df['Ram'].value_counts().values
                        )
                    ],
                    'layout': go.Layout(
                        title='Count of Mobile Phone RAM Sizes',
                        xaxis={'title': 'RAM Size'},
                        yaxis={'title': 'Count'}
                    )
                },
                style={'height': '400px'}
            ),
            width={'size': 4}
        ),

        dbc.Col(
            dcc.Graph(
                id='price-distribution',
                figure={
                    'data': [
                        go.Histogram(
                            x=df['Price'],
                            nbinsx=30
                        )
                    ],
                    'layout': go.Layout(
                        title='Distribution of Mobile Phone Prices',
                        xaxis={'title': 'Price'},
                        yaxis={'title': 'Count'}
                    )
                },
                style={'height': '400px'}
            ),
            width={'size': 4}
        ),

        dbc.Col(
               dcc.Graph(
               id='Scatterplot',
               figure=px.scatter(df, x='Display_Size', y='Price', color='Display_Type',
                          title='Mobile Phone Display Size vs Price'),
    ),
    #width={'size': 4, 'offset': 3}
    width={'size': 4}
),
]),
dbc.Row([

        dbc.Col(
             dcc.Graph(
             id='scatterplot',
             figure={
                'data': [
                go.Scatter3d(
                    x=df['Storage'],
                    y=df['Price'],
                    z=df['Ram'],
                    mode='markers',
                    marker={
                        'size': 8,
                        'opacity': 0.7,
                        'color': 'rgb(255,0,0)',
                        'colorscale': 'Viridis'
                    }
                )
            ],
            'layout': go.Layout(
                title='Mobile Phone Storage, Price and Ram',
                scene={
                    'xaxis': {'title': 'Storage (GB)'},
                    'yaxis': {'title': 'Price '},
                    'zaxis': {'title': 'Ram'}
                },
                margin={'l': 0, 'r': 0, 'b': 0, 't': 40},
                height=500
            )
        }
    ),
    #width={'size': 4, 'offset': 0}
    width={'size': 4}
),
  

    ])
])




    

# ...


@dash_app.callback(
    dash.dependencies.Output('price-graph', 'figure'),
    [dash.dependencies.Input('brand-dropdown', 'value')]
)
def update_price_graph(selected_brand):
   
    brand_data = df[df['Brand'] == selected_brand]

    
    avg_price = brand_data.groupby('Storage')['Price'].mean()

    
    trace = go.Scatter(
        x=avg_price.index,
        y=avg_price.values,
        mode='lines+markers'
    )

    
    layout = go.Layout(
        title=f'Average Price of {selected_brand} Phones by Storage Size',
        xaxis={'title': 'Storage Size (GB)'},
        yaxis={'title': 'Price (USD)'}
    )

    
    figure = {'data': [trace], 'layout': layout}

    return figure

#model = joblib.load('model.joblib')
ct = joblib.load('/home/sharanbalakrishnan/Desktop/InternProj/data/column_transformer.pkl')

#with open(r'/home/sharanbalakrishnan/Desktop/InternProj/data/my_rf.bin', 'rb') as f:
    #model = pickle.load(f)

model = tf.keras.models.load_model('/home/sharanbalakrishnan/Desktop/InternProj/data/my_model_tf.h5')



#color_mapping = {'blue': 0, 'black': 1, 'gold': 2, 'grey': 3, 'green': 4, 'white': 5, 'silver': 6,
                 #'yellow': 7, 'carbon': 8, 'purple': 9, 'orange': 10, 'pearl': 11, 'cream': 12}

#display_type_mapping = {'HD+': 0, 'AMOLED': 1, 'HD': 2, 'XDR': 3, 'Retina': 4}

'''@app.route('/')
def home():
    return render_template('index.html')'''

@app.route('/predict', methods=['POST' , 'GET'])
def predict():

    if request.method == 'POST':
        print(request.form)
        ROM = int(request.form['ROM'])
        RAM = int(request.form['RAM'])
        Color = request.form['Color']
        Front_Cam = float(request.form['Front_Cam'])
        Rear_Cam = float(request.form['Rear_Cam'])
        Display_Size = float(request.form['Display_Size'])
        Display_Type = request.form['Display_Type']

        new_data = {
                     'RAM': RAM,
                     'ROM': ROM,
                     'Color': Color,
                     'Display_Type': Display_Type,
                     'Display_Size': Display_Size,
                     'Front_Cam': Front_Cam,
                     'Rear_Cam': Rear_Cam
                   }

        new_df = pd.DataFrame([new_data])

        new_data = ct.transform(new_df)

    
        #prediction = model.predict([[storage, ram, color, front_cam, rear_cam, display, display_type]])
        prediction = model.predict(new_data)
        #predicted_mobile = prediction[0]
        predicted_mobile = float(prediction[0])
    
        #return render_template('result.html', result=predicted_mobile)
        #return render_template('index.html', prediction_text='Predicted Mobile Price: ₹ {:.2f}'.format(predicted_mobile))
        #return render_template('index.html', prediction_text=predicted_mobile)
        return render_template('index.html', prediction_text='Price is: ₹ {:.2f}'.format(predicted_mobile))

    else:
        return render_template('index.html')


@app.route('/dashboard/')
def dashboard():
    return dash_app.index()

@app.route('/static/css/style.css')
def serve_css():
    return send_from_directory('static/css', 'style.css')

@app.route("/")
def home():
    return render_template("home.html")



if __name__ == '__main__':
    app.run(debug=True)
