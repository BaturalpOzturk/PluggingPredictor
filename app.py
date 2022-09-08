import pandas as pd
import numpy as np
import sys

from bokeh.models import Button, CustomJS
from bokeh.io import curdoc
from bokeh.models.widgets import Div, Select, TextInput, Slider
from bokeh.layouts import layout, widgetbox, column
from bokeh.models import Label
from bokeh.plotting import figure

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from bokeh.client import push_session, pull_session
import nest_asyncio
from xgboost import XGBClassifier
from bokeh.models import ColumnDataSource, TableColumn, DataTable
# Scikit has many annoying warnings...
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

##############################################################################
###                          SET UP HEADER/FOOTER                          ###
##############################################################################

# Page header and footer
page_header = Div(text = """
            <style>
            h1 {
                margin: 1em 0 0 0;
                color: #2e484c;
                font-family: 'Julius Sans One', sans-serif;
                font-size: 1.8em;
                text-transform: uppercase;
            }
            a:link {
                font-weight: bold;
                text-decoration: none;
                color: #0d8ba1;
            }
            a:visited {
                font-weight: bold;
                text-decoration: none;
                color: #1a5952;
            }
            a:hover, a:focus, a:active {
                text-decoration: underline;
                color: #9685BA;
            }
            p {
                text-align: justify;
                text-justify: inter-word;
                /*font: "Libre Baskerville", sans-serif;
                width: 90%;
                max-width: 940;*/
            }
            small {
                color: #424242;
            }

            p.big {
                margin-bottom: 0.8cm;
            }

            </style>
            <h1>Pile Plugging Condition Predictor (BETA)</h1>
            <p>
            This online tool features a <em>Machine Learning</em> fed prediction for the pile plugging condition. The process is outlined in:
            <br><br>
            <blockquote>
            Kodsy, A., Ozturk, B., and Iskander, M. G. (2022). “Forecasting of Pile Plugging using Machine Learning.” 
            </blockquote>
            </p>
            <br />
            <h3>DISCLAIMER</h3>
            <p>
            This tool is offered without any warranties about the accuracy of the predicted condition. The predicted condition is a result of approximation by scientific methodologies. The authors' sole intent is to further advance the field of Geotechnical Engineering and are not offering this online tool as a design aid. <strong>Use to learn and experiment, do not design piles based on the condition you get below.</strong>
            </p>
            <br />
            <hr />
            <br />
            """
,width = 940)

page_footer = Div(text = """
            <br>
            <hr>
            <br>
            <div>
                <div align="left">
                  <a href="http://engineering.nyu.edu/academics/departments/civil" target="_blank">
                    <img src="media/tandon_cue_color.png" style="float:left" onerror="this.src='http://drive.google.com/uc?export=view&id=0B2vCK8uO_I7kWFYwbHM3UXVNalk'" width="25%">
                  </a>
                </div>
                <div align="right" style="color:gray">
                    Developed by:<br>
		    <a href="www.google.com" target="_blank">Antonio Kodsy</a><br />
                    <a href="https://www.linkedin.com/in/baturalp-%C3%B6zt%C3%BCrk-390b23155/" target="_blank">Baturalp Ozturk</a><br />		    
                    <a href="http://engineering.nyu.edu/people/magued-g-iskander/" target="_blank">Magued Iskander</a>
                </div>
            </div>
            """
,width = 940)




##############################################################################
###                             SET UP INPUTS                              ###
##############################################################################


# Soil Type dropdown
soil_type = Select(title = "Select Predominant Soil Type:",
                   value = "Sand",
                   options = ['Sand','Clay','Mixed']
                   ,width=180)
# Soil at Pile Toe dropdown
soil_at_toe = Select(title = "Select Soil Type at Pile Toe:",
                   value = "Sand",
                   options = ['Sand','Clay', 'Rock']
                   ,width=180)

# Number of Soil Layers

no_of_layers = Slider(start=1, end=120,
               value=5, step=1,
               title="Select Number of Soil Layers",
               width=180)

# Create list of soil widgets
soil_controls = [Div(text="<strong>SOIL PROPERTIES</strong>"),
                 soil_type,
                 Div(text=""),
                 soil_at_toe,
                 Div(text=""),
                 no_of_layers]

# Create soil_inputs widgetbox
soil_inputs = widgetbox(*soil_controls, width=250)



# Diameter
diameter = Slider(start=8, end=500,
                    value=24, step=0.1,
                    title="Select diameter (in)")


# Length
length = Slider(start=9, end=173,
                    value=22.5, step=1,
                    title="Select length (ft)")
# Thickness
thickness = Slider(start=0.1, end=100,
                    value=1, step=0.1,
                    title="Select wall thickness (in)")

# Create list of pile widgets
pile_controls1 = [Div(text="<strong>PILE PROPERTIES</strong>"),
                 diameter,
                 Div(text=""),
                 length,
                 Div(text=""),
                 thickness]

# Create pile_inputs widgetbox
pile_inputs1 = widgetbox(*pile_controls1, width=200)

#Button
button = Button(label="Calculate", button_type="success")



##############################################################################
###                           RUN CALCULATIONS                             ###
##############################################################################

# -- Imports ------------------------------------------------------------------
import numpy as np
import pandas as pd


from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


import scipy.stats as ss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.naive_bayes import ComplementNB



import warnings
warnings.filterwarnings(action='once')

import warnings
warnings.filterwarnings(action='once')

from IPython.display import clear_output

# from figure__shared import df_cpt, cpt_methods, plot_setup_log, df_nocpt, all_methods,\
# plot_setup_qcqm_ld, plot_setup_qcqm, plot_setup_qcqm_l

# -----------------------------------------------------------------------------
# -- SI -----------------------------------------------------------------------
# -----------------------------------------------------------------------------
def kip2mn(x):
    return x * 0.0044482216

def mn2kip(x):
    return x * 224.80894387096

def in2m(x):
    return x * 0.0254

def m2in(x):
    return x * 39.3701

def ft2m(x):
    return x * 0.3048

def m2ft(x):
    return x * 3.28084

# feature_names_dict = {'O.D.':'D', 
#                       'pile_length':'L', 
#                       'L/D':'L/D'}

feature_names_dict = {'O.D.':'Diameter', 
                      'pile_length':'Length', 
                      'L/D':'L/D', 
                      'thickness': 'Thickness',
                      'Soil@Tip_cat': 'Soil @ Tip', 
                      'Soil_type_cat': 'Soil Type'}

X_test = pd.read_excel('X_test.xlsx', engine='openpyxl')
X_train = pd.read_excel('X_train.xlsx', engine='openpyxl')
y_test = pd.read_excel('y_test.xlsx', engine='openpyxl')
y_train = pd.read_excel('y_train.xlsx', engine='openpyxl')

features_list = [
                 ['O.D.', 'L/D', 'thickness']  #16
                      #17
                ]
 
# create loocv procedure
cv = LeaveOneOut()


i = 1
data = []
model_results = []


    

for features in features_list:

    #----------------------------------------------------------------------------
    #----SVM---------------------------------------------------------------------
    #----------------------------------------------------------------------------
    svm  = SVC(max_iter=10000, kernel='linear', C=1, probability=True)

    #----------------------------------------------------------------------------
    #----Logistic Regression-----------------------------------------------------
    #----------------------------------------------------------------------------
    lr   = LogisticRegression(max_iter=100000, 
                              random_state=1,
                              C=0.001,
                              solver='liblinear')

    #----------------------------------------------------------------------------
    #----kNN---------------------------------------------------------------------
    #----------------------------------------------------------------------------

    kNN  = KNeighborsClassifier()

    #----------------------------------------------------------------------------
    #----Decision Tree-----------------------------------------------------------
    #----------------------------------------------------------------------------
    tree = DecisionTreeClassifier(min_samples_leaf=10,
                                  max_leaf_nodes=3,
                                  random_state=1)

    #----------------------------------------------------------------------------
    #----Random Forest----------------------------------------------------------
    #----------------------------------------------------------------------------
    rf   = RandomForestClassifier(min_samples_leaf=10,
                                  n_estimators=1000,
                                  max_leaf_nodes=3,
                                  n_jobs=-1,
                                  random_state=1)

    #----------------------------------------------------------------------------
    #----AdaBoost----------------------------------------------------------------
    #----------------------------------------------------------------------------
    ada  = AdaBoostClassifier(n_estimators=1000)

    #----------------------------------------------------------------------------
    #----XGBoost-----------------------------------------------------------------
    #----------------------------------------------------------------------------
    xgb  = XGBClassifier(eval_metric='mlogloss', use_label_encoder =False)

    #----------------------------------------------------------------------------
    #----MLP---------------------------------------------------------------------
    #----------------------------------------------------------------------------
    mlp  = MLPClassifier(max_iter=10000,
                         random_state=1,
                         hidden_layer_sizes=15,
                         learning_rate='constant',
                         activation='logistic',
                         solver='lbfgs')

    #----------------------------------------------------------------------------
    #----Soft Voting Classifier--------------------------------------------------
    #----------------------------------------------------------------------------
    voting_C = VotingClassifier(estimators=[
                                            ('svc', svm), 
                                            ('lr', lr), 
                                            ('kNN', kNN),
                                            ('Decision Tree', tree), 
                                            ('Random Forrest', rf),
                                            ('AdaBoost', ada),
                                            ('XGBoost', xgb),
                                            ('MLP', mlp)
                                           ], 
                                weights= [1,1,1,1,1,1,1,1],
                                voting='soft', 
    #                                 n_jobs=-1
                           )
    #pipes = Pipeline([#('scaler', StandardScaler()),
                     #('svc', svm)
                     #('lr', lr)
                     #('kNN', kNN)
                     #('Decision Tree', tree)
                     #('Random Forrest', rf)
                     #('mlp', mlp)
                     #('AdaBoost', ada)
                     #('XGBoost', xgb)
                     #('votingC', voting_C)
                    #])
    svc = Pipeline([('scaler', StandardScaler()),
                     ('svc', svm)
                    ])
    lr = Pipeline([('scaler', StandardScaler()),
                     ('lr', lr)
                    ])
    kNN = Pipeline([('scaler', StandardScaler()),
                     ('kNN', kNN)
                    ])
    tree = Pipeline([('scaler', StandardScaler()),
                     ('Decision Tree', tree)
                    ])
    rf = Pipeline([('scaler', StandardScaler()),
                     ('Random Forrest', rf)
                    ])
    ada = Pipeline([('scaler', StandardScaler()),
                     ('AdaBoost', ada)
                    ])
    votingC = Pipeline([('scaler', StandardScaler()),
                     ('votingC', voting_C)
                    ])
    svc.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    kNN.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    ada.fit(X_train, y_train)
    votingC.fit(X_train, y_train)









##############################################################################
###                              SET UP PLOT                               ###
##############################################################################
results_df = pd.DataFrame({
	'Model_Name' : ['SVM' , 'Log. Regression', 'kNN', 'Decision Tree', 'Random Forest', 'MLP' , 'AdaBoost', 'Soft Voting'],
	'Results' : ['' , '', '','', '','',  '' ,'' ]})

source = ColumnDataSource(results_df)

columns = [
        TableColumn(field='Model_Name', title='Model'),
        TableColumn(field='Results', title='Prediction'),
    ]
myTable = DataTable(source=source, columns=columns)


plot = figure(x_range=(0, 10),
              y_range=(0, 10),
              plot_height=120,
              plot_width=940)
plot.axis.visible = False
plot.toolbar.logo = None
plot.toolbar_location = None
plot.xgrid.grid_line_color = None
plot.ygrid.grid_line_color = None

label = Label(x=5, y=2,
              text= 'Loading..',
              text_font_size='20pt',
              text_color='#7f7f7f',
              text_align='center',
              text_baseline='middle')
label2 = Label(x=5, y=5,
              text= 'Loading..',
              text_font_size='20pt',
              text_color='#7f7f7f',
              text_align='center',
              text_baseline='middle')
plot.add_layout(label)
plot.add_layout(label2)




##############################################################################
###                            SET UP CALLBACK                             ###
##############################################################################



# Define a callback function: callback
#def callback(attr, old, new):
def callback():
    # Read the current values
    new_soil_type = soil_type.value
    new_soil_at_toe = soil_at_toe.value
    new_no_of_layers = no_of_layers.value
    new_diameter = diameter.value
    new_length = length.value
    new_thickness = thickness.value
    

    # Create dummies for soil_type
    if new_soil_type == 'Sand':
        new_soil_type = [0,0,1]
    elif new_soil_type == 'Clay':
        new_soil_type = [1,0,0]
    else:
        new_soil_type = [0,1,0]

    # Create dummies for soil_at_toe
    if new_soil_at_toe == 'Sand':
        new_soil_at_toe = [0,0,1]
    elif new_soil_at_toe == 'Clay':
        new_soil_at_toe = [1,0,0]
    else:
        new_soil_at_toe = [0,1,0]

    lod = new_length/new_diameter

    # Compile new inputs

    #new_inputs2 = ([float(new_diameter)] + [float(lod)] + [float(new_length)] + [float(new_thickness)]
    #              + new_soil_type + new_soil_at_toe + [float(new_no_of_layers)])
    new_inputs2 = ([float(new_diameter)] + [float(lod)] + [float(new_thickness)])
                  


    X_testt = X_test.copy()
    X_testt.loc[len(X_testt.index)] = new_inputs2
    #df2 = X_test.iloc[[-1]]
    #HERE YOU NEED TO RERUN THE ANALYSIS BY (PREDICT) COMMAND. CHECK THE WORKING FILE

    svc_predict = str(svc.predict(X_testt.iloc[[-1]]))[2:-2] 
    lr_predict = str(lr.predict(X_testt.iloc[[-1]]))[2:-2]
    knn_predict = str(kNN.predict(X_testt.iloc[[-1]]))[2:-2] 
    dt_predict = str(tree.predict(X_testt.iloc[[-1]]))[2:-2] 
    rf_predict = str(rf.predict(X_testt.iloc[[-1]]))[2:-2] 
    mlp_predict = str(mlp.predict(X_testt.iloc[[-1]]))[2:-2]
    ada_predict = str(ada.predict(X_testt.iloc[[-1]]))[2:-2]
    votingC_predict = str(votingC.predict(X_testt.iloc[[-1]]))[2:-2]

    results_df = pd.DataFrame({
	'Model_Name' : ['SVM' , 'Log. Regression', 'kNN', 'Decision Tree', 'Random Forest', 'MLP' , 'AdaBoost', 'Soft Voting'],
	'Results' : [svc_predict , lr_predict , knn_predict , dt_predict , rf_predict , mlp_predict,  ada_predict ,votingC_predict ]})
    myTable.source.data = results_df



    label.text = str ('svm:'+ svc_predict + '\t LogReg:'+ lr_predict  + '\t kNN:'+ knn_predict )
    label2.text =str ('DT:'+dt_predict  + '\n RF:'+rf_predict  + '\n AdaBoost:'+ ada_predict  + '\n SoftVoting:'+ votingC_predict )
    
#label.text = str(len(new_inputs2))
# Call the callback function on update of these fields
#for i in [soil_type, soil_at_toe, no_of_layers,
#          diameter,length,thickness]:
#for i in [diameter,length,thickness]:
#    i.on_change('value', callback)
button.on_click(callback)


##############################################################################
###                             SET UP LAYOUT                              ###
##############################################################################

# Set up initial page layout
page_layout = layout([[page_header],
                      [soil_inputs,pile_inputs1],
		      [button],
                      [Div(text="<br><h2>RESULT</h2>")],
                      [myTable],
                      [page_footer]],
                      width = 940)


# Add the page layout to the current document
curdoc().add_root(page_layout)
curdoc().title = "Plugging Predictor"

nest_asyncio.apply()
#bokeh serve --show myapp.py
#session = push_session(curdoc())
#session.show()
# run with:
    # bokeh serve --show plugging_predictor.py
    # bokeh serve --show untitled2.py
# run forever on server with:
# nohup bokeh serve plugging_predictor.py --allow-websocket-origin cue3.engineering.nyu.edu:22 --host cue3.engineering.nyu.edu:22 --port 22
