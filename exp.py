import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
#from sklearn.pipeline import make_pipeline
from scipy.optimize import linprog
from scipy.optimize import dual_annealing

from scipy.optimize import minimize


from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

###############################################################################################
####################################### STREAMLIT UI ##########################################
###############################################################################################

st.set_page_config(page_title = 'Optimization for manufacturing', layout = 'wide')

fileup = st.sidebar.file_uploader('Upload csv file', type=['.csv'])

plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=6)  
plt.rc('figure', titlesize=8)        

if "fileup" not in st.session_state:
    st.session_state.fileup = False

if "response" not in st.session_state:
    st.session_state.response = False

if "model" not in st.session_state:
    st.session_state.model = False

if 'optclicked' not in st.session_state:
    st.session_state.optclicked = False

if 'perfclicked' not in st.session_state:
    st.session_state.perfclicked = False

if 'genclicked' not in st.session_state:
    st.session_state.genclicked = False

###############################################################################################
####################################### FUNCTION DEFS #########################################
###############################################################################################

@st.cache_data
def check_input_data(file):
    df = pd.read_csv(file)
    #missing data?
    nan_count = df.isna().sum()
    #all numerical?
    numeric = df.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
    return nan_count, numeric, df

def check_con_file(file, df, response):
    con = pd.read_csv(file)
    dfcols = [col for col in df if col!=response]
    if list(con.columns) == dfcols:
        return True, con
    else:
        return False, con


def plot_pairplots(df):
    pairplots = sns.pairplot(df)
    return pairplots

def plot_heatmap(df):
    correlation_matrix = df.corr()
    #fig, ax = plt.subplots()
    fig = plt.figure()
    #sns.heatmap(correlation_matrix, ax=ax, annot=True, cmap='coolwarm', fmt=".2f")
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    return fig

#@st.cache_data
def plot_histplots(df):
    ncols = int(len(df.columns)/3)
    fig, axes = plt.subplots(nrows=3,ncols=ncols, constrained_layout=True)
    fig.subplots_adjust(hspace=.3, wspace=.175)
    #fig, ax = plt.subplots()#nrows=len(list(df.columns)))
    #for i, col in enumerate(df.columns):
        #sns.distplot(df[col], ax=axes[i//n_cols,i%n_cols])#hist=False, rug=True, ax=ax)
    #    sns.histplot(data = df, x = col, ax=axes[i], discrete = True, kde = True)
    #sns.displot(data=df, kind="kde")#,ax=ax)
    #sns.kdeplot(data=df, ax=ax)
    cols = list(df.columns)
    for col, ax in zip(cols, axes.ravel()): #ravel = array to 1D, can also use axes.flatten() or axes.flat()
        sns.kdeplot(data=df, x=col, fill=True, ax=ax)
        ax.set(title=f'Distribution: {col}', xlabel=None)
    return fig

def get_best_degree(X_train,y_train,X_test,y_test):
    rmse= []
    degrees = np.arange(1,5)
    min_rmse , min_value = 1e10, 0

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        X_poly_train = poly_features.fit_transform(X_train)
    
        # Create and train a linear regression model
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
    
        X_poly_test = poly_features.fit_transform(X_test)

        # Make predictions
        y_pred = model.predict(X_poly_test)
    
        # Calculate RMSE
        mse = mean_squared_error(y_test, y_pred)
        current_rmse = np.sqrt(mse)
    
        # Store RMSE values
        rmse.append(current_rmse)
    
        # Check if this is the model with the lowest RMSE
        if current_rmse < min_rmse:
            min_rmse = current_rmse
            min_degree = degree
    
    return min_degree

def model_performance(y_test, y_pred_PR):
    fig = plt.figure()
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(122)
    sns.regplot(x=y_test, y=y_pred_PR, scatter_kws={"alpha":0.5}, line_kws={"color":"red"}, ax=ax1)
    ax1.set(title="Regression Plot", xlabel="Actual values", ylabel="Predicted values")
    sns.scatterplot(x=y_pred_PR, y=y_test-y_pred_PR, color='b', alpha=0.7, ax=ax2)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set(title="Residuals vs fitted values", xlabel="Fitted values", ylabel="Residuals")
    sns.histplot(y_test - y_pred_PR, kde=True, color='b', ax=ax3)
    ax3.set(title="Residuals distribution", xlabel="Residuals", ylabel="Frequency")
    return fig


@st.cache_data
def generate_poly_model(df, response):#, model_select):
    y = df[response]
    X = df.drop(response, axis=1)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # df_train = pd.concat([y_train, X_train], axis=1, sort=True)
    # df_test = pd.concat([y_test, X_test], axis=1, sort=True)
    best_degree = get_best_degree(X_train,y_train,X_test,y_test)
    poly_degree = PolynomialFeatures(degree = best_degree)
    X_train_pol = poly_degree.fit_transform(X_train)
    X_test_pol = poly_degree.transform(X_test)
    features = pd.DataFrame(poly_degree.transform(X), columns=poly_degree.get_feature_names_out(X.columns))

    PR = LinearRegression()
    PR.fit(X_train_pol, y_train)
    y_pred_PR = PR.predict(X_test_pol)
    mse_PR = mean_squared_error(y_test, y_pred_PR)# Calculate the Mean Squared Error (MSE)
    testscore=PR.score(X_test_pol, y_test)
    trainscore=PR.score(X_train_pol, y_train)
    intercept = PR.intercept_ 

    #get performance plots
    fig = model_performance(y_test, y_pred_PR)

    return best_degree, mse_PR, testscore, trainscore, intercept, features, fig, PR, poly_degree, y_pred_PR

def objective_max(x, model, poly_degree):
    px = poly_degree.transform(x)
    y = -model.predict(px)
    return y

def optimize_model(model, df, response, poly_degree):
    X = df.drop(response, axis=1)

    #get min, max, x0 for each param
    #boundlist = []
    cons=[]
    initial = []
    cols = list(X.columns)
    for el in range(len(cols)):
        lower = X[cols[el]].min()*0.8
        upper = X[cols[el]].max()*1.2
        #b = [lower, upper]
        initial.append(lower)
        #boundlist.append(b)
        l = {'type': 'ineq',
            'fun': lambda x, lb=lower, i=el: x[i] - lb}
        u = {'type': 'ineq',
            'fun': lambda x, ub=upper, i=el: ub - x[i]}
        cons.append(l)
        cons.append(u)

    # for factor in range(len(boundlist)):
    #     lower, upper = boundlist[factor]
    #     l = {'type': 'ineq',
    #         'fun': lambda x, lb=lower, i=factor: x[i] - lb}
    #     u = {'type': 'ineq',
    #         'fun': lambda x, ub=upper, i=factor: ub - x[i]}
    #     cons.append(l)
    #     cons.append(u)

    #objective = objective_max(x, model, poly_degree)
    #res = minimize(lambda x: -model.predict(poly_degree.transform[X]), initial, constraints=cons, method='COBYLA')
    #lambda x: model.predict(np.array([x]))
    res = minimize(lambda x: objective_max([x], model, poly_degree), initial, constraints=cons, method='COBYLA')
    #COBYLA method
    return res

def click_response():
    st.session_state.response = True

def click_opt():
    st.session_state.optclicked = True

def click_perf():
    st.session_state.perfclicked = True

def click_gen():
    st.session_state.genclicked = True



###############################################################################################
####################################### INPUT #################################################
###############################################################################################
#states: fileup, data viz/exp, gen model, residuals/fit, optimize
placeholder = st.empty()

if fileup!=None:
    nan_count, numeric, df = check_input_data(fileup)
    collist = list(df.columns)
    collist.insert(0,"None")
    response = st.sidebar.selectbox('Choose response variable', collist)
    if response!="None":
        st.sidebar.button("Show data exploration", on_click=click_response)

if st.session_state.response:
    placeholder.empty()
    plotsel = st.sidebar.selectbox("Select desired plot", options=["None","Seaborn pairplots","Heatmap","KDE histplots"])
    if plotsel=="Seaborn pairplots":
        placeholder.empty()
        pairplots = plot_pairplots(df)
        with placeholder.container():
            st.pyplot(pairplots)
    elif plotsel=="Heatmap":
        placeholder.empty()
        heatmap = plot_heatmap(df)
        with placeholder.container():
            st.pyplot(heatmap)
    elif plotsel=="KDE histplots":
        placeholder.empty()
        histplots = plot_histplots(df)
        with placeholder.container():
            st.pyplot(histplots)
    if plotsel!="None":
        st.sidebar.button("Estimate best degree and generate polynomial regression", on_click=click_gen)

if st.session_state.genclicked:
    placeholder.empty()
    best_degree, mse_PR, testscore, trainscore, intercept, features, perf, PR, poly_degree, y_pred_PR = generate_poly_model(df,response)
    with placeholder.container():
        st.write("Best degree: " + str(best_degree))
        st.write("MSE: " + str(mse_PR))
        st.write("Test score: " + str(testscore))
        st.write("Train score: " + str(trainscore))
        st.write("Intercept: " + str(intercept))
        st.write("Num features: " + str(len(list(features.columns))))
    perfb = st.sidebar.button("Show fit/performance?", on_click=click_perf)

if st.session_state.perfclicked: #takes perf from genclicked state
    placeholder.empty()
    with placeholder.container():
        st.pyplot(perf)
    opt = st.sidebar.button("Optimize model?", on_click=click_opt)

if st.session_state.optclicked:
    st.sidebar.write("Bounds provided will be [minimum - 20%, maximum + 20%] for each feature")
    placeholder.empty()
    with placeholder.container():
        result = optimize_model(PR, df, response, poly_degree)
        st.write(f"Status: {result['message']}")
        st.write(f"Total Evaluations: {result['nfev']}")
        st.write(f"Minimum reached: {result['fun']}")
        st.write(f"Solution vector: {result['x']}")



    