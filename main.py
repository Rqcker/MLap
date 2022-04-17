"""
RAInS Project: machine-learning analysis platform
Author: Junhao Song
Email: songjh.john@gmail.com
Website: http://junhaosong.com/
"""

import os
import mlflow
# use streamlit to achieve interactive use on the web side
import streamlit as st
import pandas as pd
# used to display the report in the web page
from streamlit_pandas_profiling import st_profile_report
# used to generate reports
from pandas_profiling import ProfileReport
# machine learning classification
import pycaret.classification as pc_cl
# machine learning regression
import pycaret.regression as pc_rg

# store some commonly used machine learning modeling techniques
ML_LIST = ['Regression', 'Classification']
RG_LIST = ['lr', 'svm', 'rf', 'xgboost', 'lightgbm']
CL_LIST = ['lr', 'dt', 'svm', 'rf', 'xgboost', 'lightgbm']


# list certain extension files in the folder
def list_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith('.' + extension)]


# read logs.log, display the number of the last
# selected line, the user can set the number of lines
def get_model_training_logs(n_lines = 10):
    file = open('logs.log', 'r')
    lines = file.read().splitlines()
    file.close()
    return lines[-n_lines:]


# get the full path of the file, used to read the dataset
def concat_file_path(file_folder, file_selected):
    if str(file_folder)[-1] != '/':
        fileSelectedPath = file_folder + '/' + file_selected
    else:
        fileSelectedPath = file_folder + file_selected
    return fileSelectedPath


# load the data set, put the data set into the cache
@st.cache(suppress_st_warning=True)
def load_csv(file_selected_path, nrows):
    try:
        if nrows == -1:
            df = pd.read_csv(file_selected_path)
        else:
            df = pd.read_csv(file_selected_path, nrows=nrows)
    except Exception as ex:
        df = pd.DataFrame([])
        st.exception(ex)
    return df


def app_main():
    st.title("Machine learning analysis platform")
    if st.sidebar.checkbox('Define Data Source'):
        filesFolder = st.sidebar.text_input('folder', value="data")
        dataList = list_files(filesFolder, 'csv')
        if len(dataList) ==0:
            st.warning('No data set available')
        else:
            file_selected = st.sidebar.selectbox(
                'Select a document', dataList)
            file_selected_path = concat_file_path(filesFolder, file_selected)
            nrows = st.sidebar.number_input('Number of lines', value=-1)
            n_rows_str = 'All' if nrows == -1 else str(nrows)
            st.info('Selected file：{file_selected_path}，The number of rows read is{n_rows_str}')
    else:
        file_selected_path = None
        nrows = 100
        st.warning('The currently selected file is empty, please select:')
    if st.sidebar.checkbox('Exploratory Analysis'):
        if file_selected_path is not None:
            if st.sidebar.button('Report Generation'):
                df = load_csv(file_selected_path, nrows)
                pr = ProfileReport(df, explorative=True)
                st_profile_report(pr)
        else:
            st.info('No file selected, analysis cannot be performed')
    if st.sidebar.checkbox('Modeling'):
        if file_selected_path is not None:
            task = st.sidebar.selectbox('Select Task', ML_LIST)
            if task == 'Regression':
                model = st.sidebar.selectbox('Select Model', RG_LIST)
            elif task == 'Classification':
                model = st.sidebar.selectbox('Select Model', RG_LIST)
            df = load_csv(file_selected_path, nrows)
            try:
                cols = df.columns.to_list()
                target_col = st.sidebar.selectbox('Select Prediction Object', cols)
            except BaseException:
                st.sidebar.warning('The data format cannot be read correctly')
                target_col = None

            if target_col is not None and st.sidebar.button('Training Model'):
                if task == 'Regression':
                    st.success('Data preprocessing...')
                    pc_rg.setup(
                        df,
                        target=target_col,
                        log_experiment=True,
                        experiment_name='ml_',
                        log_plots=True,
                        silent=True,
                        verbose=False,
                        profile=True)
                    st.success('Data preprocessing is complete')
                    st.success('Training model. . .')
                    pc_rg.create_model(model, verbose=False)
                    st.success('The model training is complete. . .')
                    #pc_rg.finalize_model(model)
                    st.success('Model has been created')
                elif task == 'Classification':
                    st.success('Data preprocessing. . .')
                    pc_cl.setup(
                        df,
                        target=target_col,
                        fix_imbalance=True,
                        log_experiment=True,
                        experiment_name='ml_',
                        log_plots=True,
                        silent=True,
                        verbose=False,
                        profile=True)
                    st.success('Data preprocessing is complete.')
                    st.success('Training model. . .')
                    pc_cl.create_model(model, verbose=False)
                    st.success('The model training is complete. . .')
                    #pc_cl.finalize_model(model)
                    st.success('Model has been created')

    if st.sidebar.checkbox('View System Log'):
        n_lines =st.sidebar.slider(label='Number of lines',min_value=3,max_value=50)
        if st.sidebar.button("Check View"):
            logs = get_model_training_logs(n_lines=n_lines)
            st.text('System log')
            st.write(logs)
    try:
        allOfRuns = mlflow.search_runs(experiment_ids=0)
    except:
        allOfRuns = []
    if len(allOfRuns) != 0:
        if st.sidebar.checkbox('Preview model'):
            ml_logs = 'http://kubernetes.docker.internal:5000/  -->Open mlflow, enter the command line: mlflow ui'
            st.markdown(ml_logs)
            st.dataframe(allOfRuns)
        if st.sidebar.checkbox('Choose a model'):
            selected_run_id = st.sidebar.selectbox('Choose from saved models', allOfRuns[allOfRuns['tags.Source'] == 'create_model']['run_id'].tolist())
            selected_run_info = allOfRuns[(
                    allOfRuns['run_id'] == selected_run_id)].iloc[0, :]
            st.code(selected_run_info)
            if st.sidebar.button('Forecast data'):
                model_uri = 'runs:/' + selected_run_id + '/model/'
                model_loaded = mlflow.sklearn.load_model(model_uri)
                df = pd.read_csv(file_selected_path, nrows=nrows)
                #st.success('Model prediction. . .')
                pred = model_loaded.predict(df)
                pred_df = pd.DataFrame(pred, columns=['Predictive Data'])
                st.dataframe(pred_df)
                pred_df.plot()
                st.pyplot()
    else:
        st.sidebar.warning('Did not find a trained model')
if __name__ == '__main__':
    app_main()

