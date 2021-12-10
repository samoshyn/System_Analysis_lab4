import streamlit as st
import numpy as np
import pandas as pd
from solver_manager import *
import time
from tqdm import tqdm
import itertools

np.seterr(divide='ignore', invalid='ignore')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

st.set_page_config(layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

def get_df(file):
    
    extension = file.name.split('.')[1]
    if extension.upper() == 'CSV':
      df = pd.read_csv(file, header=None, sep='\t')
    elif extension.upper() == 'XLSX':
      df = pd.ExcelFile(file)
      df = df.parse(df.sheet_names[0])
    dfd = df.to_numpy()
    # print(dfd)
    t = df.T.columns.values.tolist()
    return t, dfd

def color_df(s):
    if s['Ключовий індекс'] <= 3 and s['Ключовий індекс'] > 0:
        return ['background-color: orange']*len(s)
    elif s['Ключовий індекс'] >= 4:
        return ['background-color: red']*len(s)
    else:
        return ['background-color: ']*len(s)

def brute(args):
    obj, i, j, k = args
    obj.norm_data()
    obj.define_norm_vectors()
    obj.built_B()
    obj.poly_func()
    obj.built_A()
    obj.lamb()
    obj.psi()
    obj.built_a()
    obj.built_Fi()
    obj.built_c()
    obj.built_F()
    obj.built_F_()
    #print(i,j,k)
    return (i, j, k), np.linalg.norm(obj.norm_error, np.inf), obj.norm_error

def search_params(df):
    
    # print(df)
    st_x1s = [i for i in range(2, 8)]
    st_x2s = [i for i in range(2, 8)]
    st_x3s = [i for i in range(2, 8)]
    opt_err = np.inf
    opt_dct = dict()
    all_results = []
    my_bar = st.progress(0)
    combinations = [st_x1s, st_x2s, st_x3s]
    combinations = list(itertools.product(*combinations))
    b = 1 / (len(combinations) - 1)
    for percent_complete, comb in tqdm(enumerate(combinations)):
        st_x1, st_x2, st_x3 = comb
        #if st_x1 + st_x2 + st_x3 > 3:
        dct = {
        'poly_type': st.session_state['poly_type'], # 
        # 'input_file': df,
        'output_file': 'output_file',
        'pred_steps': 10,
        'samples': 10,
        'dimensions': [st.session_state[i] for i in ['dim_x1', 'dim_x2', 'dim_x3', 'dim_y']], # rozmirnist' vectoriv (x1,x2,x3,y)
        'degrees': [st_x1, st_x2, st_x3], # stepin' polinoma
        'weights': st.session_state['weights'], # vagi (scaled/average)
        'lambda_multiblock': int(st.session_state['lambda_multiblock']),
        'is_save': False
            }
        dct['dimensions'][3] = 1
        solver = Solve(dct)
        solver.load_data(df[:dct['samples'], :-2])
        err = brute((solver, st_x1, st_x2, st_x3))
        
        all_results.append({"degrees": ', '.join([str(st_x1), str(st_x2), str(st_x3)]),
                            "error": err[1]})
        # print(err)
        if err[1] < opt_err:
            opt_err = err[1]
            opt_dct = dct
        my_bar.progress(b*percent_complete)
                        
    st.success('Все готово! 10 найкращих ітерацій виведено нижче')
    st.dataframe(pd.DataFrame(all_results).sort_values(by='error').iloc[:10])
    # print('Top', opt_err)
    for i, step in zip(['st_x1', 'st_x2', 'st_x3'], opt_dct['degrees']):
        st.session_state[i] = step
        
        
    
def config_params(df, col1, col2, col3):
    
    with col1:
        dim_x1 = st.slider('Розмірність Х1', min_value=1, max_value=5, key='dim_x1')
        dim_x2 = st.slider('Розмірність Х2', min_value=1, max_value=5, key='dim_x2')
        dim_x3 = st.slider('Розмірність Х3', min_value=1, max_value=5, key='dim_x3')
        dim_y = st.slider('Розмірність Y', min_value=1, max_value=5, key='dim_y')
        
    if sum([st.session_state[i] for i in ['dim_x1', 'dim_x2', 'dim_x3', 'dim_y']])>df.shape[1]:
        st.session_state['norm_params'] = False
        st.error('Перевищена сумарна розмірність вибірки. Будь-ласка, змініть параметри для подальшої роботи')
    else:
        st.session_state['dimensions'] = [st.session_state[i] for i in ['dim_x1', 'dim_x2', 'dim_x3', 'dim_y']]
        with col1:
            custom_struct = st.radio('Вибір форми', ["Адитивна", 
                                                "Мультиплікативна"], key='custom_struct')
        st.session_state['norm_params'] = True
        with col2:
            
            method = st.radio('Вид поліномів', ["Зміщені поліноми Чебишева", "Поліноми Чебишева", 
                                                "Зміщені полномі Чебишева 2 роду"], key='poly_type')
            grid_search = st.radio('Ввести степені вручную чи підібрати найкращі автоматично?', ["Вручну", "Підібрати"], key='grid_search')
            if grid_search=='Вручну':
                if method:
                    st_x1 = st.slider('Степінь для Х1', min_value=1, max_value=12, key='st_x1')
                    st_x2 = st.slider('Степінь для Х2', min_value=1, max_value=12, key='st_x2')
                    st_x3 = st.slider('Степінь для Х3', min_value=1, max_value=12, key='st_x3')
                    
                weights = st.radio('Вид вагів', ["Нормоване", "Середнє"], key='weights')
                lambda_multiblock = st.checkbox('Визначити лямбда через 3 системи', key='lambda_multiblock')
            else:
                st.write('Оберіть додаткові параметри та натисність кнопку "Оптимізувати степені"')
                weights = st.radio('Вид вагів', ["Нормоване", "Середнє"], key='weights')
                lambda_multiblock = st.checkbox('Визначити лямбда через 3 системи', key='lambda_multiblock')
                opt = st.button('Оптимізувати степені')
                if opt:
                    search_params(df)
        with col3:
            if 'st_x1' in st.session_state:
                samples = st.slider('Розмір вибірки для відновлення', min_value=41, max_value=69, value=50, key='samples')
                pred_steps = st.slider('Крок для прогнозування', min_value=1, max_value=20, value=10, key='pred_steps')
                timestamp = st.slider('Момент часу t', min_value=1, max_value=len(df), value=1, key='current_iter')
                full_graphs = st.checkbox('Накопичувальні графіки', key='full_graphs')
                
                st.session_state['output_file'] = 'output_file'
                st.session_state['degrees'] = [st.session_state[i] for i in ['st_x1', 'st_x2', 'st_x3']]
                
                return st.session_state
            
            
            
def main():
    df = None
    buffer = None
    print_result = None
        
    st.header('Крок 1. Дані')
    st.info("Завантажте датасет для подальшого налаштування алгоритму")
    uploaded_file = st.file_uploader("Завантажити вхідні дані")
    if uploaded_file is not None:
        time_list, df = get_df(uploaded_file)
        # st.session_state['df'] = df            
        
    if df is not None:
        st.header('Крок 2. Налаштування')
        col1, col2, col3 = st.columns(3)
        params = config_params(df, col1, col2, col3)
        if st.session_state.get('norm_params'):
            st.header('Крок 3. Симуляція')
            # b = st.empty()
            start = st.button('Розпочати симуляцію')
            analyse = st.button('Проаналізувати момент часу t')
            # block = st.button("Пауза")
            if start:
                
                t = 0

                for_table, for_graphs = st.columns(2)
                
                results_table = for_table.empty()
                results_title = for_table.empty()
                results_warning = for_table.empty()
                results_metric1 = for_table.empty()
                results_metric2 = for_table.empty()
                results_metric3 = for_table.empty()
                results_metric4 = for_table.empty()
                

                results_graphs = for_graphs.empty()
                pause = for_graphs.empty()
                
                # print(st.session_state)
                params['is_save'] = False
                manager = SolverManager(params)
                manager.prepare_v2(time_list, df)
                running = True
                block = False
                # print('start')
                for i in range(len(df)):
                    # print(start, block)
                    # print('running', running)
                    # print('block', block)
                    res, func_result, print_result = manager.launch()
                    time.sleep(1)
                    if res=='normal':
                        batch_res = pd.DataFrame(manager.insert_data_df, 
                                                columns=['Момент часу', 'Бортова напруга', 'Запас палива', 
                                                         'Напруга в АБ', 'Стан', 'Ймовірність аварії', 
                                                         'Причина позашт. ситуації', 'Ключовий індекс']).iloc[1:]
                        batch_res['Момент часу'] = batch_res['Момент часу'].astype(int)
                        batch_res['Бортова напруга'] = batch_res['Бортова напруга'].astype(float)
                        batch_res['Запас палива'] = batch_res['Запас палива'].astype(float)
                        batch_res['Напруга в АБ'] = batch_res['Напруга в АБ'].astype(float)
                        batch_res['Ключовий індекс'] = batch_res['Ключовий індекс'].astype(int)
                        unnormal_res = batch_res[batch_res['Стан']!='Безпечна ситуація']
                        
                        results_title.subheader(f'Стан системи в момент часу: {manager.time[manager.batch_size + manager.current_iter -1]}')
                        if len(unnormal_res):
                            results_warning.warning(unnormal_res['Стан'].iloc[0])
                        else:
                            results_warning.info('Штатна ситуація. Все нормально')
                        results_metric1.metric("Бортова напруга", manager.y_current[0],  manager.y_current[0] -  manager.temp_old_values['y1'])
                        results_metric2.metric("Запас палива", manager.y_current[1],  manager.y_current[1] -  manager.temp_old_values['y2'])
                        results_metric3.metric("Напруга в АБ", manager.y_current[2],  manager.y_current[2] -  manager.temp_old_values['y3'])
                        results_metric4.metric("Ресурс допустимого ризику", manager.rdr,  manager.rdr -  manager.temp_old_values['risk'])
                        manager.temp_old_values = {'y1': manager.y_current[0], 
                                                'y2': manager.y_current[1], 
                                                'y3': manager.y_current[2], 
                                                'risk': manager.rdr}
                        
                        results_table.dataframe(batch_res.style.apply(color_df, axis=1))
                        # results_table.write(batch_res.style.apply(highlight_greaterthan_1, axis=1))
                        results_graphs.pyplot(manager.current_graphs)
                        
                        # block = pause.button("Пауза", key=i)
                        
                        # if block:
                        #     print('swdwfwef')
                        #     time.sleep(5)
                
                    else:
                        running = False
                        
            if analyse:
                
                for_table, for_graphs = st.columns(2)
                
                results_table = for_table.empty()
                results_title = for_table.empty()
                results_warning = for_table.empty()
                results_metric1 = for_table.empty()
                results_metric2 = for_table.empty()
                results_metric3 = for_table.empty()
                results_metric4 = for_table.empty()
                

                results_graphs = for_graphs.empty()
                pause = for_graphs.empty()
                params['is_save'] = True
                manager = SolverManager(params)
                manager.prepare_v2(time_list, df)
                running = True
                # print('start')
                res, func_result, print_result = manager.launch()
                time.sleep(1)
                if res=='normal':
                    batch_res = pd.DataFrame(manager.insert_data_df, 
                                            columns=['Момент часу', 'Бортова напруга', 'Запас палива', 
                                                     'Напруга в АБ', 'Стан', 'Ймовірність аварії', 
                                                     'Причина позашт. ситуації', 'Ключовий індекс']).iloc[1:]
                    batch_res['Момент часу'] = batch_res['Момент часу'].astype(int)
                    batch_res['Бортова напруга'] = batch_res['Бортова напруга'].astype(float)
                    batch_res['Запас палива'] = batch_res['Запас палива'].astype(float)
                    batch_res['Напруга в АБ'] = batch_res['Напруга в АБ'].astype(float)
                    batch_res['Ключовий індекс'] = batch_res['Ключовий індекс'].astype(int)
                    unnormal_res = batch_res[batch_res['Стан']!='Безпечна ситуація']
                    
                    results_title.subheader(f'Стан системи в момент часу: {manager.time[manager.batch_size + manager.current_iter -1]}')
                    if len(unnormal_res):
                        results_warning.warning(unnormal_res['Стан'].iloc[0])
                    else:
                        results_warning.info('Штатна ситуація. Все нормально')
                    results_metric1.metric("Бортова напруга", manager.y_current[0],  manager.y_current[0] -  manager.temp_old_values['y1'])
                    results_metric2.metric("Запас палива", manager.y_current[1],  manager.y_current[1] -  manager.temp_old_values['y2'])
                    results_metric3.metric("Напруга в АБ", manager.y_current[2],  manager.y_current[2] -  manager.temp_old_values['y3'])
                    results_metric4.metric("Ресурс допустимого ризику", manager.rdr,  manager.rdr -  manager.temp_old_values['risk'])
                    manager.temp_old_values = {'y1': manager.y_current[0], 
                                            'y2': manager.y_current[1], 
                                            'y3': manager.y_current[2], 
                                            'risk': manager.rdr}
                    
                    results_table.dataframe(batch_res.style.apply(color_df, axis=1))
                    # results_table.write(batch_res.style.apply(highlight_greaterthan_1, axis=1))
                    results_graphs.pyplot(manager.current_graphs)
                    st.write(print_result)
            
main()