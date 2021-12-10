import random
from solve import *
from solve_custom import SolveExpTh
from read_data import read_data
import matplotlib.pyplot as plt
from represent import *
from lightgbm import LGBMRegressor
from scipy.ndimage.interpolation import shift
from scipy import stats

import time

START_FUEL = 47
TIME_DELTA = 10
reason = [u'Мала бортова напруга\n',u'Недостатній запас палива\n',u'Низький рівень заряду АБ\n']

def calculate_rdr_delta(ycurrent, yf,  yd):
    maxl = np.max(yf[:-1] - yf[1:])
    return (ycurrent - yd)/ maxl

def lblText(lbl, text):
    lbl.setText(str(text)[:10])
    return

def prob(x, xmax, xmin):
    alpha = 1
    # res = np.fabs((x - xmax) / (xmax - xmin))
    res = (np.tanh(np.fabs((x - xmax) / (xmax - xmin))))**(alpha)
    r = np.ma.array(res, mask=np.array(x >= xmax), fill_value=0)
    return r.filled()

def insert_data(tw, row, data):
    assert len(data) <= 8
    try:
        for i, d in enumerate(data):
            item = QTableWidgetItem(d)
            item.setTextAlignment(Qt.AlignHCenter)
            tw.setItem(row, i, item)
    except Exception as e:
        raise ('insert data in table' + str(e))


def classify_danger_rating(level):
    if 0 <= level <= 0.07:
        return 0, u"Безпечна ситуація"
    elif 0.07 < level <= 0.25:
        return 1, u"Нештатна ситуація по одному параметру"
    elif 0.25 < level <= 0.375:
        return 2, u"Нештатна ситуація по декількох параметрах"
    elif 0.375 < level <= 0.5:
        return 3, u"Спостерігається загроза аварії"
    elif 0.5 < level <= 0.625:
        return 4, u"Висока загроза аварії"
    elif 0.625 < level <= 0.75:
        return 5, u"Критична ситуація"
    elif 0.75 < level <= 0.875:
        return 6, u"Шанс уникнути аварії дуже малий"
    elif 0.875 < level:
        return 7, u"Аварія"


class SolverManager(object):
    Y_C = np.array([[11.7], [1], [11.7]])  # warning value
    Y_D = np.array([[10.5], [0.0], [10.5]])  # failure value

    def __init__(self, d):
        self.custom_struct = d['custom_struct']
        d['dimensions'][3] = 1
        if d['custom_struct']=='Мультиплікативна':
            self.solver = SolveExpTh(d)
        else:
            self.solver = Solve(d)
        self.full_graphs = d['full_graphs']
        self.first_launch = True
        self.batch_size = d['samples']
        self.forecast_size = d['pred_steps']
        # remove_old =  d['remove_old']
        self.is_save = d['is_save']
        self.current_iter = d['current_iter']
        self.y_influenced = None
        self.data_window = None
        # self.current_graphs = None
        # self.tablewidget = d['tablewidget']
        self.reason = np.array([],dtype = str) # init warning reason
        self.temp_old_values = {'y1': 0, 'y2': 0, 'y3': 0, 'risk': 0}

    def prepare(self, filename):
        self.time, self.data = read_data(filename)
        increment = self.time[-1] - self.time[-2]
        self.time = np.append(self.time, np.arange(1, 1 + self.forecast_size) * increment + self.time[-1])
        self.N_all_iter = len(self.time)
        #self.operator_view.show()
        #self.operator_view.status_bar.showMessage('Loaded successfully.', 1000)
        
    def prepare_v2(self, time, data):
        self.time, self.data = time, data #read_data(filename)
        increment = self.time[-1] - self.time[-2]
        self.time = np.append(self.time, np.arange(1, 1 + self.forecast_size) * increment + self.time[-1])
        self.N_all_iter = len(self.time)
        self.boosting = self.get_fited_boost()
        #self.operator_view.show()
        #self.operator_view.status_bar.showMessage('Loaded successfully.', 1000)

    def start_machine(self):
        self.operator_view.start_process()

    def launch(self):
        res = 'normal'
        if self.current_iter + 2 * self.batch_size < len(self.data):
            # print(f'current_iter {self.current_iter}, batch_size {self.batch_size}')
            func_result, print_result = self.fit(self.current_iter, self.batch_size)
            self.current_iter += 1
            return res, func_result, print_result

        return 'end launch', None, None
            #self.operator_view.timer.stop()

        
    def plot_graphs(self, real_values, predicted_values,
                risk_values, time_ticks,
                warning, failure,
                tail):
        descriptions = ['Бортова напруга', 'Запас палива', 'Напруга в АБ']
        fig, ax = plt.subplots(3, figsize=(9,9))
        for i in range(3):
            ax[i].plot(time_ticks[:-tail], real_values[:,i])
            ax[i].plot(time_ticks[-tail - 1:], np.append(real_values[-1,i], predicted_values[i]))
            ax[i].axhline(y=warning[i], color='r', linestyle='dotted', label='warning')
            ax[i].axhline(y=failure[i], color='r', linewidth=3, label='failure')
            ax[i].axhspan(ymin=failure[i], ymax=warning[i], alpha=0.2, color='r')
            ax[i].axvline(x=time_ticks[-tail - 1], color='black', label='delta')
            ax[i].set_title(descriptions[i])
            bot = min(map(min,[real_values[:,i], predicted_values[i], ]))
            top = max(map(max,[real_values[:,i], predicted_values[i], ]))
            bot -= (top - bot)*0.1
            top += (top - bot)*0.1
            ax[i].set_ylim(bot, top)
        self.current_graphs = fig
        plt.close()
        
    def plot_graphs_full(self, real_values, predicted_values,
                risk_values, time_ticks,
                warning, failure,
                tail):
        descriptions = ['Бортова напруга', 'Запас палива', 'Напруга в АБ']
        fig, ax = plt.subplots(3, figsize=(9,9))
        for i in range(3):
            ax[i].plot(range(len(real_values)), real_values[:,i])
            ax[i].plot(time_ticks[-tail - 1:], np.append(real_values[-1,i], predicted_values[i]))
            ax[i].axhline(y=warning[i], color='r', linestyle='dotted', label='warning')
            ax[i].axhline(y=failure[i], color='r', linewidth=3, label='failure')
            ax[i].axhspan(ymin=failure[i], ymax=warning[i], alpha=0.2, color='r')
            ax[i].axvline(x=time_ticks[-tail - 1], color='black', label='delta')
            ax[i].set_title(descriptions[i])
            bot = min(map(min,[real_values[:,i], predicted_values[i], ]))
            top = max(map(max,[real_values[:,i], predicted_values[i], ]))
            bot -= (top - bot)*0.1
            top += (top - bot)*0.1
            ax[i].set_ylim(bot, top)
        self.current_graphs = fig
        plt.close()
        
    
    def fit(self, shift, n):
        # print('fit', shift, n)
        print_result = None
        self.data_window = self.data[shift:shift + n]
        self.solver.load_data(self.data_window[:, :-2])  # y2 and y3 not used
        func_result = self.solver.prepare()
        if self.is_save:
            if self.custom_struct=='Мультиплікативна':
                solution = PolynomialBuilderExpTh(self.solver)
                print_result = solution.get_results()
            else:
                solution = PolynomialBuilder(self.solver)
                print_result = solution.get_results()
        self.predict(shift, n)
        self.risk()  # really suspicious realisation
        self.y_influenced = [y * (1 - self.f) for y in self.y_forecasted]
            
        if self.full_graphs:
            self.plot_graphs_full(real_values=self.data[:shift + n, -3:],
                         predicted_values=self.y_forecasted,
                         risk_values=self.y_influenced,
                         time_ticks=self.time[shift:shift + n + self.solver.pred_step],
                         warning=self.Y_C, failure=self.Y_D,tail=self.forecast_size
                         )
        else:
            self.plot_graphs(real_values=self.data_window[:, -3:],
                          predicted_values=self.y_forecasted,
                          risk_values=self.y_influenced,
                          time_ticks=self.time[shift:shift + n + self.solver.pred_step],
                          warning=self.Y_C, failure=self.Y_D,tail=self.forecast_size
                          )
            


        fuel = (13120 - self.time[shift] ) / 13120 * self.data_window[-1, -3:][1]
        # print(f'fit fuel {fuel}')
        self.y_current = np.array([self.solver.Y_[-1,0], fuel, self.solver.X_[0][-1,0]])

        # self.y_current = np.array([self.solver.Y_[-1,0], self.solver.X_[0][-1,3], self.solver.X_[1][-1,2]])
        self.rdr_calc()
        self.table_data_forecasted()
        
        return func_result, print_result


    def risk(self):
        self.p = prob(self.y_forecasted, self.Y_C, self.Y_D)
        # print(f'risk p {self.p}')
        self.f = 1 - (1 - self.p[0, :]) * (1 - self.p[1, :]) * (1 - self.p[2, :])
        for i in range(len(self.f)):
            if self.f[i] > 1:
                self.f[i] = 1
        #calculate reason warning situation
        self.reason = np.array([],dtype = str)
        for i in range(self.forecast_size):
            string = ''
            for j in range(3):
                if self.p[j, i] > 0:
                    string = string + reason[j]
            if string == '':
                self.reason = np.append(self.reason, '-')
            else:
                self.reason = np.append(self.reason, string)
        assert len(self.reason) == self.forecast_size

        self.danger_rate = np.array([classify_danger_rating(i) for i in self.f])
        # print(f'risk danger_rate {self.danger_rate}')

        
    def get_fited_boost(self):
        
        df_xgb = pd.DataFrame(self.data[:, -3], columns=['target'])
        num_steps = self.forecast_size
        for lag in range(num_steps,num_steps+5):
            df_xgb[f'lag_{lag}'] = df_xgb['target'].shift(lag)
        df_xgb = df_xgb.dropna()
        lgbm = LGBMRegressor(n_estimators=500)
        X_xgb, y_xgb = df_xgb.drop('target', axis=1), df_xgb['target']
        lgbm.fit(X_xgb, y_xgb)
        
        return lgbm    
    
    def predict(self, shift, n):
        num_steps = self.forecast_size
        try:
            fuel = np.array(list(
                [self.data[self.current_iter + self.batch_size + i][-2] + random.randrange(-1, 1) / 50
                for i in range(self.forecast_size)]
            ))
        except:
            return
        voltage = np.array(list(
            [x[0] for x in self.solver.X_[0][-num_steps:, 0].tolist()]
        ))
        
        df_pred = pd.DataFrame(self.data_window[:,-3], columns=[f'lag_{num_steps}'])
        for lag in range(num_steps+1,num_steps+5):
        #     print(lag-10)
            df_pred[f'lag_{lag}'] = df_pred[f'lag_{num_steps}'].shift(lag-num_steps)
        df_pred = df_pred.dropna().tail(num_steps)
        main_pred = self.boosting.predict(df_pred)

        self.y_forecasted = [main_pred, fuel, voltage]
        return
        

    def table_data_forecasted(self):
        t = self.time[self.batch_size + self.current_iter: self.batch_size + self.current_iter + self.solver.pred_step]
        y1 = self.y_forecasted[0]
        y2 = self.y_forecasted[1]
        y3 = self.y_forecasted[2]
        state = self.danger_rate[:, 1]
        risk = self.f
        reason = self.reason
        rate = self.danger_rate[:, 0]
        data = np.array([t, y1, y2, y3, state, risk, reason, rate]).T
        assert data.shape == (self.solver.pred_step, 8)
        self.insert_data_df = data
      
    def rdr_calc(self):
        self.rdr = np.inf
        rdr = np.inf
        rdrs = []
        for i in range(3):
            
            if self.y_current[i]>=self.Y_C[i,0]:
                continue
            if self.y_current[i]<= self.Y_D[i,0]:
                rdr = 0
            
            s = calculate_rdr_delta(self.y_current[i], self.y_influenced[i], self.Y_D[i,0])
            if s <= 0:
                continue
            t =(self.y_current[i] - self.Y_D[i,0])/s
            if t < rdr:
                rdr = t
                
                y_full = np.concatenate([self.data_window[:, -3:][:,i], self.y_forecasted[i]])
                y_full_shift = shift(y_full, 1, cval=0)
                s = np.min((y_full[1:] + y_full_shift[1:]) / 2)
                deltas = y_full[1:] - y_full_shift[1:]
                v = self.exponential_moving_average(np.abs(deltas), len(deltas), 50)[-self.forecast_size]
                
                zscores = np.abs(stats.zscore(deltas))[:-1]
                zscores = zscores[zscores>1.5]
                if len(zscores):
                    coef = np.min((zscores - 1.5) / 1.5)
                    coef = 1 if coef>1 else coef
                    rdr_new = s * v * coef
                    rdrs.append(rdr_new)
        if len(rdrs):
            rdrs = np.min(rdrs)
            if rdr != np.inf and np.abs(rdrs - rdr) > 0.2:
                rdrs = rdr
                
        if type(rdrs)==float:
            self.rdr = rdrs
        elif rdr != np.inf:
            self.rdr = rdr
        else:
            self.rdr = np.inf
   

    # def rdr_calc(self):
    #     self.rdr = np.inf
    #     rdr = np.inf
    #     for i in range(3):
    #         if self.y_current[i]>=self.Y_C[i,0]:
    #             continue
    #         if self.y_current[i]<= self.Y_D[i,0]:
    #             rdr = 0
    #             continue
    #         s = calculate_rdr_delta(self.y_current[i], self.y_influenced[i], self.Y_D[i,0])
    #         if s <= 0:
    #             continue
    #         t =(self.y_current[i] - self.Y_D[i,0])/s
    #         if t < rdr:
    #             rdr = t
    #     if rdr != np.inf:
    #         self.rdr = rdr
            
    def exponential_moving_average(self, signal, points, smoothing=2):

        weight = smoothing / (points + 1)
        ema = np.zeros(len(signal))
        ema[0] = signal[0]
    
        for i in range(1, len(signal)):
            ema[i] = (signal[i] * weight) + (ema[i - 1] * (1 - weight))
    
        return ema


            

