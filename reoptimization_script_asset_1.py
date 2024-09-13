import tpqoa_reoptimization
import oandaenv_classification as oe
import datetime as dt
import pandas as pd
import numpy as np
import random
import time
import os
from pprint import pprint
os.environ['PYTHONHASHSEED'] = '0'
from pylab import plt
plt.style.use('seaborn')
import scipy.stats as scs
import zmq
from collections import deque
import pickle
from tensorflow.keras.models import load_model
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing

instrument_list = ['XAU_AUD']


class automation(tpqoa_reoptimization.tpqoa): 
    
    def __init__(self, # Reoptimization Class 
                 config_file = './pyalgo.cfg.txt', 
                 symbol = 'XAU_AUD', start = str(dt.datetime.now().date() - dt.timedelta(days = 142)), 
                 end = str(dt.datetime.now().date() - dt.timedelta(days = 25)), granularity = 'M30', 
                 granular_resam_wkend = '30Min', leverage = 20, localized = True, symbol_list = instrument_list, 
                 price_ask = 'A', price_bid = 'B', averaging_period = 168, no_of_elements = 168, spread = 0.00006, 
                 begining_test_env = 25, instruments_traded = 3, position_sizing_factor_after_VaR = 0.2, 
                 # Trading Class
                 granularity_resam = '30Min', units = 1000, verbose = True, log_file = 'the_project_reoptimization_XAU_AUD.log'):
        super(automation, self).__init__(config_file)
        
        self.log_file = log_file
        self.symbol = symbol
        self.start = start
        self.end = end
        self.granularity = granularity
        self.granular_resam_wkend = granular_resam_wkend
        self.leverage = leverage
        self.localized = localized
        self.symbol_list = symbol_list
        self.price_ask = price_ask
        self.price_bid = price_bid
        self.averaging_period = averaging_period
        self.no_of_elements = no_of_elements
        self.spread = spread
        self.begining_test_env = begining_test_env
        self.instruments_traded = instruments_traded
        self.position_sizing_factor_after_VaR = position_sizing_factor_after_VaR
        self.go_in_reoptimize = False
        self.time_flag_reoptimization = True
        self.gone_in_reoptimization = False
        
        self.granularity_resam = granularity_resam
        self.units = units
        self.trades = 0
        self.position = 0
        self.tick_data = pd.DataFrame()
        # Important: Trained model sees current state to predict next return
        self.min_length = self.averaging_period - 1
        self.pl = list()
        self.recent_pl = deque(maxlen = 100)
        self.verbose = verbose
        self.var_breached = False
        self.trading_time = True
        self.master_frame_build()
        self.side_kick_frame_build()
        self.generate_features()
        self.set_socket()
        
    def set_socket(self):
        
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind('tcp://127.0.0.1:5555') # IP Address of digitalocean
        self.socket = socket
        
        
    def logger_monitor(self, message, time = False, sep = False):
    
        with open(self.log_file, 'a') as f: # opens file for appending
            t = str(dt.datetime.now())
            msg = ''
            if time:
                msg += '\n' + t + '\n'
            if sep:
                msg += 66 * '=' + '\n'
            msg += message + '\n\n'

            self.socket.send_string(msg)
            f.write(msg)    
        
    def generate_features(self):
        
        self.features = []
        for self.s in self.symbol_list:
            self.features.append(self.s + '_Z_Score')
        
    def learn_env(self):
        
        self.learn_env = oe.OandaEnv(symbol = self.symbol, start = self.start,
                        end = self.end, granularity = self.granularity, # 'M1', 'S5'
                        granular_resam_wkend = self.granular_resam_wkend, features = self.features, 
                        leverage = self.leverage, localized = self.localized, symbol_list = self.symbol_list, 
                        price_ask = self.price_ask, price_bid = self.price_bid, 
                        averaging_period = self.averaging_period, no_of_elements = self.no_of_elements)
        
        self.equity = (float(self.learn_env.api.get_account_summary()['balance']) / self.instruments_traded) * 0.8 # due to margin (trade 80% of equity)

        self.margin_buffer = (float(self.learn_env.api.get_account_summary()['balance']) / self.instruments_traded) * 0.2
        
    def test_env(self):
        
        self.test_env = oe.OandaEnv(symbol = self.learn_env.symbol, 
                        start = str(dt.datetime.now().date() - dt.timedelta(days = 32)),
                        end = str(dt.datetime.now().date() - dt.timedelta(days = 1)), 
                        granularity = self.learn_env.granularity, granular_resam_wkend = self.learn_env.granular_resam_wkend,
                        features = self.learn_env.features, leverage = self.learn_env.leverage, 
                        localized = self.learn_env.localized, symbol_list = self.learn_env.symbol_list, 
                        price_ask = self.learn_env.price_ask, price_bid = self.learn_env.price_bid, 
                        averaging_period = self.learn_env.averaging_period, no_of_elements = self.learn_env.no_of_elements)
        
    def generate_all_env(self):
        
        self.retrieve_start = dt.datetime.now()
        print(f'Retrieving data @ {self.retrieve_start}')
        self.logger_monitor(f'Retrieving data @ {self.retrieve_start}')
        try:
            self.learn_env()
        except:
            print(f'V20 connection error @ {dt.datetime.now()} for self.learn_env. Retrieve again 2 seconds later.')
            self.logger_monitor(f'V20 connection error @ {dt.datetime.now()} for self.learn_env. Retrieve again 2 seconds later.')
            time.sleep(2)
            try:
                self.learn_env()
            except:
                print(f'V20 connection error @ {dt.datetime.now()} for self.learn_env. Retrieve last time 2 seconds later.')
                self.logger_monitor(f'V20 connection error @ {dt.datetime.now()} for self.learn_env. Retrieve last time 2 seconds later.')
                time.sleep(2)
                self.learn_env()
    
        try:
            self.test_env()
        except:
            print(f'V20 connection error @ {dt.datetime.now()} for self.test_env. Retrieve again 2 seconds later.')
            self.logger_monitor(f'V20 connection error @ {dt.datetime.now()} for self.test_env. Retrieve again 2 seconds later.')
            time.sleep(2)
            try: 
                self.test_env()
            except:
                print(f'V20 connection error @ {dt.datetime.now()} for self.test_env. Retrieve last time 2 seconds later.')
                self.logger_monitor(f'V20 connection error @ {dt.datetime.now()} for self.test_env. Retrieve last time 2 seconds later.')
                time.sleep(2)
                self.test_env()
            
        self.retrieve_end = dt.datetime.now()
        print(f'Finished data retrival @ {self.retrieve_end}. Time taken is {self.retrieve_end - self.retrieve_start}. All environment\
        is successfully initialized.') 
        self.logger_monitor(f'Finished data retrival @ {self.retrieve_end}. Time taken is {self.retrieve_end - self.retrieve_start}.\
        All environment is successfully initialized.')
        
        
    def set_seeds(seed=100):  
        random.seed(seed)  
        np.random.seed(int('100')) # has to pass in an int when its a function within a class  
        tf.random.set_random_seed(int('100'))
        
        
    def create_model(self, hl, hu, dropout, rate, lr):
        model = Sequential()
        model.add(Dense(hu, activation = 'relu', input_dim = self.learn_env.n_features))
        if dropout:
            model.add(Dropout(rate, seed = 100))
        for _ in range(hl):
            model.add(Dense(hu, activation = 'relu'))
            if dropout:
                model.add(Dropout(rate, seed = 100))
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr = lr), metrics = ['accuracy'])
        return model
    
    
    def hyperparameter_tuning(self):
        
        # self.net_leveraged_returns_hyper will be higher than it's supposed to be as the ptc is @ 0.00004 & does not include..
        # ...fixed cost per trade. But, it does not affect the overall selection of the best perfoming model. If the first...
        # ...model has a higher net returns relative to its 'actual' level, so will be the subsequent models. So, during
        # ...comparing to get the best models, the comparison will remain as status quo, and the final model produced will...
        # ...still be the best model.
        
        self.trading_model = load_model('trading_model_XAU_AUD.h5')
        print('trading_model_XAU_AUD.h5 has been loaded from SSD.')
        self.logger_monitor('trading_model_XAU_AUD.h5 has been loaded from SSD.')
        
        self.hidden_layers = np.arange(2, 3)
        self.learning_rates = np.linspace(0.00001, 0.1, 10)
         
        callbacks = [EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)]
        
        self.max_net_leveraged_returns_hyper = 0.0 

        for self.hl in self.hidden_layers:
            for self.lr in self.learning_rates:
                self.set_seeds()
                self.model = self.create_model(self.hl, 128, True, 0.3, self.lr)  
                self.t0 = time.time()
                self.model.fit(np.asarray(self.learn_env.data[self.features]), np.asarray(self.learn_env.data['d']), epochs = 3000,\
                               verbose = False, validation_split = 0.18, 
                            shuffle = False, callbacks = callbacks)
                K.clear_session()
                self.t1 = time.time()
                self.t = self.t1 - self.t0
                self.acc_tr = self.model.evaluate(np.asarray(self.learn_env.data[self.features]),\
                                                  np.asarray(self.learn_env.data['d']), verbose=False)[1]  
                self.acc_te = self.model.evaluate(np.asarray(self.test_env.data[self.features]),\
                                                  np.asarray(self.test_env.data['d']), verbose=False)[1]
                self.out = f'Hidden Layer: {self.hl} | Learning Rate: {self.lr} | time [s]: {self.t:.4f} | '
                self.out += f'in-sample={self.acc_tr:.4f} | out-of-sample={self.acc_te:.4f}\n'
                print(self.out)
                self.vectorized_backtesting_hyper()
                self.backtest_selected_periods_hyper()
                self.net_leveraged_returns_hyper = float((self.list_s[['s_tc']] * self.leverage).sum().apply(np.exp) - 1)
                if self.net_leveraged_returns_hyper > self.max_net_leveraged_returns_hyper:
                    self.max_net_leveraged_returns_hyper = self.net_leveraged_returns_hyper
                    self.model_persisted = self.model
                    print(f'New maximum for net leveraged returns after tc of {self.max_net_leveraged_returns_hyper} is achieved with\
                    an out-of-sample performance of {self.acc_te:.4f}')
                    self.logger_monitor(f'New maximum for net leveraged returns after tc of {self.max_net_leveraged_returns_hyper}\
                    is achieved with an out-of-sample performance of {self.acc_te:.4f}')
        try:
            self.model_persisted.save('reoptimized_model_XAU_AUD.h5')
            print(f'model_persisted is saved to SSD @ {dt.datetime.now()} as reoptimized_model_XAU_AUD.h5')
            self.logger_monitor(f'model_persisted is saved to SSD @ {dt.datetime.now()} as reoptimized_model_XAU_AUD.h5')
        except:
            self.trading_model.save('reoptimized_model_XAU_AUD.h5')
            print(f'No model that has a net return (after tc) of greater than zero is found. The previous model (last month) is used.\
            trading_model is saved to SSD as reoptimized_model_XAU_AUD.h5 @ {dt.datetime.now()}')
            self.logger_monitor(f'No model that has a net return (after tc) of greater than zero is found. The previous model\
            (last month) is used. trading_model is saved to SSD as reoptimized_model_XAU_AUD.h5 @ {dt.datetime.now()}')
            self.model_persisted = self.trading_model
            print('self.model_persisted is initialized with self.trading_model.')
            self.logger_monitor('self.model_persisted is initialized with self.trading_model.')
            
        
    def vectorized_backtesting_hyper(self):
        
        self.test_env.data['predictions'] = np.where(self.model.predict(np.asarray(self.test_env.data[self.features])) > 0.5, 1, 0) 
        self.test_env.data['p'] = np.where(self.test_env.data['predictions'] == 1, 1, -1)
        self.test_env.data['s'] = self.test_env.data['p'] * self.test_env.data['r']
        self.df = pd.DataFrame({'r' : self.test_env.data['r'].values, 's' : self.test_env.data['s'].values, 
                                'p' : self.test_env.data['p'].values}, index = self.test_env.data.index)
        self.ptc = self.spread / self.test_env.data[self.test_env.symbol].mean() 
        self.df['s_tc'] = np.where(self.df['p'].diff() != 0, self.df['s'] - self.ptc, self.df['s'])
        
    def backtest_selected_periods_hyper(self):
        # The following setting is ONLY for test_env within 1 month...
        # i.e. up to 25 days ago from the 26th. (28 - 1 = 27 [end of date range, selected dataset is entire of 26th])
        # Only work for 1 min timeframe
        
        self.list_s = pd.DataFrame()

        self.length_appendix = 0.0
                        
         
        #test_env: (21:00:00 to 02:00:00 [Current day to next day])
        for self.s in range((dt.datetime.now().date() - dt.timedelta(days = self.begining_test_env)).day, 
                            (dt.datetime.now().date() - dt.timedelta(days = 1)).day):
            
        
            try:
                self.d = '{}-{}-{} '.format((dt.datetime.now().date() - dt.timedelta(days = self.begining_test_env)).year,
                                        (dt.datetime.now().date() - dt.timedelta(days = self.begining_test_env)).month,
                                         self.s)
            
                self.d1 = '{}-{}-{} '.format((dt.datetime.now().date() - dt.timedelta(days = self.begining_test_env)).year, 
                                             (dt.datetime.now().date() - dt.timedelta(days = self.begining_test_env)).month,
                                              self.s + 1)

                self.length_appendix = len(self.df.loc[self.d + '06:30:00' : self.d1 + '06:30:00'][['r', 's_tc', 'p']])
                
                # Will append as long as there are data within the range
                # Will append even there is no data; append an empty dataframe
                self.list_s = self.list_s.append(self.df.loc[self.d + '06:30:00' : self.d1 + '06:30:00'][['r', 's_tc', 'p']])
                
            except:
                print('Looping. Data append failed.')
                self.logger_monitor('Looping. Data append failed.')
                
            
            if self.length_appendix > 0:
                
                print(81 * '=')
                self.logger_monitor(81 * '=')
                print(f'Looping. Data from [{self.d}06:30:00] to [{self.d1}06:30:00] appended.')
                self.logger_monitor(f'Looping. Data from [{self.d}06:30:00] to [{self.d1}06:30:00] appended.')
                            
            
        print(f'test_env: {self.test_env.data}')
        self.logger_monitor(f'test_env: {self.test_env.data}')
        print(96 * '=')
        self.logger_monitor(96 * '=')
        print(f'backtested_test_env_with_weekend_data: {self.df}')
        self.logger_monitor(f'backtested_test_env_with_weekend_data: {self.df}')
        print(96 * '=')
        self.logger_monitor(96 * '=')                    
        print(f'Selected_periods: {self.list_s}')
        self.logger_monitor(f'Selected_periods: {self.list_s}')
    
    
    def vectorized_backtesting(self):
        
        self.test_env.data['predictions'] = np.where(self.model_persisted.predict(np.asarray(self.test_env.data[self.features])) > 0.5, 1, 0)
        self.test_env.data['threshold_level'] = self.model_persisted.predict(np.asarray(self.test_env.data[self.features]))
        self.test_env.data['p'] = np.where(self.test_env.data['predictions'] == 1, 1, -1)
        self.test_env.data['s'] = self.test_env.data['p'] * self.test_env.data['r']
        self.df = pd.DataFrame({'r' : self.test_env.data['r'].values, 's' : self.test_env.data['s'].values, 
                                'p' : self.test_env.data['p'].values, 'threshold_level' : self.test_env.data['threshold_level'].values,                                     'price_level' : self.test_env.data[self.symbol].values}, index = self.test_env.data.index)
        self.ptc = self.spread / self.test_env.data[self.test_env.symbol].mean() 
        self.df['s_tc'] = np.where(self.df['p'].diff() != 0, self.df['s'] - self.ptc, self.df['s'])
         
           
    def backtest_selected_periods(self):
        # The following setting is ONLY for test_env within 1 month...
        # i.e. up to 25 days ago from the 26th. (28 - 1 = 27 [end of date range, selected dataset is entire of 26th])
        # Only work for 1 min timeframe
        
        self.list_s = pd.DataFrame()

        self.length_appendix = 0.0
                        
         
        #test_env: (21:00:00 to 02:00:00 [Current day to next day])
        for self.s in range((dt.datetime.now().date() - dt.timedelta(days = self.begining_test_env)).day, 
                            (dt.datetime.now().date() - dt.timedelta(days = 1)).day):
            
        
            try:
                self.d = '{}-{}-{} '.format((dt.datetime.now().date() - dt.timedelta(days = self.begining_test_env)).year,
                                        (dt.datetime.now().date() - dt.timedelta(days = self.begining_test_env)).month,
                                         self.s)
            
                self.d1 = '{}-{}-{} '.format((dt.datetime.now().date() - dt.timedelta(days = self.begining_test_env)).year, 
                                             (dt.datetime.now().date() - dt.timedelta(days = self.begining_test_env)).month,
                                              self.s + 1)

                self.length_appendix = len(self.df.loc[self.d + '06:30:00' : self.d1 + '06:30:00'][['r', 's', 's_tc', 'p', 'threshold_level']])
                
                # Will append as long as there are data within the range
                # Will append even there is no data; append an empty dataframe
                self.list_s = self.list_s.append(self.df.loc[self.d + '06:30:00' : self.d1 + '06:30:00'][['r', 's', 's_tc', 'p', 'threshold_level']])
                
            except:
                print('Looping. Data append failed.')
                self.logger_monitor('Looping. Data append failed.')
                
            
            if self.length_appendix > 0:
                
                print(81 * '=')
                self.logger_monitor(81 * '=')
                print(f'Looping. Data from [{self.d}06:30:00] to [{self.d1}06:30:00] appended.')
                self.logger_monitor(f'Looping. Data from [{self.d}06:30:00] to [{self.d1}06:30:00] appended.')
                            
            
        print(f'test_env: {self.test_env.data}')
        self.logger_monitor(f'test_env: {self.test_env.data}')
        print(96 * '=')
        self.logger_monitor(96 * '=')
        print(f'backtested_test_env_with_weekend_data: {self.df}')
        self.logger_monitor(f'backtested_test_env_with_weekend_data: {self.df}')
        print(96 * '=')
        self.logger_monitor(96 * '=')                    
        print(f'Selected_periods: {self.list_s}')
        self.logger_monitor(f'Selected_periods: {self.list_s}')
            
                  
    def analytics_test_env(self):
        
        self.bullish_bearish = self.list_s['p'].value_counts()
        print(f'Bullish or Bearish: {self.bullish_bearish}')
        self.logger_monitor(f'Bullish or Bearish: {self.bullish_bearish}')
        print(61 * '=')
        self.logger_monitor(61 * '=')
        
        self.no_trades = sum(self.list_s['p'].diff() != 0)
        print(f'Number of trades: {self.no_trades}')
        self.logger_monitor(f'Number of trades: {self.no_trades}')
        print(61 * '=')
        self.logger_monitor(61 * '=')
        
        self.gross_leveraged_returns_m = (self.list_s[['r']] * self.leverage).sum().apply(np.exp)
        print(f'Gross market returns (Leveraged): {self.gross_leveraged_returns_m}')
        self.logger_monitor(f'Gross market returns (Leveraged): {self.gross_leveraged_returns_m}')
        print(61 * '=')
        self.logger_monitor(61 * '=')
        
        self.gross_leveraged_returns_s = (self.list_s[['s_tc']] * self.leverage).sum().apply(np.exp)
        print(f'Gross strategy returns after tc (Leveraged): {self.gross_leveraged_returns_s}')
        self.logger_monitor(f'Gross strategy returns after tc (Leveraged): {self.gross_leveraged_returns_s}')
        print(61 * '=')
        self.logger_monitor(61 * '=')
        
        self.net_leveraged_returns_m = (self.list_s[['r']] * self.leverage).sum().apply(np.exp) - 1
        print(f'Net market returns (Leveraged): {self.net_leveraged_returns_m}')
        self.logger_monitor(f'Net market returns (Leveraged): {self.net_leveraged_returns_m}')
        print(61 * '=')
        self.logger_monitor(61 * '=')
        
        self.net_leveraged_returns_s = (self.list_s[['s_tc']] * self.leverage).sum().apply(np.exp) - 1
        print(f'Net strategy returns after tc (Leveraged): {self.net_leveraged_returns_s}')
        self.logger_monitor(f'Net strategy returns after tc (Leveraged): {self.net_leveraged_returns_s}')
        print(61 * '=')
        self.logger_monitor(61 * '=')
        
        
    def optimal_leverage_kelly(self):
        
        self.mean = self.list_s[['s_tc']].mean()
        self.var = self.list_s[['s_tc']].var()
        self.optimal_leverage_half_kelly = self.mean / self.var * 0.5
        print(f'Optimal leverage using half Kelly: {self.optimal_leverage_half_kelly}')
        self.logger_monitor(f'Optimal leverage using half Kelly: {self.optimal_leverage_half_kelly}')
        
        
    def various_leverage_levels(self):
        
        self.to_plot = ['r', 's_tc']
        for self.lev in [10, 20, 30, 40, 50]:
            self.label = 'L_strategy_tc_%d' % self.lev
            self.list_s[self.label] = self.list_s['s_tc'] * self.lev  
            self.to_plot.append(self.label)
        #self.list_s[self.to_plot].cumsum().apply(np.exp).plot(figsize = (10, 6), title = 'Various leverage levels');
        
        
    def drawdown_analysis(self):
        
        self.risk = pd.DataFrame(self.list_s['L_strategy_tc_20'])
        self.risk['equity'] = self.risk['L_strategy_tc_20'].cumsum().apply(np.exp) * self.equity
        self.equity_evo_lev = self.risk['equity']
        print(f'Evolution of equity (Leveraged_after_tc): {self.equity_evo_lev}')
        self.logger_monitor(f'Evolution of equity (Leveraged_after_tc): {self.equity_evo_lev}')
        print('\n' + 96 * '=' + '\n')
        self.logger_monitor('\n' + 96 * '=' + '\n')
        self.expected_equity = self.risk['equity'].iloc[-1:].values
        print(f'NET FORECASTED EQUITY: {self.expected_equity}')
        self.logger_monitor(f'NET FORECASTED EQUITY: {self.expected_equity}')
        print('\n' + 96 * '=' + '\n')
        self.logger_monitor('\n' + 96 * '=' + '\n')
        self.risk['cummax'] = self.risk['equity'].cummax()
        self.risk['drawdown'] = self.risk['cummax'] - self.risk['equity']
        self.max_drawdown = self.risk['drawdown'].max()
        self.t_max = self.risk['drawdown'].idxmax()
        print(f'Maximum drawndown of {self.max_drawdown} happens @ {self.t_max}')
        self.logger_monitor(f'Maximum drawndown of {self.max_drawdown} happens @ {self.t_max}')
        self.temp = self.risk['drawdown'][self.risk['drawdown'] == 0]
        self.periods = (self.temp.index[1:].to_pydatetime() - self.temp.index[:-1].to_pydatetime())
        try:
            self.t_per = self.periods.max()
            print(f'Longest drawdown period (hr:min:sec): {self.t_per}')
            self.logger_monitor(f'Longest drawdown period (hr:min:sec): {self.t_per}')
        except:
            print('Error. No next peak is reached.')
            self.logger_monitor('Error. No next peak is reached.')
        #self.risk[['equity', 'cummax']].plot(figsize=(10, 6), 
        #                                     title = f'Evolution of initial equity of {self.equity} & Maximum Drawdown.')
        #plt.axvline(self.t_max, c='r', alpha=0.5);
        
        
    def drawdown_analysis_realized_pl(self):
        
        self.adjusted_position_sizing_month_end_lev = self.equity * self.leverage
        self.equity_month_end = self.equity

        self.plot = pd.DataFrame({"equity" : (np.array(self.pl_realized[-480:])/self.initial_position_sizing_lev)*self.adjusted_position_sizing_month_end_lev}).cumsum() + self.equity_month_end
        self.optimal_equity_evolution = self.plot['equity']
        print(f'Expected evolution of optimal equity @ month-end (when traded) & performance is based on previous month realized returns: {self.optimal_equity_evolution}')
        self.logger_monitor(f'Expected evolution of optimal equity @ month-end (when traded) & performance is based on previous month realized returns: {self.optimal_equity_evolution}')
        self.plot['cummax'] = self.plot['equity'].cummax()
        self.plot['drawdown'] = self.plot['cummax'] - self.plot['equity']
        self.maximum_drawdown = self.plot['drawdown'].max()
        print(f'The maximum drawdown is {self.maximum_drawdown} when optimal equity level is traded.')
        self.logger_monitor(f'The maximum drawdown is {self.maximum_drawdown} when optimal equity level is traded.')
        
    
    
    def set_VaR(self):
        
        self.percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
        self.risk['returns'] = np.log(self.risk['equity'] / self.risk['equity'].shift(1))
        self.VaR = scs.scoreatpercentile(self.equity * self.risk['returns'], self.percs)
        print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
        self.logger_monitor('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
        print(33 * '-')
        self.logger_monitor(33 * '-')
        for self.pair in zip(self.percs, self.VaR):
            print('%16.2f %16.3f' % (100 - self.pair[0], -self.pair[1]))
            self.logger_monitor('%16.2f %16.3f' % (100 - self.pair[0], -self.pair[1]))
        self.v = -self.VaR[0] # amt afford to lose @ 99.99 % confidence level
        # Error here. self.units is static throughout the months???!!! NVM self.vAr not used.
        self.vAr = self.v / self.units # here will reset the self.vAr which is set manually earlier on
        
        
    def set_VaR_realized_pl(self):
        self.percs_1 = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
        self.risk_1 = pd.DataFrame({"equity" : np.log(self.plot["equity"] / self.plot["equity"].shift(1))}) 
        self.equity_1 = self.equity_month_end
        self.VaR_1 = scs.scoreatpercentile(self.equity_1 * self.risk_1["equity"], self.percs_1)
        print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
        self.logger_monitor('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
        print(33 * '-')
        self.logger_monitor(33 * '-')
        for self.pair in zip(self.percs_1, self.VaR_1):
            print('%16.2f %16.3f' % (100 - self.pair[0], -self.pair[1]))
            self.logger_monitor('%16.2f %16.3f' % (100 - self.pair[0], -self.pair[1]))
        self.v = -self.VaR_1[0] # amt afford to lose @ 99.99 % confidence level
        self.vAr = self.v / (self.equity_1 * self.leverage) # here will reset the self.vAr which is set manually earlier on
   

    def units_optimization(self):
        self.amt_afford_to_lose = self.margin_buffer
        for _ in range(10):
            self.equity_optimal = self.equity
            self.equity += (0.1 * self.margin_buffer)
            print(f'Expected equity to be traded: {self.equity}')
            self.logger_monitor(f'Expected equity to be traded: {self.equity}')
            self.risk = pd.DataFrame(self.list_s['L_strategy_tc_20'])
            self.risk['equity'] = self.risk['L_strategy_tc_20'].cumsum().apply(np.exp) * self.equity
            self.set_VaR()
            self.amt_afford_to_lose -= 0.1 * self.margin_buffer
            print(f'Expected amt afford to lose (Leveraged): {self.amt_afford_to_lose} VS Expected Losses (VaR @ 99.99%): {self.v}')
            self.logger_monitor(f'Expected amt afford to lose (Leveraged): {self.amt_afford_to_lose} VS Expected Losses (VaR @ 99.99%): {self.v}')
            if self.v > self.amt_afford_to_lose:
                print(f'Expected tail end loss of VAR @ {self.v} is greater than the amount I could afford to lose, {self.amt_afford_to_lose}. self.equity_optimal of the previous round is persisted.')
                self.logger_monitor(f'Expected tail end loss of VAR @ {self.v} is greater than the amount I could afford to lose, {self.amt_afford_to_lose}. self.equity_optimal of the previous round is persisted.')
                self.units = (self.equity_optimal * self.leverage) * self.position_sizing_factor_after_VaR
                break
                
                
    def units_optimization_realized_pl(self):
        
        print(f'units_optimization_realized_pl has started @ {dt.datetime.now()} to find optimal equity level to trade based on previous month realized returns.')
        self.logger_monitor(f'units_optimization_realized_pl has started @ {dt.datetime.now()} to find optimal equity level to trade based on previous month realized returns.')
        
        self.amt_afford_to_lose = self.margin_buffer
        
        self.pkl_file_realized_pl = open('pl_eur_usd.pkl', 'rb')
        self.pl_realized = pickle.load(self.pkl_file_realized_pl)
        print('pl_eur_usd.pkl is loaded from SSD & self.pl_realized is initialized.')
        self.logger_monitor('pl_eur_usd.pkl is loaded from SSD & self.pl_realized is initialized.')
        self.pkl_file_realized_pl.close()
        
        try: # remove units_initial_setup.pkl from drive after 1st reoptimization
            self.pkl_file_units_initial_position = open('units_initial_setup.pkl', 'rb') # 1000 units
            self.initial_position_sizing_lev = pickle.load(self.pkl_file_units_initial_position)
            print('units_initial_setup.pkl is loaded from SSD & self.initial_position_sizing_lev is initialized.')
            self.logger_monitor('units_initial_setup.pkl is loaded from SSD & self.initial_position_sizing_lev is initialized.')
            self.pkl_file_units_initial_position.close()
        except:
            self.pkl_file_units_initial_position = open('units_transfer.pkl', 'rb')
            self.initial_position_sizing_lev = pickle.load(self.pkl_file_units_initial_position)
            print('units_transfer.pkl is loaded from SSD & self.initial_position_sizing_lev is initialized.')
            self.logger_monitor('units_transfer.pkl is loaded from SSD & self.initial_position_sizing_lev is initialized.')
            self.pkl_file_units_initial_position.close()
        
        for _ in range(10):
            self.equity_optimal = self.equity
            self.equity += (0.1 * self.margin_buffer)
            print(f'Expected equity to be traded: {self.equity}')
            self.logger_monitor(f'Expected equity to be traded: {self.equity}')
            self.adjusted_position_sizing_month_end_lev = self.equity * self.leverage
            self.equity_month_end = self.equity
            # evolution of current (month's end) equity based on historical realized returns
            self.plot = pd.DataFrame({"equity" : (np.array(self.pl_realized[-480:])/self.initial_position_sizing_lev)*self.adjusted_position_sizing_month_end_lev}).cumsum() + self.equity_month_end
            
            self.set_VaR_realized_pl()
            
            self.amt_afford_to_lose -= 0.1 * self.margin_buffer
            print(f'Expected amt afford to lose (Leveraged): {self.amt_afford_to_lose} VS Expected Losses (VaR @ 99.99%): {self.v}')
            self.logger_monitor(f'Expected amt afford to lose (Leveraged): {self.amt_afford_to_lose} VS Expected Losses (VaR @ 99.99%): {self.v}')
            if self.v > self.amt_afford_to_lose:
                print(f'Expected tail end loss of VAR @ {self.v} is greater than the amount I could afford to lose, {self.amt_afford_to_lose}. self.equity_optimal of the previous round is persisted.')
                self.logger_monitor(f'Expected tail end loss of VAR @ {self.v} is greater than the amount I could afford to lose, {self.amt_afford_to_lose}. self.equity_optimal of the previous round is persisted.')
                self.units = self.equity_optimal * self.leverage
                break
        
                
                
       
    def master_frame_build(self):
        self.list_ticks = []
        for self.s in self.symbol_list:
            self.list_ticks.append(self.s + '_bid')
            self.list_ticks.append(self.s + '_ask')
        self.list_ticks.append('Time')
        self.master_frame = pd.DataFrame(columns = self.list_ticks)
        
    def side_kick_frame_build(self):
        self.side_kick_list = []
        for self.s in self.symbol_list:
            self.side_kick_list.append(self.s + '_bid_Lowest_transfer_in_progress')
            self.side_kick_list.append(self.s + '_ask_Highest_transfer_in_progress')
        self.side_kick_list.append('Time')
        self.side_kick_frame = pd.DataFrame(columns = self.side_kick_list)
        
    def _prepare_data(self):
        
        for self.s in self.symbol_list:
            self.data[self.s + '_Moving_Log_Returns'] = self.data[self.s + '_Mid_returns'].rolling(self.averaging_period).mean()
            self.data[self.s + '_Difference'] = self.data[self.s + '_Mid_returns'] - self.data[self.s + '_Moving_Log_Returns']
            self.data[self.s + '_Parkinson'] = (np.sqrt((1 / (4 * self.no_of_elements * np.log(2))) * pow(
                                    np.log(self.data[self.s + '_ask_Highest'] / self.data[self.s + '_bid_Lowest']) , 2).rolling(
                                                                                                self.no_of_elements).sum()))
            self.data[self.s + '_Z_Score'] = self.data[self.s + '_Difference'] / self.data[self.s + '_Parkinson']
            self.data.drop(columns = [self.s + '_Mid_returns', self.s + '_Moving_Log_Returns', self.s + '_Difference',
                                      self.s + '_Parkinson', self.s + '_bid', self.s + '_ask', self.s + '_Mid',
                                      self.s + '_ask_Highest', self.s + '_bid_Lowest'], inplace = True)
        # self.data will not be reflected. Maybe some memory issues 
        self.data = self.data.iloc[self.averaging_period : ]
        self.data.fillna(0, inplace = True)
        self.data_p = self.data
        
    def _resample_data(self):
        
        self.data = self.transfer.resample(self.granularity_resam, label = 'right').last().ffill().iloc[:-1]
        self.data.index = self.data.index.tz_localize(None)
        
        self.grouped = self.side_kick_transfer.groupby(pd.Grouper(freq=self.granularity_resam))
        
        for self.s in symbol_list:
            self.data[self.s + '_bid_Lowest'] = self.grouped[self.s + '_bid_Lowest_transfer_in_progress'].min()
            self.data[self.s + '_ask_Highest'] = self.grouped[self.s + '_ask_Highest_transfer_in_progress'].max()
        self.data = self.data
             
    def calculate_returns(self):
        
        for self.s in self.symbol_list:
            self.data[self.s + '_Mid'] = (self.data[self.s + '_bid'] + self.data[self.s + '_ask']) / 2
        self.data.ffill(inplace = True) # ffill() to facilitate inactive markets
        for self.s in self.symbol_list:
            self.data[self.s + '_Mid_returns'] = np.log(self.data[self.s + '_Mid'] / self.data[self.s + '_Mid'].shift(1))
        self.data.dropna(inplace = True)
        
    def VaR_check(self, df, instrument):
           
        if instrument == self.symbol:  
            if self.position == 1:
                self.stopout_price_long = self.order_price - self.vAr
                if float(df[f'{instrument}_bid'].values) <= self.stopout_price_long:
                    order = self.create_order(self.symbol,
                             units = -(1) * self.units,
                             suppress = True, ret = True)
                    self.report_trade(time, 'SHORT', order)
                    self.position = 0
                    if self.optimal_leverage_half_kelly < 20:
                        print('VaR level breached. Close out order executed. Trading suspended for 2 hrs because calculated optimal\
                        leverage (half kelly) is below 20.')
                        self.logger_monitor('VaR level breached. Close out order executed. Trading suspended for 2 hrs because calculated\
                        optimal leverage (half kelly) is below 20.')
                        self.time_sleep = dt.datetime.now()
                        self.trading_time = False
                        self.var_breached = True
                        print(f'Trading stops @ {self.time_sleep}')
                        self.logger_monitor(f'Trading stops @ {self.time_sleep}')
                        
                    else:
                        print('VaR level breached. Close out order executed. Trading suspended for 1 hr as calculated optimal leverage\
                        (half kelly) is 20 & above.')
                        self.logger_monitor('VaR level breached. Close out order executed. Trading suspended for 1 hr as calculated\
                        optimal leverage (half kelly) is 20 & above.')
                        self.time_sleep = dt.datetime.now()
                        self.trading_time = False
                        self.var_breached = True
                        print(f'Trading stops @ {self.time_sleep}')
                        self.logger_monitor(f'Trading stops @ {self.time_sleep}')
                
            elif self.position == -1:
                self.stopout_price_short = self.order_price + self.vAr
                if float(df[f'{instrument}_ask'].values) >= self.stopout_price_short:
                    order = self.create_order(self.symbol, 
                             units = (1) * self.units,
                             suppress = True, ret = True)
                    self.report_trade(time, 'LONG', order)
                    self.position = 0
                    if self.optimal_leverage_half_kelly < 20:
                        print('VaR level breached. Close out order executed. Trading suspended for 2 hrs because calculated optimal\
                        leverage (half kelly) is below 20.')
                        self.logger_monitor('VaR level breached. Close out order executed. Trading suspended for 2 hrs because calculated\
                        optimal leverage (half kelly) is below 20.')
                        self.time_sleep = dt.datetime.now()
                        self.trading_time = False
                        self.var_breached = True
                        print(f'Trading stops @ {self.time_sleep}')
                        self.logger_monitor(f'Trading stops @ {self.time_sleep}')
                    else:
                        print('VaR level breached. Close out order executed. Trading suspended for 1 hr as calculated optimal leverage\
                        (half kelly) is 20 & above.')
                        self.logger_monitor('VaR level breached. Close out order executed. Trading suspended for 1 hr as calculated\
                        optimal leverage (half kelly) is 20 & above.')
                        self.time_sleep = dt.datetime.now()
                        self.trading_time = False
                        self.var_breached = True
                        print(f'Trading stops @ {self.time_sleep}')
                        self.logger_monitor(f'Trading stops @ {self.time_sleep}')
       
                                         
    def reset_frames(self):
        
        del [[self.master_frame, self.side_kick_frame, self.transfer, self.side_kick_transfer, 
              self.data, self.grouped, self.data_p]]
        # gc.collect()
        # print(f'Frames are deleted before reset. Memory is released @ {dt.datetime.now()}.')
        self.master_frame_build()
        self.side_kick_frame_build()
        self.transfer = pd.DataFrame()
        self.side_kick_transfer = pd.DataFrame()
        self.data = pd.DataFrame()
        self.grouped = pd.DataFrame()
        self.data_p = pd.DataFrame()
          
    def _get_state(self):
        state = self.data_p[self.features].iloc[-1 : ]
        return np.asarray(state)
    
    def report_trade(self, time, side, order):
        self.trades += 1
        pl = float(order['pl'])
        self.pl.append(pl)
        self.recent_pl.append(pl)
        self.order_price = float(order['price'])
        cpl = sum(self.pl)
        self.pl_arr = np.array(self.pl)
        print('\n' + 80 * '=')
        self.logger_monitor('\n' + 80 * '=')
        print(f'{time} | *** GOING {side} ***  | TOTAL TRADES: {self.trades}')
        self.logger_monitor(f'{time} | *** GOING {side} ***  | TOTAL TRADES: {self.trades}')
        print(f'{time} | PROFIT / LOSS: {pl:.2f} | CUMULATIVE: {cpl:.2f}')
        self.logger_monitor(f'{time} | PROFIT / LOSS: {pl:.2f} | CUMULATIVE: {cpl:.2f}')
        print(f'{time} | Recent 100 trades (PROFIT / LOSS): {self.recent_pl}')
        self.logger_monitor(f'{time} | Recent 100 trades (PROFIT / LOSS): {self.recent_pl}')
        print('The following bar chart represents single trades - (All trades).')
        plt.bar(np.arange(1, len(self.pl_arr) + 1), self.pl_arr)
        print('The following line chart represents cumulative sum of previous trades for each particular trade - (All trades).')
        plt.plot(np.arange(1, len(self.pl_arr) + 1), self.pl_arr.cumsum())
        print(80 * '=')
        self.logger_monitor(80 * '=')                    
        if self.verbose:
            pprint(order)
            self.logger_monitor(f'Full description of order: {order}')
            print(80 * '=')
            self.logger_monitor(80 * '=')
        if dt.datetime.now().hour >= 2 and dt.datetime.now().minute >= 0 and dt.datetime.now().second >= 0:
            print('Closing out trade @ {}'.format(dt.datetime.now()))
            self.logger_monitor('Closing out trade @ {}'.format(dt.datetime.now()))
            
    
    def on_success(self, df, instrument):
        pass
    
    
    def on_success_heartbeat(self, heartbeat):


       # print(f'Incoming data (reoptimization_script): {instrument}')
       # self.logger_monitor(f'Incoming data (reoptimization_script): {instrument}')
        
        
        # KIV for testing: Need to execute only once, constant incoming/reactivate (sleep will then sleep continuously)
        
        # Its not really concurrent as the 1st in line process has to finish first (or parts of it), then the subsequent...
        # ...process can continue
        
        # Note: When reoptimizing, no incoming data is printed & the concurrent method (i.e. on_success) has no incoming data!
        
       # if instrument == self.symbol:
       #     self.eurusd_bid = float(df[f'{instrument}_bid'].values)
       #     self.eurusd_ask = float(df[f'{instrument}_ask'].values)
       #     self.eurusd_mid = (self.eurusd_bid + self.eurusd_ask) / 2
            
        #if True: # Manual optimization
            
        if dt.datetime.now().day == 28 and dt.datetime.now().hour == 12 and dt.datetime.now().minute == 5:
        
            if self.time_flag_reoptimization:
            
                self.time_flag_reoptimization = False
                print('self.time_flag_reoptimization is set to False')
                self.logger_monitor('self.time_flag_reoptimization is set to False')
                self.time_open = dt.datetime.now() 
                print(f'Time portal has opened @ {self.time_open} for 60 seconds to initiate reoptimization!')
                self.logger_monitor(f'Time portal has opened @ {self.time_open} for 60 seconds to initiate reoptimization!')
                # print(f'Reoptimization process starts @ {self.time_open}')
                # self.logger_monitor(f'Reoptimization process starts @ {self.time_open}')
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.generate_all_env()
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                # previous self.agent is initialized & retrained. When finished retraining, it's used for trading
                self.hyperparameter_tuning()
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.vectorized_backtesting()  
                self.pkl_file_df_test = open('df_test_transfer.pkl', 'wb')
                pickle.dump(self.df, self.pkl_file_df_test)
                self.pkl_file_df_test.close()
                print('Test dataframe, self.df, is saved on SSD.')
                self.logger_monitor('Test dataframe, self.df, is saved on SSD.')
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.backtest_selected_periods()
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.analytics_test_env()
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.optimal_leverage_kelly()
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')                    
                self.various_leverage_levels()
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.units_optimization() # has same variable names with self.units_optimization_realized_pl(), use 1 method only 
                
                
                # Another option: Historical realized returns is used to project current (month's end) equity
                # self.units_optimization_realized_pl()
                
                
                self.pkl_file_units = open('units_transfer.pkl', 'wb')
                pickle.dump(self.units, self.pkl_file_units)
                print(f'self.units is saved on SSD & {self.units} units will be traded.') # Maintain an EUR account for the correct units to be traded.
                self.logger_monitor(f'self.units is saved on SSD & {self.units} units will be traded.')
                self.pkl_file_units.close()
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.equity = self.equity_optimal
                print('self.equity is reinitialized with self.equity_optimal to proceed with drawdown_analysis & VaR setting.')
                self.logger_monitor('self.equity is reinitialized with self.equity_optimal to proceed with drawdown_analysis & VaR setting.')
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                
                
                # Another option: Historical realized returns is used to project current (month's end) equity
                # self.drawdown_analysis_realized_pl()
                # print('\n' + 96 * '=' + '\n')
                # self.logger_monitor('\n' + 96 * '=' + '\n')
                # self.set_VaR_realized_pl()
                
                
                self.drawdown_analysis()
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.set_VaR() # the new self.vAr is updated
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.pkl_file_vAr = open('vAr_transfer.pkl', 'wb')
                pickle.dump(self.vAr, self.pkl_file_vAr)
                print(f'self.vAr is saved on SSD & if market moves AGAINST my strategy by {self.vAr}, tail end (max) loss is incurred.')
                self.logger_monitor('self.vAr is saved on SSD & if market moves AGAINST my strategy by {self.vAr}, tail end (max) loss is incurred.')
                self.pkl_file_vAr.close()
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.time_close = dt.datetime.now() 
                print(f'Reoptimization process completed @ {self.time_close}. Time taken for reoptimization process\
                is {self.time_close - self.time_open}')
                self.logger_monitor(f'Reoptimization process completed @ {self.time_close}. Time taken for reoptimization process is\
                {self.time_close - self.time_open}')
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                
                self.gone_in_reoptimization = True
                print(f'self.gone_in_reoptimization is set to True @ {dt.datetime.now()}')
                self.logger_monitor(f'self.gone_in_reoptimization is set to True @ {dt.datetime.now()}')
                
        if dt.datetime.now().day == 1 and self.gone_in_reoptimization == True and dt.datetime.now().hour == 12 and\
                                                                                            dt.datetime.now().minute == 1:
            self.gone_in_reoptimization = False
            print(f'self.gone_in_reoptimization is reset to False @ {dt.datetime.now()}')
            self.logger_monitor(f'self.gone_in_reoptimization is reset to False @ {dt.datetime.now()}')
            self.time_flag_reoptimization = True
            print(f'self.time_flag_reoptimization is reset to True @ {dt.datetime.now()}')
            self.logger_monitor(f'self.time_flag_reoptimization is reset to True @ {dt.datetime.now()}')
            
            
if __name__ == '__main__':
    
    auto = automation()
    print(f'auto is initialized with automation class @ {dt.datetime.now()}')
    auto.logger_monitor(f'auto is initialized with automation class @ {dt.datetime.now()}')
    
    stream_list = str(auto.symbol_list).replace("['", "").replace("', '", ",").replace("']", "")
    print(f'Stream List: {stream_list}')
    auto.logger_monitor(f'Stream List: {stream_list}')    

    print(f'Trying to stream data for reoptimization_script @ {dt.datetime.now()}')
    auto.logger_monitor(f'Trying to stream data for reoptimization_script @ {dt.datetime.now()}')

    #auto.stream_data(stream_list)



    go_connect = True
    while go_connect:
        try:
            auto.stream_data(stream_list)
            go_connect = False
        except:
            print(f'Connection Error @ {dt.datetime.now()}. Connect again after 2 seconds.')
            auto.logger_monitor(f'Connection Error @ {dt.datetime.now()}. Connect again after 2 seconds.')
            time.sleep(2)
            go_connect = True
                

                                  
