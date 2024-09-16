import tpqoa_trading_cython
import oandaenv_classification as oe
import datetime as dt
from datetime import datetime
import pandas as pd
pd.set_option('mode.chained_assignment', None)
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


class automation(tpqoa_trading_cython.tpqoa): 
    
    def __init__(self, model_persisted, vAr, units, df_test, # Trading Class    
                 # Reoptimization Class  
                 config_file = './pyalgo.cfg.txt', 
                 symbol = 'XAU_AUD', start = str(dt.datetime.now().date() - dt.timedelta(days = 142)), 
                 end = str(dt.datetime.now().date() - dt.timedelta(days = 25)), granularity = 'M30', 
                 granular_resam_wkend = '30Min', leverage = 20, localized = True, symbol_list = instrument_list,  
                 price_ask = 'A', price_bid = 'B', averaging_period = 168, no_of_elements = 168, spread = 0.00004, 
                 begining_test_env = 25, instruments_traded = 3,   
                 # Trading Class
                 granularity_resam = '30Min', verbose = True, stop_loss_distance = 0.00120, take_profit_distance = 0.00030,  
                 log_file = 'the_project_trading_XAU_AUD.log'):
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
        self.go_in_reoptimize = False
        
        self.time_flag_unload = True
        self.model_persisted = model_persisted
        self.granularity_resam = granularity_resam
        self.units = units
        self.df_test = df_test
        self.trades = 0
        self.position = 0
        self.tick_data = pd.DataFrame()
        self.data_p = pd.DataFrame()
        # Important: Trained model sees current state to predict next return
        self.min_length = self.averaging_period - 1
        self.pl = list()
        self.recent_pl = deque(maxlen = 100)
        self.vAr = vAr # will be updated later after reoptimization
        self.verbose = verbose
        self.stop_loss_distance = stop_loss_distance
        self.take_profit_distance = take_profit_distance
        self.var_breached = False
        self.trading_time = True
        self.master_n_side_kick_frame_save_time_flag = True
        self.master_n_side_kick_frame_load_time_flag = True
        self.load_reoptimized_model_time_flag = True
        self.set_gone_in_save_master_n_side_kick_frame = False
        self.set_gone_in_load_master_n_side_kick_frame = False
        self.set_gone_in_save_trading_model = False
        self.set_gone_in_load_reoptimized_model_vAr_units = False
        self.plot_diversification_chart_time_flag = True
        self.set_gone_in_plot_diversification_chart = False
        
        self.master_frame_build()
        self.side_kick_frame_build()
        self.generate_features()
        self.set_socket()
        
    def set_socket(self):
        
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind('tcp://127.0.0.1:1900') # IP address of digitalocean
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
        
        self.equity = float(self.learn_env.api.get_account_summary()['balance']) / self.instruments_traded
        
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
            print(f'V20 connection error @ {dt.datetime.now()} for self.learn_env. Retrieve again now.')
            self.logger_monitor(f'V20 connection error @ {dt.datetime.now()} for self.learn_env. Retrieve again now.')
            try:
                self.learn_env()
            except:
                print(f'V20 connection error @ {dt.datetime.now()} for self.learn_env. Retrieve last time.')
                self.logger_monitor(f'V20 connection error @ {dt.datetime.now()} for self.learn_env. Retrieve last time.')
                self.learn_env()
    
        try:
            self.test_env()
        except:
            print(f'V20 connection error @ {dt.datetime.now()} for self.test_env. Retrieve again now.')
            self.logger_monitor(f'V20 connection error @ {dt.datetime.now()} for self.test_env. Retrieve again now.')
            try: 
                self.test_env()
            except:
                print(f'V20 connection error @ {dt.datetime.now()} for self.test_env. Retrieve last time.')
                self.logger_monitor(f'V20 connection error @ {dt.datetime.now()} for self.test_env. Retrieve last time.')
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
        
        self.hidden_layers = np.arange(2, 4)
        self.learning_rates = np.linspace(0.00001, 0.1, 20)
        
        callbacks = [EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)]
        
        self.max_acc_te = 0.0

        for self.hl in self.hidden_layers:
            for self.lr in self.learning_rates:
                self.set_seeds()
                self.model = self.create_model(self.hl, 128, True, 0.3, self.lr)  
                self.t0 = time.time()
                self.model.fit(np.asarray(self.learn_env.data[self.features]), np.asarray(self.learn_env.data['d']),\
                               epochs = 3000, verbose = False, validation_split = 0.18, 
                            shuffle = False, callbacks = callbacks)
                K.clear_session()
                self.t1 = time.time()
                self.t = self.t1 - self.t0
                self.acc_tr = self.model.evaluate(np.asarray(self.learn_env.data[self.features]),\
                                                  np.asarray(self.learn_env.data['d']), verbose=False)[1]  
                self.acc_te = self.model.evaluate(np.asarray(self.test_env.data[self.features]),\
                                                  np.asarray(self.test_env.data['d']), verbose=False)[1]
                K.clear_session()
                self.out = f'Hidden Layer: {self.hl} | Learning Rate: {self.lr} | time [s]: {self.t:.4f} | '
                self.out += f'in-sample={self.acc_tr:.4f} | out-of-sample={self.acc_te:.4f}\n'
                print(self.out)
                if self.acc_te > self.max_acc_te:
                    self.max_acc_te = self.acc_te
                    self.model_persisted = self.model
                    print(f'New max_acc_te is achieved with Hidden Layers: {self.hl} | Learning Rate: {self.lr} ||.\
                              Model is persisted.\n')
                    self.logger_monitor(f'New max_acc_te is achieved with Hidden Layers: {self.hl} | Learning Rate: {self.lr} ||. Model\
                    is persisted.\n')   
        self.model_persisted.save('reoptimized_model.h5')
        print(f'model_persisted is saved to SSD @ {dt.datetime.now()}')
        self.logger_monitor(f'model_persisted is saved to SSD @ {dt.datetime.now()}')
                            
           
    def vectorized_backtesting(self):
        
        self.test_env.data['predictions'] = self.model_persisted.predict_classes(np.asarray(self.test_env.data[self.features]))
        self.test_env.data['p'] = np.where(self.test_env.data['predictions'] == 1, 1, -1)
        self.test_env.data['s'] = self.test_env.data['p'] * self.test_env.data['r']
        self.df = pd.DataFrame({'r' : self.test_env.data['r'].values, 's' : self.test_env.data['s'].values, 
                                'p' : self.test_env.data['p'].values}, index = self.test_env.data.index)
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

                self.length_appendix = len(self.df.loc[self.d + '06:30:00' : self.d1 + '04:00:00'][['r', 's_tc', 'p']])
                
                # Will append as long as there are data within the range
                # Will append even there is no data; append an empty dataframe
                self.list_s = self.list_s.append(self.df.loc[self.d + '06:30:00' : self.d1 + '04:00:00'][['r', 's_tc', 'p']])
            
                
            except:
                print('Looping. Data append failed.')
                self.logger_monitor('Looping. Data append failed.')
                
            
            if self.length_appendix > 0:
                
                print(81 * '=')
                self.logger_monitor(81 * '=')
                print(f'Looping. Data from [{self.d}06:30:00] to [{self.d1}04:00:00] appended.')
                self.logger_monitor(f'Looping. Data from [{self.d}06:30:00] to [{self.d1}04:00:00] appended.')
                            
            
        print(f'test_env: {self.test_env.data}')
        self.logger_monitor(f'test_env: {self.test_env.data}')
        print(96 * '=')
        self.logger_monitor(96 * '=')
        print(f'backtested_test_env_without_weekend_data: {self.df}')
        self.logger_monitor(f'backtested_test_env_without_weekend_data: {self.df}')
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
        
        (self.list_s[['r', 's_tc']] * self.leverage).cumsum().apply(np.exp).plot(figsize = (10, 6), 
                title = 'Performance with Leverage & after TC (s_tc) vs Market returns with Leverage (buy & hold) - test_env\
                (Selected Periods)');
     
        
    def optimal_leverage_kelly(self):
        
        self.mean = self.list_s['s_tc'].mean()
        self.var = self.list_s['s_tc'].var()
        self.optimal_leverage_half_kelly = self.mean / self.var * 0.5
        print(f'Optimal leverage using half Kelly: {self.optimal_leverage_half_kelly}')
        self.logger_monitor(f'Optimal leverage using half Kelly: {self.optimal_leverage_half_kelly}')
        
        
    def various_leverage_levels(self):
        
        self.to_plot = ['r', 's_tc']
        for self.lev in [10, 20, 30, 40, 50]:
            self.label = 'L_strategy_tc_%d' % self.lev
            self.list_s[self.label] = self.list_s['s_tc'] * self.lev  
            self.to_plot.append(self.label)
        self.list_s[self.to_plot].cumsum().apply(np.exp).plot(figsize = (10, 6), title = 'Various leverage levels');
        
        
    def drawdown_analysis(self):
        
        self.risk = pd.DataFrame(self.list_s['L_strategy_tc_20'])
        self.risk['equity'] = self.risk['L_strategy_tc_20'].cumsum().apply(np.exp) * self.equity
        self.equity_evo_lev = self.risk['equity']
        print(f'Evolution of equity (Leveraged_after_tc): {self.equity_evo_lev}')
        self.logger_monitor(f'Evolution of equity (Leveraged_after_tc): {self.equity_evo_lev}')
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
        self.risk[['equity', 'cummax']].plot(figsize=(10, 6), 
                                             title = f'Evolution of initial equity of {self.equity} & Maximum Drawdown.')
        plt.axvline(self.t_max, c='r', alpha=0.5);
        
    
    
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
        self.vAr = self.v / self.units # here will reset the self.vAr which is set manually earlier on
    
 
        
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
        self.data.fillna(0, inplace = True)
        self.data_p = self.data
        
    def _resample_data(self):
        
        self.data = self.transfer.resample(self.granularity_resam, 
                                label = 'right').last().ffill().iloc[:-1]
        self.data.index = self.data.index.tz_localize(None)
        
        self.grouped = self.side_kick_transfer.groupby(pd.Grouper(freq=self.granularity_resam))
        
        for self.s in self.symbol_list:
            self.data[self.s + '_bid_Lowest'] = self.grouped[self.s + '_bid_Lowest_transfer_in_progress'].min()
            self.data[self.s + '_ask_Highest'] = self.grouped[self.s + '_ask_Highest_transfer_in_progress'].max()
            
            self.data[self.s + '_bid_Lowest'].fillna(500000.123456789, inplace = True)
            self.data[self.s + '_ask_Highest'].fillna(500000.123456789, inplace = True)
            self.data[self.s + '_bid_Lowest'] = np.where(self.data[self.s + '_bid_Lowest'] == 500000.123456789,
                                                         self.data[self.s + '_bid'], self.data[self.s + '_bid_Lowest'])
            self.data[self.s + '_ask_Highest'] = np.where(self.data[self.s + '_ask_Highest'] == 500000.123456789,
                                                          self.data[self.s + '_ask'], self.data[self.s + '_ask_Highest'])                 
        self.data = self.data
             
    def calculate_returns(self):
        
        for self.s in self.symbol_list:
            self.data[self.s + '_Mid'] = (self.data[self.s + '_bid'] + self.data[self.s + '_ask']) / 2
        self.data.ffill(inplace = True) # ffill() to facilitate inactive markets
        for self.s in self.symbol_list:
            self.data[self.s + '_Mid_returns'] = np.log(self.data[self.s + '_Mid'] / self.data[self.s + '_Mid'].shift(1))
        self.data.dropna(inplace = True)

        self.resampled_data_audit = self.data
        
    def VaR_check(self, df, instrument):
               
        if instrument == self.symbol:
            if self.position == 1:
                self.stopout_price_long = self.order_price - self.vAr
                if float(df[f'{instrument}_bid'].values) <= self.stopout_price_long:
                    order = self.create_order(self.symbol,
                             units = -(1) * self.units,
                             suppress = True, ret = True)
                    self.report_trade(dt.datetime.now(), 'SHORT', order)
                    self.position = 0
                    print(f'VaR has been breached @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')
                    self.logger_monitor(f'VaR has been breached @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')

                    # self.optimal_leverage_half_kelly not included as variable on top (ERROR if used)
                    # Not using sleeping function as the outcome may not tally with the backtesting results.
                    #if self.optimal_leverage_half_kelly < 20:  
                       # print('VaR level breached. Close out order executed. Trading suspended for 2 hrs because calculated\
                       # optimal leverage (half kelly) is below 20.')
                       # self.logger_monitor('VaR level breached. Close out order executed. Trading suspended for 2 hrs because\
                       # calculated optimal leverage (half kelly) is below 20.')
                       # self.time_sleep = dt.datetime.now()
                       # self.trading_time = False
                       # self.var_breached = True
                       # print(f'Trading stops @ {self.time_sleep}')
                       # self.logger_monitor(f'Trading stops @ {self.time_sleep}')
                        
                   # else:
                       # print('VaR level breached. Close out order executed. Trading suspended for 1 hr as calculated optimal\
                       # leverage (half kelly) is 20 & above.')
                       # self.logger_monitor('VaR level breached. Close out order executed. Trading suspended for 1 hr as\
                       # calculated optimal leverage (half kelly) is 20 & above.')
                       # self.time_sleep = dt.datetime.now()
                       # self.trading_time = False
                       # self.var_breached = True
                       # print(f'Trading stops @ {self.time_sleep}')
                       # self.logger_monitor(f'Trading stops @ {self.time_sleep}')
                
            elif self.position == -1:
                self.stopout_price_short = self.order_price + self.vAr
                if float(df[f'{instrument}_ask'].values) >= self.stopout_price_short:
                    order = self.create_order(self.symbol, 
                             units = (1) * self.units,
                             suppress = True, ret = True)
                    self.report_trade(dt.datetime.now(), 'LONG', order)
                    self.position = 0
                    print(f'VaR has been breached @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')
                    self.logger_monitor(f'VaR has been breached @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')

                   # if self.optimal_leverage_half_kelly < 20:
                       # print('VaR level breached. Close out order executed. Trading suspended for 2 hrs because calculated\
                       # optimal leverage (half kelly) is below 20.')
                       # self.logger_monitor('VaR level breached. Close out order executed. Trading suspended for 2 hrs because\
                       # calculated optimal leverage (half kelly) is below 20.')
                       # self.time_sleep = dt.datetime.now()
                       # self.trading_time = False
                       # self.var_breached = True
                       # print(f'Trading stops @ {self.time_sleep}')
                       # self.logger_monitor(f'Trading stops @ {self.time_sleep}')
                   # else:
                       # print('VaR level breached. Close out order executed. Trading suspended for 1 hr as calculated optimal\
                       # leverage (half kelly) is 20 & above.')
                       # self.logger_monitor('VaR level breached. Close out order executed. Trading suspended for 1 hr as\
                       # calculated optimal leverage (half kelly) is 20 & above.')
                       # self.time_sleep = dt.datetime.now()
                       # self.trading_time = False
                       # self.var_breached = True
                       # print(f'Trading stops @ {self.time_sleep}')
                       # self.logger_monitor(f'Trading stops @ {self.time_sleep}')
       
                                         
    def Hidden_SL(self, df, instrument):
        # Different from TP as: full SL distance covered(i.e. 6 pips, more leeway) + spread(i.e. more loss if realized)
               
        if instrument == self.symbol:
            if self.position == 1:
                self.stopout_price_long_sl = self.order_price - self.stop_loss_distance
                if float(df[f'{instrument}_ask'].values) <= self.stopout_price_long_sl:
                    order = self.create_order(self.symbol,
                             units = -(1) * self.units,
                             suppress = True, ret = True)
                    self.report_trade(dt.datetime.now(), 'SHORT', order)
                    self.position = 0
                    print(f'Hidden stop loss activated @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')
                    self.logger_monitor(f'Hidden stop loss activated @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')
                
            elif self.position == -1:
                self.stopout_price_short_sl = self.order_price + self.stop_loss_distance
                if float(df[f'{instrument}_bid'].values) >= self.stopout_price_short_sl:
                    order = self.create_order(self.symbol, 
                             units = (1) * self.units,
                             suppress = True, ret = True)
                    self.report_trade(dt.datetime.now(), 'LONG', order)
                    self.position = 0
                    print(f'Hidden stop loss activated @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')
                    self.logger_monitor(f'Hidden stop loss activated @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')
                    
    def Hidden_TP(self, df, instrument):
               
        if instrument == self.symbol:
            if self.position == 1:
                self.stopout_price_long_tp = self.order_price + self.take_profit_distance
                self.order_price_prev = self.order_price
                # Spread accounted for, actual take profit distance will be realized disregarding how large the spread is
                # Meaning if i set take profit distance as 3 pips, I'll get the full 3 pips AFTER accounting for spread.
                if float(df[f'{instrument}_bid'].values) >= self.stopout_price_long_tp:
                    order = self.create_order(self.symbol,
                             units = -(1) * self.units,
                             suppress = True, ret = True)
                    self.current_spread = float(df[f'{instrument}_ask'].values) - float(df[f'{instrument}_bid'].values)
                    self.current_bid = float(df[f'{instrument}_bid'].values) 
                    self.report_trade(dt.datetime.now(), 'SHORT', order)
                    self.position = 0
                    if self.big_returns_above_50_flag:
                        self.big_returns_above_50_dropped_frame = self.big_returns_above_50.iloc[self.i_above_50]
                        self.big_returns_above_50.drop(self.big_returns_above_50.index[self.i_above_50], inplace = True)
                        print(f'The following frame from big_returns_above_50 has been dropped: {self.big_returns_above_50_dropped_frame}')
                        self.logger_monitor(f'The following frame from big_returns_above_50 has been dropped: {self.big_returns_above_50_dropped_frame}')
                    print(f'Market ASK @ Order: {self.current_market_rate} ||VS|| Actual Filled Price: {self.order_price_prev} |+| Take Profit Distance: {self.take_profit_distance} |=| Stopout Price (LONG TP): {self.stopout_price_long_tp} |<=| Market Bid @ Stopout: {self.current_bid} | Current Spread: {self.current_spread} | Exit Slippage (LONG TP), +ve: eats into profit (TP distance) & vice versa: {self.current_bid - self.order_price} | Entry Slippage (LONG TP), +ve: order filled @ below Market ASK & vice versa (No impact on profits as TP distance is adjusted accordingly): {self.current_market_rate - self.order_price_prev}')
                    self.logger_monitor(f'Market ASK @ Order: {self.current_market_rate} ||VS|| Actual Filled Price: {self.order_price_prev} |+| Take Profit Distance: {self.take_profit_distance} |=| Stopout Price (LONG TP): {self.stopout_price_long_tp} |<=| Market Bid @ Stopout: {self.current_bid} | Current Spread: {self.current_spread} | Exit Slippage (LONG TP), +ve: eats into profit (TP distance) & vice versa: {self.current_bid - self.order_price} | Entry Slippage (LONG TP), +ve: order filled @ below Market ASK & vice versa (No impact on profits as TP distance is adjusted accordingly): {self.current_market_rate - self.order_price_prev}')
                    print(f'Hidden take profit activated @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')
                    self.logger_monitor(f'Hidden take profit activated @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.') 
                
            elif self.position == -1:
                self.stopout_price_short_tp = self.order_price - self.take_profit_distance
                self.order_price_prev = self.order_price
                if float(df[f'{instrument}_ask'].values) <= self.stopout_price_short_tp:
                    order = self.create_order(self.symbol, 
                             units = (1) * self.units,
                             suppress = True, ret = True)
                    self.current_spread = float(df[f'{instrument}_ask'].values) - float(df[f'{instrument}_bid'].values)
                    self.current_ask = float(df[f'{instrument}_ask'].values) 
                    self.report_trade(dt.datetime.now(), 'LONG', order)
                    self.position = 0
                    if self.big_returns_below_50_flag:
                        self.big_returns_below_50_dropped_frame = self.big_returns_below_50.iloc[self.i_below_50]
                        self.big_returns_below_50.drop(self.big_returns_below_50.index[self.i_below_50], inplace = True)
                        print(f'The following frame from big_returns_below_50 has been dropped: {self.big_returns_below_50_dropped_frame}')
                        self.logger_monitor(f'The following frame from big_returns_below_50 has been dropped: {self.big_returns_below_50_dropped_frame}')           
                    print(f'Market BID @ Order: {self.current_market_rate} ||VS|| Actual Filled Price: {self.order_price_prev} |-| Take Profit Distance: {self.take_profit_distance} |=| Stopout Price (SHORT TP): {self.stopout_price_short_tp} |>=| Market ASK @ Stopout: {self.current_ask} | Current Spread: {self.current_spread} | Exit Slippage (SHORT TP), +ve: eats into profit (TP distance) & vice versa: {self.order_price - self.current_ask} | Entry Slippage (SHORT TP), +ve: order filled above Market BID & vice versa (No impact on profit as TP distance is adjusted accordingly): {self.order_price_prev - self.current_market_rate}')
                    self.logger_monitor(f'Market BID @ Order: {self.current_market_rate} ||VS|| Actual Filled Price: {self.order_price_prev} |-| Take Profit Distance: {self.take_profit_distance} |=| Stopout Price (SHORT TP): {self.stopout_price_short_tp} |>=| Market ASK @ Stopout: {self.current_ask} | Current Spread: {self.current_spread} | Exit Slippage (SHORT TP), +ve: eats into profit (TP distance) & vice versa: {self.order_price - self.current_ask} | Entry Slippage (SHORT TP), +ve: order filled above Market BID & vice versa (No impact on profit as TP distance is adjusted accordingly): {self.order_price_prev - self.current_market_rate}')
                    print(f'Hidden take profit activated @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')
                    self.logger_monitor(f'Hidden take profit activated @ {dt.datetime.now()}. Close out order has been executed and trade will resume at the next time interval.')
                                       
    def datetime_from_utc_to_local(self, utc_datetime):
        now_timestamp = time.time()
        self.offset = datetime.fromtimestamp(now_timestamp) - datetime.utcfromtimestamp(now_timestamp)
        return utc_datetime + self.offset
    
    def calculate_TP(self, state): 
        
        self.predicted_threshold = self.model_persisted.predict(state)
        print(f'Predicted threshold level is {self.predicted_threshold}')
        self.logger_monitor(f'Predicted threshold level is {self.predicted_threshold}')
        
        self.big_returns_above_50 = self.df_test.query('s >= 0.0020 & threshold_level > 0.5 & threshold_level <= 1.0')
        self.big_returns_below_50 = self.df_test.query('s >= 0.0020 & threshold_level < 0.5')
        
        self.big_returns_above_50_flag = False #within method to reset gates before run
        self.big_returns_below_50_flag = False

        for self.i_above_50 in range(len(self.big_returns_above_50)):
            #self.local = self.datetime_from_utc_to_local(self.big_returns_above_50.index[self.i_above_50]) - dt.timedelta(minutes = 30)
            self.threshold_target = self.big_returns_above_50.iloc[self.i_above_50]['threshold_level']
            if self.predicted_threshold >= self.threshold_target and self.predicted_threshold < (self.threshold_target * 1.0001):
                self.previous_price = self.big_returns_above_50.iloc[self.i_above_50]['price_level'] / np.exp(self.big_returns_above_50.iloc[self.i_above_50]['s'])
                self.big_returns_above_50_TP = self.big_returns_above_50.iloc[self.i_above_50]['price_level'] - self.previous_price
                self.big_returns_above_50_flag = True
                break;
        
        for self.i_below_50 in range(len(self.big_returns_below_50)):
            #self.local = self.datetime_from_utc_to_local(self.big_returns_below_50.index[self.i_below_50]) - dt.timedelta(minutes = 30)
            self.threshold_target = self.big_returns_below_50.iloc[self.i_below_50]['threshold_level']
            if self.predicted_threshold <= self.threshold_target and self.predicted_threshold > (self.threshold_target - (self.threshold_target * 0.0001)):
                self.previous_price = self.big_returns_below_50.iloc[self.i_below_50]['price_level'] / np.exp(self.big_returns_below_50.iloc[self.i_below_50]['s'])
                self.big_returns_below_50_TP = self.big_returns_below_50.iloc[self.i_below_50]['price_level'] - self.previous_price
                self.big_returns_below_50_flag = True
                break;
           
        if self.big_returns_above_50_flag:
            self.take_profit_distance = self.big_returns_above_50_TP
            print('Take profit uses big_returns_above_50')
            self.logger_monitor('Take profit uses big_returns_above_50')
        elif self.big_returns_below_50_flag:
            self.take_profit_distance = self.big_returns_below_50_TP
            print('Take profit uses big_returns_below_50')
            self.logger_monitor('Take profit uses big_returns_below_50')
        elif True:
            for self.i in np.arange(len(self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])) - 1):
                if self.i == 0:
                    if self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['threshold_level'][self.i] > self.predicted_threshold >= 0.0:
                        self.previous_price = self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['price_level'][self.i] / np.exp(self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['s'][self.i])
                        self.take_profit_distance = self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['price_level'][self.i] - self.previous_price
                        self.take_profit_distance = self.take_profit_distance / 2
                        print('Take profit uses tranches. predicted_threshold is between 0.0 and threshold_level[0]')
                        self.logger_monitor('Take profit uses tranches. predicted_threshold is between 0.0 and threshold_level[0]')
                        break
                if self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['threshold_level'][self.i] <= self.predicted_threshold < self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['threshold_level'][self.i + 1]:
                    self.previous_price = self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['price_level'][self.i] / np.exp(self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['s'][self.i])
                    self.take_profit_distance = self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['price_level'][self.i] - self.previous_price
                    print('Take profit uses tranches.')
                    self.logger_monitor('Take profit uses tranches.')
                    break
                if self.i + 1 == len(self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])) - 1:
                    if self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['threshold_level'][self.i + 1] <= self.predicted_threshold < 1.0:
                        self.previous_price = self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['price_level'][self.i + 1] / np.exp(self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['s'][self.i + 1])
                        self.take_profit_distance = self.df_test.query('0 < s < 0.0020').sort_values(by = ['threshold_level'])['price_level'][self.i + 1] - self.previous_price
                        print('Take profit uses tranches. Reached end of Dataframe.')
                        self.logger_monitor('Take profit uses tranches. Reached end of Dataframe.')
        else:
            self.take_profit_distance = 0.00010
            print('Take profit defaulted to 1 pip')
            self.logger_monitor('Take profit defaulted to 1 pip')
    
        print(f'Take Profit Distance: {self.take_profit_distance}')
        self.logger_monitor(f'Take Profit Distance: {self.take_profit_distance}')                
        
        
    def reset_frames(self):
        
        del [[self.master_frame, self.side_kick_frame, self.transfer, self.side_kick_transfer, 
              self.data, self.grouped, self.data_p]]
        print(f'Frames are deleted before reset to empty frames @ {dt.datetime.now()}.')
        self.logger_monitor(f'Frames are deleted before reset to empty frames @ {dt.datetime.now()}.')
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
        plt.bar(np.arange(1, len(self.pl_arr) + 1), self.pl_arr, width = 0.1)
        plt.plot(np.arange(1, len(self.pl_arr) + 1), self.pl_arr.cumsum(), 'b--')
        plt.title('Bar chart (Single trades). Line chart (Cumulative Performance). All trades shown.')
        plt.savefig('Performance Chart - (All Trades)')
        print(80 * '=')
        self.logger_monitor(80 * '=')                    
        if self.verbose:
            pprint(order)
            self.logger_monitor(f'Full description of order: {order}')
            print(80 * '=')
            self.logger_monitor(80 * '=')
            
            
    
    def on_success(self, df, instrument):
        
     
       # self.VaR_check(df, instrument)
        
        #self.Hidden_SL(df, instrument)
        
        self.Hidden_TP(df, instrument)
        
        self.master_frame = self.master_frame.append({f'{instrument}_bid' : float(df[f'{instrument}_bid'].values), 
                            f'{instrument}_ask' : float(df[f'{instrument}_ask'].values),
                            'Time' : str(df[f'{instrument}_Time'].values
                                        ).replace('[', '').replace(']', '').replace("'", "").replace("'", "")},
                                                            ignore_index = True)

        self.side_kick_frame = self.side_kick_frame.append(
            {f'{instrument}_bid_Lowest_transfer_in_progress' : float(df[f'{instrument}_bid'].values),
                f'{instrument}_ask_Highest_transfer_in_progress' : float(df[f'{instrument}_ask'].values),
                'Time' : str(df[f'{instrument}_Time'].values
                                        ).replace('[', '').replace(']', '').replace("'", "").replace("'", "")},
                                                            ignore_index = True)

        self.transfer = self.master_frame.set_index('Time')
        self.transfer.index = pd.to_datetime(self.transfer.index, infer_datetime_format = True, errors = 'coerce')

        self.side_kick_transfer = self.side_kick_frame.set_index('Time')
        self.side_kick_transfer.index = pd.to_datetime(self.side_kick_transfer.index, infer_datetime_format = True,
                                                                                  errors = 'coerce')
        self._resample_data()
        self.calculate_returns()

        print(f'Length of master_frame: {len(self.master_frame)}')
        self.logger_monitor(f'Length of master_frame: {len(self.master_frame)}')

        print(f'Length of side_kick_frame: {len(self.side_kick_frame)}')
        self.logger_monitor(f'Length of side_kick_frame: {len(self.side_kick_frame)}')

        print(f'B4 Check: Data Length: {len(self.data)}')
        self.logger_monitor(f'B4 Check: Data Length: {len(self.data)}')

        print(f'B4 Check: Min Length: {self.min_length}')
        self.logger_monitor(f'B4 Check: Min Length: {self.min_length}')
        
        
        if len(self.data) > self.min_length:
            
            self.min_length += 1
            self._prepare_data()
            state = self._get_state()

            print(f'Min Length requirement is met @ {dt.datetime.now()}. The following features will be used for prediction depending on the time: {state}')
            self.logger_monitor(f'Min Length requirement is met @ {dt.datetime.now()}. The following features will be used for prediction depending on the time: {state}')


            if (dt.datetime.today().weekday() in [0] and dt.datetime.now().hour >= 6 and dt.datetime.now().hour <= 11):

                print('Model for Monday morning 6 am has taken over.')
                self.logger_monitor('Model for Monday morning 6 am has taken over.')

                if self.var_breached:
                    if self.optimal_leverage_half_kelly < 20:
                        if (dt.datetime.now() - self.time_sleep).seconds >= 7200:
                            self.trading_time = True
                            self.var_breached = False
                            self.resumed_time = dt.datetime.now()
                            print(f'Trading has resumed @ {self.resumed_time} after 2 hrs of trade suspension.')
                            self.logger_monitor(f'Trading has resumed @ {self.resumed_time} after 2 hrs of trade suspension.')
                    else:
                        if (dt.datetime.now() - self.time_sleep).seconds >= 3600:
                            self.trading_time = True
                            self.var_breached = False
                            self.resumed_time = dt.datetime.now()
                            print(f'Trading has resumed @ {self.resumed_time} after 1 hr of trade suspension.')
                            self.logger_monitor(f'Trading has resumed @ {self.resumed_time} after 1 hr of trade suspension.')

                if self.trading_time:
                    prediction = np.where(self.model_persisted.predict(state) > 0.5, 1, 0)
                    signal = 1 if prediction == 1 else -1
                    if self.position in [0, -1] and signal == 1:
                        try:
                            order = self.create_order(self.symbol, 
                                    units = (1 - self.position) * self.units,
                                    suppress = True, ret = True)
                            self.current_market_rate = float(df[f'{instrument}_ask'].values)
                            self.report_trade(dt.datetime.now(), 'LONG', order)
                            self.position = 1
                            self.calculate_TP(state)
                        except:
                            print('LONG order did not go through. Most likely market is closed.')
                            self.logger_monitor('LONG order did not go through. Most likely market is closed.')

                    elif self.position in [0, 1] and signal == -1:
                        try:
                            order = self.create_order(self.symbol, 
                                units = -(1 + self.position) * self.units,
                                suppress = True, ret = True)
                            self.current_market_rate = float(df[f'{instrument}_bid'].values)
                            self.report_trade(dt.datetime.now(), 'SHORT', order)
                            self.position = -1
                            self.calculate_TP(state)
                        except:
                            print('SHORT order did not go through. Most likely market is closed.')
                            self.logger_monitor('SHORT order did not go through. Most likely market is closed.')


            elif (dt.datetime.today().weekday() in [0, 1, 2, 3, 4] and dt.datetime.now().hour >= 12 and dt.datetime.now().hour <= 23) or\
                    (dt.datetime.today().weekday() in [1, 2, 3, 4] and dt.datetime.now().hour >= 0 and dt.datetime.now().hour <= 11):

                print('Model for trading on weekdays has taken over.')  
                self.logger_monitor('Model for trading on weekdays has taken over.') 

                if self.var_breached:
                    if self.optimal_leverage_half_kelly < 20:
                        if (dt.datetime.now() - self.time_sleep).seconds >= 7200:
                            self.trading_time = True
                            self.var_breached = False
                            self.resumed_time = dt.datetime.now()
                            print(f'Trading has resumed @ {self.resumed_time} after 2 hrs of trade suspension.')
                            self.logger_monitor(f'Trading has resumed @ {self.resumed_time} after 2 hrs of trade suspension.')
                    else:
                        if (dt.datetime.now() - self.time_sleep).seconds >= 3600:
                            self.trading_time = True
                            self.var_breached = False
                            self.resumed_time = dt.datetime.now()
                            print(f'Trading has resumed @ {self.resumed_time} after 1 hr of trade suspension.')
                            self.logger_monitor(f'Trading has resumed @ {self.resumed_time} after 1 hr of trade suspension.')

                if self.trading_time:
                    prediction = np.where(self.model_persisted.predict(state) > 0.5, 1, 0)
                    signal = 1 if prediction == 1 else -1
                    if self.position in [0, -1] and signal == 1:
                        try:
                            order = self.create_order(self.symbol, 
                            units = (1 - self.position) * self.units,
                            suppress = True, ret = True)
                            self.current_market_rate = float(df[f'{instrument}_ask'].values)
                            self.report_trade(dt.datetime.now(), 'LONG', order)
                            self.position = 1
                            self.calculate_TP(state)
                        except:
                            print('LONG order did not go through. Most likely market is closed.')
                            self.logger_monitor('LONG order did not go through. Most likely market is closed.')

                    elif self.position in [0, 1] and signal == -1:
                        try:
                            order = self.create_order(self.symbol, 
                                units = -(1 + self.position) * self.units,
                                suppress = True, ret = True)
                            self.current_market_rate = float(df[f'{instrument}_bid'].values)
                            self.report_trade(dt.datetime.now(), 'SHORT', order)
                            self.position = -1
                            self.calculate_TP(state)
                        except:
                            print('SHORT order did not go through. Most likely market is closed.')
                            self.logger_monitor('SHORT order did not go through. Most likely market is closed.')
                        
                        
           # elif dt.datetime.today().weekday() in [1, 2, 3, 4] and dt.datetime.now().hour == 12:
                                                                
               # print('Model for closing out at 12 pm has taken over.')
               # self.logger_monitor('Model for closing out at 12 pm has taken over.')

               # if self.var_breached:
                   # if self.optimal_leverage_half_kelly < 20:
                       # if (dt.datetime.now() - self.time_sleep).seconds >= 7200:
                           # self.trading_time = True
                           # self.var_breached = False
                           # self.resumed_time = dt.datetime.now()
                           # print(f'Trading has resumed @ {self.resumed_time} after 2 hrs of trade suspension.')
                           # self.logger_monitor(f'Trading has resumed @ {self.resumed_time} after 2 hrs of trade suspension.')
                   # else:
                       # if (dt.datetime.now() - self.time_sleep).seconds >= 3600:
                           # self.trading_time = True
                           # self.var_breached = False
                           # self.resumed_time = dt.datetime.now()
                           # print(f'Trading has resumed @ {self.resumed_time} after 1 hr of trade suspension.')
                           # self.logger_monitor(f'Trading has resumed @ {self.resumed_time} after 1 hr of trade suspension.')

               # if self.trading_time:    
                   # if self.position in [-1]:
                       # order = self.create_order(self.symbol, 
                           # units = (1) * self.units,
                           # suppress = True, ret = True)
                       # self.report_trade(dt.datetime.now(), 'LONG', order)   
                       # self.position = 0

                   # elif self.position in [1]:
                       # order = self.create_order(self.symbol, 
                               # units = -(1) * self.units,
                               # suppress = True, ret = True)
                       # self.report_trade(dt.datetime.now(), 'SHORT', order)
                       # self.position = 0 
            
            
            
            elif dt.datetime.today().weekday() in [5] and dt.datetime.now().hour >= 0 and dt.datetime.now().hour <= 3:
                print('Model for Saturday morning has taken over.')
                self.logger_monitor('Model for Saturday morning has taken over.')

                if self.var_breached:
                    if self.optimal_leverage_half_kelly < 20:
                        if (dt.datetime.now() - self.time_sleep).seconds >= 7200:
                            self.trading_time = True
                            self.var_breached = False
                            self.resumed_time = dt.datetime.now()
                            print(f'Trading has resumed @ {self.resumed_time} after 2 hrs of trade suspension.')
                            self.logger_monitor(f'Trading has resumed @ {self.resumed_time} after 2 hrs of trade suspension.')
                    else:
                        if (dt.datetime.now() - self.time_sleep).seconds >= 3600:
                            self.trading_time = True
                            self.var_breached = False
                            self.resumed_time = dt.datetime.now()
                            print(f'Trading has resumed @ {self.resumed_time} after 1 hr of trade suspension.')
                            self.logger_monitor(f'Trading has resumed @ {self.resumed_time} after 1 hr of trade suspension.')

                if self.trading_time:
                    prediction = np.where(self.model_persisted.predict(state) > 0.5, 1, 0)
                    signal = 1 if prediction == 1 else -1
                    if self.position in [0, -1] and signal == 1:
                        try:
                            order = self.create_order(self.symbol, 
                            units = (1 - self.position) * self.units,
                            suppress = True, ret = True)
                            self.current_market_rate = float(df[f'{instrument}_ask'].values)
                            self.report_trade(dt.datetime.now(), 'LONG', order)
                            self.position = 1
                            self.calculate_TP(state)
                        except:
                            print('LONG order did not go through. Most likely market is closed.')
                            self.logger_monitor('LONG order did not go through. Most likely market is closed.')

                    elif self.position in [0, 1] and signal == -1:
                        try:
                            order = self.create_order(self.symbol, 
                                units = -(1 + self.position) * self.units,
                                suppress = True, ret = True)
                            self.current_market_rate = float(df[f'{instrument}_bid'].values)
                            self.report_trade(dt.datetime.now(), 'SHORT', order)
                            self.position = -1
                            self.calculate_TP(state)
                        except:
                            print('SHORT order did not go through. Most likely market is closed.')
                            self.logger_monitor('SHORT order did not go through. Most likely market is closed.')
                        
                                   
            elif dt.datetime.today().weekday() in [5] and dt.datetime.now().hour == 4 and dt.datetime.now().minute >= 0 and dt.datetime.now().minute <= 29:
                                                                
                print('Model for closing out on Saturday at 4 am has taken over.')
                self.logger_monitor('Model for closing out on Saturday at 4 am has taken over.')

                if self.var_breached:
                    if self.optimal_leverage_half_kelly < 20:
                        if (dt.datetime.now() - self.time_sleep).seconds >= 7200:
                            self.trading_time = True
                            self.var_breached = False
                            self.resumed_time = dt.datetime.now()
                            print(f'Trading has resumed @ {self.resumed_time} after 2 hrs of trade suspension.')
                            self.logger_monitor(f'Trading has resumed @ {self.resumed_time} after 2 hrs of trade suspension.')
                    else:
                        if (dt.datetime.now() - self.time_sleep).seconds >= 3600:
                            self.trading_time = True
                            self.var_breached = False
                            self.resumed_time = dt.datetime.now()
                            print(f'Trading has resumed @ {self.resumed_time} after 1 hr of trade suspension.')
                            self.logger_monitor(f'Trading has resumed @ {self.resumed_time} after 1 hr of trade suspension.')

                if self.trading_time:
                    if self.position in [-1]:
                        try:
                            order = self.create_order(self.symbol, 
                            units = (1) * self.units,
                            suppress = True, ret = True)
                            self.report_trade(dt.datetime.now(), 'LONG', order)   
                            self.position = 0
                        except:
                            print('Close out LONG order did not go through. Most likely market is closed.')
                            self.logger_monitor('Close out LONG order did not go through. Most likely market is closed.')

                    elif self.position in [1]:
                        try:
                            order = self.create_order(self.symbol, 
                                units = -(1) * self.units,
                                suppress = True, ret = True)
                            self.report_trade(dt.datetime.now(), 'SHORT', order)
                            self.position = 0
                        except:
                            print('Close out SHORT order did not go through. Most likely market is closed.')
                            self.logger_monitor('Close out SHORT order did not go through. Most likely market is closed.')
                         
                            
    def on_success_heartbeat(self, heartbeat):

        print(f'Heartbeat @ {dt.datetime.now()}')
        self.logger_monitor(f'Heartbeat @ {dt.datetime.now()}')
        
        # Deals with logging off
        # save data 5 minutes before market closes, minimal impact due to decrease liquidity after 4 pm, less price fluctuations 
        
        if dt.datetime.today().weekday() == 5 and dt.datetime.now().hour == 4 and dt.datetime.now().minute == 55:
            if self.master_n_side_kick_frame_save_time_flag:
                self.pkl_file_master_frame = open('master_frame.pkl', 'wb')
                self.pkl_file_side_kick_frame = open('side_kick_frame.pkl', 'wb')
                self.pkl_file_resampled_data_audit = open('resampled_data_audit.pkl', 'wb')
                self.pkl_file_eur_usd = open('pl_eur_usd.pkl', 'wb')
                self.pkl_file_eur_usd_trades = open('trades_eur_usd.pkl', 'wb')
                pickle.dump(self.master_frame.iloc[-300000 : ], self.pkl_file_master_frame)
                pickle.dump(self.side_kick_frame.iloc[-300000 : ], self.pkl_file_side_kick_frame)
                pickle.dump(self.resampled_data_audit.iloc[-240 : ], self.pkl_file_resampled_data_audit)
                pickle.dump(self.pl, self.pkl_file_eur_usd)
                pickle.dump(self.trades, self.pkl_file_eur_usd_trades)
                self.pkl_file_master_frame.close()
                self.pkl_file_side_kick_frame.close()
                self.pkl_file_resampled_data_audit.close()
                self.pkl_file_eur_usd.close()
                self.pkl_file_eur_usd_trades.close()
                print(f'self.master_frame, self.side_kick_frame, self.pl, self.trades and self.resampled_data_audit are saved to SSD @ {dt.datetime.now()}. Its safe to log off (after 5:15 AM when all activities ceased) but MUST log back in before Sunday, 5:01 PM.')
                self.logger_monitor(f'self.master_frame, self.side_kick_frame, self.pl, self.trades and self.resampled_data_audit are saved to SSD @ {dt.datetime.now()}. Its safe to log off (after 5:15 AM when all activities ceased)\
                        but MUST log back in before Sunday, 5:01 PM.')
                self.master_n_side_kick_frame_save_time_flag = False
                print(f'self.master_n_side_kick_frame_save_time_flag is set to False @ {dt.datetime.now()}')
                self.logger_monitor(f'self.master_n_side_kick_frame_save_time_flag is set to False @ {dt.datetime.now()}')
                self.set_gone_in_save_master_n_side_kick_frame = True
                print(f'self.set_gone_in_save_master_n_side_kick_frame is set to True @ {dt.datetime.now()}')
                self.logger_monitor(f'self.set_gone_in_save_master_n_side_kick_frame is set to True @ {dt.datetime.now()}')
                
                
        # KIV: Set time gate, time_go_in = True, so that there is only 1 printing below
        if dt.datetime.today().weekday() == 5 and dt.datetime.now().hour == 4 and dt.datetime.now().minute == 59 and\
                                                self.set_gone_in_save_master_n_side_kick_frame == True:
            self.master_n_side_kick_frame_save_time_flag = True
            print(f'self.master_n_side_kick_frame_save_time_flag is reset to True @ {dt.datetime.now()}')
            self.logger_monitor(f'self.master_n_side_kick_frame_save_time_flag is reset to True @ {dt.datetime.now()}')
            self.set_gone_in_save_master_n_side_kick_frame = False
            print(f'self.set_gone_in_save_master_n_side_kick_frame is reset to False @ {dt.datetime.now()}')
            self.logger_monitor(f'self.set_gone_in_save_master_n_side_kick_frame is reset to False @ {dt.datetime.now()}')


        if dt.datetime.today().weekday() == 5 and dt.datetime.now().hour == 5 and dt.datetime.now().minute == 1:
            if self.plot_diversification_chart_time_flag:
                plt.clf()
                self.pkl_file_eur_usd = open('pl_eur_usd.pkl', 'rb')
                self.pkl_file_gbp_usd = open('pl_gbp_usd.pkl', 'rb')
                self.pkl_file_usd_jpy = open('pl_usd_jpy.pkl', 'rb')
                self.pl_eur_usd = pickle.load(self.pkl_file_eur_usd)
                self.pl_gbp_usd = pickle.load(self.pkl_file_gbp_usd)
                self.pl_usd_jpy = pickle.load(self.pkl_file_usd_jpy)
                self.pkl_file_eur_usd.close()
                self.pkl_file_gbp_usd.close()
                self.pkl_file_usd_jpy.close()

                self.pl_eur_usd = np.array(self.pl_eur_usd)
                self.pl_gbp_usd = np.array(self.pl_gbp_usd)
                self.pl_usd_jpy = np.array(self.pl_usd_jpy)

                self.pl_consolidated = np.concatenate((self.pl_eur_usd, self.pl_gbp_usd, self.pl_usd_jpy))

                self.plot_eur_usd = plt.scatter(self.pl_eur_usd.std(), self.pl_eur_usd.mean(), marker = 'o', color = 'b')
                self.plot_gbp_usd = plt.scatter(self.pl_gbp_usd.std(), self.pl_gbp_usd.mean(), marker = '<', color = 'violet')
                self.plot_usd_jpy = plt.scatter(self.pl_usd_jpy.std(), self.pl_usd_jpy.mean(), marker = '>', color = 'purple')
                self.plot_consolidated = plt.scatter(self.pl_consolidated.std(), self.pl_consolidated.mean(), marker = 'x', color = 'g')
                plt.legend((self.plot_eur_usd, self.plot_gbp_usd, self.plot_usd_jpy, self.plot_consolidated), ('EUR_USD', 'GBP_USD', 'USD_JPY', 'Diversified'))
                plt.grid(True)
                plt.xlabel('Risk - std')
                plt.ylabel('Average Return - mean')
                plt.title('Diversification Effect')
                plt.savefig('Diversification Effect')
                self.plot_diversification_chart_time_flag = False
                print(f'self.plot_diversification_chart_time_flag is set to False @ {dt.datetime.now()}')
                self.logger_monitor(f'self.plot_diversification_chart_time_flag is set to False @ {dt.datetime.now()}')
                self.set_gone_in_plot_diversification_chart = True
                print(f'self.set_gone_in_plot_diversification_chart is set to True @ {dt.datetime.now()}')
                self.logger_monitor(f'self.set_gone_in_plot_diversification_chart is set to True @ {dt.datetime.now()}')


        if dt.datetime.today().weekday() == 5 and dt.datetime.now().hour == 5 and dt.datetime.now().minute == 5 and\
                                                self.set_gone_in_plot_diversification_chart == True:

            self.set_gone_in_plot_diversification_chart = False
            print(f'self.set_gone_in_plot_diversification_chart is reset to False @ {dt.datetime.now()}')
            self.logger_monitor(f'self.set_gone_in_plot_diversification_chart is reset to False @ {dt.datetime.now()}')
            self.plot_diversification_chart_time_flag = True
            print(f'self.plot_diversification_chart_time_flag is reset to True @ {dt.datetime.now()}')
            self.logger_monitor(f'self.plot_diversification_chart_time_flag is reset to True @ {dt.datetime.now()}')
                

            
            
        if dt.datetime.today().weekday() == 0 and dt.datetime.now().hour == 5 and dt.datetime.now().minute == 0:
            if self.master_n_side_kick_frame_load_time_flag:
                
                self.reset_frames()
                self.pkl_file_master_frame = open('master_frame.pkl', 'rb')
                self.pkl_file_side_kick_frame = open('side_kick_frame.pkl', 'rb')
                self.pkl_file_eur_usd = open('pl_eur_usd.pkl', 'rb')
                self.pkl_file_eur_usd_trades = open('trades_eur_usd.pkl', 'rb')
                self.master_frame = pickle.load(self.pkl_file_master_frame)
                self.side_kick_frame = pickle.load(self.pkl_file_side_kick_frame)
                self.pl = pickle.load(self.pkl_file_eur_usd)
                self.trades = pickle.load(self.pkl_file_eur_usd_trades)
                self.pkl_file_master_frame.close()
                self.pkl_file_side_kick_frame.close()
                self.pkl_file_eur_usd.close()
                self.pkl_file_eur_usd_trades.close()
                print(f'self.master_frame and self.side_kick_frame are loaded from SSD @ {dt.datetime.now()}')
                self.logger_monitor(f'self.master_frame and self.side_kick_frame are loaded from SSD @ {dt.datetime.now()}')
                
                self.master_frame = self.master_frame.append({'EUR_USD_bid' : np.nan, 'EUR_USD_ask' : np.nan,
                            'Time' : str(dt.datetime.now())}, ignore_index = True)

                self.side_kick_frame = self.side_kick_frame.append(
                    {'EUR_USD_bid_Lowest_transfer_in_progress' : np.nan, 'EUR_USD_ask_Highest_transfer_in_progress' : np.nan,
                        'Time' : str(dt.datetime.now())}, ignore_index = True)
                
                self.transfer = self.master_frame.set_index('Time')
                self.transfer.index = pd.to_datetime(self.transfer.index, infer_datetime_format = True, errors = 'coerce')

                self.side_kick_transfer = self.side_kick_frame.set_index('Time')
                self.side_kick_transfer.index = pd.to_datetime(self.side_kick_transfer.index, infer_datetime_format = True,
                                                                                          errors = 'coerce')
                self._resample_data()
                self.calculate_returns()
                   
                if len(self.data) < self.averaging_period - 1:
                        
                    self.min_length = self.averaging_period - 1
                    print('len(self.data) is less than self.min_length (self.averaging_period - 1), so self.min_length is set\
                    as (self.averaging_period - 1).')
                    self.logger_monitor('len(self.data) is less than self.min_length (self.averaging_period - 1), so self.min_length\
                    is set as (self.averaging_period - 1).')
                else:
                    self.min_length = len(self.data)
                    print('len(self.data) hits self.min_length (self.averaging_period - 1) and above, self.min_length is set\
                    as len(self.data).')
                    self.logger_monitor('len(self.data) hits self.min_length (self.averaging_period - 1) and above, self.min_length is\
                    set as len(self.data).')

                self.master_n_side_kick_frame_load_time_flag = False
                print(f'self.master_n_side_kick_frame_load_time_flag is set to False @ {dt.datetime.now()}')
                self.logger_monitor(f'self.master_n_side_kick_frame_load_time_flag is set to False @ {dt.datetime.now()}')
                
                self.set_gone_in_load_master_n_side_kick_frame = True
                print(f'self.set_gone_in_load_master_n_side_kick_frame is set to True @ {dt.datetime.now()}')
                self.logger_monitor(f'self.set_gone_in_load_master_n_side_kick_frame is set to True @ {dt.datetime.now()}')
                
                
        if dt.datetime.today().weekday() == 0 and dt.datetime.now().hour == 5 and dt.datetime.now().minute == 10 and\
                                        self.set_gone_in_load_master_n_side_kick_frame == True:
            self.master_n_side_kick_frame_load_time_flag = True
            print(f'self.master_n_side_kick_frame_load_time_flag is reset to True @ {dt.datetime.now()}')
            self.logger_monitor(f'self.master_n_side_kick_frame_load_time_flag is reset to True @ {dt.datetime.now()}')
            self.set_gone_in_load_master_n_side_kick_frame = False
            print(f'self.set_gone_in_load_master_n_side_kick_frame is reset to False @ {dt.datetime.now()}')
            self.logger_monitor(f'self.set_gone_in_load_master_n_side_kick_frame is reset to False @ {dt.datetime.now()}')
               
            
        # Deals with Reoptimization 
        
        # set 28 & on a weekday
            
        if dt.datetime.now().day == 28 and dt.datetime.now().hour == 12 and dt.datetime.now().minute == 1:
            if self.time_flag_unload:
                print('\n' + 96 * '=' + '\n')
                self.logger_monitor('\n' + 96 * '=' + '\n')
                self.model_persisted.save('trading_model.h5')
                print('Trading model is saved on SSD.')
                self.logger_monitor('Trading model is saved on SSD.')
                self.time_flag_unload = False
                print('self.time_flag_unload is set to False.')
                self.logger_monitor('self.time_flag_unload is set to False.')
                self.set_gone_in_save_trading_model = True
                print(f'self.set_gone_in_save_trading_model is set to True @ {dt.datetime.now()}')
                self.logger_monitor(f'self.set_gone_in_save_trading_model is set to True @ {dt.datetime.now()}')
                
        if dt.datetime.now().day == 28 and dt.datetime.now().hour == 12 and dt.datetime.now().minute == 2 and\
                                                                self.set_gone_in_save_trading_model == True:
            self.time_flag_unload = True
            print('self.time_flag_unload is reset to True.')
            self.logger_monitor('self.time_flag_unload is reset to True.')
            self.set_gone_in_save_trading_model = False
            print(f'self.set_gone_in_save_trading_model is reset to False @ {dt.datetime.now()}')
            self.logger_monitor(f'self.set_gone_in_save_trading_model is reset to False @ {dt.datetime.now()}')
                
                
        # Make sure there is a reoptimized_model.h5 first, else the previous model would have been deleted.
        # So start the script on the beginning of the month, and by the beginning of next month there would be a reoptimized_model.h5

        if dt.datetime.now().day >= 1 and dt.datetime.now().day <= 8 and dt.datetime.today().weekday() in [0] and dt.datetime.now().hour == 4 and dt.datetime.now().minute == 45:
            
            if self.load_reoptimized_model_time_flag:
                
                try:
                    self.load_reoptimized_model_time_flag = False
                    print(f'self.load_reoptimized_model_time_flag is set to False @ {dt.datetime.now()}')
                    self.logger_monitor(f'self.load_reoptimized_model_time_flag is set to False @ {dt.datetime.now()}')
                    self.set_gone_in_load_reoptimized_model_vAr_units = True
                    print(f'self.set_gone_in_load_reoptimized_model_vAr_units is set to True @ {dt.datetime.now()}')
                    self.logger_monitor(f'self.set_gone_in_load_reoptimized_model_vAr_units is set to True @ {dt.datetime.now()}')
                    print('\n' + 96 * '=' + '\n')
                    self.logger_monitor('\n' + 96 * '=' + '\n')
                    self.model_persisted_backup = self.model_persisted
                    print(f'self.model_persisted is backed up with self.model_persisted_backup @ {dt.datetime.now()}')
                    self.logger_monitor(f'self.model_persisted is backed up with self.model_persisted_backup @ {dt.datetime.now()}')
                    print('\n' + 96 * '=' + '\n')
                    self.logger_monitor('\n' + 96 * '=' + '\n')
                    del [self.model_persisted]
                    self.model_persisted = None
                    print('Previous trading model has been deleted and set to None.')
                    self.logger_monitor('Previous trading model has been deleted and set to None.')
                    print('\n' + 96 * '=' + '\n')
                    self.logger_monitor('\n' + 96 * '=' + '\n')
                    self.model_persisted = load_model('reoptimized_model.h5')
                    print('Reoptimized model is loaded from SSD and self.model_persisted is reinitialized with the reoptimized model.')
                    self.logger_monitor('Reoptimized model is loaded from SSD and self.model_persisted is reinitialized with the\
                    reoptimized model.')
                    print('\n' + 96 * '=' + '\n')
                    self.logger_monitor('\n' + 96 * '=' + '\n')
                    self.pkl_file_units = open('units_transfer.pkl', 'rb')
                    self.units = pickle.load(self.pkl_file_units)
                    self.pkl_file_units.close()
                    print(f'units is loaded from SSD & self.units is reinitialized with {self.units} units.')
                    self.logger_monitor(f'units is loaded from SSD & self.units is reinitialized with {self.units} units.')
                    print('\n' + 96 * '=' + '\n')
                    self.logger_monitor('\n' + 96 * '=' + '\n')
                    self.pkl_file_vAr = open('vAr_transfer.pkl', 'rb')
                    self.vAr = pickle.load(self.pkl_file_vAr)
                    self.pkl_file_vAr.close()
                    print(f'vAr is loaded from SSD & self.vAr is reinitialized with {self.vAr}')
                    self.logger_monitor(f'vAr is loaded from SSD & self.vAr is reinitialized with {self.vAr}')
                    print('\n' + 96 * '=' + '\n')
                    self.logger_monitor('\n' + 96 * '=' + '\n')
                    self.pkl_file_df_test = open('df_test_transfer.pkl', 'rb')
                    self.df_test = pickle.load(self.pkl_file_df_test)
                    self.pkl_file_df_test.close()
                    print('df_test_transfer is loaded from SSD. Use jupyter notebook to check if needed.')
                    self.logger_monitor('df_test_transfer is loaded from SSD. Use jupyter notebook to check if needed.')
                except:
                    self.model_persisted = self.model_persisted_backup
                    print(f'Error in loading reoptimized model. No such file found. self.model_persisted is initialized with self.model_persisted_backup @ {dt.datetime.now()}')
                    self.logger_monitor(f'Error in loading reoptimized model. No such file found. self.model_persisted is initialized with self.model_persisted_backup @ {dt.datetime.now()}')
                
                


        if dt.datetime.now().day >= 1 and dt.datetime.now().day <= 8 and dt.datetime.today().weekday() in [0] and dt.datetime.now().hour == 4 and\
                                            dt.datetime.now().minute == 50 and self.set_gone_in_load_reoptimized_model_vAr_units == True:

            self.load_reoptimized_model_time_flag = True
            print(f'self.load_reoptimized_model_time_flag is reset to True @ {dt.datetime.now()}')
            self.logger_monitor(f'self.load_reoptimized_model_time_flag is reset to True @ {dt.datetime.now()}')
            self.set_gone_in_load_reoptimized_model_vAr_units = False
            print(f'self.set_gone_in_load_reoptimized_model_vAr_units is reset to False @ {dt.datetime.now()}')
            self.logger_monitor(f'self.set_gone_in_load_reoptimized_model_vAr_units is reset to False @ {dt.datetime.now()}')
                     
            
                                
if __name__ == '__main__':
    
    model_persisted = load_model('trading_model.h5')
    print(f'model_persisted is initialized with trading_model.h5 from SSD @ {dt.datetime.now()}')
    
    pkl_file_vAr_initial_setup = open('vAr_initial_setup.pkl', 'rb')
    vAr = pickle.load(pkl_file_vAr_initial_setup)
    pkl_file_vAr_initial_setup.close()
    print(f'vAr is initialized with vAr_initial_setup.pkl from SSD @ {dt.datetime.now()}')
    
    pkl_file_units_initial_setup = open('units_initial_setup.pkl', 'rb')
    units = pickle.load(pkl_file_units_initial_setup)
    pkl_file_units_initial_setup.close()
    
    pkl_file_df_test = open('df_test_setup.pkl', 'rb')
    df_test = pickle.load(pkl_file_df_test)
    pkl_file_df_test.close()
    
    auto = automation(model_persisted = model_persisted, vAr = vAr, units = units, df_test = df_test)
    print(f'auto is initialized with automation class using model_persisted, vAr, units & df_test @ {dt.datetime.now()}')
    auto.logger_monitor(f'auto is initialized with automation class using model_persisted, vAr, units & df_test @ {dt.datetime.now()}')
    
    stream_list = str(auto.symbol_list).replace("['", "").replace("', '", ",").replace("']", "")
    print(f'Stream List: {stream_list}')
    auto.logger_monitor(f'Stream List: {stream_list}')    

    print(f'Trying to stream data for trading_script @ {dt.datetime.now()}')
    auto.logger_monitor(f'Trying to stream data for trading_script @ {dt.datetime.now()}')

#    auto.stream_data(stream_list)


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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
