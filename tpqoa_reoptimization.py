import v20

import configparser

import pandas as pd

from v20.transaction import StopLossDetails, ClientExtensions

from v20.transaction import TrailingStopLossDetails, TakeProfitDetails

import datetime as dt

import threading





class tpqoa(object):

    ''' tpqoa is a Python wrapper class for the Oanda v20 API. '''



    def __init__(self, conf_file):

        ''' Init function is expecting a configuration file with

        the following content:



        [oanda]

        account_id = XYZ-ABC-...

        access_token = ZYXCAB...

        account_type = practice (default) or live



        Parameters

        ==========

        conf_file: string

            path to and filename of the configuration file,

            e.g. '/home/me/oanda.cfg'

        '''

        self.config = configparser.ConfigParser()

        self.config.read(conf_file)

        self.access_token = self.config['oanda']['access_token']

        self.account_id = self.config['oanda']['account_id']

        self.account_type = self.config['oanda']['account_type']



        if self.account_type == 'live':

            self.hostname = 'api-fxtrade.oanda.com'

            self.stream_hostname = 'stream-fxtrade.oanda.com'

        else:

            self.hostname = 'api-fxpractice.oanda.com'

            self.stream_hostname = 'stream-fxpractice.oanda.com'



        self.ctx = v20.Context(

            hostname=self.hostname,

            port=443,

            token=self.access_token,

            poll_timeout=10

        )

        self.ctx_stream = v20.Context(

            hostname=self.stream_hostname,

            port=443,

            token=self.access_token,

        )



        self.suffix = '.000000000Z'

        self.stop_stream = False



    def get_instruments(self):

        ''' Retrieves and returns all instruments for the given account. '''

        resp = self.ctx.account.instruments(self.account_id)

        instruments = resp.get('instruments')

        instruments = [ins.dict() for ins in instruments]

        instruments = [(ins['displayName'], ins['name'])

                       for ins in instruments]

        return sorted(instruments)



    def transform_datetime(self, dati):

        ''' Transforms Python datetime object to string. '''

        if isinstance(dati, str):

            dati = pd.Timestamp(dati).to_pydatetime()

        return dati.isoformat('T') + self.suffix



    def retrieve_data(self, instrument, start, end, granularity, price):

        raw = self.ctx.instrument.candles(

            instrument=instrument,

            fromTime=start, toTime=end,

            granularity=granularity, price=price)

        raw = raw.get('candles')

        raw = [cs.dict() for cs in raw]

        if price == 'A':

            for cs in raw:

                cs.update(cs['ask'])

                del cs['ask']

        elif price == 'B':

            for cs in raw:

                cs.update(cs['bid'])

                del cs['bid']

        elif price == 'M':

            for cs in raw:

                cs.update(cs['mid'])

                del cs['mid']

        else:

            raise ValueError("price must be either 'B', 'A' or 'M'.")

        if len(raw) == 0:

            return pd.DataFrame()  # return empty DataFrame if no data

        data = pd.DataFrame(raw)

        data['time'] = pd.to_datetime(data['time'])

        data = data.set_index('time')

        data.index = pd.DatetimeIndex(data.index)

        for col in list('ohlc'):

            data[col] = data[col].astype(float)

        return data



    def get_history(self, instrument, start, end,

                    granularity, price, localize=True):

        ''' Retrieves historical data for instrument.



        Parameters

        ==========

        instrument: string

            valid instrument name

        start, end: datetime, str

            Python datetime or string objects for start and end

        granularity: string

            a string like 'S5', 'M1' or 'D'

        price: string

            one of 'A' (ask), 'B' (bid) or 'M' (middle)



        Returns

        =======

        data: pd.DataFrame

            pandas DataFrame object with data

        '''

        if granularity.startswith('S') or granularity.startswith('M'):

            if granularity.startswith('S'):

                freq = '1h'

            else:

                freq = 'D'

            data = pd.DataFrame()

            dr = pd.date_range(start, end, freq=freq)

            for t in range(len(dr) - 1):

                start = self.transform_datetime(dr[t])

                end = self.transform_datetime(dr[t + 1])

                batch = self.retrieve_data(instrument, start, end,

                                           granularity, price)

                data = data.append(batch)

        else:

            start = self.transform_datetime(start)

            end = self.transform_datetime(end)

            data = self.retrieve_data(instrument, start, end,

                                      granularity, price)

        if localize:

            data.index = data.index.tz_localize(None)



        return data[['o', 'h', 'l', 'c', 'volume', 'complete']]



    def create_order(self, instrument, units, sl_distance=None,

                     tsl_distance=None, tp_price=None, comment=None,

                     suppress=False, ret=False):

        ''' Places order with Oanda.



        Parameters

        ==========

        instrument: string

            valid instrument name

        units: int

            number of units of instrument to be bought

            (positive int, eg 'units=50')

            or to be sold (negative int, eg 'units=-100')

        sl_distance: float

            stop loss distance price, mandatory eg in Germany

        tsl_distance: float

            trailing stop loss distance

        tp_price: float

            take profit price to be used for the trade

        comment: str

            string

        '''

        client_ext = ClientExtensions(

            comment=comment) if comment is not None else None

        sl_details = (StopLossDetails(distance=sl_distance,

                                      clientExtensions=client_ext)

                      if sl_distance is not None else None)

        tsl_details = (TrailingStopLossDetails(distance=tsl_distance,

                                               clientExtensions=client_ext)

                       if tsl_distance is not None else None)

        tp_details = (TakeProfitDetails(

            price=tp_price, clientExtensions=client_ext)

            if tp_price is not None else None)



        request = self.ctx.order.market(

            self.account_id,

            instrument=instrument,

            units=units,

            stopLossOnFill=sl_details,

            trailingStopLossOnFill=tsl_details,

            takeProfitOnFill=tp_details,

        )

        try:

            order = request.get('orderFillTransaction')

        except:

            order = request.get('orderCreateTransaction')

        if not suppress:

            print('\n\n', order.dict(), '\n')

        if ret is True:

            return order.dict()


    def stream_data(self, instrument_list, stop=None, ret=False):

        ''' Starts a real-time data stream.



        Parameters

        ==========

        instrument: string

            valid instrument name

        '''

        self.stream_instrument = instrument_list
        
        self.ticks = 0
         
        response = self.ctx_stream.pricing.stream(

            self.account_id, snapshot=True,

            instruments=instrument_list)
        
        msgs = []
        
        for msg_type, msg in response.parts():
            
            msgs.append(msg)

            # print(msg_type, msg)
            
            heartbeat = f'Heartbeat @ {dt.datetime.now()}'
            
            self.on_success_heartbeat(heartbeat)

            if msg_type == 'pricing.ClientPrice' and msg.type == 'PRICE':
                
                self.ticks += 1

                self.time = msg.time

                df = pd.DataFrame({'%s_Time' % msg.instrument : dt.datetime.now(),
                                   '%s_bid' % msg.instrument : float(msg.bids[0].dict()['price']),
                                   '%s_ask' % msg.instrument : float(msg.asks[0].dict()['price'])},
                                                       index = [msg.instrument])

                self.on_success(df, msg.instrument)
        
                    
                if stop is not None:
                
                    if self.ticks >= stop:
                    
                        if ret:
                        
                            return msgs

                        break 

            if self.stop_stream:

                if ret:

                    return msgs

                break

    #def on_success(self, time, bid, ask):

        ''' Method called when new data is retrieved. '''

        #print(time, bid, ask)
        
    def on_success(self, df, instrument):

        ''' Method called when new data is retrieved. '''

        print(df, instrument)
        
        
    def on_success_heartbeat(self, heartbeat):
        
        print(heartbeat)    
        

    def get_account_summary(self, detailed=False):

        ''' Returns summary data for Oanda account.'''

        if detailed is True:

            response = self.ctx.account.get(self.account_id)

        else:

            response = self.ctx.account.summary(self.account_id)

        raw = response.get('account')

        return raw.dict()



    def get_transaction(self, tid=0):

        ''' Retrieves and returns tansaction data. '''

        response = self.ctx.transaction.get(self.account_id, tid)

        transaction = response.get('transaction')

        return transaction.dict()



    def get_transactions(self, tid=0):

        ''' Retrieves and returns transactions data. '''

        response = self.ctx.transaction.since(self.account_id, id=tid)

        transactions = response.get('transactions')

        transactions = [t.dict() for t in transactions]

        return transactions



    def print_transactions(self, tid=0):

        ''' Prints basic transactions data. '''

        transactions = self.get_transactions(tid)

        for trans in transactions:

            try:

                templ = '%5s | %s | %9s | %12s | %8s'

                print(templ % (trans['id'],

                               trans['time'],

                               trans['instrument'],

                               trans['units'],

                               trans['pl']))

            except:

                pass