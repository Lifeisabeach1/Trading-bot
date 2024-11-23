
import os
import time
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_scanner.log'),
        logging.StreamHandler()
    ]
)

class BinanceScanner:
    def __init__(self):
        
        load_dotenv()
        
        # API credentials
        self.API_KEY = os.getenv('BINANCE_API_KEY')
        self.API_SECRET = os.getenv('BINANCE_API_SECRET')
        
        if not self.API_KEY or not self.API_SECRET:
            raise ValueError("API credentials not found in environment variables")
        
        # Binance client
        self.client = Client(self.API_KEY, self.API_SECRET)
        
        
        self.sync_time()
        
        # Technical conditions
        self.RSI_PERIOD = 14
        self.RSI_OVERSOLD = 30
        self.FAST_MA_PERIOD = 20
        self.SLOW_MA_PERIOD = 50
        
        
        self.triggered_alerts = set()
    
    def sync_time(self):
        """Synchronize local time with Binance server time."""
        try:
            server_time = self.client.get_server_time()
            self.time_offset = int(server_time['serverTime']) - int(time.time() * 1000)
            self.client.timestamp_offset = self.time_offset
            logging.info(f"Time synchronized with Binance server. Offset: {self.time_offset}ms")
        except BinanceAPIException as e:
            logging.error(f"Error synchronizing time: {str(e)}")
            raise
    
    def calculate_rsi(self, closes, period=14):
        """Calculate RSI indicator."""
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ma(self, closes, period):
        """Calculate Moving Average."""
        return closes.rolling(window=period).mean()
    
    def fetch_klines_data(self, symbol, interval='1h', limit=100):
        """Fetch klines data from Binance."""
        try:
            klines = self.client.get_klines(
                symbol=symbol.replace('/', ''),  
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except BinanceAPIException as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def analyze_symbol(self, symbol):
        """Analyze a single symbol with detailed condition status."""
        df = self.fetch_klines_data(symbol)
        if df is None or len(df) < self.SLOW_MA_PERIOD:
            return None
        
        
        df['rsi'] = self.calculate_rsi(df['close'], self.RSI_PERIOD)
        df['fast_ma'] = self.calculate_ma(df['close'], self.FAST_MA_PERIOD)
        df['slow_ma'] = self.calculate_ma(df['close'], self.SLOW_MA_PERIOD)
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Conditions
        rsi_value = latest['rsi']
        rsi_distance = abs(rsi_value - self.RSI_OVERSOLD)
        rsi_fulfilled = rsi_value <= self.RSI_OVERSOLD
        
        # MA Crossover analysis
        current_diff = latest['fast_ma'] - latest['slow_ma']
        previous_diff = previous['fast_ma'] - previous['slow_ma']
        ma_fulfilled = previous_diff <= 0 and current_diff > 0
        ma_distance = abs(current_diff)
        
        # Price above MA analysis
        price_ma_diff = latest['close'] - latest['slow_ma']
        price_ma_fulfilled = price_ma_diff > 0
        
        # Volume 
        avg_volume = df['volume'].mean()
        volume_ratio = latest['volume'] / avg_volume
        volume_fulfilled = volume_ratio > 1.5
        volume_distance = abs(volume_ratio - 1.5)
        
        conditions = {
            'rsi_oversold': {
                'fulfilled': rsi_fulfilled,
                'current': round(rsi_value, 2),
                'target': self.RSI_OVERSOLD,
                'distance': round(rsi_distance, 2),
                'percentage': round((min(rsi_value, 70) / self.RSI_OVERSOLD) * 100, 1)
            },
            'ma_crossover': {
                'fulfilled': ma_fulfilled,
                'current': round(current_diff, 2),
                'previous': round(previous_diff, 2),
                'distance': round(ma_distance, 2),
                'percentage': round(
                    (1 - min(abs(current_diff), abs(previous_diff)) / 
                     max(abs(current_diff), abs(previous_diff))) * 100 
                    if max(abs(current_diff), abs(previous_diff)) > 0 else 0, 1)
            },
            'price_above_ma': {
                'fulfilled': price_ma_fulfilled,
                'current': round(price_ma_diff, 2),
                'percentage': round(
                    (latest['close'] / latest['slow_ma'] * 100 - 100), 1)
            },
            'volume_spike': {
                'fulfilled': volume_fulfilled,
                'current': round(volume_ratio, 2),
                'target': 1.5,
                'distance': round(volume_distance, 2),
                'percentage': round((volume_ratio / 1.5) * 100, 1)
            }
        }
        
        return {
            'symbol': symbol,
            'price': round(latest['close'], 2),
            'conditions': conditions,
            'met_conditions': sum(cond['fulfilled'] for cond in conditions.values()),
            'total_completion': round(
                sum(cond['percentage'] for cond in conditions.values()) / 4, 1
            )
        }

    def scan_market(self):
        """Scan all USDT pairs with RSI-focused analysis."""
        try:
            exchange_info = self.client.get_exchange_info()
            usdt_pairs = [
                symbol['symbol'] 
                for symbol in exchange_info['symbols'] 
                if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING'
            ]
            
            results = []
            for symbol in usdt_pairs:
                analysis = self.analyze_symbol(symbol)
                if analysis:
                    results.append(analysis)
            
            # Sort by RSI value 
            results.sort(key=lambda x: x['conditions']['rsi_oversold']['current'])
            return results
            
        except BinanceAPIException as e:
            logging.error(f"Error in market scan: {str(e)}")
            return []

    def format_condition_status(self, condition_data, include_value=False):
        """Format condition status for display."""
        status = "✅" if condition_data['fulfilled'] else "❌"
        if include_value and 'current' in condition_data:
            return f"{status} ({condition_data['current']})"
        return f"{status} ({condition_data['percentage']}%)"

    def run(self, interval=300):
        """Run the scanner with RSI-focused display."""
        logging.info("Starting Binance Scanner with RSI Focus...")
        
        while True:
            try:
                if int(time.time()) % 3600 == 0:
                    self.sync_time()
                
                print("\n" + "="*120)
                print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*120)
                
                results = self.scan_market()
                
                # Table header
                print(f"{'Symbol':<12} {'Price':<10} {'RSI':<15} {'MA Cross':<20} {'Price>MA':<15} {'Volume':<15} {'Conditions Met'}")
                print("-"*120)
                
                # Group results by RSI ranges
                rsi_ranges = {
                    "Extremely Oversold (RSI < 20)": [],
                    "Oversold (RSI 20-30)": [],
                    "Near Oversold (RSI 30-40)": [],
                    "Normal Range (RSI 40+)": []
                }
                
                for r in results:
                    rsi = r['conditions']['rsi_oversold']['current']
                    if rsi < 20:
                        rsi_ranges["Extremely Oversold (RSI < 20)"].append(r)
                    elif rsi <= 30:
                        rsi_ranges["Oversold (RSI 20-30)"].append(r)
                    elif rsi <= 40:
                        rsi_ranges["Near Oversold (RSI 30-40)"].append(r)
                    else:
                        rsi_ranges["Normal Range (RSI 40+)"].append(r)
                
                # Print results RSI
                for range_name, range_results in rsi_ranges.items():
                    if range_results:
                        print(f"\n{range_name}:")
                        print("-" * 120)
                        
                        for r in range_results:
                            conditions = r['conditions']
                            met_conditions = r['met_conditions']
                            
                           
                            rsi_value = conditions['rsi_oversold']['current']
                            
                            print(
                                f"{r['symbol']:<12} "
                                f"${r['price']:<9} "
                                f"RSI={rsi_value:<8.1f} | "
                                f"MA={self.format_condition_status(conditions['ma_crossover']):<15} | "
                                f"P/MA={self.format_condition_status(conditions['price_above_ma']):<12} | "
                                f"Vol={self.format_condition_status(conditions['volume_spike']):<12} | "
                                f"{met_conditions}/4 conditions"
                            )
                            
                            # Show detailes
                            if met_conditions >= 2 and rsi_value <= 40:
                                print(f"  Detailed Analysis:")
                                print(f"  • Fast MA vs Slow MA: {conditions['ma_crossover']['current']:+.2f}")
                                print(f"  • Price vs Slow MA: {conditions['price_above_ma']['current']:+.2f}")
                                print(f"  • Volume: {conditions['volume_spike']['current']:.1f}x average")
                                if met_conditions >= 3:
                                    print(f"  ⚠️ HIGH POTENTIAL SETUP - {4-met_conditions} condition(s) away from full signal")
                                print(f"  {'-'*60}")
                
                # Summary statistics
                total_symbols = sum(len(group) for group in rsi_ranges.values())
                oversold_count = len(rsi_ranges["Extremely Oversold (RSI < 20)"]) + len(rsi_ranges["Oversold (RSI 20-30)"])
                
                print("\nSummary:")
                print(f"Total Symbols Scanned: {total_symbols}")
                print(f"Oversold Symbols (RSI <= 30): {oversold_count}")
                print(f"Symbols with 3+ Conditions Met: {sum(1 for r in results if r['met_conditions'] >= 3)}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logging.info("Scanner stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    scanner = BinanceScanner()
    scanner.run()


