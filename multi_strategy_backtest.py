#!/usr/bin/env python3
"""
Multi-Strategy Backtest for SPY Historical Data

This script implements three different trading strategies:
- Strategy A: Strict ADX filter (ADX > 43)
- Strategy B: ATR + RSI breakout
- Strategy C: Dynamic ADX + BBW + DI hybrid

All strategies use EMA trend detection and proper risk management.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MultiStrategyBacktest:
    def __init__(self, initial_capital: float = 100000.0, risk_per_trade: float = 0.05):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load SPY historical data from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        bars = data['bars']['SPY']
        df = pd.DataFrame(bars)
        
        # Rename columns for clarity
        df.columns = ['close', 'high', 'low', 'n', 'open', 'timestamp', 'volume', 'vwap']
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Rename for compatibility with strategy code
        df['Close'] = df['close']
        df['High'] = df['high']
        df['Low'] = df['low']
        df['Open'] = df['open']
        
        return df
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, and -DI"""
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        # Filter positive and negative directional movement
        dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
        dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
        
        # Convert to Series
        dm_plus = pd.Series(dm_plus, index=high.index)
        dm_minus = pd.Series(dm_minus, index=high.index)
        
        # Calculate smoothed values
        atr_smooth = tr.ewm(span=period, adjust=False).mean()
        dm_plus_smooth = dm_plus.ewm(span=period, adjust=False).mean()
        dm_minus_smooth = dm_minus.ewm(span=period, adjust=False).mean()
        
        # Calculate Directional Indicators
        di_plus = 100 * (dm_plus_smooth / atr_smooth)
        di_minus = 100 * (dm_minus_smooth / atr_smooth)
        
        # Calculate DX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # Calculate ADX
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx, di_plus, di_minus
    
    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, close: pd.Series, window: int = 20, window_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        bb_mid = close.rolling(window=window).mean()
        bb_std = close.rolling(window=window).std()
        bb_upper = bb_mid + (bb_std * window_dev)
        bb_lower = bb_mid - (bb_std * window_dev)
        return bb_upper, bb_lower, bb_mid
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # EMAs for trend detection
        df['ema_30'] = self.calculate_ema(df['Close'], 30)
        df['ema_46'] = self.calculate_ema(df['Close'], 46)
        df['ema_80'] = self.calculate_ema(df['Close'], 80)
        
        # Trend condition
        df['ema_up'] = (df['ema_30'] > df['ema_46']) & (df['ema_46'] > df['ema_80']) & (df['Close'] > df['ema_30'])
        
        # ADX + DI
        df['adx'], df['plus_di'], df['minus_di'] = self.calculate_adx(df['High'], df['Low'], df['Close'], 14)
        df['adx_rising'] = df['adx'] > df['adx'].shift(1)
        
        # ATR + RSI
        df['atr'] = self.calculate_atr(df['High'], df['Low'], df['Close'], 14)
        df['rsi'] = self.calculate_rsi(df['Close'], 14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_lower'], df['bb_mid'] = self.calculate_bollinger_bands(df['Close'], 20, 2)
        df['bbw'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bbw_expanding'] = df['bbw'] > df['bbw'].shift(1)
        
        return df
    
    def create_strategy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create entry and exit signals for all strategies"""
        # Strategy A: Strict ADX filter (lowered threshold)
        df['entry_A'] = df['ema_up'] & (df['adx'] > 35)
        df['exit_A'] = (~df['ema_up']) | (df['adx'] < 30)
        
        # Strategy B: ATR + RSI breakout
        atr_threshold = df['atr'].rolling(50).mean()
        df['entry_B'] = df['ema_up'] & (df['atr'] > atr_threshold) & (df['rsi'] > 55)
        df['exit_B'] = (~df['ema_up']) | (df['rsi'] < 45)
        
        # Strategy C: Dynamic ADX + BBW + DI hybrid
        df['entry_C'] = df['ema_up'] & (df['adx'] > 25) & df['adx_rising'] & (df['plus_di'] > df['minus_di']) & df['bbw_expanding']
        df['add_on_C'] = df['ema_up'] & (df['adx'] > 43) & df['adx_rising']
        df['exit_C'] = (~df['ema_up']) | (df['adx'] < 35) | (~df['adx_rising'])
        
        return df
    
    def backtest_strategy(self, df: pd.DataFrame, entry_col: str, exit_col: str, name: str) -> Dict:
        """Backtest a single strategy with proper risk management"""
        print(f"\nRunning {name}...")
        
        # Initialize tracking variables
        current_position = None
        trades = []
        equity_curve = []
        
        # Wait for indicators to stabilize
        start_idx = 80
        
        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            
            # Skip if any required indicator is NaN
            if pd.isna(row[entry_col]) or pd.isna(row[exit_col]):
                continue
            
            current_price = row['Close']
            
            # Check for exit conditions first
            if current_position and row[exit_col]:
                # Close position
                pnl = (current_price - current_position['entry_price']) * current_position['size']
                self.current_capital += pnl
                
                trade = {
                    'entry_date': current_position['entry_date'],
                    'exit_date': row['timestamp'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': current_price,
                    'size': current_position['size'],
                    'pnl': pnl,
                    'capital_after': self.current_capital
                }
                
                trades.append(trade)
                current_position = None
            
            # Check for entry conditions
            if not current_position and row[entry_col]:
                # Calculate position size based on risk management
                atr_value = row['atr']
                stop_loss_distance = 1.8 * atr_value
                risk_amount = self.current_capital * self.risk_per_trade
                
                if stop_loss_distance > 0:
                    position_size = int(risk_amount / stop_loss_distance)
                    
                    if position_size > 0:
                        current_position = {
                            'entry_price': current_price,
                            'size': position_size,
                            'entry_date': row['timestamp']
                        }
            
            # Record equity curve
            equity_curve.append({
                'date': row['timestamp'],
                'capital': self.current_capital,
                'price': current_price
            })
        
        # Analyze results
        return self.analyze_strategy_results(trades, equity_curve, name)
    
    def analyze_strategy_results(self, trades: List[Dict], equity_curve: List[Dict], name: str) -> Dict:
        """Analyze results for a single strategy"""
        if not trades:
            return {
                'name': name,
                'total_trades': 0,
                'final_capital': self.current_capital,
                'total_return': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'trades': []
            }
        
        trades_df = pd.DataFrame(trades)
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate max drawdown
        equity_df = pd.DataFrame(equity_curve)
        equity_df['peak'] = equity_df['capital'].cummax()
        equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Calculate Sharpe ratio (simplified)
        if len(trades_df) > 1:
            returns = trades_df['pnl'] / self.initial_capital
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        return {
            'name': name,
            'total_trades': total_trades,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades_df
        }
    
    def run_all_strategies(self, df: pd.DataFrame) -> Dict:
        """Run backtest for all three strategies"""
        print("Calculating technical indicators...")
        df = self.calculate_indicators(df)
        
        print("Creating strategy signals...")
        df = self.create_strategy_signals(df)
        
        results = {}
        
        # Reset capital for each strategy
        original_capital = self.current_capital
        
        # Strategy A: Strict ADX filter
        self.current_capital = original_capital
        results['A'] = self.backtest_strategy(df, 'entry_A', 'exit_A', "Strategy A: ADX Filter (35 threshold)")
        
        # Strategy B: ATR + RSI breakout
        self.current_capital = original_capital
        results['B'] = self.backtest_strategy(df, 'entry_B', 'exit_B', "Strategy B: ATR + RSI Breakout")
        
        # Strategy C: Dynamic ADX + BBW + DI hybrid
        self.current_capital = original_capital
        results['C'] = self.backtest_strategy(df, 'entry_C', 'exit_C', "Strategy C: Dynamic ADX + BBW + DI Hybrid")
        
        return results
    
    def print_comparison(self, results: Dict):
        """Print comparison of all strategies"""
        print("\n" + "="*80)
        print("STRATEGY COMPARISON RESULTS")
        print("="*80)
        
        print(f"{'Strategy':<30} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Sharpe':<8} {'Max DD':<10}")
        print("-" * 80)
        
        for strategy_key, result in results.items():
            print(f"{result['name']:<30} "
                  f"{result['total_return']:>8.2%} "
                  f"{result['total_trades']:>6} "
                  f"{result['win_rate']:>8.2%} "
                  f"{result['sharpe_ratio']:>6.2f} "
                  f"{result['max_drawdown']:>8.2%}")
        
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for strategy_key, result in results.items():
            print(f"\n{result['name']}:")
            print(f"  Final Capital: ${result['final_capital']:,.2f}")
            print(f"  Total Return: {result['total_return']:.2%}")
            print(f"  Total Trades: {result['total_trades']}")
            print(f"  Win Rate: {result['win_rate']:.2%}")
            print(f"  Average Win: ${result['avg_win']:,.2f}")
            print(f"  Average Loss: ${result['avg_loss']:,.2f}")
            print(f"  Profit Factor: {result['profit_factor']:.2f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            
            if result['total_trades'] > 0:
                print(f"  Recent Trades:")
                recent_trades = result['trades'].tail(3)
                for _, trade in recent_trades.iterrows():
                    print(f"    {trade['entry_date'].strftime('%Y-%m-%d')} | "
                          f"Entry: ${trade['entry_price']:.2f} | "
                          f"Exit: ${trade['exit_price']:.2f} | "
                          f"PnL: ${trade['pnl']:,.2f}")
        
        # Find best strategy
        best_strategy = max(results.values(), key=lambda x: x['total_return'])
        print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy['name']}")
        print(f"   Return: {best_strategy['total_return']:.2%}")
        print(f"   Final Capital: ${best_strategy['final_capital']:,.2f}")


def main():
    """Main function to run all strategy backtests"""
    print("Multi-Strategy Backtest for SPY Historical Data")
    print("=" * 50)
    
    # Initialize backtest
    backtest = MultiStrategyBacktest(initial_capital=100000.0, risk_per_trade=0.05)
    
    # Load data
    data_file = "/Users/vamsiravi/Documents/Apps/adxema/adxema/testdata/spy-ytd.json"
    df = backtest.load_data(data_file)
    
    print(f"Loaded {len(df)} data points from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Run all strategies
    results = backtest.run_all_strategies(df)
    
    # Print comparison
    backtest.print_comparison(results)
    
    return results


if __name__ == "__main__":
    results = main()
