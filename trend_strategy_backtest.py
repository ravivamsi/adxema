#!/usr/bin/env python3
"""
Trend Strategy Backtest for SPY Historical Data

This script implements a trend-following strategy using:
- 3 EMAs (30, 46, 80 periods) for trend detection
- ADX (Average Directional Index) for trend strength
- ATR (Average True Range) for stop loss and take profit levels
- Risk management with 5% risk per trade

Strategy Rules:
- Go long when uptrend (30EMA > 46EMA > 80EMA) and ADX > 43
- Go short when downtrend (80EMA > 46EMA > 30EMA) and ADX > 43
- Stop loss: Entry ± 1.8 * ATR
- Take profit: Entry ± 3.3 * ATR
- Risk: 5% of account per trade
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class TrendStrategyBacktest:
    def __init__(self, initial_capital: float = 100000.0, risk_per_trade: float = 0.05):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
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
        
        return df
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the EMA of True Range
        atr = true_range.ewm(span=period, adjust=False).mean()
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
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
        
        return adx
    
    def detect_trend(self, ema_30: float, ema_46: float, ema_80: float, current_price: float) -> str:
        """Detect trend direction based on EMA alignment"""
        if ema_30 > ema_46 > ema_80 and current_price > ema_30:
            return 'uptrend'
        elif ema_80 > ema_46 > ema_30 and current_price < ema_30:
            return 'downtrend'
        else:
            return 'sideways'
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management"""
        risk_amount = self.current_capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        position_size = int(risk_amount / risk_per_share)
        return max(0, position_size)
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run the complete backtest"""
        print("Running trend strategy backtest...")
        
        # Calculate technical indicators
        df['ema_30'] = self.calculate_ema(df['close'], 30)
        df['ema_46'] = self.calculate_ema(df['close'], 46)
        df['ema_80'] = self.calculate_ema(df['close'], 80)
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'])
        
        # Wait for indicators to stabilize (need at least 80 periods for EMAs)
        start_idx = 80
        
        current_position = None
        
        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            
            # Skip if any indicator is NaN
            if pd.isna(row['ema_30']) or pd.isna(row['ema_46']) or pd.isna(row['ema_80']) or \
               pd.isna(row['atr']) or pd.isna(row['adx']):
                continue
            
            current_price = row['close']
            trend = self.detect_trend(row['ema_30'], row['ema_46'], row['ema_80'], current_price)
            
            # Check for exit conditions first
            if current_position:
                exit_price = None
                exit_reason = None
                
                if current_position['side'] == 'long':
                    # Long position exit conditions
                    if current_price <= current_position['stop_loss']:
                        exit_price = current_position['stop_loss']
                        exit_reason = 'stop_loss'
                    elif current_price >= current_position['take_profit']:
                        exit_price = current_position['take_profit']
                        exit_reason = 'take_profit'
                    elif trend != 'uptrend' or row['adx'] <= 43:
                        exit_price = current_price
                        exit_reason = 'trend_change'
                
                elif current_position['side'] == 'short':
                    # Short position exit conditions
                    if current_price >= current_position['stop_loss']:
                        exit_price = current_position['stop_loss']
                        exit_reason = 'stop_loss'
                    elif current_price <= current_position['take_profit']:
                        exit_price = current_position['take_profit']
                        exit_reason = 'take_profit'
                    elif trend != 'downtrend' or row['adx'] <= 43:
                        exit_price = current_price
                        exit_reason = 'trend_change'
                
                if exit_price:
                    # Close position
                    pnl = (exit_price - current_position['entry_price']) * current_position['size'] * \
                          (1 if current_position['side'] == 'long' else -1)
                    
                    self.current_capital += pnl
                    
                    trade = {
                        'entry_date': current_position['entry_date'],
                        'exit_date': row['timestamp'],
                        'side': current_position['side'],
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'size': current_position['size'],
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'capital_after': self.current_capital
                    }
                    
                    self.trades.append(trade)
                    current_position = None
            
            # Check for entry conditions
            if not current_position and row['adx'] > 43:
                if trend == 'uptrend':
                    # Long entry
                    entry_price = current_price
                    stop_loss = entry_price - 1.8 * row['atr']
                    take_profit = entry_price + 3.3 * row['atr']
                    
                    position_size = self.calculate_position_size(entry_price, stop_loss)
                    
                    if position_size > 0:
                        current_position = {
                            'side': 'long',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'size': position_size,
                            'entry_date': row['timestamp']
                        }
                
                elif trend == 'downtrend':
                    # Short entry
                    entry_price = current_price
                    stop_loss = entry_price + 1.8 * row['atr']
                    take_profit = entry_price - 3.3 * row['atr']
                    
                    position_size = self.calculate_position_size(entry_price, stop_loss)
                    
                    if position_size > 0:
                        current_position = {
                            'side': 'short',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'size': position_size,
                            'entry_date': row['timestamp']
                        }
            
            # Record equity curve
            self.equity_curve.append({
                'date': row['timestamp'],
                'capital': self.current_capital,
                'price': current_price,
                'position': current_position['side'] if current_position else None
            })
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        """Analyze backtest results"""
        if not self.trades:
            return {
                'total_trades': 0,
                'final_capital': self.current_capital,
                'total_return': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
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
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['capital'].cummax()
        equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        return {
            'total_trades': total_trades,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades': trades_df
        }
    
    def plot_results(self, df: pd.DataFrame, results: Dict):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price and EMAs
        axes[0].plot(df['timestamp'], df['close'], label='SPY Close', alpha=0.7)
        axes[0].plot(df['timestamp'], df['ema_30'], label='EMA 30', alpha=0.8)
        axes[0].plot(df['timestamp'], df['ema_46'], label='EMA 46', alpha=0.8)
        axes[0].plot(df['timestamp'], df['ema_80'], label='EMA 80', alpha=0.8)
        axes[0].set_title('SPY Price and EMAs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: ADX
        axes[1].plot(df['timestamp'], df['adx'], label='ADX', color='purple')
        axes[1].axhline(y=43, color='red', linestyle='--', alpha=0.7, label='ADX Threshold (43)')
        axes[1].set_title('ADX (Average Directional Index)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Equity Curve
        equity_df = pd.DataFrame(self.equity_curve)
        axes[2].plot(equity_df['date'], equity_df['capital'], label='Account Value', color='green')
        axes[2].axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        axes[2].set_title(f'Equity Curve - Final Value: ${self.current_capital:,.2f}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, results: Dict):
        """Print backtest summary"""
        print("\n" + "="*60)
        print("TREND STRATEGY BACKTEST RESULTS")
        print("="*60)
        
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        
        if results['total_trades'] > 0:
            print(f"Win Rate: {results['win_rate']:.2%}")
            print(f"Average Win: ${results['avg_win']:,.2f}")
            print(f"Average Loss: ${results['avg_loss']:,.2f}")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        print("\n" + "="*60)
        
        # Show recent trades
        if results['total_trades'] > 0:
            print("\nRecent Trades:")
            print("-" * 80)
            recent_trades = results['trades'].tail(10)
            for _, trade in recent_trades.iterrows():
                print(f"{trade['entry_date'].strftime('%Y-%m-%d')} | "
                      f"{trade['side'].upper()} | "
                      f"Entry: ${trade['entry_price']:.2f} | "
                      f"Exit: ${trade['exit_price']:.2f} | "
                      f"PnL: ${trade['pnl']:,.2f} | "
                      f"Reason: {trade['exit_reason']}")


def main():
    """Main function to run the backtest"""
    # Initialize backtest
    backtest = TrendStrategyBacktest(initial_capital=100000.0, risk_per_trade=0.05)
    
    # Load data
    data_file = "/Users/vamsiravi/Documents/Apps/adxema/adxema/testdata/spy-ytd.json"
    df = backtest.load_data(data_file)
    
    print(f"Loaded {len(df)} data points from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Run backtest
    results = backtest.run_backtest(df)
    
    # Print results
    backtest.print_summary(results)
    
    # Plot results
    backtest.plot_results(df, results)
    
    return results


if __name__ == "__main__":
    results = main()
