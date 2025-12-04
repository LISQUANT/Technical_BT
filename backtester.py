#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-04T23:29:46.903Z
"""

import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import ipywidgets as widgets
import os
from datetime import datetime, timedelta

# ## Yfinance version


# If needed: !pip install yfinance ipywidgets matplotlib pandas numpy --quiet

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import ipywidgets as widgets
import os
from datetime import datetime

# ================================
# 1. Fetch and clean stock data
# ================================
def fetch_stock_data(ticker: str, years: int):
    """Download stock data and ensure proper column format"""
    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
   
    if df is None or df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
   
    # Fix: Remove MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
   
    df.index = pd.to_datetime(df.index)
   
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
   
    return df

# ================================
# 2. Add technical indicators
# ================================
def add_indicators(df, ma_window, rsi_window, sr_window):
    """Calculate technical indicators"""
    df = df.copy()
   
    # Moving Average
    df['MA'] = df['Close'].rolling(window=ma_window).mean()
   
    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
   
    # Support and Resistance
    df['Support'] = df['Low'].rolling(window=sr_window).min().shift(1)
    df['Resistance'] = df['High'].rolling(window=sr_window).max().shift(1)
   
    return df

# ================================
# 3. Backtest the strategy
# ================================
def backtest_strategy(ticker, ma_window, rsi_window, sr_window, rsi_lower, rsi_upper, years):
    """Run backtest simulation"""
    df = fetch_stock_data(ticker, years)
    df = add_indicators(df, ma_window, rsi_window, sr_window)
   
    # Drop NaN rows from indicators
    df = df.dropna()
   
    if len(df) == 0:
        raise ValueError("No data left after calculating indicators")
   
    # Initialize backtest variables
    initial_capital = 10000.0
    capital = initial_capital
    shares = 0.0
    in_position = False
   
    equity_curve = []
    buy_signals = []
    sell_signals = []
    trades = []  # Track individual trades
   
    # Buy & Hold benchmark
    shares_bh = initial_capital / df['Close'].iloc[0]
    bh_curve = []
   
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]
       
        # Current equity
        current_equity = (shares * price) if in_position else capital
        equity_curve.append(current_equity)
        bh_curve.append(shares_bh * price)
       
        if i == 0:
            continue
       
        # BUY SIGNAL: RSI crosses above lower threshold AND price above MA
        if not in_position:
            if (df['RSI'].iloc[i] > rsi_lower and
                df['RSI'].iloc[i-1] <= rsi_lower and
                price > df['MA'].iloc[i]):
               
                shares = capital / price
                buy_price = price
                buy_date = date
                capital = 0.0
                in_position = True
                buy_signals.append((date, price))
       
        # SELL SIGNAL: RSI crosses below upper threshold OR price below MA
        elif in_position:
            if ((df['RSI'].iloc[i] < rsi_upper and df['RSI'].iloc[i-1] >= rsi_upper) or
                price < df['MA'].iloc[i]):
               
                capital = shares * price
                sell_price = price
                sell_date = date
               
                # Record trade
                profit_loss = (sell_price - buy_price) / buy_price
                trades.append({
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': profit_loss
                })
               
                shares = 0.0
                in_position = False
                sell_signals.append((date, price))
   
    # Close final position if still open
    if in_position:
        final_price = df['Close'].iloc[-1]
        capital = shares * final_price
        profit_loss = (final_price - buy_price) / buy_price
        trades.append({
            'buy_date': buy_date,
            'sell_date': df.index[-1],
            'buy_price': buy_price,
            'sell_price': final_price,
            'return': profit_loss
        })
        sell_signals.append((df.index[-1], final_price))
   
    equity_series = pd.Series(equity_curve, index=df.index)
    bh_series = pd.Series(bh_curve, index=df.index)
   
    return {
        "data": df,
        "equity_curve": equity_series,
        "buy_hold_curve": bh_series,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "trades": trades,
        "initial_capital": initial_capital
    }

# ================================
# 4. Calculate performance metrics
# ================================
def calculate_metrics(result):
    """Calculate performance statistics"""
    equity = result["equity_curve"]
    bh = result["buy_hold_curve"]
    trades = result["trades"]
    initial = result["initial_capital"]
    data = result["data"]
   
    # Calculate number of years
    days = (data.index[-1] - data.index[0]).days
    years = days / 365.25
   
    # Total Return
    final_value = equity.iloc[-1]
    total_return = (final_value - initial) / initial * 100
   
    # Buy & Hold Return
    bh_final = bh.iloc[-1]
    bh_return = (bh_final - initial) / initial * 100
   
    # Annualized Return
    if years > 0:
        annualized_return = ((final_value / initial) ** (1 / years) - 1) * 100
        bh_annualized = ((bh_final / initial) ** (1 / years) - 1) * 100
    else:
        annualized_return = 0
        bh_annualized = 0
   
    # Win Rate & Average Trade Return
    if len(trades) > 0:
        winning_trades = sum(1 for t in trades if t['return'] > 0)
        win_rate = (winning_trades / len(trades)) * 100
        avg_trade_return = (sum(t['return'] for t in trades) / len(trades)) * 100
    else:
        win_rate = 0
        avg_trade_return = 0
   
    # Maximum Drawdown
    cumulative = equity.cummax()
    drawdown = (equity - cumulative) / cumulative * 100
    max_drawdown = abs(drawdown.min())
   
    # Sharpe Ratio (simplified, assuming daily returns)
    returns = equity.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    else:
        sharpe = 0
   
    return {
        'total_return': round(total_return, 2),
        'bh_return': round(bh_return, 2),
        'annualized_return': round(annualized_return, 2),
        'bh_annualized': round(bh_annualized, 2),
        'num_trades': len(trades),
        'win_rate': round(win_rate, 2),
        'avg_trade_return': round(avg_trade_return, 2),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe': round(sharpe, 2),
        'final_value': round(final_value, 2)
    }

# ================================
# 5. Save results to files
# ================================
def save_results(result, ticker, ma_window, rsi_window, sr_window, rsi_lower, rsi_upper, save_dir="backtest_results"):
    """Save backtest results to files"""
   
    # Create directory if doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
   
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{ticker}_{timestamp}"
   
    # 1. Save metrics to TXT file
    metrics = calculate_metrics(result)
    txt_path = os.path.join(save_dir, f"{base_filename}_metrics.txt")
   
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"BACKTEST RESULTS FOR {ticker}\n")
        f.write("="*70 + "\n\n")
       
        f.write("PARAMETERS USED:\n")
        f.write(f"  MA Window: {ma_window}\n")
        f.write(f"  RSI Window: {rsi_window}\n")
        f.write(f"  S/R Window: {sr_window}\n")
        f.write(f"  RSI Lower: {rsi_lower}\n")
        f.write(f"  RSI Upper: {rsi_upper}\n\n")
       
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Total Return: {metrics['total_return']:.2f}%\n")
        f.write(f"  Buy & Hold Return: {metrics['bh_return']:.2f}%\n")
        f.write(f"  Annualized Return: {metrics['annualized_return']:.2f}% (Buy & Hold: {metrics['bh_annualized']:.2f}%)\n")
        f.write(f"  Win Rate: {metrics['win_rate']:.2f}%\n")
        f.write(f"  Average Trade Return: {metrics['avg_trade_return']:.2f}%\n")
        f.write(f"  Maximum Drawdown: {metrics['max_drawdown']:.2f}%\n")
        f.write(f"  Number of Trades: {metrics['num_trades']}\n")
        f.write(f"  Sharpe Ratio: {metrics['sharpe']:.2f}\n")
        f.write(f"  Final Portfolio Value: ${metrics['final_value']:,.2f}\n")
        f.write("\n" + "="*70 + "\n")
       
        # Trade log
        f.write("\nTRADE LOG:\n")
        f.write("-"*70 + "\n")
        trades = result["trades"]
        for i, trade in enumerate(trades, 1):
            f.write(f"\nTrade #{i}:\n")
            f.write(f"  Buy Date:  {trade['buy_date'].strftime('%Y-%m-%d')}\n")
            f.write(f"  Buy Price: ${trade['buy_price']:.2f}\n")
            f.write(f"  Sell Date: {trade['sell_date'].strftime('%Y-%m-%d')}\n")
            f.write(f"  Sell Price: ${trade['sell_price']:.2f}\n")
            f.write(f"  Return: {trade['return']*100:.2f}%\n")
   
    # 2. Save trades to CSV
    csv_path = os.path.join(save_dir, f"{base_filename}_trades.csv")
    trades_df = pd.DataFrame(result["trades"])
    if not trades_df.empty:
        trades_df['return_pct'] = trades_df['return'] * 100
        trades_df.to_csv(csv_path, index=False)
   
    # 3. Save equity curve to CSV
    equity_csv_path = os.path.join(save_dir, f"{base_filename}_equity.csv")
    equity_data = pd.DataFrame({
        'Date': result["equity_curve"].index,
        'Strategy_Value': result["equity_curve"].values,
        'BuyHold_Value': result["buy_hold_curve"].values
    })
    equity_data.to_csv(equity_csv_path, index=False)
   
    # 4. Save chart as PNG
    png_path = os.path.join(save_dir, f"{base_filename}_chart.png")
   
    return {
        'txt_path': txt_path,
        'csv_path': csv_path,
        'equity_csv_path': equity_csv_path,
        'png_path': png_path,
        'timestamp': timestamp
    }

# ================================
# 6. Plot results
# ================================
def plot_results(result, ticker, ma_window, rsi_window, sr_window, rsi_lower, rsi_upper, save_chart=False):
    """Visualize backtest results"""
    df = result["data"]
    equity_curve = result["equity_curve"]
    bh_curve = result["buy_hold_curve"]
    buy_signals = result["buy_signals"]
    sell_signals = result["sell_signals"]
   
    # Calculate metrics
    metrics = calculate_metrics(result)
   
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
   
    # Subplot 1: Price + Signals + Indicators
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df["Close"], label="Close Price", color="blue", linewidth=1.5)
    ax1.plot(df.index, df["MA"], label=f"MA ({ma_window})", color="orange", linewidth=1.5)
    ax1.plot(df.index, df["Support"], label="Support", linestyle="--", color="green", alpha=0.5)
    ax1.plot(df.index, df["Resistance"], label="Resistance", linestyle="--", color="red", alpha=0.5)
   
    if buy_signals:
        b_dates, b_prices = zip(*buy_signals)
        ax1.scatter(b_dates, b_prices, marker="^", color="lime", s=120,
                   label="Buy Signal", zorder=5, edgecolors='darkgreen', linewidths=1.5)
   
    if sell_signals:
        s_dates, s_prices = zip(*sell_signals)
        ax1.scatter(s_dates, s_prices, marker="v", color="red", s=120,
                   label="Sell Signal", zorder=5, edgecolors='darkred', linewidths=1.5)
   
    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.set_title(f"{ticker} - Technical Analysis Backtest", fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
   
    # Subplot 2: RSI
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df["RSI"], label="RSI", color="purple", linewidth=1.5)
    ax2.axhline(rsi_lower, linestyle="--", color="green", label=f"Oversold ({rsi_lower})", alpha=0.7)
    ax2.axhline(rsi_upper, linestyle="--", color="red", label=f"Overbought ({rsi_upper})", alpha=0.7)
    ax2.axhline(50, linestyle=":", color="gray", alpha=0.5)
    ax2.fill_between(df.index, 0, 100, where=(df["RSI"] < rsi_lower), alpha=0.2, color='green')
    ax2.fill_between(df.index, 0, 100, where=(df["RSI"] > rsi_upper), alpha=0.2, color='red')
    ax2.set_ylabel("RSI", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
   
    # Subplot 3: Equity Curve
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(equity_curve.index, equity_curve, label="Strategy", color="blue", linewidth=2)
    ax3.plot(bh_curve.index, bh_curve, label="Buy & Hold", color="gray",
            linewidth=2, linestyle='--', alpha=0.7)
    ax3.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
   
    plt.tight_layout()
   
    # Save chart if requested
    if save_chart:
        saved_files = save_results(result, ticker, ma_window, rsi_window, sr_window, rsi_lower, rsi_upper)
        plt.savefig(saved_files['png_path'], dpi=300, bbox_inches='tight')
        print(f"\nResults saved:")
        print(f"   Metrics: {saved_files['txt_path']}")
        print(f"   Trades CSV: {saved_files['csv_path']}")
        print(f"   Equity CSV: {saved_files['equity_csv_path']}")
        print(f"   Chart: {saved_files['png_path']}")
   
    plt.show()
   
    # Print performance metrics in formatted style
    m = metrics
   
    print("\n" + "="*60)
    print(f"Backtest Results for {ticker}:")
    print("="*60)
    print(f"Total Return: {m['total_return']:>6.2f}%")
    print(f"Buy & Hold Return: {m['bh_return']:>6.2f}%")
    print(f"Annualized Return: {m['annualized_return']:>6.2f}%  (Buy & Hold: {m['bh_annualized']:.2f}%)")
    print(f"Win Rate: {m['win_rate']:>6.2f}%")
    print(f"Average Trade Return: {m['avg_trade_return']:>6.2f}%")
    print(f"Maximum Drawdown: {m['max_drawdown']:>6.2f}%")
    print(f"Number of Trades: {m['num_trades']}")
    print(f"Sharpe Ratio: {m['sharpe']:>6.2f}")
    print(f"Final Portfolio Value: ${m['final_value']:,.2f}")
    print("="*60)

# ================================
# 7. Interactive dashboard
# ================================
def run_dashboard():
    """Launch interactive parameter tuning dashboard"""
    ticker_widget = widgets.Text(value='AAPL', description='Ticker:')
    years_widget = widgets.IntSlider(value=5, min=1, max=20, description='Years:')
    ma_widget = widgets.IntSlider(value=50, min=5, max=200, step=5, description='MA Window:')
    rsi_widget = widgets.IntSlider(value=14, min=2, max=50, step=1, description='RSI Window:')
    sr_widget = widgets.IntSlider(value=20, min=5, max=100, step=5, description='S/R Window:')
    rsi_lower_widget = widgets.IntSlider(value=30, min=5, max=50, description='RSI Lower:')
    rsi_upper_widget = widgets.IntSlider(value=70, min=50, max=95, description='RSI Upper:')
    save_checkbox = widgets.Checkbox(value=False, description='Save Results')
   
    output_area = widgets.Output()
   
    def _update(change):
        with output_area:
            clear_output(wait=True)
            try:
                ticker = ticker_widget.value
                years = years_widget.value
                ma_window = ma_widget.value
                rsi_window = rsi_widget.value
                sr_window = sr_widget.value
                rsi_lower = rsi_lower_widget.value
                rsi_upper = rsi_upper_widget.value
                save_results_flag = save_checkbox.value
               
                print(f"Running backtest for {ticker}...")
                result = backtest_strategy(ticker, ma_window, rsi_window, sr_window,
                                         rsi_lower, rsi_upper, years)
                plot_results(result, ticker, ma_window, rsi_window, sr_window,
                            rsi_lower, rsi_upper, save_chart=save_results_flag)
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
   
    # Attach observers to all widgets
    ticker_widget.observe(_update, names='value')
    years_widget.observe(_update, names='value')
    ma_widget.observe(_update, names='value')
    rsi_widget.observe(_update, names='value')
    sr_widget.observe(_update, names='value')
    rsi_lower_widget.observe(_update, names='value')
    rsi_upper_widget.observe(_update, names='value')
    save_checkbox.observe(_update, names='value')
   
    ui = widgets.VBox([
        widgets.HTML("<h3>Stock Backtesting Dashboard</h3>"),
        widgets.HBox([ticker_widget, years_widget]),
        widgets.HBox([ma_widget, rsi_widget, sr_widget]),
        widgets.HBox([rsi_lower_widget, rsi_upper_widget]),
        save_checkbox,
        output_area
    ])
   
    display(ui)
   
    # Run initial backtest
    _update(None)

# Launch the interactive dashboard
print("System ready! Launching dashboard...")
run_dashboard()

# ## Alpha Vantage


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import ipywidgets as widgets
import os
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries

API_KEY = "D3AAL06Y1J22VJVE"

# ================================
# 1. Fetch and clean stock data
# ================================
def fetch_stock_data(ticker: str, years: int):
    """Download stock data and ensure proper column format"""
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    df, meta = ts.get_daily(symbol=ticker, outputsize='compact')
   
    if df is None or df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
   
    # Rename Alpha Vantage columns to standard format
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
   
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
   
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
   
    return df

# ================================
# 2. Add technical indicators
# ================================
def add_indicators(df, ma_window, rsi_window, sr_window):
    """Calculate technical indicators"""
    df = df.copy()
   
    # Moving Average
    df['MA'] = df['Close'].rolling(window=ma_window).mean()
   
    # RSI Calculation (Simplified - more reliable)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
   
    # Support and Resistance
    df['Support'] = df['Low'].rolling(window=sr_window).min().shift(1)
    df['Resistance'] = df['High'].rolling(window=sr_window).max().shift(1)
   
    return df

# ================================
# 3. Backtest the strategy
# ================================
def backtest_strategy(ticker, ma_window, rsi_window, sr_window, rsi_lower, rsi_upper, years):
    """Run backtest simulation - Simple RSI + MA strategy"""
    df = fetch_stock_data(ticker, years)
    df = add_indicators(df, ma_window, rsi_window, sr_window)
   
    # Drop NaN rows from indicators
    df = df.dropna()
   
    if len(df) == 0:
        raise ValueError("No data left after calculating indicators")
   
    print(f"Data range: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    print(f"RSI range: {df['RSI'].min():.2f} to {df['RSI'].max():.2f}")
    
    # Initialize backtest variables
    initial_capital = 10000.0
    capital = initial_capital
    shares = 0.0
    in_position = False
    buy_price = 0
    buy_date = None
   
    equity_curve = []
    buy_signals = []
    sell_signals = []
    trades = []
   
    # Buy & Hold benchmark
    shares_bh = initial_capital / df['Close'].iloc[0]
    bh_curve = []
   
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]
        rsi_val = df['RSI'].iloc[i]
        ma_val = df['MA'].iloc[i]
       
        # Current equity
        current_equity = (shares * price) if in_position else capital
        equity_curve.append(current_equity)
        bh_curve.append(shares_bh * price)
       
        # BUY SIGNAL: RSI < lower threshold AND price > MA
        if not in_position and rsi_val < rsi_lower and price > ma_val:
            shares = capital / price
            buy_price = price
            buy_date = date
            capital = 0.0
            in_position = True
            buy_signals.append((date, price))
            print(f"BUY at {date.date()}: Price=${price:.2f}, RSI={rsi_val:.2f}")
       
        # SELL SIGNAL: RSI > upper threshold OR price < MA
        elif in_position and (rsi_val > rsi_upper or price < ma_val):
            capital = shares * price
            sell_price = price
            sell_date = date
           
            # Record trade
            profit_loss = (sell_price - buy_price) / buy_price
            trades.append({
                'buy_date': buy_date,
                'sell_date': sell_date,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'return': profit_loss
            })
           
            shares = 0.0
            in_position = False
            sell_signals.append((date, price))
            print(f"SELL at {date.date()}: Price=${price:.2f}, RSI={rsi_val:.2f}, Return={profit_loss*100:.2f}%")
   
    # Close final position if still open
    if in_position:
        final_price = df['Close'].iloc[-1]
        capital = shares * final_price
        profit_loss = (final_price - buy_price) / buy_price
        trades.append({
            'buy_date': buy_date,
            'sell_date': df.index[-1],
            'buy_price': buy_price,
            'sell_price': final_price,
            'return': profit_loss
        })
        sell_signals.append((df.index[-1], final_price))
        print(f"FINAL SELL at {df.index[-1].date()}: Price=${final_price:.2f}, Return={profit_loss*100:.2f}%")
   
    equity_series = pd.Series(equity_curve, index=df.index)
    bh_series = pd.Series(bh_curve, index=df.index)
   
    return {
        "data": df,
        "equity_curve": equity_series,
        "buy_hold_curve": bh_series,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "trades": trades,
        "initial_capital": initial_capital
    }

# ================================
# 4. Calculate performance metrics
# ================================
def calculate_metrics(result):
    """Calculate performance statistics"""
    equity = result["equity_curve"]
    bh = result["buy_hold_curve"]
    trades = result["trades"]
    initial = result["initial_capital"]
    data = result["data"]
   
    # Calculate number of years
    days = (data.index[-1] - data.index[0]).days
    years = days / 365.25
   
    # Total Return
    final_value = equity.iloc[-1]
    total_return = (final_value - initial) / initial * 100
   
    # Buy & Hold Return
    bh_final = bh.iloc[-1]
    bh_return = (bh_final - initial) / initial * 100
   
    # Annualized Return
    if years > 0:
        annualized_return = ((final_value / initial) ** (1 / years) - 1) * 100
        bh_annualized = ((bh_final / initial) ** (1 / years) - 1) * 100
    else:
        annualized_return = 0
        bh_annualized = 0
   
    # Win Rate & Average Trade Return
    if len(trades) > 0:
        winning_trades = sum(1 for t in trades if t['return'] > 0)
        win_rate = (winning_trades / len(trades)) * 100
        avg_trade_return = (sum(t['return'] for t in trades) / len(trades)) * 100
    else:
        win_rate = 0
        avg_trade_return = 0
   
    # Maximum Drawdown
    cumulative = equity.cummax()
    drawdown = (equity - cumulative) / cumulative * 100
    max_drawdown = abs(drawdown.min())
   
    # Sharpe Ratio (simplified, assuming daily returns)
    returns = equity.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0
   
    return {
        'total_return': round(total_return, 2),
        'bh_return': round(bh_return, 2),
        'annualized_return': round(annualized_return, 2),
        'bh_annualized': round(bh_annualized, 2),
        'num_trades': len(trades),
        'win_rate': round(win_rate, 2),
        'avg_trade_return': round(avg_trade_return, 2),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe': round(sharpe, 2),
        'final_value': round(final_value, 2)
    }

# ================================
# 5. Save results to files
# ================================
def save_results(result, ticker, ma_window, rsi_window, sr_window, rsi_lower, rsi_upper, save_dir="backtest_results"):
    """Save backtest results to files"""
   
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{ticker}_{timestamp}"
   
    metrics = calculate_metrics(result)
    txt_path = os.path.join(save_dir, f"{base_filename}_metrics.txt")
   
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"BACKTEST RESULTS FOR {ticker}\n")
        f.write("="*70 + "\n\n")
       
        f.write("PARAMETERS USED:\n")
        f.write(f"  MA Window: {ma_window}\n")
        f.write(f"  RSI Window: {rsi_window}\n")
        f.write(f"  S/R Window: {sr_window}\n")
        f.write(f"  RSI Lower: {rsi_lower}\n")
        f.write(f"  RSI Upper: {rsi_upper}\n\n")
       
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Total Return: {metrics['total_return']:.2f}%\n")
        f.write(f"  Buy & Hold Return: {metrics['bh_return']:.2f}%\n")
        f.write(f"  Annualized Return: {metrics['annualized_return']:.2f}% (Buy & Hold: {metrics['bh_annualized']:.2f}%)\n")
        f.write(f"  Win Rate: {metrics['win_rate']:.2f}%\n")
        f.write(f"  Average Trade Return: {metrics['avg_trade_return']:.2f}%\n")
        f.write(f"  Maximum Drawdown: {metrics['max_drawdown']:.2f}%\n")
        f.write(f"  Number of Trades: {metrics['num_trades']}\n")
        f.write(f"  Sharpe Ratio: {metrics['sharpe']:.2f}\n")
        f.write(f"  Final Portfolio Value: ${metrics['final_value']:,.2f}\n")
        f.write("\n" + "="*70 + "\n")
       
        f.write("\nTRADE LOG:\n")
        f.write("-"*70 + "\n")
        trades = result["trades"]
        for i, trade in enumerate(trades, 1):
            f.write(f"\nTrade #{i}:\n")
            f.write(f"  Buy Date:  {trade['buy_date'].strftime('%Y-%m-%d')}\n")
            f.write(f"  Buy Price: ${trade['buy_price']:.2f}\n")
            f.write(f"  Sell Date: {trade['sell_date'].strftime('%Y-%m-%d')}\n")
            f.write(f"  Sell Price: ${trade['sell_price']:.2f}\n")
            f.write(f"  Return: {trade['return']*100:.2f}%\n")
   
    csv_path = os.path.join(save_dir, f"{base_filename}_trades.csv")
    trades_df = pd.DataFrame(result["trades"])
    if not trades_df.empty:
        trades_df['return_pct'] = trades_df['return'] * 100
        trades_df.to_csv(csv_path, index=False)
   
    equity_csv_path = os.path.join(save_dir, f"{base_filename}_equity.csv")
    equity_data = pd.DataFrame({
        'Date': result["equity_curve"].index,
        'Strategy_Value': result["equity_curve"].values,
        'BuyHold_Value': result["buy_hold_curve"].values
    })
    equity_data.to_csv(equity_csv_path, index=False)
   
    png_path = os.path.join(save_dir, f"{base_filename}_chart.png")
   
    return {
        'txt_path': txt_path,
        'csv_path': csv_path,
        'equity_csv_path': equity_csv_path,
        'png_path': png_path,
        'timestamp': timestamp
    }

# ================================
# 6. Plot results
# ================================
def plot_results(result, ticker, ma_window, rsi_window, sr_window, rsi_lower, rsi_upper, save_chart=False):
    """Visualize backtest results"""
    df = result["data"]
    equity_curve = result["equity_curve"]
    bh_curve = result["buy_hold_curve"]
    buy_signals = result["buy_signals"]
    sell_signals = result["sell_signals"]
   
    metrics = calculate_metrics(result)
   
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)
   
    # Subplot 1: Price + Signals + Indicators
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df["Close"], label="Close Price", color="blue", linewidth=1.5)
    ax1.plot(df.index, df["MA"], label=f"MA ({ma_window})", color="orange", linewidth=1.5)
    ax1.plot(df.index, df["Support"], label="Support", linestyle="--", color="green", alpha=0.5)
    ax1.plot(df.index, df["Resistance"], label="Resistance", linestyle="--", color="red", alpha=0.5)
   
    if buy_signals:
        b_dates, b_prices = zip(*buy_signals)
        ax1.scatter(b_dates, b_prices, marker="^", color="lime", s=120,
                   label="Buy Signal", zorder=5, edgecolors='darkgreen', linewidths=1.5)
   
    if sell_signals:
        s_dates, s_prices = zip(*sell_signals)
        ax1.scatter(s_dates, s_prices, marker="v", color="red", s=120,
                   label="Sell Signal", zorder=5, edgecolors='darkred', linewidths=1.5)
   
    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.set_title(f"{ticker} - Technical Analysis Backtest", fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
   
    # Subplot 2: RSI
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df["RSI"], label="RSI", color="purple", linewidth=1.5)
    ax2.axhline(rsi_lower, linestyle="--", color="green", label=f"Oversold ({rsi_lower})", alpha=0.7)
    ax2.axhline(rsi_upper, linestyle="--", color="red", label=f"Overbought ({rsi_upper})", alpha=0.7)
    ax2.axhline(50, linestyle=":", color="gray", alpha=0.5)
    ax2.fill_between(df.index, 0, 100, where=(df["RSI"] < rsi_lower), alpha=0.2, color='green')
    ax2.fill_between(df.index, 0, 100, where=(df["RSI"] > rsi_upper), alpha=0.2, color='red')
    ax2.set_ylabel("RSI", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
   
    # Subplot 3: Equity Curve
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(equity_curve.index, equity_curve, label="Strategy", color="blue", linewidth=2)
    ax3.plot(bh_curve.index, bh_curve, label="Buy & Hold", color="gray",
            linewidth=2, linestyle='--', alpha=0.7)
    ax3.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
   
    plt.subplots_adjust(hspace=0.35)
   
    if save_chart:
        saved_files = save_results(result, ticker, ma_window, rsi_window, sr_window, rsi_lower, rsi_upper)
        plt.savefig(saved_files['png_path'], dpi=300, bbox_inches='tight')
        print(f"\nResults saved:")
        print(f"   Metrics: {saved_files['txt_path']}")
        print(f"   Trades CSV: {saved_files['csv_path']}")
        print(f"   Equity CSV: {saved_files['equity_csv_path']}")
        print(f"   Chart: {saved_files['png_path']}")
   
    plt.show()
   
    m = metrics
   
    print("\n" + "="*60)
    print(f"Backtest Results for {ticker}:")
    print("="*60)
    print(f"Total Return: {m['total_return']:>6.2f}%")
    print(f"Buy & Hold Return: {m['bh_return']:>6.2f}%")
    print(f"Annualized Return: {m['annualized_return']:>6.2f}%  (Buy & Hold: {m['bh_annualized']:.2f}%)")
    print(f"Win Rate: {m['win_rate']:>6.2f}%")
    print(f"Average Trade Return: {m['avg_trade_return']:>6.2f}%")
    print(f"Maximum Drawdown: {m['max_drawdown']:>6.2f}%")
    print(f"Number of Trades: {m['num_trades']}")
    print(f"Sharpe Ratio: {m['sharpe']:>6.2f}")
    print(f"Final Portfolio Value: ${m['final_value']:,.2f}")
    print("="*60)

# ================================
# 7. Interactive dashboard
# ================================
def run_dashboard():
    """Launch interactive parameter tuning dashboard"""
    ticker_widget = widgets.Text(value='AAPL', description='Ticker:')
    years_widget = widgets.IntSlider(value=5, min=1, max=20, description='Years:')
    ma_widget = widgets.IntSlider(value=15, min=5, max=50, step=1, description='MA Window:')
    rsi_widget = widgets.IntSlider(value=8, min=2, max=20, step=1, description='RSI Window:')
    sr_widget = widgets.IntSlider(value=10, min=5, max=30, step=1, description='S/R Window:')
    rsi_lower_widget = widgets.IntSlider(value=35, min=15, max=45, description='RSI Lower:')
    rsi_upper_widget = widgets.IntSlider(value=65, min=55, max=85, description='RSI Upper:')
    save_checkbox = widgets.Checkbox(value=False, description='Save Results')
   
    output_area = widgets.Output()
   
    def _update(change):
        with output_area:
            clear_output(wait=True)
            try:
                ticker = ticker_widget.value.upper()
                years = years_widget.value
                ma_window = ma_widget.value
                rsi_window = rsi_widget.value
                sr_window = sr_widget.value
                rsi_lower = rsi_lower_widget.value
                rsi_upper = rsi_upper_widget.value
                save_results_flag = save_checkbox.value
               
                print(f"Running backtest for {ticker}...\n")
                result = backtest_strategy(ticker, ma_window, rsi_window, sr_window,
                                         rsi_lower, rsi_upper, years)
                plot_results(result, ticker, ma_window, rsi_window, sr_window,
                            rsi_lower, rsi_upper, save_chart=save_results_flag)
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
   
    ticker_widget.observe(_update, names='value')
    years_widget.observe(_update, names='value')
    ma_widget.observe(_update, names='value')
    rsi_widget.observe(_update, names='value')
    sr_widget.observe(_update, names='value')
    rsi_lower_widget.observe(_update, names='value')
    rsi_upper_widget.observe(_update, names='value')
    save_checkbox.observe(_update, names='value')
   
    ui = widgets.VBox([
        widgets.HTML("<h3>Stock Backtesting Dashboard</h3>"),
        widgets.HBox([ticker_widget, years_widget]),
        widgets.HBox([ma_widget, rsi_widget, sr_widget]),
        widgets.HBox([rsi_lower_widget, rsi_upper_widget]),
        save_checkbox,
        output_area
    ])
   
    display(ui)
   
    _update(None)

print("System ready! Launching dashboard...")
run_dashboard()
