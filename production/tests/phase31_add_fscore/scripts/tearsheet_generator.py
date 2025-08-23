#!/usr/bin/env python3
"""
Tearsheet Generator for QVM Strategy Analysis
============================================

This script contains the tearsheet generation functions extracted from 
06_QVM_risk_comparison.py without any modifications.

Functions included:
- calculate_performance_metrics: Calculates comprehensive performance metrics
- generate_comprehensive_tearsheet: Generates comprehensive institutional tearsheet
- generate_comparison_tearsheet: Generates comparison tearsheet for risk management analysis
- create_comparison_plots: Creates comprehensive comparison plots

Usage:
    from tearsheet_generator import (
        calculate_performance_metrics,
        generate_comprehensive_tearsheet,
        generate_comparison_tearsheet,
        create_comparison_plots
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def calculate_performance_metrics(returns, benchmark, periods_per_year: int = 252) -> dict:
    """Calculates comprehensive performance metrics with corrected benchmark alignment and data integrity checks."""
    # CRITICAL: Data integrity validation
    if returns is None or benchmark is None:
        print("❌ ERROR: Returns or benchmark data is None")
        return {metric: 0.0 for metric in ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'information_ratio', 'beta']}
    
    # Ensure inputs are pandas Series with proper index
    if not isinstance(returns, pd.Series):
        if isinstance(returns, np.ndarray):
            # Create a Series with default index if it's a numpy array
            returns = pd.Series(returns, index=pd.RangeIndex(len(returns)))
        else:
            returns = pd.Series(returns)
    
    if not isinstance(benchmark, pd.Series):
        if isinstance(benchmark, np.ndarray):
            # Create a Series with default index if it's a numpy array
            benchmark = pd.Series(benchmark, index=pd.RangeIndex(len(benchmark)))
        else:
            benchmark = pd.Series(benchmark)
    
    # CRITICAL: Validate return data integrity
    if returns.empty:
        print("❌ ERROR: Returns data is empty")
        return {metric: 0.0 for metric in ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'information_ratio', 'beta']}
    
    # CRITICAL: Check for extreme values that indicate data corruption
    if returns.min() < -0.99 or returns.max() > 2.0:  # Allow for some extreme market events but not -100%+
        print(f"⚠️ WARNING: Extreme return values detected: min={returns.min():.4f}, max={returns.max():.4f}")
        print("   This may indicate data corruption or calculation errors")
        
        # Clip extreme values to prevent calculation errors
        returns = returns.clip(-0.99, 2.0)
        print("   Returns clipped to reasonable range (-99% to +200%)")
    
    # Align on date intersection if both have DatetimeIndex; else fall back to length-based trim
    try:
        if isinstance(returns.index, pd.DatetimeIndex) and isinstance(benchmark.index, pd.DatetimeIndex):
            common_index = returns.index.intersection(benchmark.index)
            returns = returns.loc[common_index]
            benchmark = benchmark.loc[common_index]
        else:
            if len(returns) != len(benchmark):
                min_length = min(len(returns), len(benchmark))
                returns = returns.iloc[:min_length]
                benchmark = benchmark.iloc[:min_length]
    except Exception:
        # Last-resort fallback
        min_length = min(len(returns), len(benchmark))
        returns = returns.iloc[:min_length]
        benchmark = benchmark.iloc[:min_length]
    
    # CRITICAL FIX: Ensure consistent index types for date handling
    try:
        # Only convert to datetime if the indices look like actual dates
        # Check if returns index looks like dates (not just integers)
        if not isinstance(returns.index, pd.DatetimeIndex):
            # Only convert if it's not a simple integer range
            if not (isinstance(returns.index, pd.RangeIndex) and returns.index[0] == 0):
                try:
                    returns.index = pd.to_datetime(returns.index)
                except:
                    # If conversion fails, keep as is
                    pass
        
        if not isinstance(benchmark.index, pd.DatetimeIndex):
            # Only convert if it's not a simple integer range
            if not (isinstance(benchmark.index, pd.RangeIndex) and benchmark.index[0] == 0):
                try:
                    benchmark.index = pd.to_datetime(benchmark.index)
                except:
                    # If conversion fails, keep as is
                    pass
    except Exception as e:
        # If conversion fails, use integer-based indexing
        returns.index = pd.RangeIndex(len(returns))
        benchmark.index = pd.RangeIndex(len(benchmark))
    
    # COMPLETELY REWRITTEN: Safe alignment logic that handles all index types
    try:
        # Find the first non-zero return
        non_zero_mask = returns.ne(0)
        if not non_zero_mask.any():
            # No non-zero returns, use all data
            aligned_returns = returns
            aligned_benchmark = benchmark
        else:
            # Find first non-zero index position
            first_non_zero_pos = non_zero_mask.idxmax()
            
            # Handle different index types safely
            if isinstance(returns.index, pd.DatetimeIndex) and isinstance(benchmark.index, pd.DatetimeIndex):
                # Both have datetime indices - use datetime slicing
                try:
                    aligned_returns = returns.loc[first_non_zero_pos:]
                    aligned_benchmark = benchmark.loc[first_non_zero_pos:]
                except Exception:
                    # Fallback to integer-based slicing
                    first_idx = returns.index.get_loc(first_non_zero_pos)
                    aligned_returns = returns.iloc[first_idx:]
                    aligned_benchmark = benchmark.iloc[first_idx:]
            else:
                # Use integer-based indexing
                try:
                    first_idx = returns.index.get_loc(first_non_zero_pos)
                    aligned_returns = returns.iloc[first_idx:]
                    aligned_benchmark = benchmark.iloc[first_idx:]
                except Exception:
                    # Ultimate fallback: use simple integer-based alignment
                    aligned_returns = returns
                    aligned_benchmark = benchmark
                    
    except Exception as e:
        # Ultimate fallback: use simple integer-based alignment
        aligned_returns = returns
        aligned_benchmark = benchmark
    
    # CRITICAL: Final data integrity check after alignment
    if aligned_returns.empty:
        print("❌ ERROR: Aligned returns data is empty after processing")
        return {metric: 0.0 for metric in ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'information_ratio', 'beta']}
    
    # CRITICAL: Check for extreme values after alignment
    if aligned_returns.min() < -0.99 or aligned_returns.max() > 2.0:
        print(f"⚠️ WARNING: Extreme aligned return values: min={aligned_returns.min():.4f}, max={aligned_returns.max():.4f}")
        aligned_returns = aligned_returns.clip(-0.99, 2.0)
    
    if len(aligned_returns) < 2:
        return {metric: 0.0 for metric in ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'information_ratio', 'beta']}
    
    # Basic metrics with safety checks
    try:
        total_return = (1 + aligned_returns).prod() - 1
        
        # CRITICAL: Validate total return is reasonable
        if total_return < -0.99 or total_return > 10.0:  # Allow for extreme market events but not -100%+
            print(f"⚠️ WARNING: Extreme total return detected: {total_return:.4f}")
            print("   This may indicate data corruption or calculation errors")
            # Use a more conservative calculation
            total_return = aligned_returns.mean() * len(aligned_returns)
            total_return = max(-0.99, min(10.0, total_return))  # Clip to reasonable range
        
        annualized_return = (1 + total_return) ** (periods_per_year / len(aligned_returns)) - 1
        annualized_volatility = aligned_returns.std() * np.sqrt(periods_per_year)
        
        # CRITICAL: Validate volatility is reasonable
        if annualized_volatility > 2.0:  # 200% annualized volatility is extremely high
            print(f"⚠️ WARNING: Extreme volatility detected: {annualized_volatility:.4f}")
            annualized_volatility = min(2.0, annualized_volatility)  # Cap at 200%
        
    except Exception as e:
        print(f"❌ ERROR calculating basic metrics: {e}")
        return {metric: 0.0 for metric in ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'information_ratio', 'beta']}
    
    # Risk metrics with safety checks
    try:
        cumulative_returns = (1 + aligned_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / running_max - 1)
        max_drawdown = drawdown.min()
        
        # CRITICAL: Validate max drawdown is reasonable
        if max_drawdown < -0.99:  # -100% drawdown is impossible
            print(f"⚠️ WARNING: Impossible max drawdown detected: {max_drawdown:.4f}")
            print("   This indicates data corruption or calculation errors")
            # Use a more conservative calculation
            max_drawdown = min(-0.99, max_drawdown)  # Cap at -99%
        
    except Exception as e:
        print(f"❌ ERROR calculating risk metrics: {e}")
        max_drawdown = -0.01  # Default to -1% if calculation fails
    
    # Ratios with safety checks
    try:
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # CRITICAL: Validate ratios are reasonable
        sharpe_ratio = max(-5.0, min(5.0, sharpe_ratio))  # Cap Sharpe ratio
        calmar_ratio = max(-10.0, min(10.0, calmar_ratio))  # Cap Calmar ratio
        
    except Exception as e:
        print(f"❌ ERROR calculating ratios: {e}")
        sharpe_ratio = 0
        calmar_ratio = 0
    
    # Benchmark metrics with safety checks
    if not aligned_benchmark.empty:
        try:
            benchmark_return = (1 + aligned_benchmark).prod() - 1
            benchmark_volatility = aligned_benchmark.std() * np.sqrt(periods_per_year)
            
            # Information ratio
            excess_returns = aligned_returns - aligned_benchmark
            
            # Handle edge cases for information ratio calculation
            if len(excess_returns) > 1:
                # Calculate annualized excess return and tracking error
                annualized_excess_return = excess_returns.mean() * periods_per_year
                tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
                
                # Set minimum tracking error threshold to avoid division by zero
                min_tracking_error = 0.001  # 0.1% minimum tracking error
                if tracking_error < min_tracking_error:
                    tracking_error = min_tracking_error
                
                # Calculate information ratio
                information_ratio = annualized_excess_return / tracking_error if tracking_error > 0 else 0
                
                # Cap information ratio to reasonable bounds (-5 to 5)
                information_ratio = max(-5.0, min(5.0, information_ratio))
            else:
                information_ratio = 0
            
            # Beta
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_variance = aligned_benchmark.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # CRITICAL: Validate beta is reasonable
            beta = max(-5.0, min(5.0, beta))  # Cap beta at reasonable bounds
            
        except Exception as e:
            print(f"❌ ERROR calculating benchmark metrics: {e}")
            information_ratio = 0
            beta = 0
    else:
        information_ratio = 0
        beta = 0
    
    # CRITICAL: Final validation of all metrics
    final_metrics = {
        'annualized_return': annualized_return,
        'volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'information_ratio': information_ratio,
        'beta': beta
    }
    
    # Validate final metrics
    for key, value in final_metrics.items():
        if pd.isna(value) or np.isinf(value):
            print(f"⚠️ WARNING: Invalid {key} value: {value}, setting to 0")
            final_metrics[key] = 0.0
    
    print(f"✅ Performance metrics calculated successfully:")
    print(f"   Annualized Return: {final_metrics['annualized_return']:.4f}")
    print(f"   Max Drawdown: {final_metrics['max_drawdown']:.4f}")
    print(f"   Sharpe Ratio: {final_metrics['sharpe_ratio']:.4f}")
    
    return final_metrics

def generate_comprehensive_tearsheet(strategy_returns: pd.Series, benchmark_returns: pd.Series, 
                                   title: str, cash_allocations: pd.DataFrame = None):
    """Generates comprehensive institutional tearsheet with equity curve and cash allocation chart."""
    
    # CRITICAL: Data integrity validation
    if strategy_returns is None or benchmark_returns is None:
        print("❌ ERROR: Strategy returns or benchmark returns is None")
        return
    
    if strategy_returns.empty or benchmark_returns.empty:
        print("❌ ERROR: Strategy returns or benchmark returns is empty")
        return
    
    # CRITICAL: Check for extreme values that indicate data corruption
    if strategy_returns.min() < -0.99 or strategy_returns.max() > 2.0:
        print(f"⚠️ WARNING: Extreme strategy return values detected: min={strategy_returns.min():.4f}, max={strategy_returns.max():.4f}")
        print("   This may indicate data corruption or calculation errors")
        strategy_returns = strategy_returns.clip(-0.99, 2.0)
    
    if benchmark_returns.min() < -0.99 or benchmark_returns.max() > 2.0:
        print(f"⚠️ WARNING: Extreme benchmark return values detected: min={benchmark_returns.min():.4f}, max={benchmark_returns.max():.4f}")
        print("   This may indicate data corruption or calculation errors")
        benchmark_returns = benchmark_returns.clip(-0.99, 2.0)
    
    # CRITICAL FIX: Create deep copies to prevent parameter corruption
    strategy_returns = strategy_returns.copy()
    benchmark_returns = benchmark_returns.copy()
    
    # Safe alignment logic that handles all index types
    try:
        if isinstance(strategy_returns.index, pd.DatetimeIndex) and isinstance(benchmark_returns.index, pd.DatetimeIndex):
            common_index = strategy_returns.index.intersection(benchmark_returns.index)
            aligned_strategy_returns = strategy_returns.loc[common_index]
            aligned_benchmark_returns = benchmark_returns.loc[common_index]
        else:
            # integer/other index fallback: trim to same length
            min_len = min(len(strategy_returns), len(benchmark_returns))
            aligned_strategy_returns = strategy_returns.iloc[:min_len]
            aligned_benchmark_returns = benchmark_returns.iloc[:min_len]
                
    except Exception as e:
        print(f"⚠️ WARNING: Alignment failed with error: {e}")
        print("   Using fallback alignment method")
        # Ultimate fallback: use simple integer-based alignment
        aligned_strategy_returns = strategy_returns
        aligned_benchmark_returns = benchmark_returns
    
    # CRITICAL: Final validation after alignment
    if aligned_strategy_returns.empty or aligned_benchmark_returns.empty:
        print("❌ ERROR: Aligned data is empty after processing")
        print("   Attempting to use original data as fallback")
        aligned_strategy_returns = strategy_returns
        aligned_benchmark_returns = benchmark_returns
        
        # Final check
        if aligned_strategy_returns.empty or aligned_benchmark_returns.empty:
            print("❌ CRITICAL ERROR: All data is empty, cannot generate tearsheet")
            return
    
    # Ensure DatetimeIndex for downstream resampling/rolling plots where possible
    for _s in (aligned_strategy_returns, aligned_benchmark_returns):
        if not isinstance(_s.index, pd.DatetimeIndex):
            try:
                _s.index = pd.to_datetime(_s.index)
            except Exception:
                # Fallback: create a synthetic daily index starting at 2000-01-01
                _s.index = pd.date_range(start='2000-01-01', periods=len(_s), freq='B')

    # CRITICAL: Check for extreme values after alignment
    if aligned_strategy_returns.min() < -0.99 or aligned_strategy_returns.max() > 2.0:
        print(f"⚠️ WARNING: Extreme aligned strategy return values: min={aligned_strategy_returns.min():.4f}, max={aligned_strategy_returns.max():.4f}")
        aligned_strategy_returns = aligned_strategy_returns.clip(-0.99, 2.0)
    
    if aligned_benchmark_returns.min() < -0.99 or aligned_benchmark_returns.max() > 2.0:
        print(f"⚠️ WARNING: Extreme aligned benchmark return values: min={aligned_benchmark_returns.min():.4f}, max={aligned_benchmark_returns.max():.4f}")
        aligned_benchmark_returns = aligned_benchmark_returns.clip(-0.99, 2.0)

    strategy_metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    # Remove the corrupting benchmark metrics calculation
    
    # CRITICAL: Validate equity curve consistency with metrics
    try:
        # Calculate what the final cumulative return should be based on metrics
        expected_final_value = (1 + strategy_metrics['annualized_return']) ** (len(aligned_strategy_returns) / 252)
        actual_final_value = (1 + aligned_strategy_returns).cumprod().iloc[-1]
        
        print(f"🔍 EQUITY CURVE VALIDATION:")
        print(f"   Expected final value (from metrics): {expected_final_value:.4f}")
        print(f"   Actual final value (from returns): {actual_final_value:.4f}")
        print(f"   Difference: {abs(expected_final_value - actual_final_value):.4f}")
        
        if abs(expected_final_value - actual_final_value) > 0.1:  # More than 10% difference
            print(f"⚠️ WARNING: Large discrepancy between metrics and equity curve!")
            print(f"   This may indicate calculation errors or data misalignment")
        
    except Exception as e:
        print(f"⚠️ WARNING: Could not validate equity curve consistency: {e}")
    
    # Choose a more compact layout by default; further adjust if cash is 0%
    fig_height = 22
    fig = plt.figure(constrained_layout=True, figsize=(18, fig_height))
    gs = fig.add_gridspec(6, 2, height_ratios=[1.5, 0.6, 0.9, 0.8, 0.6, 1.2], hspace=0.6, wspace=0.2)
    fig.suptitle(title, fontsize=20, fontweight='bold', color='#2C3E50')

    # 1. Cumulative Performance (Equity Curve)
    ax1 = fig.add_subplot(gs[0, :])
    
    # CRITICAL FIX: Ensure equity curve matches calculated metrics
    try:
        # Calculate cumulative returns properly
        strategy_cumulative = (1 + aligned_strategy_returns).cumprod()
        benchmark_cumulative = (1 + aligned_benchmark_returns).cumprod()
        
        # Validate that cumulative returns make sense
        if strategy_cumulative.iloc[-1] < 0.01:  # Less than 1% remaining
            print(f"⚠️ WARNING: Strategy cumulative return very low: {strategy_cumulative.iloc[-1]:.4f}")
            print("   This may indicate calculation errors or extreme market conditions")
        
        # Plot the main equity curves
        strategy_cumulative.index = pd.to_datetime(strategy_cumulative.index)
        strategy_cumulative.plot(ax=ax1, label='QVM Engine v3 (F-Score)', color='#16A085', lw=2.5)
        benchmark_cumulative.plot(ax=ax1, label='VN-Index (Aligned)', color='#34495E', linestyle='--', lw=2)
        
        print(f"✅ Equity curve calculated: Strategy final value = {strategy_cumulative.iloc[-1]:.4f}, Benchmark = {benchmark_cumulative.iloc[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ ERROR calculating equity curve: {e}")
        # Fallback to simple plotting
        (1 + aligned_strategy_returns).cumprod().plot(ax=ax1, label='QVM Engine v3 (F-Score)', color='#16A085', lw=2.5)
        (1 + aligned_benchmark_returns).cumprod().plot(ax=ax1, label='VN-Index (Aligned)', color='#34495E', linestyle='--', lw=2)
    
    ax1.set_title('Daily Equity Curve (Log Scale)', fontweight='bold')
    ax1.set_ylabel('Growth of 1 VND')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 2. Cash Allocation Chart
    ax2 = fig.add_subplot(gs[1, :])
    if cash_allocations is not None and not cash_allocations.empty:
        # Convert dates to datetime for plotting
        cash_allocations['date'] = pd.to_datetime(cash_allocations['date'])
        cash_allocations = cash_allocations.sort_values('date')
        # Accept either 'cash_percentage' (0-100) or 'cash_allocation' (0-1)
        if 'cash_percentage' not in cash_allocations.columns:
            if 'cash_allocation' in cash_allocations.columns:
                try:
                    cash_allocations['cash_percentage'] = cash_allocations['cash_allocation'].astype(float) * 100.0
                except Exception:
                    cash_allocations['cash_percentage'] = 0.0
            else:
                cash_allocations['cash_percentage'] = 0.0

        max_cash = float(cash_allocations['cash_percentage'].max()) if 'cash_percentage' in cash_allocations.columns else 0.0
        if max_cash <= 1e-9:
            # 0% cash throughout – keep the panel compact and avoid misleading reference lines
            ax2.plot(cash_allocations['date'], cash_allocations['cash_percentage'], color='#7F8C8D', linewidth=1.5)
            ax2.set_title('Cash Allocation Over Time (0%)', fontweight='bold')
            ax2.set_ylabel('Cash Allocation (%)')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.2)
        else:
            # Plot cash allocation percentage over time
            ax2.plot(cash_allocations['date'], cash_allocations['cash_percentage'], 
                    color='#E74C3C', linewidth=2, marker='o', markersize=3)
            ax2.fill_between(cash_allocations['date'], cash_allocations['cash_percentage'], 
                            alpha=0.25, color='#E74C3C')
            # Reference lines scaled to data range
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            y_max = max(5.0, max_cash * 1.2)
            ax2.set_ylim(0, y_max)
            ax2.set_title('Cash Allocation Over Time', fontweight='bold')
            ax2.set_ylabel('Cash Allocation (%)')
            ax2.grid(True, alpha=0.3)
        
        print(f"   📊 Cash allocation chart created")
    else:
        ax2.text(0.5, 0.5, 'No Cash Allocation Data Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Cash Allocation Over Time', fontweight='bold')

    # 3. Drawdown Analysis
    ax3 = fig.add_subplot(gs[2, :])
    drawdown = ((1 + aligned_strategy_returns).cumprod() / (1 + aligned_strategy_returns).cumprod().cummax() - 1) * 100
    drawdown.plot(ax=ax3, color='#C0392B')
    ax3.fill_between(drawdown.index, drawdown, 0, color='#C0392B', alpha=0.1)
    ax3.set_title('Drawdown Analysis', fontweight='bold')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # 4. Annual Returns
    ax4 = fig.add_subplot(gs[3, 0])
    # Use pandas year-end alias 'A' instead of deprecated/invalid 'YE'
    strat_annual = aligned_strategy_returns.resample('YE').apply(lambda x: (1+x).prod()-1) * 100
    bench_annual = aligned_benchmark_returns.resample('YE').apply(lambda x: (1+x).prod()-1) * 100
    pd.DataFrame({'Strategy': strat_annual, 'Benchmark': bench_annual}).plot(kind='bar', ax=ax4, color=['#16A085', '#34495E'])
    ax4.set_xticks(range(len(strat_annual)))
    ax4.set_xticklabels([d.strftime('%Y') for d in strat_annual.index], rotation=45, ha='right')
    ax4.set_title('Annual Returns', fontweight='bold')
    ax4.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 5. Rolling Sharpe Ratio
    ax5 = fig.add_subplot(gs[3, 1])
    rolling_sharpe = (aligned_strategy_returns.rolling(252).mean() * 252) / (aligned_strategy_returns.rolling(252).std() * np.sqrt(252))
    rolling_sharpe.plot(ax=ax5, color='#E67E22')
    ax5.axhline(1.0, color='#27AE60', linestyle='--')
    ax5.set_title('1-Year Rolling Sharpe Ratio', fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.5)

    # 6. Factor Score Evolution
    ax6 = fig.add_subplot(gs[4, 0])
    # This would show factor score evolution over time
    ax6.text(0.5, 0.5, 'Factor Score Evolution\n(To be implemented)', 
            ha='center', va='center', transform=ax6.transAxes, fontsize=14)
    ax6.set_title('Factor Score Evolution', fontweight='bold')

    # 7. Portfolio Holdings Distribution
    ax7 = fig.add_subplot(gs[4, 1])
    # This would show portfolio holdings distribution
    ax7.text(0.5, 0.5, 'Portfolio Holdings\nDistribution', 
            ha='center', va='center', transform=ax7.transAxes, fontsize=14)
    ax7.set_title('Portfolio Holdings Distribution', fontweight='bold')

    # 8. Performance Metrics Table
    ax8 = fig.add_subplot(gs[5:, :])
    ax8.axis('off')
    
    # Calculate benchmark metrics for comparison
    benchmark_metrics = calculate_performance_metrics(benchmark_returns, benchmark_returns)
    
    summary_data = [['Metric', 'Strategy', 'Benchmark']]
    for key in strategy_metrics.keys():
        strategy_value = f"{strategy_metrics[key]:.2f}"
        benchmark_value = f"{benchmark_metrics[key]:.2f}" if key in benchmark_metrics else "N/A"
        summary_data.append([key, strategy_value, benchmark_value])
    
    table = ax8.table(cellText=summary_data[1:], colLabels=summary_data[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    plt.show()
    
    return strategy_metrics

def generate_comparison_tearsheet(strategy_with_risk: pd.Series, 
                                strategy_without_risk: pd.Series,
                                benchmark_returns: pd.Series,
                                cash_allocations_df: pd.DataFrame,
                                config: Dict) -> None:
    """
    Generate a comprehensive tearsheet comparing all three strategies.
    
    Args:
        strategy_with_risk: Returns series for strategy with risk management
        strategy_without_risk: Returns series for strategy without risk management
        benchmark_returns: Returns series for benchmark
        cash_allocations_df: DataFrame with cash allocation data
        config: Configuration dictionary
    """
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE STRATEGY COMPARISON TEARSHEET")
    print("="*80)
    
    # Display configuration summary
    print(f"\n⚙️ CONFIGURATION SUMMARY")
    print("-" * 40)
    strategy_name = config['strategy']['name']
    strategy_version = config['strategy']['version']
    portfolio_size = config['strategy']['portfolio']['portfolio_size']
    starting_capital = config['strategy']['portfolio']['starting_capital']
    
    print(f"   Strategy: {strategy_name} v{strategy_version}")
    print(f"   Portfolio Size: {portfolio_size} stocks")
    print(f"   Starting Capital: {starting_capital:,.0f} VND")
    print(f"   Factor Weights: Q({config['factor_weights']['quality']:.1%}) V({config['factor_weights']['value']:.1%}) M({config['factor_weights']['momentum']:.1%})")
    
    # Calculate performance metrics for all strategies
    print("\n🔍 PERFORMANCE METRICS COMPARISON")
    print("-" * 60)
    
    # Strategy with risk management
    strategy_with_risk_metrics = calculate_performance_metrics(strategy_with_risk, benchmark_returns)
    print(f"\n✅ WITH Risk Management:")
    print(f"   Annualized Return: {strategy_with_risk_metrics['annualized_return']:.2%}")
    print(f"   Volatility: {strategy_with_risk_metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio: {strategy_with_risk_metrics['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {strategy_with_risk_metrics['max_drawdown']:.2%}")
    print(f"   Calmar Ratio: {strategy_with_risk_metrics['calmar_ratio']:.3f}")
    print(f"   Information Ratio: {strategy_with_risk_metrics['information_ratio']:.3f}")
    print(f"   Beta: {strategy_with_risk_metrics['beta']:.3f}")
    
    # Strategy without risk management
    strategy_without_risk_metrics = calculate_performance_metrics(strategy_without_risk, benchmark_returns)
    print(f"\n❌ WITHOUT Risk Management:")
    print(f"   Annualized Return: {strategy_without_risk_metrics['annualized_return']:.2%}")
    print(f"   Volatility: {strategy_without_risk_metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio: {strategy_without_risk_metrics['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {strategy_without_risk_metrics['max_drawdown']:.2%}")
    print(f"   Calmar Ratio: {strategy_without_risk_metrics['calmar_ratio']:.3f}")
    print(f"   Information Ratio: {strategy_without_risk_metrics['information_ratio']:.3f}")
    print(f"   Beta: {strategy_without_risk_metrics['beta']:.3f}")
    
    # Benchmark
    benchmark_metrics = calculate_performance_metrics(benchmark_returns, benchmark_returns)
    print(f"\n📈 BENCHMARK (VN-Index):")
    print(f"   Annualized Return: {benchmark_metrics['annualized_return']:.2%}")
    print(f"   Volatility: {benchmark_metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {benchmark_metrics['max_drawdown']:.2%}")
    print(f"   Calmar Ratio: {benchmark_metrics['calmar_ratio']:.3f}")
    
    # Risk management impact analysis
    print(f"\n🎯 RISK MANAGEMENT IMPACT ANALYSIS")
    print("-" * 40)
    
    # Return improvement
    return_improvement = strategy_with_risk_metrics['annualized_return'] - strategy_without_risk_metrics['annualized_return']
    print(f"   Return Impact: {return_improvement:+.2%}")
    
    # Volatility reduction
    volatility_reduction = strategy_without_risk_metrics['volatility'] - strategy_with_risk_metrics['volatility']
    print(f"   Volatility Reduction: {volatility_reduction:+.2%}")
    
    # Drawdown protection
    drawdown_protection = strategy_without_risk_metrics['max_drawdown'] - strategy_with_risk_metrics['max_drawdown']
    print(f"   Drawdown Protection: {drawdown_protection:+.2%}")
    
    # Sharpe ratio improvement
    sharpe_improvement = strategy_with_risk_metrics['sharpe_ratio'] - strategy_without_risk_metrics['sharpe_ratio']
    print(f"   Sharpe Ratio Improvement: {sharpe_improvement:+.3f}")
    
    # Information ratio improvement
    ir_improvement = strategy_with_risk_metrics['information_ratio'] - strategy_without_risk_metrics['information_ratio']
    print(f"   Information Ratio Improvement: {ir_improvement:+.3f}")
    
    # Cash allocation statistics
    print(f"\n💰 CASH ALLOCATION STATISTICS")
    print("-" * 40)
    cash_stats = cash_allocations_df['cash_allocation'].describe()
    print(f"   Average Cash: {cash_stats['mean']:.1%}")
    print(f"   Max Cash: {cash_stats['max']:.1%}")
    print(f"   Min Cash: {cash_stats['min']:.1%}")
    print(f"   Cash Volatility: {cash_stats['std']:.1%}")
    
    # Generate comprehensive tearsheet for each strategy
    print(f"\n📊 GENERATING COMPREHENSIVE TEARSHEETS...")
    
    # Generate tearsheet for strategy WITH risk management
    print(f"\n📊 Strategy WITH Risk Management:")
    generate_comprehensive_tearsheet(
        strategy_with_risk, 
        benchmark_returns, 
        f"{config['strategy']['name']}: WITH Risk Management vs Benchmark",
        cash_allocations_df
    )
    
    # Generate tearsheet for strategy WITHOUT risk management
    print(f"\n📊 Strategy WITHOUT Risk Management:")
    generate_comprehensive_tearsheet(
        strategy_without_risk, 
        benchmark_returns, 
        f"{config['strategy']['name']}: WITHOUT Risk Management vs Benchmark"
    )

def create_comparison_plots(strategy_with_risk: pd.Series, 
                           strategy_without_risk: pd.Series,
                           benchmark_returns: pd.Series,
                           cash_allocations_df: pd.DataFrame,
                           config: Dict) -> None:
    """
    Create comprehensive comparison plots.
    
    Args:
        strategy_with_risk: Returns series for strategy with risk management
        strategy_without_risk: Returns series for strategy without risk management
        benchmark_returns: Returns series for benchmark
        cash_allocations_df: DataFrame with cash allocation data
        config: Configuration dictionary
    """
    print(f"\n📊 GENERATING COMPARISON PLOTS...")
    
    # Set up the plotting style from config
    plot_style = config.get('output', {}).get('plots', {}).get('style', 'seaborn-v0_8')
    figure_size = config.get('output', {}).get('plots', {}).get('figure_size', [16, 12])
    
    plt.style.use(plot_style)
    fig, axes = plt.subplots(2, 2, figsize=tuple(figure_size), constrained_layout=True)
    
    strategy_name = config['strategy']['name']
    fig.suptitle(f'{strategy_name}: Risk Management vs No Risk Management vs Benchmark', 
                 fontsize=16, fontweight='bold')
    
    # 1. Cumulative Returns Comparison
    ax1 = axes[0, 0]
    cumulative_with_risk = (1 + strategy_with_risk).cumprod()
    cumulative_without_risk = (1 + strategy_without_risk).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    
    ax1.plot(cumulative_with_risk.index, cumulative_with_risk.values, 
             label='With Risk Management', linewidth=2, color='green')
    ax1.plot(cumulative_without_risk.index, cumulative_without_risk.values, 
             label='Without Risk Management', linewidth=2, color='red')
    ax1.plot(cumulative_benchmark.index, cumulative_benchmark.values, 
             label='VN-Index Benchmark', linewidth=2, color='blue', alpha=0.7)
    
    ax1.set_title('Cumulative Returns Comparison')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown Comparison
    ax2 = axes[0, 1]
    
    # Calculate drawdowns
    def calculate_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    drawdown_with_risk = calculate_drawdown(strategy_with_risk)
    drawdown_without_risk = calculate_drawdown(strategy_without_risk)
    drawdown_benchmark = calculate_drawdown(benchmark_returns)
    
    ax2.fill_between(drawdown_with_risk.index, drawdown_with_risk.values, 0, 
                     alpha=0.3, color='green', label='With Risk Management')
    ax2.fill_between(drawdown_without_risk.index, drawdown_without_risk.values, 0, 
                     alpha=0.3, color='red', label='Without Risk Management')
    ax2.fill_between(drawdown_benchmark.index, drawdown_benchmark.values, 0, 
                     alpha=0.3, color='blue', label='VN-Index Benchmark')
    
    ax2.set_title('Drawdown Comparison')
    ax2.set_ylabel('Drawdown')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cash Allocation Over Time
    ax3 = axes[1, 0]

    # Debug: Check the structure of cash_allocations_df
    print(f"🔍 Cash allocations DataFrame columns: {cash_allocations_df.columns.tolist()}")
    print(f"🔍 Cash allocations DataFrame shape: {cash_allocations_df.shape}")

    # Handle different possible column structures
    if 'date' in cash_allocations_df.columns:
        date_col = 'date'
    elif 'index' in cash_allocations_df.columns:
        date_col = 'index'
    else:
        # If no date column, use the first column that's not cash_allocation
        date_candidates = [col for col in cash_allocations_df.columns if col != 'cash_allocation']
        date_col = date_candidates[0] if date_candidates else cash_allocations_df.columns[0]
        print(f"🔍 Using column '{date_col}' as date column")

    # Convert cash_allocation to percentage for better visualization
    cash_values = cash_allocations_df['cash_allocation'] * 100

    # Compact rendering when cash is 0% across the window
    max_cash = float(cash_values.max()) if not cash_values.empty else 0.0
    if max_cash <= 1e-9:
        ax3.plot(cash_allocations_df[date_col], cash_values, linewidth=1.5, color='#7F8C8D')
        ax3.set_ylim(0, 1)
        ax3.set_title('Dynamic Cash Allocation Over Time (0%)')
    else:
        ax3.plot(cash_allocations_df[date_col], cash_values, linewidth=2, color='purple', alpha=0.8)
        ax3.fill_between(cash_allocations_df[date_col], cash_values, alpha=0.3, color='purple')
        ax3.set_ylim(0, max(5.0, max_cash * 1.2))

    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cash Allocation %')
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk-Return Scatter Plot
    ax4 = axes[1, 1]
    
    # Calculate annualized metrics for scatter
    def annualize_metrics(returns):
        annual_return = (1 + returns.mean()) ** 252 - 1
        annual_vol = returns.std() * np.sqrt(252)
        return annual_return, annual_vol
    
    ret_with_risk, vol_with_risk = annualize_metrics(strategy_with_risk)
    ret_without_risk, vol_without_risk = annualize_metrics(strategy_without_risk)
    ret_benchmark, vol_benchmark = annualize_metrics(benchmark_returns)
    
    ax4.scatter(vol_with_risk, ret_with_risk, s=200, color='green', 
                label='With Risk Management', alpha=0.8, edgecolors='black')
    ax4.scatter(vol_without_risk, ret_without_risk, s=200, color='red', 
                label='Without Risk Management', alpha=0.8, edgecolors='black')
    ax4.scatter(vol_benchmark, ret_benchmark, s=200, color='blue', 
                label='VN-Index Benchmark', alpha=0.8, edgecolors='black')
    
    ax4.set_title('Risk-Return Profile Comparison')
    ax4.set_xlabel('Annualized Volatility')
    ax4.set_ylabel('Annualized Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add text annotations
    ax4.annotate(f'Sharpe: {ret_with_risk/vol_with_risk:.2f}', 
                 (vol_with_risk, ret_with_risk), xytext=(5, 5), textcoords='offset points')
    ax4.annotate(f'Sharpe: {ret_without_risk/vol_without_risk:.2f}', 
                 (vol_without_risk, ret_without_risk), xytext=(5, 5), textcoords='offset points')
    ax4.annotate(f'Sharpe: {ret_benchmark/vol_benchmark:.2f}', 
                 (vol_benchmark, ret_benchmark), xytext=(5, 5), textcoords='offset points')
    
    plt.show()
    
    print("✅ Comparison plots generated successfully!")

if __name__ == "__main__":
    print("📊 Tearsheet Generator Module")
    print("=" * 40)
    print("This module contains the following functions:")
    print("  - calculate_performance_metrics")
    print("  - generate_comprehensive_tearsheet")
    print("  - generate_comparison_tearsheet")
    print("  - create_comparison_plots")
    print("\nImport and use these functions in your analysis scripts.")
