"""
Simple Regime Detection System
Based on Phase 20's volatility and return approach
Validation procedures from Phase 21

Author: Factor Investing Team
Date: July 30, 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import yaml
import pickle
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimpleRegimeDetection:
    """
    Simple regime detection based on volatility and returns
    Mirrors Phase 20's successful approach
    """
    
    def __init__(self, config_path="../../../config/database.yml"):
        """Initialize with database configuration"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        db_config = config['production']
        self.engine = create_engine(
            f"mysql+pymysql://{db_config['username']}:{db_config['password']}@"
            f"{db_config['host']}:3306/{db_config['schema_name']}"
        )
        
        # Regime detection parameters (Phase 20 style)
        self.lookback_period = 60  # 60-day rolling window
        self.vol_threshold_high = 0.75  # 75th percentile for high volatility
        self.return_threshold_bull = 0.10  # 10% annualized return for bull market
        self.return_threshold_bear = -0.10  # -10% annualized return for bear market
        
    def load_benchmark_data(self, start_date='2016-01-01', end_date='2025-07-28'):
        """Load VNINDEX benchmark data"""
        query = """
        SELECT date, close
        FROM etf_history 
        WHERE ticker = 'VNINDEX' 
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """
        
        df = pd.read_sql(query.format(start_date=start_date, end_date=end_date), self.engine)
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        return df
    
    def detect_regimes(self, benchmark_data):
        """
        Detect market regimes based on volatility and returns
        Phase 20 style simple approach
        """
        df = benchmark_data.copy()
        
        # Calculate rolling volatility (annualized)
        df['rolling_vol'] = df['returns'].rolling(self.lookback_period).std() * np.sqrt(252)
        
        # Calculate rolling returns (annualized)
        df['rolling_returns'] = df['returns'].rolling(self.lookback_period).mean() * 252
        
        # Calculate volatility thresholds
        vol_75th = df['rolling_vol'].quantile(self.vol_threshold_high)
        
        # Regime classification (Phase 20 style)
        conditions = [
            (df['rolling_vol'] > vol_75th) & (df['rolling_returns'] < self.return_threshold_bear),
            (df['rolling_vol'] > vol_75th) & (df['rolling_returns'] >= self.return_threshold_bear),
            (df['rolling_vol'] <= vol_75th) & (df['rolling_returns'] >= self.return_threshold_bull),
            (df['rolling_vol'] <= vol_75th) & (df['rolling_returns'] < self.return_threshold_bull)
        ]
        
        choices = ['Stress', 'Bear', 'Bull', 'Sideways']
        df['regime'] = np.select(conditions, choices, default='Sideways')
        
        return df
    
    def calculate_regime_statistics(self, regime_data):
        """Calculate regime statistics and characteristics"""
        stats = {}
        
        # Regime distribution
        regime_counts = regime_data['regime'].value_counts()
        regime_pct = regime_data['regime'].value_counts(normalize=True) * 100
        
        stats['regime_distribution'] = {
            'counts': regime_counts.to_dict(),
            'percentages': regime_pct.to_dict()
        }
        
        # Regime characteristics
        regime_stats = {}
        for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
            regime_mask = regime_data['regime'] == regime
            if regime_mask.sum() > 0:
                regime_data_subset = regime_data[regime_mask]
                
                regime_stats[regime] = {
                    'count': len(regime_data_subset),
                    'avg_return': regime_data_subset['returns'].mean() * 252,
                    'avg_vol': regime_data_subset['rolling_vol'].mean(),
                    'sharpe': (regime_data_subset['returns'].mean() * 252) / 
                             (regime_data_subset['returns'].std() * np.sqrt(252)) if regime_data_subset['returns'].std() > 0 else 0,
                    'max_drawdown': self.calculate_max_drawdown(regime_data_subset['close']),
                    'avg_duration': self.calculate_avg_regime_duration(regime_data, regime)
                }
        
        stats['regime_characteristics'] = regime_stats
        
        return stats
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def calculate_avg_regime_duration(self, regime_data, regime):
        """Calculate average duration of a regime"""
        regime_changes = regime_data['regime'] != regime_data['regime'].shift(1)
        regime_starts = regime_changes & (regime_data['regime'] == regime)
        
        if regime_starts.sum() == 0:
            return 0
        
        durations = []
        start_idx = None
        
        for i, is_start in enumerate(regime_starts):
            if is_start:
                if start_idx is not None:
                    durations.append(i - start_idx)
                start_idx = i
        
        # Handle last regime
        if start_idx is not None:
            durations.append(len(regime_data) - start_idx)
        
        return np.mean(durations) if durations else 0
    
    def validate_regime_detection(self, regime_data):
        """
        Validate regime detection using Phase 21 style procedures
        """
        validation_results = {}
        
        # 1. Regime identification accuracy
        validation_results['regime_accuracy'] = self.calculate_regime_accuracy(regime_data)
        
        # 2. Regime persistence
        validation_results['regime_persistence'] = self.calculate_regime_persistence(regime_data)
        
        # 3. Regime transition probabilities
        validation_results['transition_matrix'] = self.calculate_transition_matrix(regime_data)
        
        # 4. Regime predictability
        validation_results['regime_predictability'] = self.calculate_regime_predictability(regime_data)
        
        # 5. Economic significance
        validation_results['economic_significance'] = self.calculate_economic_significance(regime_data)
        
        return validation_results
    
    def calculate_regime_accuracy(self, regime_data):
        """Calculate regime identification accuracy"""
        # Simple accuracy based on regime consistency
        regime_changes = (regime_data['regime'] != regime_data['regime'].shift(1)).sum()
        total_periods = len(regime_data)
        
        # Higher accuracy means fewer regime changes (more stable regimes)
        accuracy = 1 - (regime_changes / total_periods)
        
        return {
            'accuracy_score': accuracy,
            'regime_changes': regime_changes,
            'total_periods': total_periods,
            'avg_regime_duration': total_periods / (regime_changes + 1)
        }
    
    def calculate_regime_persistence(self, regime_data):
        """Calculate regime persistence metrics"""
        persistence_metrics = {}
        
        for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
            regime_mask = regime_data['regime'] == regime
            if regime_mask.sum() > 0:
                # Calculate consecutive periods in regime
                consecutive_periods = []
                current_count = 0
                
                for is_in_regime in regime_mask:
                    if is_in_regime:
                        current_count += 1
                    else:
                        if current_count > 0:
                            consecutive_periods.append(current_count)
                        current_count = 0
                
                # Handle last regime
                if current_count > 0:
                    consecutive_periods.append(current_count)
                
                if consecutive_periods:
                    persistence_metrics[regime] = {
                        'avg_duration': np.mean(consecutive_periods),
                        'max_duration': np.max(consecutive_periods),
                        'min_duration': np.min(consecutive_periods),
                        'total_occurrences': len(consecutive_periods)
                    }
        
        return persistence_metrics
    
    def calculate_transition_matrix(self, regime_data):
        """Calculate regime transition probability matrix"""
        regimes = ['Bull', 'Bear', 'Stress', 'Sideways']
        transition_counts = pd.DataFrame(0, index=regimes, columns=regimes)
        
        for i in range(1, len(regime_data)):
            current_regime = regime_data['regime'].iloc[i-1]
            next_regime = regime_data['regime'].iloc[i]
            
            if current_regime in regimes and next_regime in regimes:
                transition_counts.loc[current_regime, next_regime] += 1
        
        # Convert to probabilities
        transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
        
        return {
            'counts': transition_counts.to_dict(),
            'probabilities': transition_probs.to_dict()
        }
    
    def calculate_regime_predictability(self, regime_data):
        """Calculate regime predictability metrics"""
        # Calculate how predictable regime changes are
        regime_changes = regime_data['regime'] != regime_data['regime'].shift(1)
        change_dates = regime_data[regime_changes].index
        
        if len(change_dates) < 2:
            return {'predictability_score': 0, 'change_frequency': 0}
        
        # Calculate time between regime changes
        change_intervals = []
        for i in range(1, len(change_dates)):
            interval = (change_dates[i] - change_dates[i-1]).days
            change_intervals.append(interval)
        
        # Predictability based on consistency of intervals
        if change_intervals:
            interval_std = np.std(change_intervals)
            interval_mean = np.mean(change_intervals)
            predictability_score = 1 / (1 + interval_std / interval_mean) if interval_mean > 0 else 0
        else:
            predictability_score = 0
        
        return {
            'predictability_score': predictability_score,
            'avg_change_interval': np.mean(change_intervals) if change_intervals else 0,
            'change_frequency': len(change_dates) / len(regime_data)
        }
    
    def calculate_economic_significance(self, regime_data):
        """Calculate economic significance of regime detection"""
        economic_metrics = {}
        
        for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
            regime_mask = regime_data['regime'] == regime
            if regime_mask.sum() > 0:
                regime_data_subset = regime_data[regime_mask]
                
                # Calculate regime-specific metrics
                returns = regime_data_subset['returns']
                cumulative_return = (1 + returns).prod() - 1
                annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                
                economic_metrics[regime] = {
                    'cumulative_return': cumulative_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'period_count': len(returns)
                }
        
        return economic_metrics
    
    def plot_regime_analysis(self, regime_data, save_path=None):
        """Create comprehensive regime analysis plots"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Simple Regime Detection Analysis (Phase 26)', fontsize=16, fontweight='bold')
        
        # 1. Price and Regime Overlay
        ax1 = axes[0, 0]
        ax1.plot(regime_data.index, regime_data['close'], 'b-', alpha=0.7, label='VNINDEX')
        
        # Color code by regime
        colors = {'Bull': 'green', 'Bear': 'red', 'Stress': 'darkred', 'Sideways': 'gray'}
        for regime in colors:
            regime_mask = regime_data['regime'] == regime
            if regime_mask.sum() > 0:
                ax1.scatter(regime_data[regime_mask].index, 
                           regime_data[regime_mask]['close'],
                           c=colors[regime], s=10, alpha=0.6, label=regime)
        
        ax1.set_title('Price and Regime Detection')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Volatility and Returns
        ax2 = axes[0, 1]
        ax2.plot(regime_data.index, regime_data['rolling_vol'], 'orange', label='Volatility')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(regime_data.index, regime_data['rolling_returns'], 'purple', label='Returns')
        
        ax2.set_title('Rolling Volatility and Returns')
        ax2.set_ylabel('Volatility', color='orange')
        ax2_twin.set_ylabel('Returns', color='purple')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Regime Distribution
        ax3 = axes[1, 0]
        regime_counts = regime_data['regime'].value_counts()
        colors_list = [colors.get(regime, 'blue') for regime in regime_counts.index]
        ax3.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%', colors=colors_list)
        ax3.set_title('Regime Distribution')
        
        # 4. Regime Returns Distribution
        ax4 = axes[1, 1]
        regime_returns = []
        regime_labels = []
        for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
            regime_mask = regime_data['regime'] == regime
            if regime_mask.sum() > 0:
                regime_returns.extend(regime_data[regime_mask]['returns'].values)
                regime_labels.extend([regime] * regime_mask.sum())
        
        if regime_returns:
            returns_df = pd.DataFrame({'returns': regime_returns, 'regime': regime_labels})
            returns_df.boxplot(column='returns', by='regime', ax=ax4)
            ax4.set_title('Returns Distribution by Regime')
            ax4.set_ylabel('Daily Returns')
        
        # 5. Regime Duration Analysis
        ax5 = axes[2, 0]
        duration_data = []
        for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
            regime_mask = regime_data['regime'] == regime
            if regime_mask.sum() > 0:
                # Calculate consecutive periods
                consecutive_periods = []
                current_count = 0
                for is_in_regime in regime_mask:
                    if is_in_regime:
                        current_count += 1
                    else:
                        if current_count > 0:
                            consecutive_periods.append(current_count)
                        current_count = 0
                if current_count > 0:
                    consecutive_periods.append(current_count)
                
                if consecutive_periods:
                    duration_data.extend([(regime, duration) for duration in consecutive_periods])
        
        if duration_data:
            duration_df = pd.DataFrame(duration_data, columns=['regime', 'duration'])
            duration_df.boxplot(column='duration', by='regime', ax=ax5)
            ax5.set_title('Regime Duration Distribution')
            ax5.set_ylabel('Duration (Days)')
        
        # 6. Cumulative Returns by Regime
        ax6 = axes[2, 1]
        for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
            regime_mask = regime_data['regime'] == regime
            if regime_mask.sum() > 0:
                regime_data_subset = regime_data[regime_mask]
                cumulative_returns = (1 + regime_data_subset['returns']).cumprod()
                ax6.plot(regime_data_subset.index, cumulative_returns, 
                        color=colors.get(regime, 'blue'), label=regime, linewidth=2)
        
        ax6.set_title('Cumulative Returns by Regime')
        ax6.set_ylabel('Cumulative Return')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def run_complete_analysis(self, start_date='2016-01-01', end_date='2025-07-28'):
        """Run complete regime detection analysis"""
        print("=== Simple Regime Detection Analysis (Phase 26) ===")
        print(f"Date Range: {start_date} to {end_date}")
        print()
        
        # 1. Load data
        print("1. Loading benchmark data...")
        benchmark_data = self.load_benchmark_data(start_date, end_date)
        print(f"   Loaded {len(benchmark_data)} data points")
        
        # 2. Detect regimes
        print("2. Detecting regimes...")
        regime_data = self.detect_regimes(benchmark_data)
        print(f"   Regime detection completed")
        
        # 3. Calculate statistics
        print("3. Calculating regime statistics...")
        regime_stats = self.calculate_regime_statistics(regime_data)
        
        # 4. Validate regime detection
        print("4. Validating regime detection...")
        validation_results = self.validate_regime_detection(regime_data)
        
        # 5. Generate plots
        print("5. Generating analysis plots...")
        self.plot_regime_analysis(regime_data, 'phase26_regime_analysis.png')
        
        # 6. Print results
        self.print_analysis_results(regime_stats, validation_results)
        
        # 7. Save results
        results = {
            'regime_data': regime_data,
            'regime_stats': regime_stats,
            'validation_results': validation_results,
            'parameters': {
                'lookback_period': self.lookback_period,
                'vol_threshold_high': self.vol_threshold_high,
                'return_threshold_bull': self.return_threshold_bull,
                'return_threshold_bear': self.return_threshold_bear
            }
        }
        
        with open('phase26_regime_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print("\n6. Results saved to 'phase26_regime_results.pkl'")
        
        return results
    
    def print_analysis_results(self, regime_stats, validation_results):
        """Print comprehensive analysis results"""
        print("\n" + "="*60)
        print("SIMPLE REGIME DETECTION RESULTS")
        print("="*60)
        
        # Regime distribution
        print("\n1. REGIME DISTRIBUTION:")
        print("-" * 30)
        for regime, pct in regime_stats['regime_distribution']['percentages'].items():
            count = regime_stats['regime_distribution']['counts'][regime]
            print(f"{regime:10}: {pct:6.1f}% ({count:4d} periods)")
        
        # Regime characteristics
        print("\n2. REGIME CHARACTERISTICS:")
        print("-" * 30)
        print(f"{'Regime':<10} {'Count':<6} {'Avg Ret':<8} {'Avg Vol':<8} {'Sharpe':<8} {'Max DD':<8}")
        print("-" * 60)
        
        for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
            if regime in regime_stats['regime_characteristics']:
                stats = regime_stats['regime_characteristics'][regime]
                print(f"{regime:<10} {stats['count']:<6} {stats['avg_return']:<8.2%} "
                      f"{stats['avg_vol']:<8.2%} {stats['sharpe']:<8.2f} {stats['max_drawdown']:<8.2%}")
        
        # Validation results
        print("\n3. VALIDATION RESULTS:")
        print("-" * 30)
        
        accuracy = validation_results['regime_accuracy']
        print(f"Regime Accuracy Score: {accuracy['accuracy_score']:.3f}")
        print(f"Regime Changes: {accuracy['regime_changes']}")
        print(f"Average Regime Duration: {accuracy['avg_regime_duration']:.1f} days")
        
        predictability = validation_results['regime_predictability']
        print(f"Predictability Score: {predictability['predictability_score']:.3f}")
        print(f"Change Frequency: {predictability['change_frequency']:.3f}")
        
        # Economic significance
        print("\n4. ECONOMIC SIGNIFICANCE:")
        print("-" * 30)
        print(f"{'Regime':<10} {'Ann Ret':<8} {'Vol':<8} {'Sharpe':<8} {'Cum Ret':<8}")
        print("-" * 50)
        
        for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
            if regime in validation_results['economic_significance']:
                econ = validation_results['economic_significance'][regime]
                print(f"{regime:<10} {econ['annualized_return']:<8.2%} {econ['volatility']:<8.2%} "
                      f"{econ['sharpe_ratio']:<8.2f} {econ['cumulative_return']:<8.2%}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # Initialize regime detection
    regime_detector = SimpleRegimeDetection()
    
    # Run complete analysis
    results = regime_detector.run_complete_analysis()
    
    print("\nAnalysis completed successfully!")
    print("Check 'phase26_regime_analysis.png' for visualizations")
    print("Check 'phase26_regime_results.pkl' for detailed results") 