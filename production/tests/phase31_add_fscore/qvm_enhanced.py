# %% [markdown]
# # QVM ENGINE V3 F-SCORE INTEGRATION TEARSHEET - ENHANCED
#
# This notebook demonstrates the QVM (Quality, Value, Momentum) factor investing strategy with Piotroski F-Score integration and comprehensive performance analysis.
#
# **Key Changes:** 
# - Piotroski F-Score integration into Quality factor (50% weight)
# - Simplified quality factor weighting: ROAA (50%), F-Score (50%)
# - Sector-specific F-Score calculations: Non-Financial (9 tests), Banking (6 tests), Securities (5 tests)
# - Real-time F-Score calculation from database
# - Cash allocation tracking below equity curve
# - Fixed portfolio size: exactly 20 stocks per rebalancing date
# - **NEW: Historical snapshots and metrics across time**
# - **NEW: Factor score analysis and portfolio holdings evolution**

# %% [markdown]
# # IMPORTS AND SETUP

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %%
import sys
sys.path.append('/home/raymond/Documents/Projects/factor-investing-public')
from production.database.connection import DatabaseManager
from production.engine.qvm_engine_v3_fscore import QVMEngineV3FScore, PiotroskiFScoreCalculator

# %% [markdown]
# # F-SCORE INTEGRATION STRATEGY

# %%
class FScoreIntegrationStrategy:
    """
    QVM strategy with Piotroski F-Score integration into quality factor.
    """
    def __init__(self):
        self.quality_weights = {
            'roaa': 0.50,         # 50% for ROAA
            'fscore': 0.50        # 50% for Piotroski F-Score
        }
        
        self.qvm_weights = {
            'quality': 0.3333,    # 33.33% Quality (with F-Score) - matching 04c
            'value': 0.3333,      # 33.33% Value - matching 04c
            'momentum': 0.3334    # 33.34% Momentum - matching 04c
        }
        
        print(f"✅ FScoreIntegrationStrategy initialized:")
        print(f"   - Quality Factor Weights:")
        print(f"      ROAA: {self.quality_weights['roaa']:.0%}")
        print(f"      F-Score: {self.quality_weights['fscore']:.0%}")
        print(f"   - QVM Factor Weights:")
        print(f"      Quality: {self.qvm_weights['quality']:.0%}")
        print(f"      Value: {self.qvm_weights['value']:.0%}")
        print(f"      Momentum: {self.qvm_weights['momentum']:.0%}")
        print(f"   - F-Score Tests by Sector:")
        print(f"      Non-Financial: 9 tests (ROA>0, CFO>0, ΔROA>0, etc.)")
        print(f"      Banking: 6 tests (ROA>0, NIM>0, ΔROA>0, etc.)")
        print(f"      Securities: 5 tests (ROA>0, BrokerageRatio>0, ΔROA>0, etc.)")

# %% [markdown]
# # DRAWDOWN PROTECTION STRATEGY - DYNAMIC POSITION SIZING

# %%
class DrawdownProtectionStrategy:
    """
    QVM strategy with drawdown-based position sizing.
    """
    def __init__(self, step_size: float = 0.10):
        self.step_size = step_size  # 10% steps for allocation changes
        self.current_allocation = 1.0  # Start at 100%
        self.last_allocation_change = 0.0  # Track last drawdown level for allocation change
        
        print(f"✅ DrawdownProtectionStrategy initialized:")
        print(f"   - Step Size: {self.step_size:.0%} (10% increments)")
        print(f"   - Initial Allocation: {self.current_allocation:.0%}")
        print(f"   - Factor Weights: Quality 40%, Value 30%, Momentum 30%")
        print(f"   - Drawdown Protection Levels:")
        print(f"     5% drawdown: 20% allocation")
        print(f"     10% drawdown: 40% allocation") 
        print(f"     15% drawdown: 60% allocation")
        print(f"     20% drawdown: 80% allocation")
        print(f"     25% drawdown: 100% allocation")
        print(f"     30%+ drawdown: 100% allocation")
    
    def calculate_drawdown(self, benchmark_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate drawdown for the benchmark."""
        # Implementation would go here
        pass

# %% [markdown]
# # HISTORICAL SNAPSHOTS AND METRICS TRACKER

# %%
class HistoricalMetricsTracker:
    """
    Tracks historical metrics, factor scores, and portfolio holdings across time.
    """
    def __init__(self):
        self.historical_data = {
            'dates': [],
            'portfolio_values': [],
            'factor_scores': [],
            'holdings': [],
            'metrics': []
        }
        self.snapshot_dates = []
        
    def add_snapshot(self, date: datetime, portfolio_data: dict, factor_scores: dict, 
                    holdings: pd.DataFrame, metrics: dict):
        """Add a historical snapshot."""
        self.snapshot_dates.append(date)
        self.historical_data['dates'].append(date)
        self.historical_data['portfolio_values'].append(portfolio_data)
        self.historical_data['factor_scores'].append(factor_scores)
        self.historical_data['holdings'].append(holdings)
        self.historical_data['metrics'].append(metrics)
        
        print(f"📊 Snapshot added for {date.strftime('%Y-%m-%d')}")
        
    def get_latest_snapshot(self):
        """Get the most recent snapshot."""
        if not self.snapshot_dates:
            return None
        latest_idx = -1
        return {
            'date': self.historical_data['dates'][latest_idx],
            'portfolio': self.historical_data['portfolio_values'][latest_idx],
            'factors': self.historical_data['factor_scores'][latest_idx],
            'holdings': self.historical_data['holdings'][latest_idx],
            'metrics': self.historical_data['metrics'][latest_idx]
        }
        
    def get_snapshot_by_date(self, target_date: datetime):
        """Get snapshot for a specific date."""
        for i, date in enumerate(self.historical_data['dates']):
            if date.date() == target_date.date():
                return {
                    'date': self.historical_data['dates'][i],
                    'portfolio': self.historical_data['portfolio_values'][i],
                    'factors': self.historical_data['factor_scores'][i],
                    'holdings': self.historical_data['holdings'][i],
                    'metrics': self.historical_data['metrics'][i]
                }
        return None
        
    def export_snapshots(self, output_dir: Path):
        """Export all snapshots to CSV files."""
        output_dir.mkdir(exist_ok=True)
        
        # Export portfolio values
        portfolio_df = pd.DataFrame(self.historical_data['portfolio_values'])
        portfolio_df['date'] = self.historical_data['dates']
        portfolio_df.to_csv(output_dir / 'portfolio_values_history.csv', index=False)
        
        # Export factor scores
        factors_df = pd.DataFrame(self.historical_data['factor_scores'])
        factors_df['date'] = self.historical_data['dates']
        factors_df.to_csv(output_dir / 'factor_scores_history.csv', index=False)
        
        # Export metrics
        metrics_df = pd.DataFrame(self.historical_data['metrics'])
        metrics_df['date'] = self.historical_data['dates']
        metrics_df.to_csv(output_dir / 'metrics_history.csv', index=False)
        
        print(f"📁 Historical data exported to {output_dir}")
        
    def generate_summary_report(self):
        """Generate a summary report of historical performance."""
        if not self.snapshot_dates:
            return "No historical data available"
            
        report = []
        report.append("📊 HISTORICAL PERFORMANCE SUMMARY")
        report.append("=" * 50)
        
        # Portfolio value evolution
        portfolio_values = [p.get('total_value', 0) for p in self.historical_data['portfolio_values']]
        if portfolio_values:
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            report.append(f"Portfolio Value Evolution:")
            report.append(f"  Initial: {initial_value:,.0f} VND")
            report.append(f"  Final: {final_value:,.0f} VND")
            report.append(f"  Total Return: {total_return:.2f}%")
            
        # Factor score trends
        if self.historical_data['factor_scores']:
            latest_factors = self.historical_data['factor_scores'][-1]
            report.append(f"\nLatest Factor Scores:")
            for factor, score in latest_factors.items():
                report.append(f"  {factor}: {score:.3f}")
                
        return "\n".join(report)

# %% [markdown]
# # FACTOR ANALYSIS AND PORTFOLIO SIMULATION

# %%
class FactorAnalyzer:
    """
    Analyzes factor scores and generates insights for portfolio construction.
    """
    def __init__(self, qvm_engine: QVMEngineV3FScore):
        self.qvm_engine = qvm_engine
        self.factor_history = []
        
    def analyze_ticker_factors(self, ticker: str, analysis_date: datetime) -> dict:
        """Analyze all factors for a specific ticker."""
        try:
            # Get quality factor (ROAA + F-Score)
            quality_score = self.qvm_engine.calculate_enhanced_quality_factor([ticker], analysis_date)
            
            # Get value factor
            value_score = self.qvm_engine.calculate_value_factor([ticker], analysis_date)
            
            # Get momentum factor
            momentum_score = self.qvm_engine.calculate_momentum_factor([ticker], analysis_date)
            
            # Calculate composite score
            composite_score = (
                0.3333 * quality_score.get(ticker, 0.0) +
                0.3333 * value_score.get(ticker, 0.0) +
                0.3334 * momentum_score.get(ticker, 0.0)
            )
            
            return {
                'ticker': ticker,
                'quality_score': quality_score.get(ticker, 0.0),
                'value_score': value_score.get(ticker, 0.0),
                'momentum_score': momentum_score.get(ticker, 0.0),
                'composite_score': composite_score,
                'analysis_date': analysis_date
            }
            
        except Exception as e:
            print(f"❌ Error analyzing factors for {ticker}: {e}")
            return {
                'ticker': ticker,
                'quality_score': 0.0,
                'value_score': 0.0,
                'momentum_score': 0.0,
                'composite_score': 0.0,
                'analysis_date': analysis_date
            }
    
    def analyze_universe_factors(self, tickers: list, analysis_date: datetime) -> pd.DataFrame:
        """Analyze factors for all tickers in universe."""
        results = []
        
        for ticker in tickers:
            ticker_analysis = self.analyze_ticker_factors(ticker, analysis_date)
            results.append(ticker_analysis)
            
        return pd.DataFrame(results)
    
    def generate_factor_insights(self, factor_df: pd.DataFrame) -> dict:
        """Generate insights from factor analysis."""
        insights = {}
        
        # Quality factor insights
        quality_scores = factor_df['quality_score'].dropna()
        if not quality_scores.empty:
            insights['quality'] = {
                'mean': quality_scores.mean(),
                'std': quality_scores.std(),
                'min': quality_scores.min(),
                'max': quality_scores.max(),
                'top_10_pct': quality_scores.quantile(0.9),
                'bottom_10_pct': quality_scores.quantile(0.1)
            }
        
        # Value factor insights
        value_scores = factor_df['value_score'].dropna()
        if not value_scores.empty:
            insights['value'] = {
                'mean': value_scores.mean(),
                'std': value_scores.std(),
                'min': value_scores.min(),
                'max': value_scores.max(),
                'top_10_pct': value_scores.quantile(0.9),
                'bottom_10_pct': value_scores.quantile(0.1)
            }
        
        # Momentum factor insights
        momentum_scores = factor_df['momentum_score'].dropna()
        if not momentum_scores.empty:
            insights['momentum'] = {
                'mean': momentum_scores.mean(),
                'std': momentum_scores.std(),
                'min': momentum_scores.min(),
                'max': momentum_scores.max(),
                'top_10_pct': momentum_scores.quantile(0.9),
                'bottom_10_pct': momentum_scores.quantile(0.1)
            }
        
        # Composite score insights
        composite_scores = factor_df['composite_score'].dropna()
        if not composite_scores.empty:
            insights['composite'] = {
                'mean': composite_scores.mean(),
                'std': composite_scores.std(),
                'min': composite_scores.min(),
                'max': composite_scores.max(),
                'top_10_pct': composite_scores.quantile(0.9),
                'bottom_10_pct': composite_scores.quantile(0.1)
            }
        
        return insights

# %% [markdown]
# # PORTFOLIO CONSTRUCTION AND REBALANCING

# %%
class PortfolioConstructor:
    """
    Constructs and rebalances portfolios based on factor scores.
    """
    def __init__(self, target_size: int = 20):
        self.target_size = target_size
        self.portfolio_history = []
        
    def construct_portfolio(self, factor_df: pd.DataFrame, cash_allocation: float = 0.05) -> dict:
        """Construct portfolio from factor scores."""
        # Sort by composite score and select top tickers
        sorted_df = factor_df.sort_values('composite_score', ascending=False)
        selected_tickers = sorted_df.head(self.target_size)
        
        # Calculate position sizes (equal weight for now)
        position_weight = (1 - cash_allocation) / self.target_size
        
        portfolio = {
            'tickers': selected_tickers['ticker'].tolist(),
            'weights': [position_weight] * self.target_size,
            'factor_scores': selected_tickers['composite_score'].tolist(),
            'cash_allocation': cash_allocation,
            'construction_date': datetime.now()
        }
        
        return portfolio
    
    def rebalance_portfolio(self, current_portfolio: dict, new_factor_df: pd.DataFrame, 
                          rebalance_threshold: float = 0.10) -> dict:
        """Rebalance portfolio based on new factor scores."""
        # Check if rebalancing is needed
        current_tickers = set(current_portfolio['tickers'])
        new_sorted_df = new_factor_df.sort_values('composite_score', ascending=False)
        new_top_tickers = set(new_sorted_df.head(self.target_size)['ticker'])
        
        # Calculate overlap
        overlap = len(current_tickers.intersection(new_top_tickers))
        overlap_ratio = overlap / self.target_size
        
        if overlap_ratio >= (1 - rebalance_threshold):
            print(f"📊 Portfolio overlap: {overlap_ratio:.1%} - No rebalancing needed")
            return current_portfolio
        
        print(f"📊 Portfolio overlap: {overlap_ratio:.1%} - Rebalancing portfolio")
        
        # Construct new portfolio
        new_portfolio = self.construct_portfolio(new_factor_df)
        
        # Track rebalancing
        rebalance_info = {
            'old_portfolio': current_portfolio,
            'new_portfolio': new_portfolio,
            'overlap_ratio': overlap_ratio,
            'rebalance_date': datetime.now()
        }
        
        self.portfolio_history.append(rebalance_info)
        return new_portfolio

# %% [markdown]
# # SAMPLE DATA GENERATION FOR DEMONSTRATION

# %%
def generate_sample_universe_data(n_tickers: int = 100) -> pd.DataFrame:
    """Generate sample universe data for demonstration purposes."""
    np.random.seed(42)  # For reproducibility
    
    # Generate sample tickers
    tickers = [f"TICKER_{i:03d}" for i in range(1, n_tickers + 1)]
    
    # Generate sample factor scores
    data = {
        'ticker': tickers,
        'quality_score': np.random.normal(0.5, 0.2, n_tickers).clip(0, 1),
        'value_score': np.random.normal(0.5, 0.2, n_tickers).clip(0, 1),
        'momentum_score': np.random.normal(0.5, 0.2, n_tickers).clip(0, 1),
        'market_cap': np.random.lognormal(10, 1, n_tickers),
        'sector': np.random.choice(['Banking', 'Securities', 'Non-Financial'], n_tickers)
    }
    
    # Calculate composite scores
    data['composite_score'] = (
        0.3333 * data['quality_score'] +
        0.3333 * data['value_score'] +
        0.3334 * data['momentum_score']
    )
    
    return pd.DataFrame(data)

def generate_sample_portfolio_snapshots(n_snapshots: int = 12) -> list:
    """Generate sample portfolio snapshots over time."""
    snapshots = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(n_snapshots):
        snapshot_date = base_date + timedelta(days=30*i)
        
        # Generate sample portfolio data
        portfolio_data = {
            'total_value': 1000000000 + i * 50000000,  # 1B + 50M monthly growth
            'equity_value': 950000000 + i * 47500000,
            'cash_value': 50000000 + i * 2500000,
            'num_positions': 20
        }
        
        # Generate sample factor scores
        factor_scores = {
            'quality_avg': 0.6 + np.random.normal(0, 0.05),
            'value_avg': 0.55 + np.random.normal(0, 0.05),
            'momentum_avg': 0.65 + np.random.normal(0, 0.05),
            'composite_avg': 0.6 + np.random.normal(0, 0.03)
        }
        
        # Generate sample holdings
        holdings_data = []
        for j in range(20):
            ticker = f"TICKER_{j+1:03d}"
            weight = 0.0475 + np.random.normal(0, 0.005)  # ~4.75% weight
            holdings_data.append({
                'ticker': ticker,
                'weight': weight,
                'market_value': weight * portfolio_data['equity_value'],
                'quality_score': factor_scores['quality_avg'] + np.random.normal(0, 0.1),
                'value_score': factor_scores['value_avg'] + np.random.normal(0, 0.1),
                'momentum_score': factor_scores['momentum_avg'] + np.random.normal(0, 0.1)
            })
        
        holdings_df = pd.DataFrame(holdings_data)
        
        # Generate sample metrics
        metrics = {
            'sharpe_ratio': 1.2 + np.random.normal(0, 0.1),
            'max_drawdown': 0.08 + np.random.normal(0, 0.02),
            'volatility': 0.15 + np.random.normal(0, 0.02),
            'beta': 0.95 + np.random.normal(0, 0.05)
        }
        
        snapshot = {
            'date': snapshot_date,
            'portfolio_data': portfolio_data,
            'factor_scores': factor_scores,
            'holdings': holdings_df,
            'metrics': metrics
        }
        
        snapshots.append(snapshot)
    
    return snapshots

# %% [markdown]
# # VISUALIZATION AND ANALYSIS FUNCTIONS

# %%
def plot_factor_score_distributions(factor_df: pd.DataFrame, save_path: Path = None):
    """Plot distributions of factor scores."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Factor Score Distributions', fontsize=16, fontweight='bold')
    
    # Quality factor distribution
    axes[0, 0].hist(factor_df['quality_score'].dropna(), bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Quality Factor Distribution')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(factor_df['quality_score'].mean(), color='red', linestyle='--', label=f'Mean: {factor_df["quality_score"].mean():.3f}')
    axes[0, 0].legend()
    
    # Value factor distribution
    axes[0, 1].hist(factor_df['value_score'].dropna(), bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Value Factor Distribution')
    axes[0, 1].set_xlabel('Value Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(factor_df['value_score'].mean(), color='red', linestyle='--', label=f'Mean: {factor_df["value_score"].mean():.3f}')
    axes[0, 1].legend()
    
    # Momentum factor distribution
    axes[1, 0].hist(factor_df['momentum_score'].dropna(), bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Momentum Factor Distribution')
    axes[1, 0].set_xlabel('Momentum Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(factor_df['momentum_score'].mean(), color='red', linestyle='--', label=f'Mean: {factor_df["momentum_score"].mean():.3f}')
    axes[1, 0].legend()
    
    # Composite score distribution
    axes[1, 1].hist(factor_df['composite_score'].dropna(), bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Composite Score Distribution')
    axes[1, 1].set_xlabel('Composite Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(factor_df['composite_score'].mean(), color='red', linestyle='--', label=f'Mean: {factor_df["composite_score"].mean():.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'factor_distributions.png', dpi=300, bbox_inches='tight')
        print(f"📊 Factor distributions plot saved to {save_path / 'factor_distributions.png'}")
    
    plt.show()

def plot_portfolio_evolution(historical_data: dict, save_path: Path = None):
    """Plot portfolio evolution over time."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Portfolio Evolution Over Time', fontsize=16, fontweight='bold')
    
    dates = [d.strftime('%Y-%m') for d in historical_data['dates']]
    
    # Portfolio value evolution
    portfolio_values = [p.get('total_value', 0) for p in historical_data['portfolio_values']]
    axes[0, 0].plot(dates, portfolio_values, marker='o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Portfolio Total Value')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value (VND)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Factor score evolution
    if historical_data['factor_scores']:
        quality_scores = [f.get('quality_avg', 0) for f in historical_data['factor_scores']]
        value_scores = [f.get('value_avg', 0) for f in historical_data['factor_scores']]
        momentum_scores = [f.get('momentum_avg', 0) for f in historical_data['factor_scores']]
        
        axes[0, 1].plot(dates, quality_scores, marker='o', label='Quality', linewidth=2)
        axes[0, 1].plot(dates, value_scores, marker='s', label='Value', linewidth=2)
        axes[0, 1].plot(dates, momentum_scores, marker='^', label='Momentum', linewidth=2)
        axes[0, 1].set_title('Average Factor Scores')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Cash vs Equity allocation
    equity_values = [p.get('equity_value', 0) for p in historical_data['portfolio_values']]
    cash_values = [p.get('cash_value', 0) for p in historical_data['portfolio_values']]
    
    axes[1, 0].stackplot(dates, [equity_values, cash_values], 
                         labels=['Equity', 'Cash'], alpha=0.7)
    axes[1, 0].set_title('Portfolio Allocation')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Value (VND)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance metrics
    if historical_data['metrics']:
        sharpe_ratios = [m.get('sharpe_ratio', 0) for m in historical_data['metrics']]
        max_drawdowns = [m.get('max_drawdown', 0) for m in historical_data['metrics']]
        
        ax2 = axes[1, 1].twinx()
        line1 = axes[1, 1].plot(dates, sharpe_ratios, marker='o', color='blue', label='Sharpe Ratio', linewidth=2)
        line2 = ax2.plot(dates, max_drawdowns, marker='s', color='red', label='Max Drawdown', linewidth=2)
        
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Sharpe Ratio', color='blue')
        ax2.set_ylabel('Max Drawdown', color='red')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[1, 1].legend(lines, labels, loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'portfolio_evolution.png', dpi=300, bbox_inches='tight')
        print(f"📊 Portfolio evolution plot saved to {save_path / 'portfolio_evolution.png'}")
    
    plt.show()

def plot_holdings_analysis(holdings_df: pd.DataFrame, save_path: Path = None):
    """Plot holdings analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Portfolio Holdings Analysis', fontsize=16, fontweight='bold')
    
    # Top holdings by weight
    top_holdings = holdings_df.nlargest(10, 'weight')
    axes[0, 0].barh(range(len(top_holdings)), top_holdings['weight'] * 100)
    axes[0, 0].set_yticks(range(len(top_holdings)))
    axes[0, 0].set_yticklabels(top_holdings['ticker'])
    axes[0, 0].set_title('Top 10 Holdings by Weight')
    axes[0, 0].set_xlabel('Weight (%)')
    axes[0, 0].invert_yaxis()
    
    # Factor scores by holding
    scatter = axes[0, 1].scatter(holdings_df['quality_score'], holdings_df['value_score'], 
                       c=holdings_df['momentum_score'], s=100, alpha=0.7, cmap='viridis')
    axes[0, 1].set_xlabel('Quality Score')
    axes[0, 1].set_ylabel('Value Score')
    axes[0, 1].set_title('Factor Score Scatter Plot')
    plt.colorbar(scatter, ax=axes[0, 1], label='Momentum Score')
    
    # Weight vs Composite Score
    axes[1, 0].scatter(holdings_df['weight'] * 100, holdings_df['quality_score'], alpha=0.7)
    axes[1, 0].set_xlabel('Weight (%)')
    axes[1, 0].set_ylabel('Quality Score')
    axes[1, 0].set_title('Weight vs Quality Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sector allocation (if available)
    if 'sector' in holdings_df.columns:
        sector_allocation = holdings_df.groupby('sector')['weight'].sum() * 100
        axes[1, 1].pie(sector_allocation.values, labels=sector_allocation.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Sector Allocation')
    else:
        # Market value distribution
        axes[1, 1].hist(holdings_df['market_value'], bins=15, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Market Value (VND)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Market Value Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'holdings_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📊 Holdings analysis plot saved to {save_path / 'holdings_analysis.png'}")
    
    plt.show()

# %% [markdown]
# # COMPREHENSIVE TEARSHEET GENERATION

# %%
def calculate_performance_metrics(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculates comprehensive performance metrics with corrected benchmark alignment."""
    # Align benchmark
    first_trade_date = returns.loc[returns.ne(0)].index.min()
    if pd.isna(first_trade_date):
        return {metric: 0.0 for metric in ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Calmar Ratio', 'Information Ratio', 'Beta']}
    
    aligned_returns = returns.loc[first_trade_date:]
    aligned_benchmark = benchmark.loc[first_trade_date:]
    
    if len(aligned_returns) < 2:
        return {metric: 0.0 for metric in ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Calmar Ratio', 'Information Ratio', 'Beta']}
    
    # Basic metrics
    total_return = (1 + aligned_returns).prod() - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(aligned_returns)) - 1
    annualized_volatility = aligned_returns.std() * np.sqrt(periods_per_year)
    
    # Risk metrics
    cumulative_returns = (1 + aligned_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Ratios
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Benchmark metrics
    if not aligned_benchmark.empty:
        benchmark_return = (1 + aligned_benchmark).prod() - 1
        benchmark_volatility = aligned_benchmark.std() * np.sqrt(periods_per_year)
        
        # Information ratio
        excess_returns = aligned_returns - aligned_benchmark
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year) if excess_returns.std() > 0 else 0
        
        # Beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
    else:
        information_ratio = 0
        beta = 0
    
    return {
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Calmar Ratio': calmar_ratio,
        'Information Ratio': information_ratio,
        'Beta': beta
    }

def generate_comprehensive_tearsheet(strategy_returns: pd.Series, benchmark_returns: pd.Series, 
                                   title: str, cash_allocations: pd.DataFrame = None):
    """Generates comprehensive institutional tearsheet with equity curve and cash allocation chart."""
    
    # Align benchmark for plotting & metrics
    first_trade_date = strategy_returns.loc[strategy_returns.ne(0)].index.min()
    if pd.isna(first_trade_date):
        print("❌ No valid strategy returns data available")
        return
        
    aligned_strategy_returns = strategy_returns.loc[first_trade_date:]
    aligned_benchmark_returns = benchmark_returns.loc[first_trade_date:]

    strategy_metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns, benchmark_returns)
    
    fig = plt.figure(figsize=(18, 30))  # Increased height for cash allocation chart
    gs = fig.add_gridspec(6, 2, height_ratios=[1.2, 0.8, 0.8, 0.8, 0.8, 1.2], hspace=0.7, wspace=0.2)
    fig.suptitle(title, fontsize=20, fontweight='bold', color='#2C3E50')

    # 1. Cumulative Performance (Equity Curve)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot the main equity curves
    (1 + aligned_strategy_returns).cumprod().plot(ax=ax1, label='QVM Engine v3 (F-Score)', color='#16A085', lw=2.5)
    (1 + aligned_benchmark_returns).cumprod().plot(ax=ax1, label='VN-Index (Aligned)', color='#34495E', linestyle='--', lw=2)
    
    ax1.set_title('Cumulative Performance (Log Scale)', fontweight='bold')
    ax1.set_ylabel('Growth of 1 VND')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 2. Cash Allocation Chart (below equity curve)
    ax2 = fig.add_subplot(gs[1, :])
    if cash_allocations is not None and not cash_allocations.empty:
        # Convert dates to datetime for plotting
        cash_allocations['date'] = pd.to_datetime(cash_allocations['date'])
        cash_allocations = cash_allocations.sort_values('date')
        
        # Plot cash allocation percentage over time
        ax2.plot(cash_allocations['date'], cash_allocations['cash_percentage'], 
                color='#E74C3C', linewidth=2, marker='o', markersize=4)
        ax2.fill_between(cash_allocations['date'], cash_allocations['cash_percentage'], 
                        alpha=0.3, color='#E74C3C')
        
        # Add horizontal lines for reference
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20% Cash')
        ax2.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='40% Cash')
        
        ax2.set_title('Cash Allocation Over Time', fontweight='bold')
        ax2.set_ylabel('Cash Allocation (%)')
        ax2.set_ylim(0, max(cash_allocations['cash_percentage'].max() * 1.1, 50))
        ax2.legend(fontsize=10)
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
    strat_annual = aligned_strategy_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
    bench_annual = aligned_benchmark_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
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
    summary_data = [['Metric', 'Strategy', 'Benchmark']]
    for key in strategy_metrics.keys():
        summary_data.append([key, f"{strategy_metrics[key]:.2f}", f"{benchmark_metrics.get(key, 0.0):.2f}"])
    
    table = ax8.table(cellText=summary_data[1:], colLabels=summary_data[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
    return strategy_metrics

def generate_sample_returns_data(n_days: int = 1000) -> tuple:
    """Generate sample returns data for tearsheet demonstration."""
    np.random.seed(42)
    
    # Generate sample dates
    start_date = pd.Timestamp('2020-01-01')
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate sample strategy returns (slightly better than benchmark)
    strategy_returns = pd.Series(np.random.normal(0.0008, 0.015, n_days), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0006, 0.014, n_days), index=dates)
    
    # Add some trend and volatility clustering
    strategy_returns = strategy_returns + 0.0001 * np.arange(n_days) / n_days
    benchmark_returns = benchmark_returns + 0.00005 * np.arange(n_days) / n_days
    
    return strategy_returns, benchmark_returns

def generate_sample_cash_allocations(n_days: int = 1000) -> pd.DataFrame:
    """Generate sample cash allocation data for tearsheet demonstration."""
    np.random.seed(42)
    
    start_date = pd.Timestamp('2020-01-01')
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate realistic cash allocations (mostly 5%, some periods with higher cash)
    base_cash = 5.0
    cash_allocations = []
    
    for i, date in enumerate(dates):
        if i % 100 == 0:  # Every 100 days, simulate a rebalancing
            # Random cash allocation between 5% and 40%
            cash_pct = np.random.choice([5, 10, 15, 20, 25, 30, 35, 40], p=[0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02])
        else:
            # Gradual drift back to base cash
            cash_pct = max(base_cash, cash_pct - np.random.normal(0.1, 0.05))
        
        cash_allocations.append({
            'date': date,
            'cash_percentage': cash_pct
        })
    
    return pd.DataFrame(cash_allocations)

# %% [markdown]
# # MAIN EXECUTION AND DEMONSTRATION

# %%
def main():
    """Main execution function for demonstration."""
    print("🚀 Starting QVM Engine V3 with F-Score Integration - Enhanced Demo")
    print("=" * 80)
    
    # Initialize strategies
    fscore_strategy = FScoreIntegrationStrategy()
    drawdown_strategy = DrawdownProtectionStrategy()
    
    # Initialize historical tracker
    historical_tracker = HistoricalMetricsTracker()
    
    # Generate sample data
    print("\n📊 Generating sample universe data...")
    universe_df = generate_sample_universe_data(n_tickers=100)
    print(f"   ✅ Generated {len(universe_df)} tickers")
    
    # Generate sample portfolio snapshots
    print("\n📊 Generating sample portfolio snapshots...")
    snapshots = generate_sample_portfolio_snapshots(n_snapshots=12)
    print(f"   ✅ Generated {len(snapshots)} monthly snapshots")
    
    # Add snapshots to tracker
    print("\n📊 Adding snapshots to historical tracker...")
    for snapshot in snapshots:
        historical_tracker.add_snapshot(
            snapshot['date'],
            snapshot['portfolio_data'],
            snapshot['factor_scores'],
            snapshot['holdings'],
            snapshot['metrics']
        )
    
    # Generate insights
    print("\n📊 Generating factor insights...")
    factor_analyzer = FactorAnalyzer(None)  # Pass None since we're using sample data
    factor_insights = factor_analyzer.generate_factor_insights(universe_df)
    
    # Display summary
    print("\n" + "=" * 80)
    print(historical_tracker.generate_summary_report())
    print("=" * 80)
    
    # Create output directory - handle different working directories
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export historical data
    print(f"\n📁 Exporting historical data to {output_dir}...")
    historical_tracker.export_snapshots(output_dir)
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    
    # Factor distributions
    plot_factor_score_distributions(universe_df, save_path=output_dir)
    
    # Portfolio evolution
    plot_portfolio_evolution(historical_tracker.historical_data, save_path=output_dir)
    
    # Latest holdings analysis
    latest_snapshot = historical_tracker.get_latest_snapshot()
    if latest_snapshot:
        plot_holdings_analysis(latest_snapshot['holdings'], save_path=output_dir)
    
    # Generate comprehensive tearsheet
    print(f"\n📊 Generating comprehensive tearsheet...")
    strategy_returns, benchmark_returns, diagnostics_df, cash_allocations_df = generate_sample_tearsheet_data()
    
    # Generate the comprehensive tearsheet
    generate_comprehensive_tearsheet_with_cash_allocation(
        strategy_returns, 
        benchmark_returns, 
        diagnostics_df, 
        cash_allocations_df,
        "QVM ENGINE V3 F-SCORE: COMPREHENSIVE TEARSHEET DEMO"
    )
    
    # Calculate and display performance metrics
    print(f"\n📊 Performance Metrics Summary:")
    strategy_metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns, benchmark_returns)
    
    print("Strategy Metrics:")
    for key, value in strategy_metrics.items():
        print(f"   {key}: {value:.2f}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"📊 Generated {len(snapshots)} portfolio snapshots")
    print(f"📈 Analyzed {len(universe_df)} tickers")
    print(f"📊 Generated comprehensive tearsheet with performance metrics")

# %% [markdown]
# # COMPREHENSIVE TEARSHEET GENERATION

# %%
def calculate_performance_metrics(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculates comprehensive performance metrics with corrected benchmark alignment."""
    # Align benchmark
    first_trade_date = returns.loc[returns.ne(0)].index.min()
    if pd.isna(first_trade_date):
        return {metric: 0.0 for metric in ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Calmar Ratio', 'Information Ratio', 'Beta']}
    
    aligned_returns = returns.loc[first_trade_date:]
    aligned_benchmark = benchmark.loc[first_trade_date:]
    
    # Basic metrics
    total_return = (1 + aligned_returns).prod() - 1
    years = len(aligned_returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Volatility
    annualized_volatility = aligned_returns.std() * np.sqrt(periods_per_year) if len(aligned_returns) > 1 else 0
    
    # Sharpe Ratio
    risk_free_rate = 0.05  # Assume 5% risk-free rate
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
    
    # Drawdown
    cumulative_returns = (1 + aligned_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Information Ratio
    excess_returns = aligned_returns - aligned_benchmark
    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year) if excess_returns.std() > 0 else 0
    
    # Beta
    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
    benchmark_variance = aligned_benchmark.var()
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    return {
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Information Ratio': information_ratio,
        'Beta': beta
    }

def generate_sample_tearsheet_data():
    """Generate sample data for tearsheet demonstration."""
    # Generate sample dates
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    
    # Generate sample strategy returns (with some volatility and trend)
    np.random.seed(42)
    daily_returns = np.random.normal(0.0008, 0.02, len(dates))  # 0.08% daily return, 2% volatility
    # Add some trend
    trend = np.linspace(0, 0.0002, len(dates))
    daily_returns += trend
    
    # Generate benchmark returns (VN-Index like)
    benchmark_returns = np.random.normal(0.0006, 0.018, len(dates))  # 0.06% daily return, 1.8% volatility
    benchmark_trend = np.linspace(0, 0.0001, len(dates))
    benchmark_returns += benchmark_trend
    
    # Create series
    strategy_returns = pd.Series(daily_returns, index=dates)
    benchmark_returns = pd.Series(benchmark_returns, index=dates)
    
    # Generate sample diagnostics data
    diagnostics_data = []
    for i in range(0, len(dates), 30):  # Monthly rebalancing
        date = dates[i]
        allocation = 1.0 if np.random.random() > 0.2 else np.random.choice([0.2, 0.4, 0.6, 0.8])
        drawdown_status = 'normal' if allocation == 1.0 else 'reduced'
        
        diagnostics_data.append({
            'date': date,
            'allocation': allocation,
            'drawdown_status': drawdown_status
        })
    
    diagnostics_df = pd.DataFrame(diagnostics_data)
    
    # Generate sample cash allocations
    cash_data = []
    for i in range(0, len(dates), 30):  # Monthly rebalancing
        date = dates[i]
        cash_percentage = np.random.uniform(5, 25)  # 5-25% cash allocation
        
        cash_data.append({
            'date': date,
            'cash_percentage': cash_percentage
        })
    
    cash_allocations_df = pd.DataFrame(cash_data)
    
    return strategy_returns, benchmark_returns, diagnostics_df, cash_allocations_df

def generate_comprehensive_tearsheet_with_cash_allocation(strategy_returns: pd.Series, benchmark_returns: pd.Series, 
                                                        diagnostics: pd.DataFrame, cash_allocations: pd.DataFrame, 
                                                        title: str):
    """Generates comprehensive institutional tearsheet with equity curve and cash allocation chart."""
    
    # Align benchmark for plotting & metrics
    first_trade_date = strategy_returns.loc[strategy_returns.ne(0)].index.min()
    aligned_strategy_returns = strategy_returns.loc[first_trade_date:]
    aligned_benchmark_returns = benchmark_returns.loc[first_trade_date:]

    strategy_metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns, benchmark_returns)
    
    fig = plt.figure(figsize=(18, 30))  # Increased height for cash allocation chart
    gs = fig.add_gridspec(6, 2, height_ratios=[1.2, 0.8, 0.8, 0.8, 0.8, 1.2], hspace=0.7, wspace=0.2)
    fig.suptitle(title, fontsize=20, fontweight='bold', color='#2C3E50')

    # 1. Cumulative Performance (Equity Curve) with F-Score Integration
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot the main equity curves
    (1 + aligned_strategy_returns).cumprod().plot(ax=ax1, label='QVM Engine v3 (F-Score)', color='#16A085', lw=2.5)
    (1 + aligned_benchmark_returns).cumprod().plot(ax=ax1, label='VN-Index (Aligned)', color='#34495E', linestyle='--', lw=2)
    
    # Add drawdown protection shading
    if not diagnostics.empty and 'drawdown_status' in diagnostics.columns:
        # Convert diagnostics dates to datetime if they're not already
        diagnostics_copy = diagnostics.copy()
        if not pd.api.types.is_datetime64_any_dtype(diagnostics_copy.index):
            diagnostics_copy.index = pd.to_datetime(diagnostics_copy.index)
        
        # Get drawdown data aligned with the returns
        dd_data = diagnostics_copy.reindex(aligned_strategy_returns.index, method='ffill')
        
        # Shade periods with reduced allocation (red with low alpha)
        reduced_allocation_periods = dd_data[dd_data['allocation'] < 1.0]
        if not reduced_allocation_periods.empty:
            for i, date in enumerate(reduced_allocation_periods.index):
                if i == 0 or (date - reduced_allocation_periods.index[i-1]).days > 1:
                    # Start of a new reduced allocation period
                    start_date = date
                    # Find the end of this reduced allocation period
                    end_date = date
                    for j in range(i+1, len(reduced_allocation_periods.index)):
                        if (reduced_allocation_periods.index[j] - reduced_allocation_periods.index[j-1]).days == 1:
                            end_date = reduced_allocation_periods.index[j]
                        else:
                            break
                    allocation = reduced_allocation_periods.loc[date, 'allocation']
                    ax1.axvspan(start_date, end_date, alpha=0.1, color='red', 
                               label=f'Reduced Allocation ({allocation:.0%})' if i == 0 else "")
        
        print(f"   📊 Drawdown protection shading applied")
    else:
        print("   📊 No drawdown data available for shading")
    
    ax1.set_title('Cumulative Performance (Log Scale)', fontweight='bold')
    ax1.set_ylabel('Growth of 1 VND')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 2. Cash Allocation Chart (NEW - below equity curve)
    ax2 = fig.add_subplot(gs[1, :])
    if not cash_allocations.empty:
        # Convert dates to datetime for plotting
        cash_allocations['date'] = pd.to_datetime(cash_allocations['date'])
        cash_allocations = cash_allocations.sort_values('date')
        
        # Plot cash allocation percentage over time
        ax2.plot(cash_allocations['date'], cash_allocations['cash_percentage'], 
                color='#E74C3C', linewidth=2, marker='o', markersize=4)
        ax2.fill_between(cash_allocations['date'], cash_allocations['cash_percentage'], 
                        alpha=0.3, color='#E74C3C')
        
        # Add horizontal lines for reference
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20% Cash')
        ax2.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='40% Cash')
        
        ax2.set_title('Cash Allocation Over Time (Actual Allocation)', fontweight='bold')
        ax2.set_ylabel('Cash Allocation (%)')
        ax2.set_ylim(0, max(cash_allocations['cash_percentage'].max() * 1.1, 50))
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        print(f"   📊 Cash allocation chart created - showing actual allocation values")
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
    strat_annual = aligned_strategy_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
    bench_annual = aligned_benchmark_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
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

    # 6. Allocation Distribution (Drawdown Protection)
    ax6 = fig.add_subplot(gs[4, 0])
    if not diagnostics.empty and 'allocation' in diagnostics.columns:
        allocation_counts = diagnostics['allocation'].value_counts().sort_index()
        allocation_counts.plot(kind='bar', ax=ax6, color=['#27AE60', '#E74C3C', '#F39C12', '#3498DB', '#9B59B6', '#E67E22'])
        ax6.set_title('Drawdown Protection Allocation Distribution', fontweight='bold')
        ax6.set_ylabel('Number of Rebalances')
        ax6.set_xlabel('Allocation Level')
        ax6.grid(True, axis='y', linestyle='--', alpha=0.5)
    else:
        ax6.text(0.5, 0.5, 'No Allocation Data Available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)
        ax6.set_title('Allocation Distribution', fontweight='bold')

    # 7. Cash Allocation Distribution
    ax7 = fig.add_subplot(gs[4, 1])
    if not cash_allocations.empty:
        # Create cash allocation bins
        cash_bins = [0, 10, 20, 30, 40, 50, 100]
        cash_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']
        cash_allocations['cash_bin'] = pd.cut(cash_allocations['cash_percentage'], bins=cash_bins, labels=cash_labels)
        cash_distribution = cash_allocations['cash_bin'].value_counts().sort_index()
        
        cash_distribution.plot(kind='bar', ax=ax7, color='#E74C3C')
        ax7.set_title('Cash Allocation Distribution', fontweight='bold')
        ax7.set_ylabel('Number of Rebalances')
        ax7.set_xlabel('Cash Allocation Range')
        ax7.grid(True, axis='y', linestyle='--', alpha=0.5)
    else:
        ax7.text(0.5, 0.5, 'No Cash Allocation Data Available', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=14)
        ax7.set_title('Cash Allocation Distribution', fontweight='bold')

    # 8. Performance Metrics Table
    ax8 = fig.add_subplot(gs[5:, :])
    ax8.axis('off')
    summary_data = [['Metric', 'Strategy', 'Benchmark']]
    for key in strategy_metrics.keys():
        if key not in ['avg_cash_allocation', 'fscore_effective_weight']:  # Exclude F-Score specific metrics
            summary_data.append([key, f"{strategy_metrics[key]:.2f}", f"{benchmark_metrics.get(key, 0.0):.2f}"])
    
    # Add F-Score specific metrics
    if 'avg_cash_allocation' in strategy_metrics:
        summary_data.append(['Avg Cash Allocation (%)', f"{strategy_metrics['avg_cash_allocation']:.1f}", "N/A"])
    if 'fscore_effective_weight' in strategy_metrics:
        summary_data.append(['F-Score Effective Weight', f"{strategy_metrics['fscore_effective_weight']:.1%}", "N/A"])
    
    # Add drawdown protection metrics
    if 'avg_allocation' in strategy_metrics:
        summary_data.append(['Avg Allocation', f"{strategy_metrics['avg_allocation']:.1%}", "N/A"])
    if 'allocation_volatility' in strategy_metrics:
        summary_data.append(['Allocation Volatility', f"{strategy_metrics['allocation_volatility']:.3f}", "N/A"])
    
    table = ax8.table(cellText=summary_data[1:], colLabels=summary_data[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# %% [markdown]
# # EXECUTION CELL

# %%
if __name__ == "__main__":
    main()
else:
    print("📚 QVM Engine V3 with F-Score Integration - Enhanced Demo loaded")
    print("   Run main() to execute the full demonstration")
    print("   Run demonstrate_tearsheet() to see the comprehensive tearsheet")
