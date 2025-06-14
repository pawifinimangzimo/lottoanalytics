#!/usr/bin/env python3
"""
COMPLETE LOTTERY OPTIMIZER WITH DASHBOARD
- SQLite storage
- Pandas analysis
- Self-contained HTML dashboard
"""

import sqlite3
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime
import argparse
from typing import Dict, List, Tuple
import logging

# ======================
# DEFAULT CONFIGURATION
# ======================
DEFAULT_CONFIG = {
    'data': {
        'db_path': 'data/lottery.db',
        'historical_csv': 'data/historical.csv',
        'results_dir': 'results/'
    },
    'analysis': {
        'hot_days': 30,
        'cold_threshold': 60,
        'top_range': 10,  # Now using top_range instead of top_n_results
        'min_display_matches': 1,
        'combination_analysis': {
            'pairs': True,
            'triplets': True,
            'quadruplets': False,
            'quintuplets': False,
            'sixtuplets': False
        }
    },
    'strategy': {
        'numbers_to_select': 6,
        'number_pool': 55
    }
}

# ======================
# CORE ANALYZER CLASS
# ======================
class LotteryAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self._validate_config() 
        # Add validation
        if 'sets_to_generate' not in self.config['output']:
            self.config['output']['sets_to_generate'] = 4
        # Rest of your init code
        #number pool initialization
        self.number_pool = list(range(1, config['strategy']['number_pool'] + 1))
        self.weights = pd.Series(1.0, index=self.number_pool) 
        #number pool initialization end 
        #mode handler 
        self._init_mode_handler()  # Add this line
        self.conn = self._init_db()
        self._prepare_filesystem()

    # ======================
    # NEW CONFIG VALIDATION
    # ======================
    def _validate_config(self):
        """Validate combination analysis config"""
        if 'combination_analysis' not in self.config['analysis']:
            return
            
        valid_sizes = {'pairs', 'triplets', 'quadruplets', 'quintuplets', 'sixtuplets'}
        invalid = [
            size for size in self.config['analysis']['combination_analysis']
            if size not in valid_sizes
        ]
        
        if invalid:
            raise ValueError(
                f"Invalid combination_analysis keys: {invalid}. "
                f"Valid options are: {valid_sizes}"
            )

    def _init_db(self) -> sqlite3.Connection:
        """Initialize SQLite database with optimized schema"""
        Path(self.config['data']['db_path']).parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(self.config['data']['db_path'])
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS draws (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                n1 INTEGER, n2 INTEGER, n3 INTEGER,
                n4 INTEGER, n5 INTEGER, n6 INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON draws(date)")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_numbers 
            ON draws(n1,n2,n3,n4,n5,n6)
        """)
        return conn

    def _prepare_filesystem(self) -> None:
        """Ensure required directories exist"""
        Path(self.config['data']['results_dir']).mkdir(exist_ok=True)

    def load_data(self) -> None:
        """Load CSV data into SQLite with validation"""
        try:
            df = pd.read_csv(
                self.config['data']['historical_csv'],
                header=None,
                names=['date', 'numbers'],
                parse_dates=['date']
            )
            
            # Split numbers into columns
            nums = df['numbers'].str.split('-', expand=True)
            for i in range(self.config['strategy']['numbers_to_select']):
                df[f'n{i+1}'] = nums[i].astype(int)
            
            # Validate number ranges
            pool_size = self.config['strategy']['number_pool']
            for i in range(1, self.config['strategy']['numbers_to_select'] + 1):
                invalid = df[(df[f'n{i}'] < 1) | (df[f'n{i}'] > pool_size)]
                if not invalid.empty:
                    raise ValueError(f"Invalid numbers in n{i} (range 1-{pool_size})")
            
            # Store in SQLite
            df[['date'] + [f'n{i}' for i in range(1, 7)]].to_sql(
                'draws', self.conn, if_exists='replace', index=False
            )
            
        except Exception as e:
            raise ValueError(f"Data loading failed: {str(e)}")
####################
    def get_frequencies(self, count: int = None) -> pd.Series:
        """Get number frequencies using optimized SQL query"""
        top_n = count or self.config['analysis']['top_range']
        query = """
            WITH nums AS (
                SELECT n1 AS num FROM draws UNION ALL
                SELECT n2 FROM draws UNION ALL
                SELECT n3 FROM draws UNION ALL
                SELECT n4 FROM draws UNION ALL
                SELECT n5 FROM draws UNION ALL
                SELECT n6 FROM draws
            )
            SELECT num, COUNT(*) as frequency 
            FROM nums 
            GROUP BY num 
            ORDER BY frequency DESC
            LIMIT ?
        """
        result = pd.read_sql(query, self.conn, params=(top_n,))
        
        # Return empty Series with correct structure if no results
        if result.empty:
            return pd.Series(dtype=float, name='frequency')
            
        return result.set_index('num')['frequency']
##############################
# ======================
# COMBINATION ANALYSIS 
# ======================

    def get_combinations(self, size: int = 2, verbose: bool = True) -> pd.DataFrame:
        """Get frequency of number combinations with proper SQL ordering.
        Args:
            size: 2 for pairs, 3 for triplets, etc. (default=2)
            verbose: Whether to print status messages (default=True)
        Returns:
            DataFrame with columns [nX, nY, ..., frequency]
        """
        # ====== CONFIG VALIDATION ======
        combo_type = {2: 'pairs', 3: 'triplets', 4: 'quadruplets', 
                      5: 'quintuplets', 6: 'sixtuplets'}.get(size)
        if not combo_type:
            if verbose:
                print(f"⚠️  Invalid combination size: {size} (must be 2-6)")
            return pd.DataFrame()
        
        if not hasattr(self, 'config'):
            if verbose:
                print("⚠️  Config not loaded - combination analysis unavailable")
            return pd.DataFrame()
        
        if not self.config['analysis']['combination_analysis'].get(combo_type, False):
            if verbose:
                print(f"ℹ️  {combo_type.capitalize()} analysis disabled in config")
            return pd.DataFrame()

        # ====== PARAMETERS ======
        top_n = self.config['analysis']['top_range']
        min_count = self.config['analysis'].get('min_combination_count', 2)  # Default to 2 if missing
        cols = [f'n{i}' for i in range(1, self.config['strategy']['numbers_to_select'] + 1)]
        
        if verbose:
            print(f"🔍 Analyzing {combo_type} (min {min_count} appearances)...", end=' ', flush=True)
        # ====== QUERY GENERATION ======
        queries = []
        for combo in combinations(cols, size):
            select_cols = ', '.join(combo)
            queries.append(f"""
                SELECT {select_cols}, COUNT(*) as frequency
                FROM draws
                GROUP BY {select_cols}
                HAVING frequency >= {min_count}  
            """)
        
        full_query = " UNION ALL ".join(queries)
        full_query += f"\nORDER BY frequency DESC\nLIMIT {top_n}"

        try:
            result = pd.read_sql(full_query, self.conn)
            if verbose:
                print(f"found {len(result)} combinations")
            return result
        except sqlite3.Error as e:
            if verbose:
                print("failed")
            raise RuntimeError(f"SQL query failed: {str(e)}")

#=======================

    def get_temperature_stats(self) -> Dict[str, List[int]]:
        """Classify numbers as hot/cold using SQL"""
        hot_query = f"""
            SELECT n1 as num FROM draws 
            WHERE date >= date('now', '-{self.config['analysis']['hot_days']} days')
            GROUP BY n1 ORDER BY COUNT(*) DESC LIMIT ?
        """
        cold_query = f"""
            SELECT DISTINCT n1 as num FROM draws
            WHERE n1 NOT IN (
                SELECT DISTINCT n1 FROM draws
                WHERE date >= date('now', '-{self.config['analysis']['cold_threshold']} days')
            ) LIMIT ?
        """
        
        hot = pd.read_sql(hot_query, self.conn, 
                         params=(self.config['analysis']['top_range'],))
        cold = pd.read_sql(cold_query, self.conn,
                          params=(self.config['analysis']['top_range'],))
        
        return {
            'hot': hot['num'].tolist(),
            'cold': cold['num'].tolist()
        }
#======================
# Start Set generator
#======================

    def generate_sets(self, strategy: str = None) -> List[List[int]]:
        """Generate sets using current mode"""
        strategy = strategy or self.config.get('strategy', {}).get('default_strategy', 'balanced')
        num_sets = self.config['output'].get('sets_to_generate', 4)
        
        sets = []
        for _ in range(num_sets):
            if self.mode == 'auto':
                # Auto-mode generates fresh weights each time
                self._init_weights()
            if strategy == 'balanced':
                hot = self.get_temperature_stats()['hot'][:3]
                cold = self.get_temperature_stats()['cold'][:2]
                remaining = self.config['strategy']['numbers_to_select'] - len(hot) - len(cold)
                random_nums = np.random.choice(
                    [n for n in self.number_pool if n not in hot + cold],
                    size=remaining,
                    replace=False
                )
                sets.append(sorted(hot + cold + random_nums.tolist()))
            
            elif strategy == 'frequent':
                top_n = self.config['strategy']['numbers_to_select']
                freqs = self.get_frequencies()
                sets.append(freqs.head(top_n).index.tolist())
        
        return sets

#===================
# end set generator
#===================
    def save_results(self, sets: List[List[int]]) -> str:
        """Save generated sets to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.config['data']['results_dir']) / f"sets_{timestamp}.csv"
        
        pd.DataFrame({
            'numbers': ['-'.join(map(str, s)) for s in sets],
            'generated_at': datetime.now()
        }).to_csv(path, index=False)
        
        return str(path)
#==============================
# Mode Handler 
#===============================

    def _init_mode_handler(self):
        """Initialize the mode handling system"""
        self.mode = self.config.get('mode', 'auto')
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights based on current mode"""
        if self.mode == 'auto':
            # Auto-mode defaults
            self.weights = pd.Series(1.0, index=self.number_pool)
            self.learning_rate = self.config.get('auto', {}).get('learning_rate', 0.01)
            self.decay_factor = self.config.get('auto', {}).get('decay_factor', 0.97)
        else:
            # Manual mode weights from config
            weights_config = self.config.get('manual', {}).get('strategy', {}).get('weighting', {})
            self.weights = pd.Series(
                weights_config.get('frequency', 0.4) * self._get_frequency_weights() +
                weights_config.get('recency', 0.3) * self._get_recent_weights() +
                weights_config.get('randomness', 0.3) * np.random.rand(len(self.number_pool))
            )
            
            # Apply cold number bonus
            cold_bonus = weights_config.get('resurgence', 0.1)
            cold_nums = self.get_temperature_stats()['cold']
            self.weights[cold_nums] *= (1 + cold_bonus)
            
            self.weights /= self.weights.sum()
    
    def set_mode(self, mode: str):
        """Change modes dynamically"""
        valid_modes = ['auto', 'manual']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Choose from: {valid_modes}")
        self.mode = mode
        self._init_weights()


#==============================
#End mode handler
#=============================

    def _is_prime(self, n: int) -> bool:
        """Helper method to check if a number is prime"""
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def detect_patterns(self) -> Dict:
        """Analyze historical draws for common number patterns.
        Returns: {
            'consecutive': float (percentage),
            'same_ending': float,
            'all_even_odd': float,
            'avg_primes': float,
            'prime_count': list[int]
        }
        """
        # Feature gate check
        if not self.config.get('features', {}).get('enable_pattern_analysis', False):
            return {}

        try:
            # Get last 100 draws (configurable amount)
            limit = self.config.get('pattern_settings', {}).get('sample_size', 100)
            query = f"""
                SELECT n1, n2, n3, n4, n5, n6 FROM draws
                ORDER BY date DESC
                LIMIT {limit}
            """
            recent = pd.read_sql(query, self.conn)
            
            # Initialize counters
            patterns = {
                'consecutive': 0,
                'same_ending': 0,
                'all_even_odd': 0,
                'prime_count': []
            }

            for _, row in recent.iterrows():
                nums = sorted(row.tolist())
                diffs = [nums[i+1] - nums[i] for i in range(5)]
                
                # Check for consecutive numbers
                if any(d == 1 for d in diffs):
                    patterns['consecutive'] += 1
                    
                # Check for same last digits
                last_digits = [n % 10 for n in nums]
                if len(set(last_digits)) < 3:  # At least 3 numbers share digit
                    patterns['same_ending'] += 1
                    
                # Check all even or all odd
                if all(n % 2 == 0 for n in nums) or all(n % 2 == 1 for n in nums):
                    patterns['all_even_odd'] += 1
                    
                # Count prime numbers
                primes = [n for n in nums if self._is_prime(n)]
                patterns['prime_count'].append(len(primes))
            
            # Convert to percentages
            total_draws = len(recent)
            if total_draws > 0:
                patterns['consecutive'] = (patterns['consecutive'] / total_draws) * 100
                patterns['same_ending'] = (patterns['same_ending'] / total_draws) * 100
                patterns['all_even_odd'] = (patterns['all_even_odd'] / total_draws) * 100
                patterns['avg_primes'] = np.mean(patterns['prime_count']) if patterns['prime_count'] else 0
                
            return patterns

        except Exception as e:
            logging.warning(f"Pattern detection failed: {str(e)}")
            return {}
######

    def _get_prime_subsets(self, numbers: List[int]) -> List[int]:
        """Extract primes from any number list."""
        return [n for n in numbers if self._is_prime(n)]

    def _tag_prime_combos(self, combos: pd.DataFrame, size: int) -> pd.DataFrame:
        """Add '[All Primes]' tag to combos where all numbers are prime."""
        primes = set(self._get_prime_numbers())
        combos['is_prime_combo'] = combos[
            [f'n{i}' for i in range(1, size+1)]
        ].apply(lambda row: all(n in primes for n in row), axis=1)
        return combos

########################

    def get_combination_stats(self, size: int) -> Dict:
        """Get statistics for number combinations of given size.
        Returns: {
            'average_frequency': float,
            'most_common': {'numbers': list, 'count': int},
            'std_deviation': float,
            'coverage_pct': float,
            'top_co_occurring': list(tuple)
        }
        """
        if not self.config.get('features', {}).get('enable_combo_stats', False):
            return {}

        try:
            combos = self.get_combinations(size, verbose=False)
            if combos.empty:
                return {}

            # Calculate co-occurrence first (existing code)
            co_occurrence = defaultdict(int)
            for _, row in combos.iterrows():
                for i in range(1, size+1):
                    num = row[f'n{i}']
                    co_occurrence[num] += 1

            # Get the most common combination (modified section)
            most_common_row = combos.iloc[0]
            most_common_numbers = [most_common_row[f'n{i}'] for i in range(1, size+1)]

            # New return format (replace the existing stats dictionary)
            return {
                'average_frequency': combos['frequency'].mean(),
                'most_common': {
                    'numbers': most_common_numbers,
                    'count': most_common_row['frequency']
                },
                'std_deviation': combos['frequency'].std(),
                'coverage_pct': (len(combos) / len(list(combinations(self.number_pool, size)))) * 100,
                'top_co_occurring': sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:5]
            }

        except Exception as e:
            logging.warning(f"Combination stats failed for size {size}: {str(e)}")
            return {}

############################# New Added Section ###########################

    def get_combined_stats(self):
        """Calculate cross-category statistical relationships"""
        if not self.config['analysis'].get('show_combined_stats', False):
            return None
            
        return {
            'hot_frequent': self._get_hot_frequent_overlap(),
            'pattern_corr': self._get_pattern_correlations(),
            'coverage': self._get_coverage_stats()
        }
###########################
    def _get_hot_frequent_overlap(self):
        """Calculate overlap between hot and frequent numbers"""
        try:
            # Get hot numbers
            temp_stats = self.get_temperature_stats()
            hot_nums = set(temp_stats.get('hot', []))
            
            # Get frequent numbers
            freq_series = self.get_frequencies(20)  # Get top 20 frequent numbers
            
            # Handle empty cases
            if not hot_nums or freq_series.empty:
                return {'overlap_pct': 0, 'freq_multiplier': 0}
                
            freq_nums = set(freq_series.index.tolist())
            overlap = hot_nums.intersection(freq_nums)
            
            hot_count = len(hot_nums) or 1  # Prevent division by zero
            freq_mean = freq_series.mean()
            
            # Calculate overlap statistics
            if overlap:
                hot_freq_mean = freq_series.loc[list(overlap)].mean()
                multiplier = round(hot_freq_mean/freq_mean, 1) if freq_mean else 0
            else:
                multiplier = 0
                
            return {
                'overlap_pct': round(len(overlap)/hot_count*100, 1),
                'freq_multiplier': multiplier
            }
        except Exception as e:
            logging.warning(f"Hot-frequent analysis failed: {str(e)}")
            return {'overlap_pct': 0, 'freq_multiplier': 0}
######################################
    def _get_pattern_correlations(self):
        """Calculate pattern relationships"""
        # Implement your pattern correlation logic here
        return {
            'hot_freq_pair_rate': 62,  # Example value
            'cold_pair_reduction': 28   # Example value
        }

    def _get_coverage_stats(self):
        """Calculate coverage statistics"""
        # Implement your coverage analysis here
        return {
            'pattern_coverage': 82,  # Example value
            'never_paired_pct': 41   # Example value
        }

###########################################################################

########################

# ======================
# DASHBOARD GENERATOR
# ======================

class DashboardGenerator:
    def __init__(self, analyzer: LotteryAnalyzer):
        self.analyzer = analyzer
        self.dashboard_dir = Path(analyzer.config['data']['results_dir']) / "dashboard"
        self.dashboard_dir.mkdir(exist_ok=True)

    # ======================
    # NEW SAFE PARSER METHOD
    # ======================
    def _parse_combo_size(self, size_name: str) -> int:
        """Convert 'triplets' → 3 with validation"""
        size_map = {
            'pairs': 2,
            'triplets': 3,
            'quadruplets': 4,
            'quintuplets': 5,
            'sixtuplets': 6
        }
        size_name = size_name.lower().strip()
        if size_name not in size_map:
            raise ValueError(
                f"Invalid combo size '{size_name}'. "
                f"Must be one of: {list(size_map.keys())}"
            )
        return size_map[size_name]

    def _generate_number_card(self, title: str, numbers: list, color_class: str) -> str:
        """Generate a card with number bubbles"""
        numbers_html = "".join(
            f'<div class="number-bubble {color_class}">{num}</div>'
            for num in numbers[:15]  # Show up to 15 numbers
        )
        return f"""
        <div class="analysis-card">
            <h3>{title}</h3>
            <div class="number-grid">{numbers_html}</div>
        </div>
        """
#==============
#New Chart 
#==============

    def _generate_combination_chart(self, size: int) -> str:
        """Generate combination frequency chart"""
        top_n = self.analyzer.config['analysis']['top_range']
        combos = self.analyzer.get_combinations(size)
        
        labels = [f"{'-'.join(map(str, row[:-1]))}" for _, row in combos.iterrows()]
        counts = combos['frequency'].tolist()
        
        return f"""
        <div class="chart-card">
            <h3>Top {top_n} {size}-Number Combinations</h3>
            <div class="chart-container">
                <canvas id="comboChart{size}"></canvas>
            </div>
            <div class="chart-data" hidden>
                {json.dumps({"combinations": labels, "counts": counts})}
            </div>
        </div>
        """

#===================
    def _generate_frequency_chart(self, frequencies: pd.Series) -> str:
        """Generate the frequency chart HTML"""
        top_n = self.analyzer.config['analysis']['top_range']  # USE CONFIG VALUE
        top_numbers = frequencies.head(top_n).index.tolist()
        counts = frequencies.head(top_n).values.tolist()
        
        return f"""
        <div class="chart-card">
            <h3>Top {top_n} Frequent Numbers</h3>  <!-- DYNAMIC TITLE -->
            <div class="chart-container">
                <canvas id="frequencyChart"></canvas>
            </div>
            <div class="chart-data" hidden>
                {json.dumps({"numbers": top_numbers, "counts": counts})}
            </div>
        </div>
        """

    def _generate_recent_draws(self, count: int = 5) -> str:
        """Show recent draws"""
        recent = pd.read_sql(
            f"SELECT * FROM draws ORDER BY date DESC LIMIT {count}",
            self.analyzer.conn
        )
        rows = "".join(
            f"<tr><td>{row['date']}</td><td>{'-'.join(str(row[f'n{i}']) for i in range(1,7))}</td></tr>"
            for _, row in recent.iterrows()
        )
        return f"""
        <div class="recent-card">
            <h3>Last {count} Draws</h3>
            <table>
                <tr><th>Date</th><th>Numbers</th></tr>
                {rows}
            </table>
        </div>
        """

    def generate(self) -> str:
        """Generate complete dashboard"""
        # Get analysis data
        freqs = self.analyzer.get_frequencies()
        temps = self.analyzer.get_temperature_stats()
        
        # Generate HTML components
        cards = [
            self._generate_number_card("Hot Numbers", temps['hot'], 'hot'),
            self._generate_number_card("Cold Numbers", temps['cold'], 'cold'),
            self._generate_number_card("Frequent Numbers", freqs.index.tolist(), 'frequent'),
            self._generate_frequency_chart(freqs),
            self._generate_recent_draws()
        ]
        # ADD THIS NEW BLOCK FOR COMBINATION CHARTS
        combo_config = self.analyzer.config['analysis'].get('combination_analysis', {})
        for size_name, enabled in combo_config.items():
            try:
                if enabled:
                    size_num = self._parse_combo_size(size_name)  # Use the safe parser
                    cards.append(self._generate_combination_chart(size_num))
            except (ValueError, KeyError) as e:
                import logging
                logging.warning(f"Skipping invalid combo size '{size_name}': {str(e)}")
                
                
        # Complete HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lottery Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                }}
                .analysis-card, .chart-card, .recent-card {{
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 15px;
                }}
                .number-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(30px, 1fr));
                    gap: 8px;
                    margin-top: 10px;
                }}
                .number-bubble {{
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                }}
                .hot {{ background-color: #ff6b6b; color: white; }}
                .cold {{ background-color: #74b9ff; color: white; }}
                .frequent {{ background-color: #2ecc71; color: white; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                .chart-container {{ position: relative; height: 300px; margin-top: 15px; }}
                h3 {{ margin-top: 0; color: #2c3e50; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>Lottery Analysis Dashboard</h1>
            <div class="dashboard">
                {"".join(cards)}
            </div>
            <script>
                // Initialize frequency chart
                const freqData = JSON.parse(
                    document.querySelector('.chart-data').innerHTML
                );
                new Chart(
                    document.getElementById('frequencyChart'),
                    {{
                        type: 'bar',
                        data: {{
                            labels: freqData.numbers,
                            datasets: [{{
                                label: 'Appearances',
                                data: freqData.counts,
                                backgroundColor: '#2ecc71'
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false
                        }}
                    }}
                );
            </script>
        </body>
        </html>
        """
        
        # Save to file
        (self.dashboard_dir / "index.html").write_text(html)
        return str(self.dashboard_dir / "index.html")

# ======================
# MAIN APPLICATION
# ======================
def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load YAML config with defaults"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Deep merge
        def merge(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    merge(d1[k], v)
                else:
                    d1[k] = v
        merged = DEFAULT_CONFIG.copy()
        merge(merged, config)
        return merged
    except Exception:
        return DEFAULT_CONFIG

def main():

    parser = argparse.ArgumentParser(description='Lottery Number Optimizer')
    parser.add_argument('--mode', choices=['auto', 'manual'], 
                       help='Override config mode setting')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--strategy', default='balanced', choices=['balanced', 'frequent'])
    parser.add_argument('--no-dashboard', action='store_true', help='Disable dashboard generation')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')

    # ============ INSERT NEW ARGUMENTS HERE ============
    parser.add_argument('--show-combos', nargs='+', 
                       choices=['pairs', 'triplets', 'quadruplets', 'quintuplets', 'sixtuplets'],
                       help="Override config to show specific combinations (e.g., --show-combos pairs triplets)")
    parser.add_argument('--hide-combos', nargs='+',
                       choices=['pairs', 'triplets', 'quadruplets', 'quintuplets', 'sixtuplets'],
                       help="Override config to hide specific combinations")
    parser.add_argument('--show-patterns', action='store_true',
                       help='Enable pattern detection analysis')
    parser.add_argument('--show-stats', action='store_true',
                       help='Enable combination statistics')
    # ============ END OF NEW ARGUMENTS ============

    args = parser.parse_args()



    
    try:
        # Initialize analyzer
        config = load_config(args.config)
#================
# New section
#================

        # Apply CLI overrides to combination analysis
        if args.show_combos or args.hide_combos:
            # Initialize if missing
            config['analysis']['combination_analysis'] = config.get('analysis', {}).get('combination_analysis', {})
            
            # Set all to False first if --show-combos is used (to enforce exclusivity)
            if args.show_combos:
                for combo in ['pairs', 'triplets', 'quadruplets', 'quintuplets', 'sixtuplets']:
                    config['analysis']['combination_analysis'][combo] = False
            
            # Apply CLI selections
            for combo in (args.show_combos or []):
                config['analysis']['combination_analysis'][combo] = True
            for combo in (args.hide_combos or []):
                config['analysis']['combination_analysis'][combo] = False

#================
        
        analyzer = LotteryAnalyzer(config)
        # Load and validate data
        analyzer.load_data()

        # Get analysis results
        freqs = analyzer.get_frequencies()
        temps = analyzer.get_temperature_stats()
        sets = analyzer.generate_sets(args.strategy)
        top_range = analyzer.config['analysis']['top_range']
        
        # Console output (unless --quiet)
        
        # ============ INSERT FEATURE RESULTS INIT HERE ============
        # Initialize feature_results dictionary
        feature_results = {
            'patterns': analyzer.detect_patterns() if (args.show_patterns or 
                     config.get('features', {}).get('enable_pattern_analysis', False)) else None,
            'stats': {
                2: analyzer.get_combination_stats(2) if (args.show_stats or 
                    config.get('features', {}).get('enable_combo_stats', False)) else None,
                3: analyzer.get_combination_stats(3) if (args.show_stats or 
                    config.get('features', {}).get('enable_combo_stats', False)) else None
            }
        }
        # ============ END FEATURE RESULTS INIT ============
        hot_primes = analyzer._get_prime_subsets(temp_stats['hot'])
        cold_primes = analyzer._get_prime_subsets(temp_stats['cold'])
            
        if not args.quiet:
            print("\n" + "="*50)
            print(" LOTTERY ANALYSIS RESULTS ".center(50, "="))
            print(f"\n🔢 Top {top_range} Frequent Numbers:")
            print(freqs.to_string())
            
            print("\n🔥 Hot Numbers (last {} days):".format(
                config['analysis']['hot_days']))
            print(", ".join(map(str, temps['hot'])))
            print(f"   • Primes: {', '.join(map(str, hot_primes)) or 'None'}")
            
            print("\n❄️ Cold Numbers (not seen in {} days):".format(
                config['analysis']['cold_threshold']))
            print(", ".join(map(str, temps['cold'])))
            print(f"   • Primes: {', '.join(map(str, cold_primes)) or 'None'}")

#==================
# New Section
#==================
            print("\n🔢 Top Combinations:")
            combo_config = analyzer.config['analysis']['combination_analysis']
            
            for size, size_name in [(2, 'pairs'), (3, 'triplets'), 
                                 (4, 'quadruplets'), (5, 'quintuplets'), 
                                 (6, 'sixtuplets')]:
                if combo_config.get(size_name, False):
                    combos = analyzer.get_combinations(size)
                    if not combos.empty:
                        print(f"\nTop {len(combos)} {size_name}:")
                        for _, row in combos.iterrows():
                            nums = [str(row[f'n{i}']) for i in range(1, size+1)]
                            print(f"- {'-'.join(nums)} (appeared {row['frequency']} times)")
                        combos = analyzer._tag_prime_combos(combos, size)
                        for _, row in combos.head(3).iterrows():
                            nums = [str(row[f'n{i}']) for i in range(1, size+1)]
                            prime_tag = " [All Primes]" if row['is_prime_combo'] else ""
                            print(f"- {'-'.join(nums)} (appeared {row['frequency']} times){prime_tag}")

            # ============ INSERT NEW FEATURE OUTPUTS HERE ============
            if feature_results['patterns']:
                print("\n" + "="*50)
                print(" NUMBER PATTERNS ".center(50, "="))
                p = feature_results['patterns']
                print(f"Consecutive numbers: {p['consecutive']:.1f}%")
                print(f"Same last digit: {p['same_ending']:.1f}%")
                print(f"All even/odd: {p['all_even_odd']:.1f}%")
                print(f"Avg primes: {p['avg_primes']:.1f}")

            if feature_results['stats'][2] or feature_results['stats'][3]:
                print("\n" + "="*50)
                print(" COMBINATION STATISTICS ".center(50, "="))
                for size in [2, 3]:
                    if feature_results['stats'][size]:
                        stats = feature_results['stats'][size]
                        print(f"\n▶ {size}-Number Combinations:")
                        print(f"  Average appearances: {stats['average_frequency']:.1f}")
                        print(f"  Most frequent: {'-'.join(map(str, stats['most_common']['numbers']))} "
                            f"(appeared {stats['most_common']['count']} times)")
            # ============ END NEW OUTPUTS ============
            if config['analysis'].get('show_combined_stats', False):
                combined = analyzer.get_combined_stats()
                if combined:
                    print("\n" + "="*50)
                    print(" COMBINED STATISTICAL INSIGHTS ".center(50, "="))
                    
                    hf = combined['hot_frequent']
                    print(f"\n● Hot & Frequent Numbers:")
                    print(f"   - {hf['overlap_pct']}% of hot numbers are also top frequent")
                    print(f"   - Appear {hf['freq_multiplier']}x more often than average")
#==================
            print("\n🎰 Recommended Number Sets:")
            for i, nums in enumerate(sets, 1):
                print(f"Set {i}: {'-'.join(map(str, nums))}")
            print("\n" + "="*50)
        # Save files
        results_path = analyzer.save_results(sets)
        if not args.quiet:
            print(f"\n💾 Results saved to: {results_path}")
        
        # Generate dashboard (unless --no-dashboard)
        if not args.no_dashboard:
            dashboard = DashboardGenerator(analyzer)
            dashboard_path = dashboard.generate()
            if not args.quiet:
                print(f"🌐 Dashboard generated at: {dashboard_path}")
                print("   View with: python -m http.server --directory results/dashboard 8000")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print(f"1. Check {args.config} exists and is valid")
        print(f"2. Verify data/numbers are 1-{config.get('strategy',{}).get('number_pool',55)}")
        print("3. Ensure CSV format: date,n1-n2-n3-n4-n5-n6")

if __name__ == "__main__":
    main()