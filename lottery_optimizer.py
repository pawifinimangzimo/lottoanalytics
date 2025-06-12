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

    def get_frequencies(self, count: int = None) -> pd.Series:  # UPDATE METHOD SIGNATURE
        """Get number frequencies using optimized SQL query"""
        top_n = count or self.config['analysis']['top_range']  # USE CONFIG VALUE
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
        return pd.read_sql(query, self.conn, params=(top_n,))

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
        # ====== MODIFIED CHECK WITH MESSAGING ======
        combo_type = {2: 'pairs', 3: 'triplets', 4: 'quadruplets', 5: 'quintuplets', 6: 'sixtuplets'}.get(size)
        if not combo_type:
            if verbose:
                print(f"⚠️  Invalid combination size: {size} (must be 2-6)")
            return pd.DataFrame()
        
        if not hasattr(self, 'config') or not self.config['analysis']['combination_analysis'].get(combo_type, False):
            if verbose:
                print(f"ℹ️  {combo_type.capitalize()} analysis disabled in config")
            return pd.DataFrame()
        # ===========================================
        
        if not isinstance(size, int) or size < 2 or size > 6:
            raise ValueError("Combination size must be integer between 2-6")

        top_n = self.config['analysis']['top_range']
        cols = [f'n{i}' for i in range(1, self.config['strategy']['numbers_to_select'] + 1)]
        
        # Generate all possible column combinations
        combo_cols = list(combinations(cols, size))
        
        # Build individual queries
        queries = []
        for combo in combo_cols:
            select_cols = ', '.join(combo)
            group_cols = ', '.join(combo)
            queries.append(f"""
                SELECT {select_cols}, COUNT(*) as frequency
                FROM draws
                GROUP BY {group_cols}
            """)
        
        # Combine with single ORDER BY
        full_query = "\nUNION ALL\n".join(queries)
        full_query += f"\nORDER BY frequency DESC\nLIMIT {top_n}"
        
        # ====== ADDED EXECUTION FEEDBACK ======
        if verbose:
            print(f"🔍 Analyzing {combo_type}...", end=' ', flush=True)
        
        try:
            result = pd.read_sql(full_query, self.conn)
            if verbose:
                print(f"found {len(result)} combinations")
            return result
        except sqlite3.Error as e:
            if verbose:
                print("failed")
            raise RuntimeError(f"SQL query failed: {str(e)}")
        # ======================================

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
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        config = load_config(args.config)
        analyzer = LotteryAnalyzer(config)
        
        # Load and validate data
        analyzer.load_data()
        
        # Get analysis results
        freqs = analyzer.get_frequencies()
        temps = analyzer.get_temperature_stats()
        sets = analyzer.generate_sets(args.strategy)
        
        # Console output (unless --quiet)
        if not args.quiet:
            print("\n" + "="*50)
            print(" LOTTERY ANALYSIS RESULTS ".center(50, "="))
            print("\n🔢 Top 10 Frequent Numbers:")
            print(freqs.to_string())
            
            print("\n🔥 Hot Numbers (last {} days):".format(
                config['analysis']['hot_days']))
            print(", ".join(map(str, temps['hot'])))
            
            print("\n❄️ Cold Numbers (not seen in {} days):".format(
                config['analysis']['cold_threshold']))
            print(", ".join(map(str, temps['cold'])))
            
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