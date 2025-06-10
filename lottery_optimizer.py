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
        'top_n_results': 10
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
        self.conn = self._init_db()
        self._prepare_filesystem()

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

    def get_frequencies(self) -> pd.Series:
        """Get number frequencies using optimized SQL query"""
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
        return pd.read_sql(
            query, self.conn, 
            params=(self.config['analysis']['top_n_results'],)
        )

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
                         params=(self.config['analysis']['top_n_results'],))
        cold = pd.read_sql(cold_query, self.conn,
                          params=(self.config['analysis']['top_n_results'],))
        
        return {
            'hot': hot['num'].tolist(),
            'cold': cold['num'].tolist()
        }

    def generate_sets(self, strategy: str = 'balanced') -> List[List[int]]:
        """Generate number sets using configured strategy"""
        freqs = self.get_frequencies()
        temps = self.get_temperature_stats()
        
        if strategy == 'balanced':
            hot = temps['hot'][:3]
            cold = temps['cold'][:2]
            random = np.random.choice(
                range(1, self.config['strategy']['number_pool'] + 1),
                size=self.config['strategy']['numbers_to_select'] - len(hot) - len(cold),
                replace=False
            )
            return [sorted(hot + cold + random.tolist())]
        
        elif strategy == 'frequent':
            return [freqs.head(self.config['strategy']['numbers_to_select']).index.tolist()]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def save_results(self, sets: List[List[int]]) -> str:
        """Save generated sets to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.config['data']['results_dir']) / f"sets_{timestamp}.csv"
        
        pd.DataFrame({
            'numbers': ['-'.join(map(str, s)) for s in sets],
            'generated_at': datetime.now()
        }).to_csv(path, index=False)
        
        return str(path)

# ======================
# DASHBOARD GENERATOR
# ======================

class DashboardGenerator:
    def __init__(self, analyzer: LotteryAnalyzer):
        self.analyzer = analyzer
        self.dashboard_dir = Path(analyzer.config['data']['results_dir']) / "dashboard"
        self.dashboard_dir.mkdir(exist_ok=True)

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

    def _generate_frequency_chart(self, frequencies: pd.Series) -> str:
        """Generate the frequency chart HTML"""
        top_numbers = frequencies.head(10).index.tolist()
        counts = frequencies.head(10).values.tolist()
        
        return f"""
        <div class="chart-card">
            <h3>Top 10 Frequent Numbers</h3>
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
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--strategy', default='balanced', choices=['balanced', 'frequent'], 
                       help='Number generation strategy')
    parser.add_argument('--no-dashboard', action='store_true', 
                       help='Disable dashboard generation')
    args = parser.parse_args()
    
    try:
        # Initialize
        config = load_config(args.config)
        analyzer = LotteryAnalyzer(config)
        analyzer.load_data()
        
        # Generate number sets
        sets = analyzer.generate_sets(args.strategy)
        saved_path = analyzer.save_results(sets)
        print(f"📊 Results saved to: {saved_path}")
        
        # Dashboard generation (enabled by default)
        if not args.no_dashboard:
            dashboard = DashboardGenerator(analyzer)
            
            # Prepare dashboard data
            freqs = analyzer.get_frequencies().reset_index()
            data = {
                'frequencies': {
                    'num': freqs['num'].tolist(),
                    'frequency': freqs['frequency'].tolist()
                },
                'hot': analyzer.get_temperature_stats()['hot'],
                'cold': analyzer.get_temperature_stats()['cold'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # dashboard_path = dashboard.generate(data)
            dashboard_path = dashboard.generate()
            print(f"🌐 Dashboard generated at: {dashboard_path}")
            print(f"   Serve with: python -m http.server --directory {dashboard.dashboard_dir} 8000")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Troubleshooting:")
        print("1. Check config.yaml exists")
        print("2. Verify historical.csv format")
        print(f"3. Ensure numbers are 1-{config.get('strategy',{}).get('number_pool',55)}")

if __name__ == "__main__":
    main()