# montecarlo

Monte Carlo portfolio simulation utilities (including an options payoff simulator with Black–Scholes pricing).

## Setup

Create/activate a virtual environment (optional) and install dependencies:

```bash
python -m pip install -U pip
python -m pip install numpy pandas matplotlib yfinance
```

## What’s in here

- `montecarlo.py`: portfolio Monte Carlo simulation (and related helpers)
- `montemethod.py`: Black–Scholes pricer/greeks, price fetcher, portfolio MC, and options MC

## Run the existing portfolio simulation

`montecarlo.py` and `montemethod.py` both contain runnable examples. For the current script-style entrypoint in `montecarlo.py`:

```bash
python montecarlo.py
```

This will generate a distribution plot (ignored by git via `.gitignore`).

## Run the options portfolio Monte Carlo (Yahoo prices)

Create a small runner like `run_options.py`:

```python
import montemethod as mm

TICKERS = ["TSLA", "NVDA", "QQQ", "SPY", "PLTR"]

OPTIONS = {
    "TSLA": {"type": "call", "strike": 480.0, "expiry_days": 45, "contracts": 2},
    "NVDA": {"type": "call", "strike": 140.0, "expiry_days": 45, "contracts": 2},
    "QQQ":  {"type": "put",  "strike": 480.0, "expiry_days": 30, "contracts": 1},
    "SPY":  {"type": "put",  "strike": 540.0, "expiry_days": 30, "contracts": 1},
    "PLTR": {"type": "call", "strike": 95.0,  "expiry_days": 60, "contracts": 3},
}

prices = mm.fetch_prices(TICKERS, lookback_days=504)
mm.monte_carlo_options_sim(
    prices,
    OPTIONS,
    num_sim=20000,
    conf_level=0.95,
    risk_free=0.045,
    contract_size=100,
)
```

Then run:

```bash
python run_options.py
```

## Notes / limitations

- Options are simulated to **expiry payoff** (European-style intrinsic value at expiry).
- Volatility is currently approximated using **historical realized volatility** from the lookback window.
- This is for research/education; not financial advice.
