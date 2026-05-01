import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import yfinance as yf
from datetime import datetime, timedelta
from math import erf, exp, log, pi, sqrt


# --- 0. Black-Scholes (no SciPy dependency) ---
def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _norm_cdf(x: float) -> float:
    # Abramowitz-Stegun erf-based normal CDF
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Price a European option using Black-Scholes."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if sigma <= 0:
        # Deterministic forward under risk-neutral drift
        fwd = S * exp(r * T)
        if option_type == "call":
            return max(fwd - K, 0.0) * exp(-r * T)
        return max(K - fwd, 0.0) * exp(-r * T)

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """Return delta, gamma, vega, theta (theta per day; vega per 1% vol)."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    pdf = _norm_pdf(d1)
    gamma = pdf / (S * sigma * sqrt(T))
    vega = (S * pdf * sqrt(T)) / 100.0  # per 1% vol move

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta = (-S * pdf * sigma / (2.0 * sqrt(T)) - r * K * exp(-r * T) * _norm_cdf(d2)) / 365.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (-S * pdf * sigma / (2.0 * sqrt(T)) + r * K * exp(-r * T) * _norm_cdf(-d2)) / 365.0

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta)}

# --- 1. Monte Carlo Simulation Function ---
def monte_carlo_portfolio_sim(prices_dict, weights, time_horizon=252, num_sim=20000, conf_level=0.95):

    tickers = list(prices_dict.keys())
    weights = np.array(weights)

    try:
        prices = np.array([prices_dict[ticker] for ticker in tickers])
    except ValueError as e:
        print("ERROR: Price arrays have different lengths.")
        print(f"Details: {e}")
        return

    n_assets, n_price_points = prices.shape

    if n_price_points < 2:
        raise ValueError(f"Insufficient price data for returns. Found only {n_price_points} points.")

    log_returns = np.log(prices[:, 1:] / prices[:, :-1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.mean(log_returns, axis=1)

    cov = np.cov(log_returns)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        print("ERROR: Covariance matrix not positive definite.")
        return

    results = np.zeros((num_sim, time_horizon + 1))
    initial_prices = prices[:, -1]
    weighted_init = np.dot(weights, initial_prices)
    results[:, 0] = weighted_init

    for i in range(num_sim):
        curr_prices = initial_prices.copy()
        for t in range(1, time_horizon + 1):
            rand_normals = np.random.normal(size=n_assets)
            correlated_normals = np.dot(L, rand_normals)
            daily_returns = mu + correlated_normals
            curr_prices = curr_prices * np.exp(daily_returns)
            results[i, t] = np.dot(weights, curr_prices)

    final_vals = results[:, -1]
    val_at_risk_level = np.percentile(final_vals, (1 - conf_level) * 100)
    VaR = weighted_init - val_at_risk_level
    percentiles = np.percentile(final_vals, [10, 25, 50, 75, 90])

    print(f"\n--- Simulation Results ---")
    print(f"Initial Portfolio Value: ${weighted_init:,.2f}")
    print(f"Value at Risk ({int(conf_level*100)}%): ${VaR:,.2f}")
    print(f"Loss will not exceed this with {int(conf_level*100)}% confidence.\n")

    print("Final Value Percentiles:")
    print({f'P{p}': f'${val:,.2f}' for p, val in zip([10, 25, 50, 75, 90], percentiles)})

    plt.figure(figsize=(10, 6))
    plt.hist(final_vals, bins=60, alpha=0.7, density=True)
    plt.title(f"Portfolio Monte Carlo Distribution ({num_sim:,} runs)")
    plt.xlabel("Final Portfolio Value")
    plt.ylabel("Density")
    plt.axvline(val_at_risk_level, linestyle='dashed', linewidth=2)
    plt.axvline(weighted_init, linestyle='dotted', linewidth=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('portfolio_simulation.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'portfolio_simulation.png'")


# --- 2. Fake Price Generator ---
def generate_fake_price_data(start_price, n_days, mu, sigma):
    mu_adj = mu + np.random.normal(0, 0.00001)
    sigma_adj = sigma + np.random.normal(0, 0.0001)

    returns = np.random.normal(mu_adj, sigma_adj, n_days)
    prices_rev = start_price * np.exp(np.cumsum(-returns))
    prices_hist = prices_rev[::-1]
    return np.append(prices_hist, start_price)


# --- 2b. Fetch Real Price Data (Yahoo Finance) ---
def fetch_prices(tickers, lookback_days=504):
    print("Fetching price data from Yahoo Finance...")
    end = datetime.today()
    start = end - timedelta(days=int(lookback_days))

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw, pd.DataFrame) and "Close" in raw.columns:
        close = raw["Close"]
    else:
        close = raw  # yfinance can return a simple frame for 1 ticker

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close = close[tickers].dropna()
    print(f"Fetched {len(close)} trading days for {len(tickers)} tickers.\n")
    return close


# --- 2c. Monte Carlo with Options Pricing (vectorized, correlated) ---
def monte_carlo_options_sim(
    prices_df,
    options_config,
    weights=None,
    *,
    num_sim=20000,
    conf_level=0.95,
    risk_free=0.045,
    contract_size=100,
):
    tickers = list(options_config.keys())
    prices_df = prices_df[tickers].dropna()

    log_rets = np.log(prices_df / prices_df.shift(1)).dropna()
    mu = log_rets.mean().values  # daily drift in log space
    cov = log_rets.cov().values

    # Enforce PSD then Cholesky
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L = np.linalg.cholesky(cov)

    current_prices = prices_df.iloc[-1].values.astype(float)
    n_assets = len(tickers)

    hist_vols = log_rets.std().values * np.sqrt(252)

    # Initial option prices / greeks
    init_opt_prices = []
    init_greeks = []

    print("--- Initial Option Pricing (Black-Scholes; hist vol proxy) ---")
    for i, ticker in enumerate(tickers):
        cfg = options_config[ticker]
        S = current_prices[i]
        K = float(cfg["strike"])
        T = float(cfg["expiry_days"]) / 365.0
        sigma = float(hist_vols[i])
        otype = cfg["type"].lower()

        price = black_scholes(S, K, T, risk_free, sigma, otype)
        greeks = black_scholes_greeks(S, K, T, risk_free, sigma, otype)
        init_opt_prices.append(price)
        init_greeks.append(greeks)

        notional = price * contract_size * int(cfg["contracts"])
        print(
            f"  {ticker:5s} | {otype.upper():4s} | S=${S:.2f} K=${K:.2f} "
            f"T={int(cfg['expiry_days'])}d | sigma={sigma:.1%} | "
            f"Price=${price:.2f} | Delta={greeks['delta']:.3f} | "
            f"Notional=${notional:,.0f}"
        )

    init_notionals = np.array(
        [init_opt_prices[i] * contract_size * int(options_config[t]["contracts"]) for i, t in enumerate(tickers)],
        dtype=float,
    )
    total_init_value = float(np.sum(init_notionals))
    print(f"\n  Total Options Portfolio Value: ${total_init_value:,.2f}\n")

    sim_horizon = int(max(cfg["expiry_days"] for cfg in options_config.values()))
    print(f"Running {num_sim:,} simulations over {sim_horizon} trading days...")

    # Simulate correlated daily log-returns for all assets together
    # Z: (num_sim, horizon, n_assets) iid N(0,1)
    Z = np.random.normal(size=(num_sim, sim_horizon, n_assets))
    shocks = Z @ L.T  # correlate last dimension
    daily = shocks + mu  # broadcast mu over sims/time

    logS0 = np.log(current_prices)
    log_paths = logS0 + np.cumsum(daily, axis=1)  # (num_sim, horizon, n_assets)
    paths = np.exp(log_paths)

    final_values = np.zeros(num_sim, dtype=float)
    for i, ticker in enumerate(tickers):
        cfg = options_config[ticker]
        day_idx = int(cfg["expiry_days"]) - 1
        ST = paths[:, day_idx, i]
        K = float(cfg["strike"])
        n_cont = int(cfg["contracts"])
        otype = cfg["type"].lower()

        if otype == "call":
            payoff = np.maximum(ST - K, 0.0)
        else:
            payoff = np.maximum(K - ST, 0.0)

        final_values += payoff * contract_size * n_cont

    # Risk Metrics
    q = (1.0 - conf_level) * 100.0
    var_threshold = np.percentile(final_values, q)
    VaR_abs = total_init_value - var_threshold
    CVaR_abs = total_init_value - final_values[final_values <= var_threshold].mean()

    percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
    pct_profitable = float(np.mean(final_values > total_init_value) * 100.0)
    expected_val = float(np.mean(final_values))
    max_loss = total_init_value
    max_gain = float(np.max(final_values))

    print("\n--- Options Portfolio Simulation Results ---")
    print(f"  Initial Portfolio Value : ${total_init_value:>12,.2f}")
    print(f"  Expected Final Value    : ${expected_val:>12,.2f}")
    print(f"  VaR ({int(conf_level*100)}%)            : ${VaR_abs:>12,.2f}")
    print(f"  CVaR ({int(conf_level*100)}%)           : ${CVaR_abs:>12,.2f}")
    print(f"  Max Possible Loss       : ${max_loss:>12,.2f}  (full premium loss)")
    print(f"  Max Simulated Gain      : ${max_gain:>12,.2f}")
    print(f"  % Simulations Profitable: {pct_profitable:>11.1f}%")
    print("\n  Payoff Percentiles:")
    for p, v in zip([5, 25, 50, 75, 95], percentiles):
        print(f"    P{p:<3}: ${v:>10,.2f}")

    # Greeks Summary (rough: sums of individual option greeks at t=0)
    total_delta = sum(
        init_greeks[i]["delta"] * contract_size * int(options_config[t]["contracts"]) for i, t in enumerate(tickers)
    )
    total_vega = sum(
        init_greeks[i]["vega"] * contract_size * int(options_config[t]["contracts"]) for i, t in enumerate(tickers)
    )
    total_theta = sum(
        init_greeks[i]["theta"] * contract_size * int(options_config[t]["contracts"]) for i, t in enumerate(tickers)
    )
    print("\n--- Portfolio Greeks (t=0) ---")
    print(f"  Portfolio Delta : {total_delta:>10.2f}")
    print(f"  Portfolio Vega  : {total_vega:>10.2f}  (per 1% vol move)")
    print(f"  Portfolio Theta : {total_theta:>10.2f}  (per day)")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(final_values, bins=80, alpha=0.75, color="steelblue", density=True)
    axes[0].axvline(var_threshold, color="red", linestyle="--", linewidth=1.8, label=f"VaR {int(conf_level*100)}%")
    axes[0].axvline(total_init_value, color="orange", linestyle=":", linewidth=1.8, label="Initial Value")
    axes[0].axvline(expected_val, color="green", linestyle="-.", linewidth=1.8, label="Expected Value")
    axes[0].set_title(f"Options Portfolio Payoff Distribution\n({num_sim:,} simulations)")
    axes[0].set_xlabel("Final Portfolio Value ($)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    ticker_init = [
        init_opt_prices[i] * contract_size * int(options_config[t]["contracts"]) for i, t in enumerate(tickers)
    ]
    x = np.arange(len(tickers))
    axes[1].bar(x, ticker_init, 0.6, label="Current Value (BS)", color="seagreen", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tickers)
    axes[1].set_title("Per-Ticker Option Notional Value (t=0)")
    axes[1].set_ylabel("Value ($)")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return {
        "final_values": final_values,
        "total_init_value": total_init_value,
        "VaR_abs": float(VaR_abs),
        "CVaR_abs": float(CVaR_abs),
    }


# --- 3. Main Execution ---
if __name__ == "__main__":
    raw_tickers = "EOSE QS IREN RKLB TSLA IBIT GOOG SOFI BABA XIACY BIDU HDB TCEHY STRL SPY GLD ESGU URBN VXUS ACWI KO PEP VZ T TMUS UNH VTI O PLD"
    raw_prices = "$14.55 $13.80 $65.00 $66.15 $461.00 $63.00 $252.34 $28.61 $165.84 $29.68 $120.15 $36.91 $80.06 $349.38 $671.38 $386.33 $146.12 $66.75 $74.19 $139.00 $69.71 $151.55 $38.82 $25.14 $217.77 $364.10 $328.79 $60.00 $126.43"
    raw_holdings = "$5,621.00 $4,205.00 $20,800.00 $9,922.50 $11,525.00 $9,450.00 $11,722.95 $5,802.00 $3,494.00 $2,950.00 $2,455.20 $2,205.60 $2,054.00 $10,235.16 $47,407.50 $18,876.00 $10,376.10 $8,132.40 $5,996.00 $4,225.80 $12,199.25 $12,124.00 $9,705.00 $10,056.00 $5,444.25 $54,375.00 $13,348.40 $24,000.00 $25,286.00"

    print("--- Data Processing ---")

    def clean_currency_string(s):
        return s.replace('$', '').replace(',', '')

    try:
        tickers_list = raw_tickers.split(' ')
        prices_list = [float(clean_currency_string(p)) for p in raw_prices.split(' ')]
        holdings_list = [float(clean_currency_string(h)) for h in raw_holdings.split(' ')]

        if not (len(tickers_list) == len(prices_list) == len(holdings_list)):
            raise ValueError("Ticker/price/holding count mismatch")

        current_prices = dict(zip(tickers_list, prices_list))
        total_portfolio_value = sum(holdings_list)
        weights = [h / total_portfolio_value for h in holdings_list]

        print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")

        print("\n--- Generating Fake Data for Simulation ---")
        fake_prices_dict = {}
        n_history = 252

        for ticker, price in current_prices.items():
            fake_mu = np.random.uniform(0.0001, 0.0005)
            fake_sigma = np.random.uniform(0.01, 0.03)
            fake_prices_dict[ticker] = generate_fake_price_data(price, n_history, fake_mu, fake_sigma)

        monte_carlo_portfolio_sim(fake_prices_dict, weights, time_horizon=252, num_sim=10000)

    except Exception as e:
        print(f"Error: {e}")
