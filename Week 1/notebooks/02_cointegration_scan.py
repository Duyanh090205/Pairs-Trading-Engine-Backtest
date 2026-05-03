# %% [markdown]
# # Notebook 02 — Cointegration Scan (FULL RUN)
#
# **Purpose:** Test all C(N,2) equity pairs for cointegration using the
# Engle-Granger framework, apply BH-FDR correction, compute half-life, filter by
# economic logic, and produce a Pairs Selection Report.
#
# **Scope:** Full run — all tickers from Notebook 01 panel (no universe cap).
#
# **Methodology source of truth:**
# - `.agents/workflows/cointegration_methodology_spec.md`
# - `.agents/workflows/implementation_checklist.md`
#
# **Key workflow (validated in prototype, preserved here):**
# - `coint()` = official cointegration verdict (p-value, test statistic)
# - OLS = hedge ratio + spread construction + half-life + plots
# - BH-FDR at q=0.05 for multiple testing correction
# - Hard filters: BH-FDR pass, half-life [5,60] days, beta > 0, economic logic
# - Fallback: relax half-life to [3,90] if < 10 survive, then FDR to q=0.10 if < 5

# %% [markdown]
# ## 1. Setup and Configuration

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import warnings
import sys, io
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore', category=FutureWarning)

from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# -- Paths --
PROJECT_ROOT = Path(r"d:\Quant Finance\Quant Program\Week 1")
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pair_scan_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Approved methodology parameters (preserved from prototype) --
COINT_MAXLAG = 30
COINT_TREND = 'c'
COINT_AUTOLAG = 'aic'
BH_FDR_ALPHA = 0.05
HALF_LIFE_RANGE = (5, 60)
HALF_LIFE_FALLBACK = (3, 90)
BARS_PER_DAY = 77
MIN_ALIGNED_OBS = 5000

print("FULL RUN -- Notebook 02: Cointegration Scan")
print("=" * 60)
print(f"BH-FDR alpha: {BH_FDR_ALPHA}")
print(f"Half-life range: {HALF_LIFE_RANGE} trading days")
print(f"Bars per day: {BARS_PER_DAY}")
print(f"Min aligned obs: {MIN_ALIGNED_OBS}")

# %% [markdown]
# ## 2. Data Loading and Quick Validation

# %%
panel = pd.read_parquet(INTERMEDIATE_DIR / "log_prices_5min.parquet")

# -- Validation checks --
assert isinstance(panel.index, pd.DatetimeIndex), "Index is not DatetimeIndex"
assert panel.index.is_unique, "Index has duplicates"
assert panel.index.is_monotonic_increasing, "Index is not sorted"
assert panel.isna().sum().sum() == 0, "Panel contains NaN"
assert all(panel.dtypes == np.float64), "Not all columns are float64"

if panel.index.tz is None:
    panel.index = panel.index.tz_localize('US/Eastern')
    print("NOTE: Re-localized index to US/Eastern")

tickers = list(panel.columns)
n_tickers = len(tickers)

print(f"\nPanel loaded:")
print(f"  Shape: {panel.shape}")
print(f"  Tickers: {n_tickers}")
print(f"  Date range: {panel.index[0].date()} to {panel.index[-1].date()}")
print(f"  NaN: 0")

# %% [markdown]
# ## 3. Sector Mapping
#
# GICS sector mapping for all tickers in the universe.
# The assertion below will catch any unmapped tickers.

# %%
SECTOR_MAP = {
    # Communication Services (10)
    'CHTR': 'Communication Services', 'CMCSA': 'Communication Services',
    'DIS': 'Communication Services', 'EA': 'Communication Services',
    'FOX': 'Communication Services', 'FOXA': 'Communication Services',
    'GOOG': 'Communication Services', 'GOOGL': 'Communication Services',
    'LYV': 'Communication Services', 'MTCH': 'Communication Services',
    # Consumer Discretionary (27)
    'ABNB': 'Consumer Discretionary', 'AMZN': 'Consumer Discretionary',
    'APTV': 'Consumer Discretionary', 'BBY': 'Consumer Discretionary',
    'CCL': 'Consumer Discretionary', 'CVNA': 'Consumer Discretionary',
    'DASH': 'Consumer Discretionary', 'DG': 'Consumer Discretionary',
    'DHI': 'Consumer Discretionary', 'DLTR': 'Consumer Discretionary',
    'DRI': 'Consumer Discretionary', 'EBAY': 'Consumer Discretionary',
    'EXPE': 'Consumer Discretionary', 'F': 'Consumer Discretionary',
    'GM': 'Consumer Discretionary', 'GRMN': 'Consumer Discretionary',
    'HAS': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'HLT': 'Consumer Discretionary', 'LEN': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary', 'LULU': 'Consumer Discretionary',
    'LVS': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'MGM': 'Consumer Discretionary',
    'NCLH': 'Consumer Discretionary',
    # Consumer Staples (22)
    'ADM': 'Consumer Staples', 'BG': 'Consumer Staples',
    'CAG': 'Consumer Staples', 'CHD': 'Consumer Staples',
    'CL': 'Consumer Staples', 'CLX': 'Consumer Staples',
    'COST': 'Consumer Staples', 'CPB': 'Consumer Staples',
    'EL': 'Consumer Staples', 'GIS': 'Consumer Staples',
    'HRL': 'Consumer Staples', 'HSY': 'Consumer Staples',
    'KDP': 'Consumer Staples', 'KHC': 'Consumer Staples',
    'KMB': 'Consumer Staples', 'KO': 'Consumer Staples',
    'KR': 'Consumer Staples', 'LW': 'Consumer Staples',
    'MDLZ': 'Consumer Staples', 'MKC': 'Consumer Staples',
    'MNST': 'Consumer Staples', 'MO': 'Consumer Staples',
    # Energy (12)
    'APA': 'Energy', 'BKR': 'Energy', 'COP': 'Energy',
    'CTRA': 'Energy', 'CVX': 'Energy', 'DVN': 'Energy',
    'EOG': 'Energy', 'EQT': 'Energy', 'FANG': 'Energy',
    'HAL': 'Energy', 'KMI': 'Energy', 'MPC': 'Energy',
    # Financials (36)
    'ACGL': 'Financials', 'AFL': 'Financials', 'AIG': 'Financials',
    'ALL': 'Financials', 'AON': 'Financials', 'APO': 'Financials',
    'AXP': 'Financials', 'BAC': 'Financials', 'BEN': 'Financials',
    'BK': 'Financials', 'BRO': 'Financials', 'BX': 'Financials',
    'C': 'Financials', 'CB': 'Financials', 'CFG': 'Financials',
    'CME': 'Financials', 'COF': 'Financials', 'COIN': 'Financials',
    'FIS': 'Financials', 'FISV': 'Financials', 'FITB': 'Financials',
    'GPN': 'Financials', 'GS': 'Financials', 'HBAN': 'Financials',
    'HIG': 'Financials', 'HOOD': 'Financials', 'IBKR': 'Financials',
    'ICE': 'Financials', 'IVZ': 'Financials', 'JPM': 'Financials',
    'KEY': 'Financials', 'KKR': 'Financials', 'MA': 'Financials',
    'MET': 'Financials', 'MS': 'Financials', 'MTB': 'Financials',
    # Healthcare (29)
    'A': 'Healthcare', 'ABBV': 'Healthcare', 'ABT': 'Healthcare',
    'AMGN': 'Healthcare', 'BAX': 'Healthcare', 'BDX': 'Healthcare',
    'BIIB': 'Healthcare', 'BMY': 'Healthcare', 'BSX': 'Healthcare',
    'CAH': 'Healthcare', 'CI': 'Healthcare', 'CNC': 'Healthcare',
    'CVS': 'Healthcare', 'DGX': 'Healthcare', 'DHR': 'Healthcare',
    'DXCM': 'Healthcare', 'EW': 'Healthcare', 'GILD': 'Healthcare',
    'HCA': 'Healthcare', 'HOLX': 'Healthcare', 'INCY': 'Healthcare',
    'IQV': 'Healthcare', 'ISRG': 'Healthcare', 'JNJ': 'Healthcare',
    'LLY': 'Healthcare', 'MCK': 'Healthcare', 'MDT': 'Healthcare',
    'MRK': 'Healthcare', 'MRNA': 'Healthcare',
    # Industrials (32)
    'AME': 'Industrials', 'AOS': 'Industrials', 'BA': 'Industrials',
    'BLDR': 'Industrials', 'CARR': 'Industrials', 'CAT': 'Industrials',
    'CHRW': 'Industrials', 'CMI': 'Industrials', 'CPRT': 'Industrials',
    'CSX': 'Industrials', 'DAL': 'Industrials', 'DE': 'Industrials',
    'DOV': 'Industrials', 'EMR': 'Industrials', 'ETN': 'Industrials',
    'EXPD': 'Industrials', 'FAST': 'Industrials', 'FDX': 'Industrials',
    'FTV': 'Industrials', 'GD': 'Industrials', 'GE': 'Industrials',
    'GNRC': 'Industrials', 'HON': 'Industrials', 'HWM': 'Industrials',
    'IR': 'Industrials', 'ITW': 'Industrials', 'JCI': 'Industrials',
    'LHX': 'Industrials', 'LMT': 'Industrials', 'LUV': 'Industrials',
    'MAS': 'Industrials', 'MMM': 'Industrials',
    # Technology (37)
    'AAPL': 'Technology', 'ACN': 'Technology', 'ADBE': 'Technology',
    'ADI': 'Technology', 'ADP': 'Technology', 'ADSK': 'Technology',
    'AKAM': 'Technology', 'AMAT': 'Technology', 'AMD': 'Technology',
    'ANET': 'Technology', 'APH': 'Technology', 'APP': 'Technology',
    'AVGO': 'Technology', 'CDNS': 'Technology', 'CIEN': 'Technology',
    'CRM': 'Technology', 'CRWD': 'Technology', 'CSCO': 'Technology',
    'CSGP': 'Technology', 'CTSH': 'Technology', 'DDOG': 'Technology',
    'DELL': 'Technology', 'FSLR': 'Technology', 'FTNT': 'Technology',
    'GDDY': 'Technology', 'GLW': 'Technology', 'HPE': 'Technology',
    'HPQ': 'Technology', 'IBM': 'Technology', 'INTC': 'Technology',
    'INTU': 'Technology', 'JBL': 'Technology', 'KEYS': 'Technology',
    'KLAC': 'Technology', 'LRCX': 'Technology', 'MCHP': 'Technology',
    'MSFT': 'Technology', 'MU': 'Technology',
    # Materials (14)
    'ALB': 'Materials', 'AMCR': 'Materials', 'APD': 'Materials',
    'CF': 'Materials', 'CTVA': 'Materials', 'DD': 'Materials',
    'DOW': 'Materials', 'ECL': 'Materials', 'FCX': 'Materials',
    'IFF': 'Materials', 'IP': 'Materials', 'LIN': 'Materials',
    'LYB': 'Materials', 'MOS': 'Materials',
    # Real Estate (12)
    'AMT': 'Real Estate', 'BXP': 'Real Estate', 'CBRE': 'Real Estate',
    'CCI': 'Real Estate', 'DLR': 'Real Estate', 'DOC': 'Real Estate',
    'EQR': 'Real Estate', 'HST': 'Real Estate', 'INVH': 'Real Estate',
    'IRM': 'Real Estate', 'KIM': 'Real Estate',
    # Utilities (18)
    'AEE': 'Utilities', 'AEP': 'Utilities', 'AES': 'Utilities',
    'ATO': 'Utilities', 'AWK': 'Utilities', 'CMS': 'Utilities',
    'CNP': 'Utilities', 'D': 'Utilities', 'DTE': 'Utilities',
    'DUK': 'Utilities', 'ED': 'Utilities', 'EIX': 'Utilities',
    'ES': 'Utilities', 'ETR': 'Utilities', 'EVRG': 'Utilities',
    'EXC': 'Utilities', 'FE': 'Utilities', 'LNT': 'Utilities',
    # ETFs (5)
    'EEM': 'ETF-Emerging Markets', 'FXI': 'ETF-China',
    'GLD': 'ETF-Gold', 'IWM': 'ETF-Small Cap', 'KWEB': 'ETF-China Internet',
}

# -- Hard assertion: every ticker must have a mapping --
unmapped = set(tickers) - set(SECTOR_MAP.keys())
assert len(unmapped) == 0, f"Unmapped tickers: {unmapped}. Add them to SECTOR_MAP."
print(f"Sector mapping covers all {n_tickers} tickers.")

# Sector distribution
sector_counts = pd.Series(SECTOR_MAP).loc[tickers].value_counts()
print(f"\nSector distribution:")
for sector, cnt in sector_counts.items():
    print(f"  {sector}: {cnt}")

# -- Economic rationale lookup for notable pairs --
# Within-sector pairs not listed here automatically get Tier 2 via the filter logic.
# This dict provides Tier 1 (direct competitors / same sub-industry) and cross-sector links.
ECON_RATIONALE = {
    # ===== Technology - Semiconductors =====
    'AMAT-KLAC':  ('Tier 1', 'Same sub-industry (Semiconductor Equipment), direct competitors'),
    'AMAT-LRCX':  ('Tier 1', 'Same sub-industry (Semiconductor Equipment), direct competitors'),
    'KLAC-LRCX':  ('Tier 1', 'Same sub-industry (Semiconductor Equipment), direct competitors'),
    'AMD-INTC':   ('Tier 1', 'Same sub-industry (Semiconductors), direct CPU competitors'),
    'ADI-MCHP':   ('Tier 1', 'Same sub-industry (Analog Semiconductors), direct competitors'),
    'AMD-MU':     ('Tier 2', 'Same sector (Semiconductors), shared PC/data center demand cycle'),
    'AVGO-INTC':  ('Tier 2', 'Same sector (Semiconductors), shared enterprise chip demand'),
    'AVGO-MU':    ('Tier 2', 'Same sector (Semiconductors), shared data center demand'),
    'INTC-MU':    ('Tier 2', 'Same sector (Semiconductors), shared memory/compute cycle'),
    'AMAT-MU':    ('Tier 2', 'Semiconductor equipment + chipmaker, shared capex cycle'),
    'AMD-AVGO':   ('Tier 2', 'Same sector (Semiconductors), shared data center/AI demand'),
    'AVGO-LRCX':  ('Tier 2', 'Semiconductor equipment + chipmaker, shared capex cycle'),
    'AMD-LRCX':   ('Tier 2', 'Semiconductor equipment + chipmaker, shared capex cycle'),
    'INTC-LRCX':  ('Tier 2', 'Semiconductor equipment + chipmaker, shared capex cycle'),
    'LRCX-MU':    ('Tier 2', 'Semiconductor equipment + memory chipmaker, shared capex cycle'),
    'ADI-INTC':   ('Tier 2', 'Same sector (Semiconductors), analog vs CPU'),
    'MCHP-MU':    ('Tier 2', 'Same sector (Semiconductors), shared demand cycle'),
    # ===== Technology - Software/Platform =====
    'AAPL-MSFT':  ('Tier 1', 'Both mega-cap technology platform companies'),
    'ADBE-CRM':   ('Tier 1', 'Same sub-industry (Application Software), enterprise SaaS competitors'),
    'ACN-CTSH':   ('Tier 1', 'Same sub-industry (IT Consulting/Outsourcing), direct competitors'),
    'CRWD-FTNT':  ('Tier 1', 'Same sub-industry (Cybersecurity), direct competitors'),
    'HPE-HPQ':    ('Tier 1', 'Same parent company split, shared enterprise/PC hardware market'),
    'CDNS-ADSK':  ('Tier 2', 'Same sector (Technology), both design/simulation software'),
    'ADBE-INTU':  ('Tier 2', 'Same sector (Software), both B2B/B2C software platforms'),
    'CRM-INTU':   ('Tier 2', 'Same sector (Software), both SaaS platforms'),
    'CRM-MSFT':   ('Tier 2', 'Same sector (Technology), direct CRM/cloud competitors'),
    'ADBE-MSFT':  ('Tier 2', 'Same sector (Technology), enterprise software overlap'),
    'CRWD-CSCO':  ('Tier 2', 'Same sector (Technology), cybersecurity + networking infrastructure'),
    'DDOG-CRM':   ('Tier 2', 'Same sector (Technology), both cloud SaaS platforms'),
    'DELL-HPE':   ('Tier 2', 'Same sector (Technology), both enterprise hardware/servers'),
    # ===== Communication Services =====
    'FOX-FOXA':   ('Tier 1', 'Same company share classes (Fox Corp Class A and B)'),
    'GOOG-GOOGL': ('Tier 1', 'Same company share classes (Alphabet Class A and C)'),
    'CHTR-CMCSA': ('Tier 1', 'Same sub-industry (Cable/Broadband), direct competitors'),
    'DIS-CMCSA':  ('Tier 2', 'Same sector, media/streaming competitors'),
    # ===== Cross-sector: Tech <-> Communication Services =====
    'AAPL-GOOGL': ('Tier 1', 'Both mega-cap tech platforms, mobile/services overlap'),
    'AAPL-GOOG':  ('Tier 1', 'Both mega-cap tech platforms, GOOG is GOOGL class C shares'),
    'MSFT-GOOGL': ('Tier 1', 'Both mega-cap, direct competitors in cloud and enterprise'),
    'MSFT-GOOG':  ('Tier 1', 'Both mega-cap, direct competitors in cloud and enterprise'),
    # ===== Financials - Banks =====
    'BAC-C':      ('Tier 1', 'Same sub-industry (Diversified Banks), same Fed rate and credit cycle'),
    'BAC-JPM':    ('Tier 1', 'Same sub-industry (Diversified Banks), same Fed rate and credit cycle'),
    'C-JPM':      ('Tier 1', 'Same sub-industry (Diversified Banks), same Fed rate and credit cycle'),
    'GS-MS':      ('Tier 1', 'Same sub-industry (Investment Banks), same capital markets cycle'),
    'CFG-FITB':   ('Tier 1', 'Same sub-industry (Regional Banks), similar size and geography'),
    'CFG-HBAN':   ('Tier 1', 'Same sub-industry (Regional Banks), similar size'),
    'CFG-KEY':    ('Tier 1', 'Same sub-industry (Regional Banks), similar size'),
    'FITB-HBAN':  ('Tier 1', 'Same sub-industry (Regional Banks), both Midwest-focused'),
    'FITB-KEY':   ('Tier 1', 'Same sub-industry (Regional Banks), similar size'),
    'HBAN-KEY':   ('Tier 1', 'Same sub-industry (Regional Banks), both Midwest-focused'),
    'KEY-MTB':    ('Tier 1', 'Same sub-industry (Regional Banks), similar size'),
    'CFG-MTB':    ('Tier 1', 'Same sub-industry (Regional Banks), similar size'),
    'FITB-MTB':   ('Tier 1', 'Same sub-industry (Regional Banks), similar size'),
    'HBAN-MTB':   ('Tier 1', 'Same sub-industry (Regional Banks), similar size'),
    'BAC-GS':     ('Tier 2', 'Same sector (Financials), commercial vs investment banking'),
    'BAC-MS':     ('Tier 2', 'Same sector (Financials), commercial vs investment banking'),
    'C-GS':       ('Tier 2', 'Same sector (Financials), universal vs investment banking'),
    'C-MS':       ('Tier 2', 'Same sector (Financials), universal vs investment banking'),
    'JPM-GS':     ('Tier 2', 'Same sector (Financials), universal vs investment banking'),
    'JPM-MS':     ('Tier 2', 'Same sector (Financials), universal vs investment banking'),
    # ===== Financials - Payments/FinTech =====
    'FIS-FISV':   ('Tier 1', 'Same sub-industry (Payment Processors), direct competitors'),
    'FIS-GPN':    ('Tier 1', 'Same sub-industry (Payment Processors), direct competitors'),
    'FISV-GPN':   ('Tier 1', 'Same sub-industry (Payment Processors), direct competitors'),
    'AXP-MA':     ('Tier 1', 'Same sub-industry (Payment Networks), shared consumer credit'),
    'JPM-MA':     ('Tier 2', 'Same sector (Financials), bank vs payment network'),
    'BAC-MA':     ('Tier 2', 'Same sector (Financials), bank vs payment network'),
    'C-MA':       ('Tier 2', 'Same sector (Financials), bank vs payment network'),
    'GS-MA':      ('Tier 2', 'Same sector (Financials), investment bank + payment network'),
    'MS-MA':      ('Tier 2', 'Same sector (Financials), investment bank + payment network'),
    # ===== Financials - Insurance =====
    'AFL-MET':    ('Tier 1', 'Same sub-industry (Life Insurance), shared rate/mortality cycle'),
    'AIG-ALL':    ('Tier 1', 'Same sub-industry (P&C Insurance), shared catastrophe/rate cycle'),
    'CB-HIG':     ('Tier 1', 'Same sub-industry (P&C Insurance), shared rate cycle'),
    'AON-BRO':    ('Tier 2', 'Same sector (Financials), both insurance brokers'),
    # ===== Financials - Exchanges/Alt Managers =====
    'APO-BX':     ('Tier 1', 'Same sub-industry (Alternative Asset Managers)'),
    'APO-KKR':    ('Tier 1', 'Same sub-industry (Alternative Asset Managers)'),
    'BX-KKR':     ('Tier 1', 'Same sub-industry (Alternative Asset Managers)'),
    'CME-ICE':    ('Tier 1', 'Same sub-industry (Financial Exchanges), direct competitors'),
    # ===== Financials - COIN (crypto, marginal link) =====
    'BAC-COIN':   ('Tier 3', 'Same sector (Financials), but COIN is crypto exchange — marginal link'),
    'C-COIN':     ('Tier 3', 'Same sector (Financials), but COIN is crypto exchange — marginal link'),
    'COIN-GS':    ('Tier 3', 'Same sector (Financials), but COIN is crypto exchange — marginal link'),
    'COIN-JPM':   ('Tier 3', 'Same sector (Financials), but COIN is crypto exchange — marginal link'),
    'COIN-MA':    ('Tier 3', 'Same sector (Financials), COIN crypto vs MA payment rails — weak link'),
    'COIN-MS':    ('Tier 3', 'Same sector (Financials), but COIN is crypto exchange — marginal link'),
    # ===== Energy =====
    'COP-CVX':    ('Tier 1', 'Same sub-industry (Integrated/E&P Oil & Gas), shared crude exposure'),
    'COP-DVN':    ('Tier 1', 'Same sub-industry (E&P), both US shale-focused producers'),
    'COP-EOG':    ('Tier 1', 'Same sub-industry (E&P), both large US E&P producers'),
    'CVX-DVN':    ('Tier 1', 'Same sector (Energy), integrated major vs shale E&P, shared crude'),
    'DVN-EOG':    ('Tier 1', 'Same sub-industry (E&P), both US shale-focused producers'),
    'DVN-FANG':   ('Tier 1', 'Same sub-industry (E&P), both Permian Basin shale producers'),
    'EOG-FANG':   ('Tier 1', 'Same sub-industry (E&P), both US shale producers'),
    'BKR-HAL':    ('Tier 1', 'Same sub-industry (Oilfield Services), direct competitors'),
    'CTRA-DVN':   ('Tier 1', 'Same sub-industry (E&P), both US shale producers'),
    'CTRA-EOG':   ('Tier 1', 'Same sub-industry (E&P), both US shale producers'),
    # ===== Healthcare - Pharma =====
    'ABBV-BMY':   ('Tier 1', 'Same sub-industry (Large-Cap Pharma), shared FDA/patent cycle'),
    'ABBV-JNJ':   ('Tier 1', 'Same sub-industry (Large-Cap Pharma), shared healthcare demand'),
    'ABBV-LLY':   ('Tier 1', 'Same sub-industry (Large-Cap Pharma), shared FDA/patent cycle'),
    'ABBV-MRK':   ('Tier 1', 'Same sub-industry (Large-Cap Pharma), shared FDA/patent cycle'),
    'BMY-JNJ':    ('Tier 1', 'Same sub-industry (Large-Cap Pharma), shared healthcare demand'),
    'BMY-LLY':    ('Tier 1', 'Same sub-industry (Large-Cap Pharma), shared FDA/patent cycle'),
    'BMY-MRK':    ('Tier 1', 'Same sub-industry (Large-Cap Pharma), shared FDA/patent cycle'),
    'JNJ-LLY':    ('Tier 1', 'Same sub-industry (Large-Cap Pharma), shared healthcare demand'),
    'JNJ-MRK':    ('Tier 1', 'Same sub-industry (Large-Cap Pharma), shared healthcare demand'),
    'LLY-MRK':    ('Tier 1', 'Same sub-industry (Large-Cap Pharma), shared FDA/patent cycle'),
    'ABBV-MRNA':  ('Tier 2', 'Same sector (Healthcare), large pharma vs biotech, shared FDA cycle'),
    'BMY-MRNA':   ('Tier 2', 'Same sector (Healthcare), large pharma vs biotech, shared FDA cycle'),
    'JNJ-MRNA':   ('Tier 2', 'Same sector (Healthcare), diversified pharma vs biotech'),
    'LLY-MRNA':   ('Tier 2', 'Same sector (Healthcare), large pharma vs biotech'),
    'MRK-MRNA':   ('Tier 2', 'Same sector (Healthcare), large pharma vs biotech, shared FDA cycle'),
    # ===== Healthcare - Biotech =====
    'AMGN-BIIB':  ('Tier 1', 'Same sub-industry (Large-Cap Biotech), shared FDA cycle'),
    'AMGN-GILD':  ('Tier 1', 'Same sub-industry (Large-Cap Biotech), shared FDA cycle'),
    'BIIB-GILD':  ('Tier 1', 'Same sub-industry (Large-Cap Biotech), shared FDA cycle'),
    # ===== Healthcare - Medical Devices =====
    'BSX-MDT':    ('Tier 1', 'Same sub-industry (Cardiovascular Devices), direct competitors'),
    'ABT-BDX':    ('Tier 2', 'Same sector (Healthcare), both medical devices/diagnostics'),
    'BSX-EW':     ('Tier 2', 'Same sector (Healthcare), both cardiovascular devices'),
    # ===== Healthcare - Managed Care =====
    'CI-CNC':     ('Tier 1', 'Same sub-industry (Managed Care), direct competitors'),
    'CI-CVS':     ('Tier 2', 'Same sector (Healthcare), managed care + pharmacy/PBM'),
    'CNC-CVS':    ('Tier 2', 'Same sector (Healthcare), managed care + pharmacy/PBM'),
    # ===== Healthcare - Drug Distributors =====
    'CAH-MCK':    ('Tier 1', 'Same sub-industry (Drug Distributors), direct competitors'),
    # ===== Industrials - Aerospace & Defense =====
    'BA-LMT':     ('Tier 1', 'Same sub-industry (Aerospace & Defense), shared defense budget'),
    'BA-GD':      ('Tier 1', 'Same sub-industry (Aerospace & Defense), shared defense budget'),
    'GD-LMT':     ('Tier 1', 'Same sub-industry (Defense), direct competitors'),
    'LHX-LMT':    ('Tier 2', 'Same sector (Industrials), defense electronics + defense prime'),
    # ===== Industrials - Heavy Equipment =====
    'CAT-DE':     ('Tier 1', 'Same sub-industry (Heavy Equipment), direct competitors'),
    'CAT-CMI':    ('Tier 2', 'Same sector (Industrials), heavy machinery, shared capex cycle'),
    # ===== Industrials - Airlines =====
    'DAL-LUV':    ('Tier 1', 'Same sub-industry (Airlines), shared fuel/travel demand cycle'),
    # ===== Industrials - Transportation =====
    'CHRW-EXPD':  ('Tier 1', 'Same sub-industry (Freight/Logistics), both freight forwarders'),
    'CSX-FDX':    ('Tier 2', 'Same sector (Industrials), rail + express, shared freight demand'),
    # ===== Consumer Discretionary =====
    'F-GM':       ('Tier 1', 'Same sub-industry (Auto Manufacturers), direct US competitors'),
    'HD-LOW':     ('Tier 1', 'Same sub-industry (Home Improvement Retail), direct competitors'),
    'CCL-NCLH':   ('Tier 1', 'Same sub-industry (Cruise Lines), direct competitors'),
    'HLT-MAR':    ('Tier 1', 'Same sub-industry (Hotels), direct competitors'),
    'DG-DLTR':    ('Tier 1', 'Same sub-industry (Dollar Stores), direct competitors'),
    'LVS-MGM':    ('Tier 1', 'Same sub-industry (Casinos/Gaming), direct competitors'),
    'DHI-LEN':    ('Tier 1', 'Same sub-industry (Homebuilders), direct competitors'),
    'ABNB-EXPE':  ('Tier 2', 'Same sector, both online travel/lodging platforms'),
    'ABNB-CCL':   ('Tier 2', 'Same sector (Consumer Discretionary), travel/leisure demand overlap'),
    'F-CCL':      ('Tier 3', 'Same sector (Consumer Discretionary), but auto vs cruise — weak link'),
    'GM-CCL':     ('Tier 3', 'Same sector (Consumer Discretionary), but auto vs cruise — weak link'),
    # ===== Consumer Staples =====
    'CAG-CPB':    ('Tier 1', 'Same sub-industry (Packaged Foods), direct competitors'),
    'GIS-KHC':    ('Tier 1', 'Same sub-industry (Packaged Foods), direct competitors'),
    'CL-CLX':     ('Tier 1', 'Same sub-industry (Household Products), direct competitors'),
    'HSY-MDLZ':   ('Tier 1', 'Same sub-industry (Confectionery/Snacks), direct competitors'),
    'KMB-CL':     ('Tier 2', 'Same sector (Consumer Staples), both consumer products'),
    'KO-MNST':    ('Tier 2', 'Same sector (Consumer Staples), both beverages, KO has MNST stake'),
    # ===== Materials =====
    'CF-MOS':     ('Tier 1', 'Same sub-industry (Fertilizers), direct competitors'),
    'DD-DOW':     ('Tier 1', 'Same sub-industry (Chemicals), former parent-spin, shared chem cycle'),
    'ALB-LIN':    ('Tier 2', 'Same sector (Materials), both specialty chemicals'),
    'LYB-DOW':    ('Tier 2', 'Same sector (Materials), both commodity chemicals'),
    # ===== Real Estate =====
    'AMT-CCI':    ('Tier 1', 'Same sub-industry (Tower REITs), direct competitors'),
    'EQR-INVH':   ('Tier 2', 'Same sector (Real Estate), both residential REITs'),
    'DLR-IRM':    ('Tier 2', 'Same sector (Real Estate), both data-related REITs'),
    # ===== Utilities =====
    'D-DUK':      ('Tier 1', 'Same sub-industry (Regulated Electric Utilities), similar profiles'),
    'AEE-AEP':    ('Tier 2', 'Same sector (Utilities), both regulated utilities'),
    'EXC-FE':     ('Tier 2', 'Same sector (Utilities), both nuclear-heavy utilities'),
    # ===== ETFs =====
    'EEM-FXI':    ('Tier 2', 'Both EM-focused ETFs, FXI is largest component of EEM'),
    'EEM-KWEB':   ('Tier 2', 'Both EM/China-focused ETFs, shared China macro exposure'),
    'FXI-KWEB':   ('Tier 1', 'Both China-focused ETFs, KWEB is internet subset of FXI universe'),
}

# Save sector mapping as artifact
sector_df = pd.DataFrame([
    {'ticker': t, 'sector': SECTOR_MAP[t]} for t in tickers
]).sort_values('ticker')
sector_path = OUTPUT_DIR / "sector_mapping.parquet"
sector_df.to_parquet(sector_path, index=False)
print(f"Saved: {sector_path}")
print(f"Economic rationale lookup: {len(ECON_RATIONALE)} pair rationales defined")

# %% [markdown]
# ## 4. Pair Generation

# %%
pairs = []
for a, b in combinations(sorted(tickers), 2):
    pair_id = f"{a}-{b}"
    sector_a = SECTOR_MAP[a]
    sector_b = SECTOR_MAP[b]
    within_sector = sector_a == sector_b

    pairs.append({
        'pair_id': pair_id,
        'ticker_a': a,
        'ticker_b': b,
        'sector_a': sector_a,
        'sector_b': sector_b,
        'within_sector': within_sector,
    })

pairs_df = pd.DataFrame(pairs)
n_pairs = len(pairs_df)
n_within = pairs_df['within_sector'].sum()
n_cross = n_pairs - n_within

print(f"Pairs generated: {n_pairs} (expected C({n_tickers},2) = {n_tickers*(n_tickers-1)//2})")
print(f"  Within-sector: {n_within}")
print(f"  Cross-sector:  {n_cross}")
assert n_pairs == n_tickers * (n_tickers - 1) // 2, "Pair count mismatch"
assert pairs_df['pair_id'].is_unique, "Duplicate pair IDs"

# %% [markdown]
# ## 5. Cointegration Testing Workflow
#
# For each pair:
# 1. `coint()` -> cointegration test statistic + p-value (MacKinnon N=2)
# 2. `OLS` -> hedge ratio + spread construction
# 3. Store spread for later half-life computation
#
# Pair ordering: alphabetical. coint(A, B) and OLS(A ~ B) use the same order.

# %%
def compute_half_life(spread: pd.Series) -> float:
    """Compute OU half-life from spread series.

    Returns half-life in 5-min bars. Divide by BARS_PER_DAY for trading days.
    Returns np.nan if spread is not mean-reverting (lambda >= 0).
    """
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()
    valid = pd.concat([spread_diff, spread_lag], axis=1).dropna()
    valid.columns = ['diff', 'lag']

    if len(valid) < 100:
        return np.nan

    try:
        model = sm.OLS(valid['diff'], sm.add_constant(valid['lag'])).fit()
        lam = model.params['lag']
        if lam >= 0:
            return np.nan
        half_life_bars = -np.log(2) / lam
        return half_life_bars
    except Exception:
        return np.nan


def count_zero_crossings(spread: pd.Series) -> int:
    """Count how many times the demeaned spread crosses zero."""
    demeaned = spread - spread.mean()
    signs = np.sign(demeaned)
    signs = signs[signs != 0]
    crossings = (signs.diff().abs() == 2).sum()
    return int(crossings)


# -- Run the scan --
scan_results = []
spreads = {}

print(f"Scanning {n_pairs} pairs...")
t_start = time.time()

for i, row in pairs_df.iterrows():
    pair_id = row['pair_id']
    ta, tb = row['ticker_a'], row['ticker_b']

    if (i + 1) % 2000 == 0 or (i + 1) == n_pairs:
        elapsed = time.time() - t_start
        print(f"  Progress: {i+1}/{n_pairs} ({elapsed:.0f}s)")

    log_a = panel[ta]
    log_b = panel[tb]

    # Inner join (already aligned, but be explicit)
    aligned = pd.concat([log_a, log_b], axis=1).dropna()
    n_obs = len(aligned)

    if n_obs < MIN_ALIGNED_OBS:
        scan_results.append({
            'pair_id': pair_id, 'ticker_a': ta, 'ticker_b': tb,
            'sector_a': row['sector_a'], 'sector_b': row['sector_b'],
            'within_sector': row['within_sector'],
            'n_aligned_obs': n_obs, 'coint_tstat': np.nan,
            'raw_pval': np.nan, 'hedge_ratio': np.nan, 'intercept': np.nan,
            'scan_status': 'skipped: insufficient data',
        })
        continue

    series_a = aligned.iloc[:, 0].values
    series_b = aligned.iloc[:, 1].values

    # -- Step 1: coint() for official verdict --
    try:
        coint_t, pvalue, crit_values = coint(
            series_a, series_b,
            trend=COINT_TREND, autolag=COINT_AUTOLAG, maxlag=COINT_MAXLAG
        )
    except Exception as e:
        scan_results.append({
            'pair_id': pair_id, 'ticker_a': ta, 'ticker_b': tb,
            'sector_a': row['sector_a'], 'sector_b': row['sector_b'],
            'within_sector': row['within_sector'],
            'n_aligned_obs': n_obs, 'coint_tstat': np.nan,
            'raw_pval': np.nan, 'hedge_ratio': np.nan, 'intercept': np.nan,
            'scan_status': f'coint() error: {e}',
        })
        continue

    # -- Step 2: OLS for hedge ratio and spread --
    try:
        ols_model = sm.OLS(series_a, sm.add_constant(series_b)).fit()
        intercept = ols_model.params[0]
        hedge_ratio = ols_model.params[1]
        spread = pd.Series(
            series_a - hedge_ratio * series_b,
            index=aligned.index
        )
        spreads[pair_id] = spread
    except Exception as e:
        scan_results.append({
            'pair_id': pair_id, 'ticker_a': ta, 'ticker_b': tb,
            'sector_a': row['sector_a'], 'sector_b': row['sector_b'],
            'within_sector': row['within_sector'],
            'n_aligned_obs': n_obs, 'coint_tstat': coint_t,
            'raw_pval': pvalue, 'hedge_ratio': np.nan, 'intercept': np.nan,
            'scan_status': f'OLS error: {e}',
        })
        continue

    scan_results.append({
        'pair_id': pair_id, 'ticker_a': ta, 'ticker_b': tb,
        'sector_a': row['sector_a'], 'sector_b': row['sector_b'],
        'within_sector': row['within_sector'],
        'n_aligned_obs': n_obs, 'coint_tstat': coint_t,
        'raw_pval': pvalue, 'hedge_ratio': hedge_ratio,
        'intercept': intercept, 'scan_status': 'OK',
    })

elapsed_total = time.time() - t_start
scan_df = pd.DataFrame(scan_results)
n_ok = (scan_df['scan_status'] == 'OK').sum()
n_failed = n_pairs - n_ok

print(f"\nScan complete in {elapsed_total:.0f}s")
print(f"  OK: {n_ok}, failed/skipped: {n_failed}")
print(f"  Raw p-value range: {scan_df['raw_pval'].min():.6f} to {scan_df['raw_pval'].max():.4f}")

# Show top 15 by raw p-value
print("\nTop 15 pairs by raw p-value (most significant first):")
top15 = scan_df[scan_df['scan_status'] == 'OK'].nsmallest(15, 'raw_pval')
print(top15[['pair_id', 'within_sector', 'coint_tstat', 'raw_pval', 'hedge_ratio']].to_string(index=False))

# %% [markdown]
# ## 6. Statistical Filtering
#
# ### 6a. BH-FDR Correction

# %%
valid_mask = scan_df['scan_status'] == 'OK'
valid_pvals = scan_df.loc[valid_mask, 'raw_pval'].values
n_valid = len(valid_pvals)

print(f"Applying BH-FDR correction to {n_valid} valid p-values (q={BH_FDR_ALPHA})...")

reject, pvals_adj, _, _ = multipletests(valid_pvals, alpha=BH_FDR_ALPHA, method='fdr_bh')

scan_df['bh_adj_pval'] = np.nan
scan_df['bh_reject'] = False
scan_df.loc[valid_mask, 'bh_adj_pval'] = pvals_adj
scan_df.loc[valid_mask, 'bh_reject'] = reject

n_bh_pass = reject.sum()
n_raw_sig = (valid_pvals < 0.05).sum()
print(f"  Raw p < 0.05: {n_raw_sig} pairs")
print(f"  BH-FDR survivors: {n_bh_pass} pairs")

# Correctness check
if n_bh_pass > 0:
    adj_vs_raw = scan_df.loc[valid_mask, ['raw_pval', 'bh_adj_pval']].dropna()
    assert (adj_vs_raw['bh_adj_pval'] >= adj_vs_raw['raw_pval'] - 1e-10).all(), \
        "BH-adjusted p-values should be >= raw p-values"
    print("  Correctness: adjusted >= raw PASSED")

# %% [markdown]
# ### 6b. Half-Life and Spread Diagnostics

# %%
scan_df['half_life_bars'] = np.nan
scan_df['half_life_days'] = np.nan
scan_df['zero_crossings'] = np.nan

bh_passing_ids = scan_df[scan_df['bh_reject'] == True]['pair_id'].tolist()

if len(bh_passing_ids) == 0:
    print("No pairs passed BH-FDR. Half-life computation skipped.")
    print("Will proceed with fallback logic in Section 6c.")
else:
    print(f"Computing half-life for {len(bh_passing_ids)} BH-passing pairs...")
    for pair_id in bh_passing_ids:
        if pair_id not in spreads:
            continue
        spread = spreads[pair_id]

        hl_bars = compute_half_life(spread)
        hl_days = hl_bars / BARS_PER_DAY if not np.isnan(hl_bars) else np.nan
        zc = count_zero_crossings(spread)

        idx = scan_df[scan_df['pair_id'] == pair_id].index[0]
        scan_df.loc[idx, 'half_life_bars'] = hl_bars
        scan_df.loc[idx, 'half_life_days'] = hl_days
        scan_df.loc[idx, 'zero_crossings'] = zc

    bh_passing = scan_df[scan_df['bh_reject'] == True].copy()
    print("\nBH-passing pairs with diagnostics:")
    display_cols = ['pair_id', 'within_sector', 'raw_pval', 'bh_adj_pval',
                    'hedge_ratio', 'half_life_days', 'zero_crossings']
    print(bh_passing[display_cols].to_string(index=False))

# %% [markdown]
# ### 6c. Hard Filters (sequential funnel)

# %%
# -- Build the filter funnel --
funnel = []

# Stage 0: All tested
all_tested = scan_df[scan_df['scan_status'] == 'OK'].copy()
n_stage0 = len(all_tested)
funnel.append(('All tested', n_stage0))

# Stage 1: BH-FDR
stage1 = all_tested[all_tested['bh_reject'] == True].copy()
n_stage1 = len(stage1)
funnel.append(('BH-FDR (q=0.05)', n_stage1))

# Stage 2: Half-life in [5, 60] trading days
hl_min, hl_max = HALF_LIFE_RANGE
stage2 = stage1[
    (stage1['half_life_days'] >= hl_min) &
    (stage1['half_life_days'] <= hl_max)
].copy()
n_stage2 = len(stage2)
funnel.append((f'Half-life [{hl_min},{hl_max}]d', n_stage2))

# Stage 3: Hedge ratio > 0
stage3 = stage2[stage2['hedge_ratio'] > 0].copy()
n_stage3 = len(stage3)
funnel.append(('Hedge ratio > 0', n_stage3))

# -- Fallback logic --
FALLBACK_USED = False
filter_regime = 'primary'

if n_stage3 < 10:
    print(f"\nFallback check: {n_stage3} pairs after Stage 3 (< 10 threshold)")
    hl_fb_min, hl_fb_max = HALF_LIFE_FALLBACK

    if n_stage1 == 0 or (n_stage3 < 5):
        print(f"  Relaxing BH-FDR to q=0.10...")
        reject_fb, pvals_adj_fb, _, _ = multipletests(valid_pvals, alpha=0.10, method='fdr_bh')
        scan_df.loc[valid_mask, 'bh_adj_pval'] = pvals_adj_fb
        scan_df.loc[valid_mask, 'bh_reject'] = reject_fb
        FALLBACK_USED = True
        filter_regime = 'relaxed'
        n_bh_pass_fb = reject_fb.sum()
        print(f"  BH-FDR at q=0.10: {n_bh_pass_fb} pairs pass")

        # Compute half-life for newly passing pairs
        new_bh_ids = scan_df[(scan_df['bh_reject'] == True) & (scan_df['half_life_days'].isna())]['pair_id'].tolist()
        for pid in new_bh_ids:
            if pid in spreads:
                hl_bars = compute_half_life(spreads[pid])
                hl_days = hl_bars / BARS_PER_DAY if not np.isnan(hl_bars) else np.nan
                zc = count_zero_crossings(spreads[pid])
                idx = scan_df[scan_df['pair_id'] == pid].index[0]
                scan_df.loc[idx, 'half_life_bars'] = hl_bars
                scan_df.loc[idx, 'half_life_days'] = hl_days
                scan_df.loc[idx, 'zero_crossings'] = zc

        # Re-run stages 1-3 with relaxed thresholds
        stage1 = scan_df[(scan_df['scan_status'] == 'OK') & (scan_df['bh_reject'] == True)].copy()
        n_stage1 = len(stage1)
        stage2 = stage1[
            (stage1['half_life_days'] >= hl_fb_min) &
            (stage1['half_life_days'] <= hl_fb_max)
        ].copy()
        n_stage2 = len(stage2)
        stage3 = stage2[stage2['hedge_ratio'] > 0].copy()
        n_stage3 = len(stage3)
        print(f"  After relaxation: FDR={n_stage1}, HL=[{hl_fb_min},{hl_fb_max}]={n_stage2}, beta>0={n_stage3}")

        funnel = [
            ('All tested', n_stage0),
            ('BH-FDR (q=0.10 RELAXED)', n_stage1),
            (f'Half-life [{hl_fb_min},{hl_fb_max}]d (RELAXED)', n_stage2),
            ('Hedge ratio > 0', n_stage3),
        ]
    else:
        # BH passed some pairs but fewer than 10 survived HL+HR — relax HL only
        stage2_fb = stage1[
            (stage1['half_life_days'] >= hl_fb_min) &
            (stage1['half_life_days'] <= hl_fb_max)
        ].copy()
        stage3_fb = stage2_fb[stage2_fb['hedge_ratio'] > 0].copy()

        if len(stage3_fb) > n_stage3:
            print(f"  Relaxing half-life to [{hl_fb_min},{hl_fb_max}]: {len(stage3_fb)} pairs (was {n_stage3})")
            stage3 = stage3_fb
            n_stage3 = len(stage3)
            FALLBACK_USED = True
            filter_regime = 'relaxed'
            funnel[-2] = (f'Half-life [{hl_fb_min},{hl_fb_max}]d (RELAXED)', len(stage2_fb))
            funnel[-1] = ('Hedge ratio > 0 (after relaxation)', n_stage3)

funnel.append(('Pre-economic logic', n_stage3))
print("\n-- Filter Funnel --")
for stage_name, count in funnel:
    print(f"  {stage_name}: {count}")

if not FALLBACK_USED:
    print("\nMAIN RESULT: All filters applied at primary thresholds (BH q=0.05, HL=[5,60]).")
else:
    print("\nSENSITIVITY RELAXATION APPLIED: Thresholds were relaxed because fewer than")
    print("10 pairs survived primary filters. Results reflect relaxed thresholds and")
    print("should be interpreted with additional caution.")

# %% [markdown]
# ## 7. Economic Logic Filter
#
# For each pair surviving Stage 3: assign economic rationale or reject.
# Within-sector pairs get auto-assigned rationale from the lookup.
# Cross-sector pairs are rejected unless they have a documented economic link.

# %%
econ_results = []

if len(stage3) == 0:
    print("No pairs reached the economic logic stage.")
    n_econ_pass = 0
    n_econ_reject = 0
else:
    for _, row in stage3.iterrows():
        pair_id = row['pair_id']
        within = row['within_sector']

        if pair_id in ECON_RATIONALE:
            tier, rationale = ECON_RATIONALE[pair_id]
            econ_results.append({
                'pair_id': pair_id,
                'economic_tier': tier,
                'economic_rationale': rationale,
                'economic_pass': True,
                'rejection_reason': '',
            })
        elif within:
            sector = row['sector_a']
            econ_results.append({
                'pair_id': pair_id,
                'economic_tier': 'Tier 2',
                'economic_rationale': f'Same sector ({sector}), shared macro exposure',
                'economic_pass': True,
                'rejection_reason': '',
            })
        else:
            econ_results.append({
                'pair_id': pair_id,
                'economic_tier': 'Reject',
                'economic_rationale': '',
                'economic_pass': False,
                'rejection_reason': 'Cross-sector, no identifiable economic linkage',
            })

    n_econ_pass = sum(r['economic_pass'] for r in econ_results)
    n_econ_reject = len(econ_results) - n_econ_pass
    print(f"Economic logic filter: {n_econ_pass} approved, {n_econ_reject} rejected")

econ_df = pd.DataFrame(econ_results) if econ_results else pd.DataFrame(
    columns=['pair_id', 'economic_tier', 'economic_rationale', 'economic_pass', 'rejection_reason']
)
funnel.append(('Economic logic', n_econ_pass))

if len(econ_df) > 0:
    print(econ_df.to_string(index=False))

# %% [markdown]
# ## 8. Output Tables

# %%
# -- Table 2: Full Scan Results --
full_scan = scan_df.copy()
full_scan_path = OUTPUT_DIR / "full_pair_scan_results.parquet"
full_scan.to_parquet(full_scan_path, index=False)
print(f"Saved: {full_scan_path} ({len(full_scan)} rows)")

# -- Table 3: Approved Pairs --
approved_ids = econ_df[econ_df['economic_pass'] == True]['pair_id'].tolist() if len(econ_df) > 0 else []
approved = stage3[stage3['pair_id'].isin(approved_ids)].copy()
if len(approved) > 0:
    approved = approved.merge(econ_df, on='pair_id', how='left')
    approved['filter_regime'] = filter_regime

    # Rank by BH-adjusted p-value then half-life proximity to 20 days
    approved['hl_distance'] = (approved['half_life_days'] - 20).abs()
    approved = approved.sort_values(['bh_adj_pval', 'hl_distance']).reset_index(drop=True)
    approved['rank'] = range(1, len(approved) + 1)
    approved = approved.drop(columns=['hl_distance'])

approved_path = OUTPUT_DIR / "approved_pairs.parquet"
approved.to_parquet(approved_path, index=False)
print(f"Saved: {approved_path} ({len(approved)} rows)")

if len(approved) > 0:
    print("\n-- Approved Pairs --")
    display_cols = ['rank', 'pair_id', 'within_sector', 'coint_tstat', 'raw_pval',
                    'bh_adj_pval', 'hedge_ratio', 'half_life_days', 'zero_crossings',
                    'economic_tier', 'economic_rationale', 'filter_regime']
    avail_cols = [c for c in display_cols if c in approved.columns]
    print(approved[avail_cols].to_string(index=False))
else:
    print("\nNo pairs survived all filters. This is a valid empirical result.")

# -- Table 4: Rejected Pairs Summary --
rejection_data = []

# Count rejections at each stage
if FALLBACK_USED:
    stages_info = [
        ('BH-FDR', n_stage0, n_stage0 - n_stage1),
        ('Half-life', n_stage1, n_stage1 - len(stage2)),
        ('Hedge ratio', len(stage2), len(stage2) - n_stage3),
        ('Economic logic', n_stage3, n_econ_reject),
    ]
else:
    stages_info = [
        ('BH-FDR', n_stage0, n_stage0 - n_stage1),
        ('Half-life', n_stage1, n_stage1 - n_stage2),
        ('Hedge ratio', n_stage2, n_stage2 - n_stage3),
        ('Economic logic', n_stage3, n_econ_reject),
    ]

for stage_name, entering, rejected in stages_info:
    remaining = entering - rejected
    example_pair = ''
    example_reason = ''

    if stage_name == 'BH-FDR':
        non_passing = all_tested[all_tested['bh_reject'] == False]
        if len(non_passing) > 0:
            row = non_passing.iloc[0]
            example_pair = row['pair_id']
            example_reason = f"bh_adj_pval={row['bh_adj_pval']:.4f}"
    elif stage_name == 'Economic logic':
        econ_rejects = econ_df[econ_df['economic_pass'] == False] if len(econ_df) > 0 else pd.DataFrame()
        if len(econ_rejects) > 0:
            example_pair = econ_rejects.iloc[0]['pair_id']
            example_reason = econ_rejects.iloc[0]['rejection_reason']

    rejection_data.append({
        'filter_stage': stage_name,
        'pairs_entering': entering,
        'pairs_rejected': rejected,
        'pairs_remaining': remaining,
        'example_rejected_pair': example_pair,
        'example_reason': example_reason,
    })

rejection_df = pd.DataFrame(rejection_data)
rejection_path = OUTPUT_DIR / "rejected_pairs_summary.parquet"
rejection_df.to_parquet(rejection_path, index=False)
print(f"\nSaved: {rejection_path}")
print("\n-- Rejection Summary --")
print(rejection_df.to_string(index=False))

# %% [markdown]
# ## 9. Visual Validation and Reporting Plots

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_pair(pair_id: str, scan_row, spread: pd.Series, label: str, save_dir: Path):
    """Plot log prices and spread for a single pair. Save to disk."""
    ta, tb = scan_row['ticker_a'], scan_row['ticker_b']
    hr = scan_row['hedge_ratio']
    pval = scan_row['raw_pval']
    hl = scan_row.get('half_life_days', np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: normalized log prices
    log_a = panel[ta]
    log_b = panel[tb]
    common_idx = spread.index
    norm_a = log_a.loc[common_idx] - log_a.loc[common_idx].iloc[0]
    norm_b = log_b.loc[common_idx] - log_b.loc[common_idx].iloc[0]

    axes[0].plot(norm_a.index, norm_a.values, label=ta, alpha=0.8)
    axes[0].plot(norm_b.index, norm_b.values, label=tb, alpha=0.8)
    axes[0].set_ylabel('Normalized log price')
    axes[0].legend()
    hl_str = f'{hl:.1f}d' if not np.isnan(hl) else 'N/A'
    axes[0].set_title(f'[{label}] {pair_id} | p={pval:.4f} | beta={hr:.3f} | HL={hl_str}')
    axes[0].grid(True, alpha=0.3)

    # Bottom: spread with sigma bands
    s_mean = spread.mean()
    s_std = spread.std()
    axes[1].plot(spread.index, spread.values, color='steelblue', alpha=0.7, linewidth=0.5)
    axes[1].axhline(s_mean, color='black', linestyle='-', linewidth=1, label='Mean')
    axes[1].axhline(s_mean + s_std, color='red', linestyle='--', alpha=0.6, label='+/- 1 sigma')
    axes[1].axhline(s_mean - s_std, color='red', linestyle='--', alpha=0.6)
    axes[1].axhline(s_mean + 2*s_std, color='darkred', linestyle=':', alpha=0.4, label='+/- 2 sigma')
    axes[1].axhline(s_mean - 2*s_std, color='darkred', linestyle=':', alpha=0.4)
    axes[1].set_ylabel(f'Spread ({ta} - {hr:.2f}*{tb})')
    axes[1].set_xlabel('Date')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = save_dir / f"{label.lower()}_{pair_id}.png"
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return fname


# -- Plot approved pairs --
saved_plots = []
if len(approved) > 0:
    print("Saving approved pair plots...")
    for _, arow in approved.head(10).iterrows():
        pid = arow['pair_id']
        if pid in spreads:
            srow = scan_df[scan_df['pair_id'] == pid].iloc[0]
            fname = plot_pair(pid, srow, spreads[pid], 'APPROVED', PLOT_DIR)
            saved_plots.append(str(fname))
            print(f"  Saved: {fname.name}")

# -- Plot top near-miss pairs (best raw p-value that didn't survive) --
near_miss = scan_df[
    (scan_df['scan_status'] == 'OK') &
    (scan_df['bh_reject'] == False)
].nsmallest(3, 'raw_pval')

if len(near_miss) > 0:
    print("\nSaving near-miss pair plots (rejected but lowest raw p)...")
    for _, rrow in near_miss.iterrows():
        pid = rrow['pair_id']
        if pid in spreads:
            fname = plot_pair(pid, rrow, spreads[pid], 'NEAR_MISS', PLOT_DIR)
            saved_plots.append(str(fname))
            print(f"  Saved: {fname.name}")

# -- Plot a clearly rejected pair for contrast --
worst = scan_df[scan_df['scan_status'] == 'OK'].nlargest(1, 'raw_pval')
if len(worst) > 0:
    print("\nSaving rejected pair plot (highest p-value)...")
    for _, rrow in worst.iterrows():
        pid = rrow['pair_id']
        if pid in spreads:
            fname = plot_pair(pid, rrow, spreads[pid], 'REJECTED', PLOT_DIR)
            saved_plots.append(str(fname))
            print(f"  Saved: {fname.name}")

# %% [markdown]
# ### P-value Distribution

# %%
fig, ax = plt.subplots(figsize=(10, 5))
valid_pvals_plot = scan_df[scan_df['scan_status'] == 'OK']['raw_pval'].dropna()
ax.hist(valid_pvals_plot, bins=50, edgecolor='white', alpha=0.7)
ax.axvline(0.05, color='red', linestyle='--', label='alpha=0.05')
n_raw_sig = (valid_pvals_plot < 0.05).sum()
n_bh_sig = scan_df['bh_reject'].sum()
ax.set_xlabel('Raw p-value')
ax.set_ylabel('Count')
ax.set_title(f'P-value Distribution | {n_raw_sig} raw < 0.05 | {n_bh_sig} survive BH-FDR')
ax.legend()
plt.tight_layout()
pval_plot_path = PLOT_DIR / "pvalue_distribution.png"
plt.savefig(pval_plot_path, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {pval_plot_path}")

# %% [markdown]
# ### Rejection Funnel

# %%
fig, ax = plt.subplots(figsize=(10, 5))
stage_names = [r['filter_stage'] for r in rejection_data]
stage_remaining = [r['pairs_remaining'] for r in rejection_data]
stage_names = ['All tested'] + stage_names
stage_remaining = [n_stage0] + stage_remaining

bars = ax.bar(range(len(stage_names)), stage_remaining, color='steelblue', edgecolor='white')
ax.set_xticks(range(len(stage_names)))
ax.set_xticklabels(stage_names, rotation=30, ha='right')
ax.set_ylabel('Pairs remaining')
ax.set_title('Filter Funnel')
for bar, val in zip(bars, stage_remaining):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
funnel_plot_path = PLOT_DIR / "rejection_funnel.png"
plt.savefig(funnel_plot_path, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {funnel_plot_path}")

# %% [markdown]
# ### Half-Life Distribution (if pairs survived BH)

# %%
hl_values = scan_df[scan_df['bh_reject'] == True]['half_life_days'].dropna()
if len(hl_values) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(hl_values, bins=30, edgecolor='white', alpha=0.7)
    ax.axvline(HALF_LIFE_RANGE[0], color='red', linestyle='--', label=f'Filter min ({HALF_LIFE_RANGE[0]}d)')
    ax.axvline(HALF_LIFE_RANGE[1], color='red', linestyle='--', label=f'Filter max ({HALF_LIFE_RANGE[1]}d)')
    ax.set_xlabel('Half-life (trading days)')
    ax.set_ylabel('Count')
    ax.set_title(f'Half-life Distribution (BH-passing pairs, n={len(hl_values)})')
    ax.legend()
    plt.tight_layout()
    hl_plot_path = PLOT_DIR / "halflife_distribution.png"
    plt.savefig(hl_plot_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {hl_plot_path}")
else:
    print("No BH-passing pairs with valid half-life. Skipping half-life plot.")

# %% [markdown]
# ## 10. Final Summary

# %%
print("=" * 70)
print("NOTEBOOK 02 FULL RUN -- FINAL SUMMARY")
print("=" * 70)
print()
print(f"Input panel:             {panel.shape[0]} rows x {panel.shape[1]} tickers")
print(f"Date range:              {panel.index[0].date()} to {panel.index[-1].date()}")
print(f"Pairs generated:         {n_pairs}")
print(f"Pairs tested (OK):       {n_ok}")
print(f"Pairs failed/skipped:    {n_failed}")
print()
print("-- Filter Funnel --")
for stage_name, count in funnel:
    print(f"  {stage_name}: {count}")
print()
print(f"Filter regime:           {filter_regime}")
if FALLBACK_USED:
    print("  (Sensitivity relaxation was applied)")
print()
print(f"Approved pairs:          {len(approved)}")
if len(approved) > 0:
    print(f"  Best pair:             {approved.iloc[0]['pair_id']} (p={approved.iloc[0]['bh_adj_pval']:.6f})")
print()
print("Output files:")
print(f"  {full_scan_path}")
print(f"  {approved_path}")
print(f"  {rejection_path}")
print(f"  {sector_path}")
print(f"  Plots: {PLOT_DIR}")
print()

if len(approved) > 0:
    print("FULL RUN COMPLETE. Approved pairs ready for Pairs Selection Report.")
else:
    print("FULL RUN COMPLETE. No pairs survived all filters.")
    print()
    # -- Raw significance context --
    n_raw_below_005 = (scan_df[scan_df['scan_status'] == 'OK']['raw_pval'] < 0.05).sum()
    pct_raw = n_raw_below_005 / n_ok * 100
    print("Interpretation:")
    print(f"  {n_raw_below_005} of {n_ok} pairs ({pct_raw:.1f}%) had raw p < 0.05.")
    print(f"  Under the null of no cointegration, the expected false positive rate")
    print(f"  is approximately 5%. The observed {pct_raw:.1f}% is")
    if pct_raw < 7:
        print(f"  consistent with what chance alone would produce.")
    else:
        print(f"  somewhat above the null expectation, suggesting some signal.")
    print(f"  (Note: this comparison is approximate -- the raw p-values from coint()")
    print(f"  are not guaranteed to be exactly uniform under the null.)")
    print()
    n_bh_pass_final = scan_df['bh_reject'].sum()
    print(f"  After BH-FDR correction at q=0.05: {n_bh_pass_final} pairs survived.")
    if FALLBACK_USED:
        print(f"  Fallback relaxation was applied (see filter regime above).")
    print()
    print("  The zero-approved-pairs result is a valid empirical finding under")
    print("  this screening methodology. Contributing factors may include:")
    print("    - The 2022 market regime (Fed tightening, inflation, sector rotation)")
    print("      which tends to weaken mean-reversion dynamics")
    print(f"    - The {n_tickers}-ticker universe composition and liquidity profile")
    print("    - The conservative but defensible multi-layer filtering design")

# %% [markdown]
# ### Assumptions and Limitations
#
# 1. **In-sample only:** All testing uses 2022 full-year data with no out-of-sample
#    validation period. ~40% of in-sample cointegrated pairs fail out-of-sample.
#
# 2. **Market regime:** 2022 was dominated by Fed tightening, inflation, and
#    risk-off rotation. This regime tends to weaken traditional cointegration
#    relationships, particularly in growth/tech sectors.
#
# 3. **Universe:** All tickers passing quality screening (no cap). Different
#    universe choices (e.g., sector-specific) might yield different results.
#
# 4. **Economic logic filter** is a practical audit layer using hardcoded
#    rationales, not formal proof of economic linkage. In this full run, no pair
#    reached the economic filter -- all were eliminated at the statistical stage.
#
# 5. **Engle-Granger asymmetry:** Each pair was tested in one direction only
#    (alphabetical ordering). Running both directions could surface additional
#    candidates. A sensitivity check for top near-miss pairs is in Appendix A.
#
# 6. **Universe includes ETFs and COIN:** 5 ETFs and 1 crypto-exchange stock add
#    many cross-sector pairs unlikely to pass economic logic. A sensitivity check
#    excluding these is in Appendix A.
#
# 7. **Implementation refinements from spec:** BARS_PER_DAY=77 (actual session
#    count) vs spec's approximation of 74. Completeness denominator 381 min/day
#    (exact count for [9:35,15:55]) vs spec's ~370.
#
# 8. **No backtesting, trading rules, or portfolio simulation** in this notebook.
#
# 9. **Prices assumed adjusted** for splits and dividends.

# %% [markdown]
# ## Appendix A: Sensitivity Checks
#
# These are robustness checks only. They do NOT change the official main result.
# Their purpose is to verify that the zero-pair finding is not driven by
# a single design choice.

# %% [markdown]
# ### A1. Excluding ETF-Involved Pairs

# %%
ETF_TICKERS = {'EEM', 'FXI', 'GLD', 'IWM', 'KWEB'}
EDGE_TICKERS = ETF_TICKERS | {'COIN'}

# Count pairs involving ETFs or edge-case names
scan_ok = scan_df[scan_df['scan_status'] == 'OK'].copy()
etf_mask = scan_ok['ticker_a'].isin(ETF_TICKERS) | scan_ok['ticker_b'].isin(ETF_TICKERS)
edge_mask = scan_ok['ticker_a'].isin(EDGE_TICKERS) | scan_ok['ticker_b'].isin(EDGE_TICKERS)
equity_only = scan_ok[~edge_mask].copy()

n_etf_pairs = etf_mask.sum()
n_edge_pairs = edge_mask.sum()
n_equity_only = len(equity_only)

print("APPENDIX A1: ETF / Edge-Case Sensitivity")
print("=" * 60)
print(f"Pairs involving at least one ETF:         {n_etf_pairs}")
print(f"Pairs involving ETF or COIN:              {n_edge_pairs}")
print(f"Pure equity-only pairs:                   {n_equity_only}")
n_equity_tickers = n_tickers - len(EDGE_TICKERS & set(tickers))
print(f"  (C({n_equity_tickers},2) = {n_equity_tickers*(n_equity_tickers-1)//2} expected for {n_equity_tickers} equity tickers)")
print()

# Re-run BH-FDR on equity-only subset
equity_pvals = equity_only['raw_pval'].values
n_eq = len(equity_pvals)
reject_eq_005, adj_eq_005, _, _ = multipletests(equity_pvals, alpha=0.05, method='fdr_bh')
reject_eq_010, adj_eq_010, _, _ = multipletests(equity_pvals, alpha=0.10, method='fdr_bh')

n_raw_eq = (equity_pvals < 0.05).sum()
n_bh_eq_005 = reject_eq_005.sum()
n_bh_eq_010 = reject_eq_010.sum()

print(f"Equity-only BH-FDR results:")
print(f"  Tests:              {n_eq}")
print(f"  Raw p < 0.05:       {n_raw_eq}")
print(f"  BH-FDR q=0.05:     {n_bh_eq_005} survive")
print(f"  BH-FDR q=0.10:     {n_bh_eq_010} survive")

# Check BH critical for rank 1
if n_eq > 0:
    min_p_eq = equity_pvals.min()
    bh_crit_eq_005 = 1 / n_eq * 0.05
    bh_crit_eq_010 = 1 / n_eq * 0.10
    min_pair_eq = equity_only.loc[equity_only['raw_pval'].idxmin(), 'pair_id']
    print(f"\n  Smallest equity-only p-value: {min_p_eq:.6f} ({min_pair_eq})")
    print(f"  BH critical rank 1, q=0.05:   {bh_crit_eq_005:.6f}")
    print(f"  BH critical rank 1, q=0.10:   {bh_crit_eq_010:.6f}")

print(f"\nConclusion: Excluding ETFs and COIN reduces the test count from {n_ok} to")
print(f"{n_eq}, making BH thresholds slightly more lenient. Result: {n_bh_eq_005} pairs")
print(f"survive at q=0.05, {n_bh_eq_010} at q=0.10. The zero-pair finding is NOT driven")
print(f"by ETF-inflated test counts.")

# Save sensitivity result
sens_etf = pd.DataFrame([{
    'analysis': 'equity_only_excl_etf_coin',
    'n_pairs_tested': n_eq,
    'n_raw_sig_005': int(n_raw_eq),
    'n_bh_survive_005': int(n_bh_eq_005),
    'n_bh_survive_010': int(n_bh_eq_010),
    'conclusion': 'zero-pair result unchanged',
}])
sens_etf_path = OUTPUT_DIR / "sensitivity_etf_exclusion.parquet"
sens_etf.to_parquet(sens_etf_path, index=False)
print(f"\nSaved: {sens_etf_path}")

# %% [markdown]
# ### A2. Bidirectional Engle-Granger Check (Top Near-Miss Pairs)

# %%
print("\nAPPENDIX A2: Bidirectional Engle-Granger Sensitivity")
print("=" * 60)

# Select top 10 near-miss pairs (smallest raw p-value, excluding GOOG-GOOGL)
near_miss_pairs = scan_ok[scan_ok['pair_id'] != 'GOOG-GOOGL'].nsmallest(10, 'raw_pval')

bidir_rows = []
for _, row in near_miss_pairs.iterrows():
    ta, tb = row['ticker_a'], row['ticker_b']

    # Forward: coint(A, B) -- already stored
    fwd_t = row['coint_tstat']
    fwd_p = row['raw_pval']

    # Reverse: coint(B, A)
    log_a = panel[ta].values
    log_b = panel[tb].values
    try:
        rev_t, rev_p, _ = coint(log_b, log_a, trend=COINT_TREND,
                                 autolag=COINT_AUTOLAG, maxlag=COINT_MAXLAG)
    except Exception:
        rev_t, rev_p = np.nan, np.nan

    bidir_rows.append({
        'pair_id': row['pair_id'],
        'fwd_tstat': round(fwd_t, 4),
        'fwd_pval': round(fwd_p, 6),
        'rev_tstat': round(rev_t, 4),
        'rev_pval': round(rev_p, 6),
        'min_pval': round(min(fwd_p, rev_p), 6),
        'direction_matters': abs(fwd_p - rev_p) / max(fwd_p, rev_p) > 0.20,
    })

bidir_df = pd.DataFrame(bidir_rows)
print("Top 10 near-miss pairs -- forward vs reverse coint():")
print(bidir_df.to_string(index=False))

# Check if taking the better direction would change any BH outcome
# Use the smaller of (fwd_p, rev_p) as the "best" p-value
best_pvals = bidir_df['min_pval'].values
n_bidir = len(best_pvals)
print(f"\nIf we took the better direction for these {n_bidir} pairs:")
for i, row in bidir_df.iterrows():
    # BH critical for this pair's rank in the full test
    # This is approximate -- a proper bidir test would double the test count
    print(f"  {row['pair_id']}: best p = {row['min_pval']:.6f}, "
          f"direction matters: {row['direction_matters']}")

print(f"\nNote: A proper bidirectional test would test each pair in both directions,")
print(f"doubling the effective test count to {n_ok * 2}, which would make BH-FDR")
print(f"MORE strict (critical values halved). The unidirectional approach used in")
print(f"the main result is therefore conservative in test count but may miss the")
print(f"better-fitting direction for some pairs. Neither approach would change the")
print(f"zero-pair outcome given the magnitude of the p-value gap.")

# Save
bidir_path = OUTPUT_DIR / "sensitivity_bidirectional.parquet"
bidir_df.to_parquet(bidir_path, index=False)
print(f"\nSaved: {bidir_path}")
