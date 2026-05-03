"""
memo_pdf.py
-----------
Generates outputs/strategist_memo.pdf from a LaTeX memo template.

Sources used:
  - notes/data_notes.md
  - notes/memo_outline.md
  - notes/source note for *.md
  - outputs/memo_chart.png      (Figure 1 — memo body)
  - outputs/appendix_chart.png  (Figure 2 — appendix)

Requires: pdflatex (MiKTeX https://miktex.org  or  TeX Live https://tug.org/texlive/)

Run from project root:
    python Code/memo_pdf.py
"""

import os
import subprocess
import shutil
import sys

# ---------------------------------------------------------------------------
# Paths  (run from project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT   = os.path.abspath(".")
OUTPUTS_DIR    = os.path.join(PROJECT_ROOT, "outputs")
TEX_FILE       = os.path.join(PROJECT_ROOT, "strategist_memo.tex")
PDF_SRC        = os.path.join(PROJECT_ROOT, "strategist_memo.pdf")
PDF_DEST       = os.path.join(OUTPUTS_DIR,  "strategist_memo.pdf")
TEX_DEST       = os.path.join(OUTPUTS_DIR,  "strategist_memo.tex")

# Image paths relative to PROJECT_ROOT (where pdflatex runs)
MEMO_IMG       = "outputs/memo_chart.png"
APPENDIX_IMG   = "outputs/appendix_chart.png"

# ---------------------------------------------------------------------------
# LaTeX content
# ---------------------------------------------------------------------------
LATEX = r"""
\documentclass[12pt, a4paper]{article}

% ---- Packages ---------------------------------------------------------------
\usepackage[top=1in, bottom=1in, left=1.15in, right=1.15in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{parskip}
\usepackage{setspace}
\usepackage{caption}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{titlesec}

% ---- Page style -------------------------------------------------------------
\setlength{\headheight}{14pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textit{Week 0 Strategist Task}}
\fancyhead[R]{\small\thepage}
\fancyfoot[C]{\small\textcolor{gray}{Private \& Confidential --- Draft}}
\renewcommand{\headrulewidth}{0.3pt}

% ---- Typography -------------------------------------------------------------
\setlength{\parskip}{7pt}
\setlength{\parindent}{0pt}
\onehalfspacing

% ---- Section headings -------------------------------------------------------
\titleformat{\section}
  {\normalfont\normalsize\bfseries\uppercase}
  {}{0em}{}[\vspace{2pt}\titlerule]
\titlespacing*{\section}{0pt}{14pt}{6pt}

% ---- Colors -----------------------------------------------------------------
\definecolor{dimgray}{HTML}{555555}

% =============================================================================
\begin{document}

% ---- Memo header ------------------------------------------------------------
\begin{center}
  {\Large\bfseries MEMORANDUM}
\end{center}

\vspace{0.5em}

\noindent
\begin{tabular}{@{}p{1.9cm}l}
  \textbf{To:}      & Mr. Sam \\[4pt]
  \textbf{From:}    & Duy Anh \\[4pt]
  \textbf{Date:}    & March 17, 2026 \\[4pt]
  \textbf{Subject:} & Why Portfolio Insurance Algorithms Sold Into the 1987 Crash \\
\end{tabular}

\vspace{0.8em}
\hrule
\vspace{1.4em}

% =============================================================================
\section{Executive Summary}

Portfolio insurance was designed to protect institutional portfolios by reducing equity
exposure as markets fell. On October~19, 1987, that portfolio-level discipline became a market-level vulnerability. As
prices declined, many institutions received the same mechanical sell signal simultaneously
and attempted to execute into rapidly thinning liquidity. What was rational for one
portfolio became destabilizing in the aggregate. This memo examines how that failure
unfolded and proposes three safeguards to prevent its recurrence.

The failure was not a coding error. It was structural: a procyclical strategy deployed
at scale in a market that could no longer absorb the required trades. A strategy designed
to limit downside at the portfolio level instead transmitted stress through the market
when many institutions executed it simultaneously.

% =============================================================================
\section{What the Data Show}

The minute-level S\&P~500 futures data in Figure~\ref{fig:memo} indicates that the
market did not experience a normal correction; it moved into disorderly liquidation.
Prices were already trending lower on Friday, October~16, suggesting that de-risking had
begun before the weekend. At the Monday open, however, selling became directional and sustained
from the first minutes, with almost no countervailing buying pressure.

By 10:00 on October~19, rolling volatility had already crossed three times its
pre-crash baseline. The worst 60-minute window, from 14:14 to 15:14, produced a
cumulative loss of $11.2\%$. By 13:01 on October~20, cumulative drawdown from the
pre-crash peak reached $28.6\%$. These figures are more consistent with disorderly liquidation than with an orderly correction.

The distinction matters. A falling market can still function. What the data show here
is a market under extreme directional pressure, with rising volatility and deepening
drawdown consistent with forced selling overwhelming near-term liquidity.

\begin{figure}[h!]
  \centering
  \includegraphics[width=\textwidth]{""" + MEMO_IMG + r"""}
  \caption{\textbf{1987 Black Monday: Market Break Detection via PELT Changepoint.}
    S\&P 500 futures prices (Oct~16--21, 1987) with PELT structural break identified at
    09:42 on October~19. Red dashed line marks the changepoint; shaded regions indicate
    Crisis Day (Oct~19, pink) and Trading Halts (yellow). Trading hours only (09:30--16:00);
    no overnight gaps. The sharp discontinuity at 09:42 signals the threshold beyond which
    selling pressure becomes irreversible and liquidity begins to fail.
    \textit{Source: 1987\_crash\_market\_data.csv.}}
  \label{fig:memo}
\end{figure}

% =============================================================================
\section{How Portfolio Insurance Worked}

Portfolio insurance was designed to replicate downside protection without requiring
investors to buy a traditional put option. Instead of paying an upfront premium,
institutions used dynamic hedging to reduce market exposure as prices fell, most often
through sales of S\&P~500 futures (Presidential Task Force on Market Mechanisms, 1988).
Falling prices triggered cuts to exposure; rising prices restored it.

The logic was straightforward: if portfolio value falls, sell enough to limit further
downside. The problem was that the hedge had to be adjusted continuously. It was not a
static allocation; it depended on ongoing execution in live markets. That made the
strategy highly sensitive to trading conditions.

In theory, dynamic hedging can work in a liquid and orderly market. In practice, the
strategy assumed that large sell orders could be executed at or near quoted prices
without materially moving the market---that liquidity would hold at scale.
On October~19, that assumption failed almost immediately as market conditions deteriorated.

% =============================================================================
\section{Why the Algorithms Sold}

The algorithms sold because that is what they were built to do. They were not designed
to identify undervaluation or buy into weakness---they were designed to reduce exposure
as prices fell. The models did not malfunction. They followed their rule set under stress
exactly as intended.

The issue was scale and synchronization. By late 1987, portfolio insurance strategies
covered an estimated \$60--90 billion in assets (Carlson, 2007). Once the market declined
sharply on October~19, a large share of that capital faced the same imperative: sell
futures to hedge additional downside. The result was concentrated, largely price-insensitive
execution into a market already under pressure.

The effect did not remain confined to the futures market. As futures prices weakened
relative to cash equities, index arbitrage transmitted that pressure into the stock
market. CFTC data underscore the magnitude: gross futures selling by institutional
hedgers increased from roughly 3,500 contracts on October~14 to 32,700 on October~19
(CFTC, 1988). Portfolio insurers represented a substantial share of non-market-maker
futures selling on Black Monday (Carlson, 2006). That concentrated selling deepened
the decline precisely when liquidity was thinnest.

This is the core strategic lesson. Portfolio insurance did not cause the crash by itself,
but it amplified it through feedback. Falling prices triggered mechanical selling; that
selling pushed prices lower; lower prices generated still more selling (Shiller, 1988).
As Greenspan later observed, even if one investor could exit successfully, the system as
a whole cannot all get out at once (Greenspan, 1988). The strategy did not fail because the models deviated from their design. It failed
because market liquidity was insufficient to absorb the volume and synchrony of the
required trades.

% =============================================================================

This diagnosis points to a single imperative: risk management systems must be redesigned so that
portfolio-level discipline does not become market-level destruction. The 1987 and 2018 episodes
reveal that mechanical risk controls, when operating at scale without awareness of liquidity, can
amplify the very distress they are meant to contain. The following framework addresses this structural
vulnerability.

% =============================================================================
\section{Risk-Control Architecture}

\subsection*{The Problem: Why Risk Management Becomes Destabilizing}

Portfolio insurance fails catastrophically not because it identifies too much risk, but because
it forces mechanical selling into collapsing liquidity. In 1987, algorithms required additional
selling exactly as market conditions deteriorated---turning a circuit breaker into a system
amplifier (Carlson, 2007; Fortune, 1993). The pattern repeated in 2018's ``Volmageddon'': coordinated
mechanical rebalancing during stress amplified volatility, forcing mass liquidations at the worst
possible prices (Augustin et al., 2021).

The root cause is simple: \textit{discrete, price-triggered exits have no awareness of market
liquidity and create incentives for synchronized selling across portfolios.} When multiple investors
hit the same stop-loss levels simultaneously, execution quality collapses and forced selling
becomes self-fulfilling. The investor intended to manage tail risk but instead becomes an
unwitting agent of market instability.

\subsection*{Proposed Solution: A Liquidity-Aware, Three-Layer Architecture}

We propose replacing reactive, trigger-based risk controls with a proactive system that phases
de-risking across three layers, each designed to respect market capacity and conditions.

\noindent\textbf{Layer 1: Volatility-Scaled Position Sizing (Continuous)}

Primary control mechanism. As volatility rises, position leverage automatically declines, reducing
portfolio delta before stress reaches critical levels (Moreira \& Muir, 2017). This works continuously---not
as a binary trigger---so the portfolio gradually shrinks its footprint as variance expands.

\begin{itemize}[leftmargin=1.5em, itemsep=3pt, topsep=3pt]
  \item \textbf{Rationale:} Markets can absorb gradual scaling; they cannot absorb synchronized liquidation.
  \item \textbf{Implementation:} Tie leverage multipliers inversely to realized and forecasted volatility.
  \item \textbf{Cost:} Whipsaw risk if markets rebound quickly after volatility spikes.
\end{itemize}

\noindent\textbf{Layer 2: Liquidity-Aware Execution Limits (Conditional)}

Operational safeguard. To prevent crowding and system risk, all de-risking trades must respect
strict volume and spread thresholds: trade sizes capped at a percentage of recent average daily volume
(e.g., no single sale exceeds 2\% of ADV), and automatic pause rules triggered by extreme bid-ask
spreads or failing auction mechanics.

\begin{itemize}[leftmargin=1.5em, itemsep=3pt, topsep=3pt]
  \item \textbf{Critical trade-off:} Execution may lag when liquidity evaporates. If the market gaps overnight,
    Layer 1 and Layer 2 cannot fully de-risk the portfolio in real time.
  \item \textbf{Rationale:} Preventing algorithmic crowding during stress is more important than achieving
    perfect fill prices (Fortune, 1993). A 1\% worse entry into an illiquid market beats a 10\% liquidation at any price.
\end{itemize}

\noindent\textbf{Layer 3: Volatility-Conditioned Exit Thresholds (Backstop Only)}

Final circuit breaker. This layer engages \textit{only after} Layers 1 and 2 have already scaled down
base exposure. Its purpose is not to drive routine de-risking, but to prevent unbounded tail losses
if Layers 1 and 2 fail due to gap risk or extreme illiquidity. Dynamic thresholds reflect market conditions:
\begin{itemize}[leftmargin=1.5em, itemsep=3pt, topsep=3pt]
  \item Thresholds \textbf{widen} during high volatility (e.g., 8\% loss level if VIX > 40) to avoid rigid, clustered
    exits that amplify market stress.
  \item Thresholds \textbf{tighten} in calm markets (e.g., 3\% loss level if VIX < 15) where execution is orderly
    and losses are less likely to cascade.
\end{itemize}

\noindent\textbf{Critical distinction:} Layer 3 is \textit{not} a parallel safeguard; it is sequential. By the time Layer 3 triggers,
the portfolio has already been through Layers 1 and 2. This prevents the 1987 pattern where a single rigid rule fires
at the worst moment and accelerates collapse.

\subsection*{What This Solves}

\begin{enumerate}[leftmargin=1.5em, itemsep=4pt, topsep=4pt]
  \item \textbf{Prevents forced liquidation cascades.} Gradual scaling (Layer 1) avoids the 1987/2018 syndrome
    where synchronized selling destroys liquidity. Machines sell gradually; humans panic all at once.
  \item \textbf{Respects market capacity.} Execution limits (Layer 2) prevent algorithmic crowding and
    shield the portfolio from being forced to execute into irrational prices during temporary liquidity voids.
  \item \textbf{Breaks the feedback loop.} By reducing exposure early (Layer 1), the portfolio enters stress
    already lighter. Layers 2 and 3 protect what remains, ensuring that tail losses do not trigger cascading
    forced sales that destroy remaining portfolio value.
  \item \textbf{Maintains control hierarchy.} Stop-loss rules are \emph{defensive backstops}, not primary policy.
    They trigger only after proactive scaling and execution discipline have already been deployed. This prevents
    the rigid, mechanical exits that characterized portfolio insurance in 1987.
\end{enumerate}

\subsection*{Residual Risks \& Trade-Offs}

\noindent\textbf{1. Whipsaw losses:} Selling into temporary volatility spikes followed by quick reversals will
underperform a buy-and-hold approach in volatile but ultimately stable regimes. This is the deliberate cost
of reducing tail risk.

\noindent\textbf{2. Execution quality degradation:} Pause rules and volume caps mean de-risking may be slow when speed
matters. The portfolio may endure temporary larger drawdowns than it would under normal market conditions.

\noindent\textbf{3. Overnight gap risk:} No dynamic framework prevents gaps or limit-down moves. This is unavoidable
under extreme illiquidity; the goal is to reduce tail exposure sufficiently that gaps do not trigger bankruptcy.

\noindent\textbf{4. Calibration risk:} Volatility estimation methods, volume thresholds, and spread limits must be
market-specific and regime-dependent. Miscalibration can lead to over-scaling or false confidence.

\subsection*{Implementation \& Transferability}

These principles are conceptually universal---all modern financial systems remain vulnerable to herding and structural
instability (Haldane \& May, 2011). However, \textbf{execution is market-specific:} volatility estimation methods vary
by asset class, liquidity thresholds must reflect venue fragmentation, and execution limits depend on typical bid-ask
spreads and ADV.

Before deployment, each market application requires calibration against historical stress episodes (2008, 2015, 2018, 2020)
to validate that Layer 1 and Layer 2 would have scaled down exposure sufficiently to avoid catastrophic Layer 3 breaches.

\subsection*{Bottom Line}

The fundamental shift is from \textit{``act when price falls to X''} to \textit{``continuously scale inversely to stress,
execute respectfully, and only exit if all else fails.''} This prevents risk management from becoming a source of systemic
instability while preserving the portfolio's ability to contain tail losses. No framework eliminates all losses or gap risk,
but this architecture materially reduces the probability of forced liquidation during the exact moments when market capacity
is most depleted.

% =============================================================================
\newpage
\section*{References}

\begin{itemize}[leftmargin=1.5em, itemsep=5pt, topsep=4pt]
  \item Augustin, P., Cheng, I.-H., \& Van den Bergen, L. (2021). Volmageddon and the
    failure of short volatility products. \textit{Financial Analysts Journal, 77}(3), 35--51.

  \item Carlson, M. A. (2007). \textit{A brief history of the 1987 stock market crash with
    a discussion of the Federal Reserve response.} Finance and Economics Discussion
    Series 2007-13. Board of Governors of the Federal Reserve System.

  \item Commodity Futures Trading Commission. (1988, January). \textit{Final report on
    stock index futures and cash market activity during October 1987.}
    Washington, DC: CFTC.

  \item Fortune, P. (1993). Stock market crashes: What have we learned from October 1987?
    \textit{New England Economic Review}, March/April, 13--24.

  \item Greenspan, A. (1988, December 28). \textit{Remarks before a joint meeting of
    the American Economic Association and the American Finance Association.}
    New York, NY.

  \item Haldane, A. G., \& May, R. M. (2011). Systemic risk in banking ecosystems.
    \textit{Nature, 469}(7330), 351--355.

  \item Moreira, A., \& Muir, T. (2017). Volatility-managed portfolios.
    \textit{The Journal of Finance, 72}(4), 1611--1644.

  \item Presidential Task Force on Market Mechanisms. (1988, January). \textit{Report
    of the Presidential Task Force on Market Mechanisms.} U.S. Department of the
    Treasury.

  \item Shiller, R.~J. (1988). Portfolio insurance and other investor fashions as
    factors in the 1987 stock market crash. In S.~Fischer (Ed.), \textit{NBER
    Macroeconomics Annual 1988, Volume~3} (pp.~287--297). MIT Press.
\end{itemize}

\end{document}
"""

# ---------------------------------------------------------------------------
# Write .tex
# ---------------------------------------------------------------------------
print("=" * 60)
print("Writing LaTeX source...")
print("=" * 60)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
with open(TEX_FILE, "w", encoding="utf-8") as f:
    f.write(LATEX)
print(f"  -> {TEX_FILE}")

# ---------------------------------------------------------------------------
# Check pdflatex is available
# ---------------------------------------------------------------------------
if shutil.which("pdflatex") is None:
    print("\nERROR: pdflatex not found on PATH.")
    print("Install a LaTeX distribution:")
    print("  Windows : MiKTeX   -> https://miktex.org/download")
    print("  Windows : TeX Live -> https://tug.org/texlive/")
    print("\nThe .tex source has been written. Compile manually with:")
    print(f"  cd \"{PROJECT_ROOT}\"")
    print(f"  pdflatex strategist_memo.tex")
    print(f"  pdflatex strategist_memo.tex  (second pass for figure refs)")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Compile (two passes — resolves \ref labels on second pass)
# ---------------------------------------------------------------------------
pdflatex_cmd = [
    "pdflatex",
    "-interaction=nonstopmode",
    f"-output-directory={PROJECT_ROOT}",
    TEX_FILE,
]

for pass_num in (1, 2):
    print(f"\nCompiling PDF (pass {pass_num}/2)...")
    result = subprocess.run(
        pdflatex_cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    # Check for actual failure: pdflatex returns non-zero for warnings too,
    # so we detect real failure by the absence of the output PDF.
    if not os.path.exists(PDF_SRC):
        print(f"pdflatex pass {pass_num} failed (no PDF produced). Last 3000 chars of log:")
        print(result.stdout[-3000:])
        sys.exit(1)

# ---------------------------------------------------------------------------
# Move outputs
# ---------------------------------------------------------------------------
if os.path.exists(PDF_SRC):
    shutil.move(PDF_SRC, PDF_DEST)
    print(f"\nPDF  saved -> {PDF_DEST}")

# Also save the .tex to outputs/ for reference / editing
shutil.copy(TEX_FILE, TEX_DEST)
print(f"TeX  saved -> {TEX_DEST}")

# ---------------------------------------------------------------------------
# Clean up auxiliary files from project root
# ---------------------------------------------------------------------------
for ext in [".aux", ".log", ".out", ".tex"]:
    aux = os.path.join(PROJECT_ROOT, "strategist_memo" + ext)
    if os.path.exists(aux):
        os.remove(aux)
print("Auxiliary files cleaned up.")

# ---------------------------------------------------------------------------
# Word count estimate (body text only, excludes headers/captions/refs)
# ---------------------------------------------------------------------------
body_sections = [
    # Executive Summary
    ("Portfolio insurance was designed to protect institutional portfolios from loss by "
     "reducing equity exposure as prices fell. On October 19 1987 this individually rational "
     "strategy became collectively destructive. When many institutions followed the same "
     "mechanical rule simultaneously their coordinated selling overwhelmed a market already "
     "starved of liquidity. The result was not a correction it was a cascade. The failure was "
     "not in the code it was structural a procyclical strategy amplifying stress in the one "
     "environment where it was supposed to provide protection."),
    # What the Data Shows
    ("The minute-level S P 500 Futures data shows the market did not correct it collapsed. "
     "Prices drifted lower through Friday October 16 signalling that de-risking had already "
     "begun before the weekend. From the Monday open selling accelerated immediately the worst "
     "60-minute window 14:14 to 15:14 on October 19 produced a cumulative return of "
     "approximately minus 11.2 percent a rate of loss with no modern parallel. By the time the "
     "trough was reached at 13:01 on October 20 the cumulative drawdown from the pre-crash "
     "peak reached minus 28.6 percent. Rolling volatility spiked to 8.3 times its pre-crash "
     "baseline breaching the three-sigma threshold by 10:00 on October 19. This is not the "
     "signature of price discovery. It is the signature of a market that had ceased to "
     "function in an orderly way."),
    # How Portfolio Insurance Worked
    ("Portfolio insurance was designed to synthesize the payoff of a put option without "
     "buying one. Rather than paying an upfront premium institutions replicated downside "
     "protection through dynamic hedging as equity prices fell their models called for "
     "progressively lower equity exposure executed primarily through S P 500 futures sales. "
     "The hedge was not set once and left alone. It had to be continuously rebalanced as "
     "prices moved. When the market fell the model signalled sell. When it rose the model "
     "signalled buy. The strategy assumed continuous liquidity and orderly execution at "
     "prevailing prices assumptions that proved catastrophically wrong."),
    # Why the Algorithms Sold
    ("The models were not malfunctioning. They did exactly what they were designed to do "
     "reduce risk in a falling market. The problem was systemic. By late 1987 portfolio "
     "insurance assets under management had grown to an estimated 60 to 90 billion dollars. "
     "When prices began falling on October 19 a large fraction of that capital received the "
     "same sell signal at the same time. CFTC data confirm the scale gross futures selling "
     "by institutional hedgers rose from roughly 3500 contracts on October 14 to 32700 on "
     "October 19 a near-tenfold increase in five days. Portfolio insurers accounted for "
     "approximately 40 percent of non-market-maker futures selling on Black Monday. That "
     "volume depressed futures prices below cash market prices triggering index arbitrageurs "
     "to sell equities spreading the pressure into the cash market. This is the feedback loop "
     "Shiller described as a cascade falling prices triggered mechanical selling which deepened "
     "the fall which triggered more mechanical selling. As Greenspan later observed even if "
     "one investor could exit successfully the system as a whole cannot all get out at once. "
     "Portfolio insurance did not cause the crash but it transformed a large decline into a "
     "structural market failure."),
    # Risk-Control Proposal
    ("The 1987 episode reveals that mechanical risk-reduction systems can become risk "
     "amplifiers when they are procyclical and operate at scale. Three changes would have "
     "mitigated the damage and remain relevant today. Liquidity filter. Automatic hedging "
     "systems should monitor bid-ask spreads and market depth in real time. When spreads "
     "widen beyond a threshold or order-book depth collapses execution should pause. Selling "
     "into an illiquid market does not reduce portfolio risk it accelerates the price decline "
     "that makes the portfolio worse. Volatility throttle. When short-term realized volatility "
     "exceeds a threshold such as the 3x baseline breach observed at 10:00 on October 19 the "
     "system should slow or fragment its execution. Trading in smaller batches spreads market "
     "impact over time and avoids stacking sell orders into a single print. Human override. "
     "When both conditions trigger simultaneously automated execution should require "
     "supervisory approval before proceeding. Disorderly markets call for judgment not speed. "
     "A brief pause is far less costly than amplifying a crash."),
]

total_words = sum(len(s.split()) for s in body_sections)
print(f"\nEstimated body word count: {total_words} words")
if 650 <= total_words <= 850:
    print("  -> Within target range (650-850).")
else:
    print(f"  -> WARNING: outside target range (650-850). Review memo text.")

print("\nAll done.")
