import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #

@dataclass
class UE:
    """Single User Equipment agent."""
    ue_id:      int
    valuation:  float          # v_i  (scalar, abstracted from rate * alpha)
    budget:     float          # psi_i
    n_channels: int            # N_i  (sub-channels required)
    strategy:   str            # 'truthful' | 'heuristic' | 'llm' | 'opt_linear'

    # runtime state
    remaining_budget: float    = field(init=False)
    accumulated_utility: float = field(init=False, default=0.0)
    wins: int                  = field(init=False, default=0)
    active: bool               = field(init=False, default=True)

    # for heuristic
    consecutive_failures: int  = field(init=False, default=0)

    # for opt_linear (set after offline optimisation)
    w0_star: Optional[float]   = field(init=False, default=None)

    def __post_init__(self):
        self.remaining_budget = self.budget


@dataclass
class AuctionRound:
    """Records the outcome of one VCG auction round."""
    round_id:       int
    clearing_price: float          # minimum price that secured a channel
    winners:        list           # list of UE ids
    payments:       dict           # {ue_id: payment per channel}
    bids:           dict           # {ue_id: submitted bid}


# ------------------------------------------------------------------ #
#  VCG Auction Mechanism (BS-side)                                    #
# ------------------------------------------------------------------ #

def run_vcg_auction(
    ues: list[UE],
    bids: dict,          # {ue_id: bid value kappa_i}
    K: int,              # total sub-channels available
    r: float,            # reservation price per sub-channel
) -> tuple[dict, dict, float]:
    r"""
    VCG allocation and payment as defined in the paper (Eqs. 9-10).

    Allocation A* = argmax_{A subset F} sum_{i in A} N_i * (kappa_i - r)
                    s.t. sum_{i in A} N_i <= K

    Payment for UE i:
        p_i = sum_{j in A^{-i}} N_j*(kappa_j - r)
            - sum_{j in A* \ {i}}  N_j*(kappa_j - r)

    Returns
    -------
    winners  : {ue_id: True}
    payments : {ue_id: payment_per_channel}  (total = N_i * payment_per_channel)
    clearing : minimum per-channel bid among winners (or r if no winner)
    """
    # Filter active UEs that can afford at least one channel
    candidates = {
        uid: b for uid, b in bids.items()
        if b >= r and ues[uid].active
    }

    if not candidates:
        return {}, {}, r

    # Sort by bid descending; greedy knapsack (unit demands handled per UE)
    sorted_ues = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    # Greedy allocation respecting channel capacity
    allocation = []
    channels_used = 0
    for uid, bid in sorted_ues:
        if channels_used + ues[uid].n_channels <= K:
            allocation.append(uid)
            channels_used += ues[uid].n_channels

    if not allocation:
        return {}, {}, r

    # Social welfare of full allocation
    def welfare(alloc):
        return sum(ues[uid].n_channels * (bids[uid] - r) for uid in alloc)

    sw_full = welfare(allocation)

    # VCG payments
    payments = {}
    for i in allocation:
        # Optimal allocation *without* UE i
        alloc_without_i = []
        ch = 0
        for uid, bid in sorted_ues:
            if uid == i:
                continue
            if ch + ues[uid].n_channels <= K:
                alloc_without_i.append(uid)
                ch += ues[uid].n_channels

        sw_without_i   = welfare(alloc_without_i)
        sw_others_in_A = welfare([uid for uid in allocation if uid != i])

        # Externality payment (per channel)
        total_payment = sw_without_i - sw_others_in_A   # total for UE i
        payments[i]   = max(total_payment / ues[i].n_channels, r)

    # Clearing price = minimum per-channel payment among winners
    clearing = min(payments.values()) if payments else r

    return {uid: True for uid in allocation}, payments, clearing


# ------------------------------------------------------------------ #
#  Bidding strategies                                                  #
# ------------------------------------------------------------------ #

def bid_truthful(ue: UE) -> float:
    """Bid true valuation (dominant strategy in single-shot VCG)."""
    if not ue.active:
        return 0.0
    return ue.valuation


def bid_heuristic(ue: UE, clearing_price: float, F_max: int = 5) -> float:
    """
    Shaded bidding heuristic (Eq. 8 in the paper).
    Interpolates between clearing price and true valuation based on
    consecutive failures.
    """
    if not ue.active:
        return 0.0
    f = ue.consecutive_failures
    beta = np.log(f + 1) / np.log(F_max + 1)   # in [0, 1)
    bid = beta * ue.valuation + (1 - beta) * clearing_price
    return min(bid, ue.valuation)               # cap at valuation


def bid_opt_linear(ue: UE) -> float:
    """
    Linear bid using the offline-optimised pacing parameter w0*.
    kappa_i = w0* * v_i   (Eq. 11 in paper / revision)
    w0* must be set beforehand via compute_opt_linear_w0().
    """
    if not ue.active or ue.w0_star is None:
        return 0.0
    return ue.w0_star * ue.valuation


# ------------------------------------------------------------------ #
#  Core OPT-Linear computation                                         #
# ------------------------------------------------------------------ #

def compute_opt_linear_w0(
    pilot_clearing_prices: np.ndarray,   # shape (T,)
    pilot_payments:        np.ndarray,   # shape (T,)  VCG payment per channel
    pilot_won:             np.ndarray,   # shape (T,)  bool: did UE win in pilot?
    valuation:             float,        # v_i
    budget:                float,        # psi_i
    n_channels:            int,          # N_i
    grid_step:             float = 0.01,
    w0_min:                float = 0.01,
    w0_max:                float = 2.00,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve ex-post optimal pacing parameter (Eqs. 12-14 in revision).

    For each candidate w0, simulate which rounds UE would have won
    (by comparing w0*v_i against the pilot clearing price, holding
    other bidders fixed), then pick w0 that maximises total utility
    subject to the budget constraint.

    Parameters
    ----------
    pilot_clearing_prices : clearing price in each round from pilot run
    pilot_payments        : VCG payment per channel in each round
                            (used when UE wins; approximated by clearing
                             price when UE did not win in pilot)
    pilot_won             : whether UE actually won in the pilot run
    valuation             : v_i
    budget                : psi_i (total budget, not per-round)
    n_channels            : N_i
    grid_step             : resolution of w0 search
    w0_min, w0_max        : search bounds

    Returns
    -------
    w0_star        : optimal pacing parameter
    w0_grid        : all candidate values evaluated
    utilities      : total utility for each w0 (NaN if budget violated)
    total_payments : total payment for each w0
    """
    w0_grid = np.arange(w0_min, w0_max + grid_step / 2, grid_step)
    T = len(pilot_clearing_prices)

    utilities      = np.full(len(w0_grid), np.nan)
    total_payments = np.zeros(len(w0_grid))

    for idx, w0 in enumerate(w0_grid):
        bid = w0 * valuation
        P_tot = 0.0
        U_tot = 0.0

        for t in range(T):
            # UE wins round t if its linear bid >= clearing price
            # (other bidders' bids held fixed as in the pilot)
            if bid >= pilot_clearing_prices[t]:
                # Payment: use pilot VCG payment if available, else clearing
                payment_t = pilot_payments[t]
                P_tot += n_channels * payment_t
                U_tot += n_channels * (valuation - payment_t)

                # Stop early if budget exhausted mid-horizon
                if P_tot > budget:
                    break

        total_payments[idx] = P_tot

        if P_tot <= budget:
            utilities[idx] = U_tot
        # else: leave as NaN (budget violated)

    # Best feasible w0
    feasible_mask = ~np.isnan(utilities)
    if not feasible_mask.any():
        # All w0 violate budget; pick smallest (most conservative)
        w0_star = w0_grid[0]
    else:
        best_idx = np.nanargmax(utilities)
        w0_star  = w0_grid[best_idx]

    return w0_star, w0_grid, utilities, total_payments


# ------------------------------------------------------------------ #
#  Pilot run (truthful bidding) to collect clearing prices            #
# ------------------------------------------------------------------ #

def run_pilot(
    ues:        list[UE],
    T:          int,
    K:          int,
    r:          float,
    static_budget: bool = True,
    seed:       int = 42,
) -> tuple[np.ndarray, dict, dict]:
    """
    Run T rounds of truthful bidding to collect pilot data.

    Returns
    -------
    clearing_prices : shape (T,)
    per_round_payments : {ue_id: np.ndarray shape (T,)}
                         payment per channel each round (0 if lost)
    per_round_won      : {ue_id: np.ndarray shape (T,) bool}
    """
    rng = np.random.default_rng(seed)

    # Reset UE state
    for ue in ues:
        ue.remaining_budget  = ue.budget
        ue.accumulated_utility = 0.0
        ue.wins              = 0
        ue.active            = True
        ue.consecutive_failures = 0

    clearing_prices = np.zeros(T)
    per_round_payments = {ue.ue_id: np.zeros(T) for ue in ues}
    per_round_won      = {ue.ue_id: np.zeros(T, dtype=bool) for ue in ues}

    for t in range(T):
        bids = {ue.ue_id: bid_truthful(ue) for ue in ues}
        winners, payments, clearing = run_vcg_auction(ues, bids, K, r)

        clearing_prices[t] = clearing

        for ue in ues:
            uid = ue.ue_id
            if uid in winners:
                pay_per_ch = payments[uid]
                total_pay  = ue.n_channels * pay_per_ch
                utility    = ue.n_channels * (ue.valuation - pay_per_ch)

                if static_budget and ue.remaining_budget < total_pay:
                    # Cannot afford -- treat as loss
                    ue.consecutive_failures += 1
                    continue

                per_round_payments[uid][t] = pay_per_ch
                per_round_won[uid][t]      = True
                ue.wins                   += 1
                ue.accumulated_utility    += utility
                ue.consecutive_failures    = 0

                if static_budget:
                    ue.remaining_budget -= total_pay
                    if ue.remaining_budget <= 0:
                        ue.active = False
            else:
                ue.consecutive_failures += 1

    return clearing_prices, per_round_payments, per_round_won


# ------------------------------------------------------------------ #
#  Full simulation with all four strategies                            #
# ------------------------------------------------------------------ #

def simulate_strategy(
    ues_template:   list[UE],
    strategy:       str,
    T:              int,
    K:              int,
    r:              float,
    static_budget:  bool,
    # OPT-Linear specific
    pilot_clearing: Optional[np.ndarray] = None,
    pilot_payments: Optional[dict]       = None,
    pilot_won:      Optional[dict]       = None,
    seed:           int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate T auction rounds for a given bidding strategy.

    Returns
    -------
    acc_utilities   : shape (|U|,) accumulated utility per UE
    win_frequencies : shape (|U|,) fraction of rounds won
    """
    rng = np.random.default_rng(seed)

    # Deep-copy UE state
    ues = []
    for u in ues_template:
        ue = UE(
            ue_id=u.ue_id,
            valuation=u.valuation,
            budget=u.budget,
            n_channels=u.n_channels,
            strategy=strategy,
        )
        ues.append(ue)

    # Pre-compute w0* for OPT-Linear UEs
    if strategy == 'opt_linear' and pilot_clearing is not None:
        for ue in ues:
            w0_star, _, _, _ = compute_opt_linear_w0(
                pilot_clearing_prices = pilot_clearing,
                pilot_payments        = pilot_payments[ue.ue_id],
                pilot_won             = pilot_won[ue.ue_id],
                valuation             = ue.valuation,
                budget                = ue.budget,
                n_channels            = ue.n_channels,
            )
            ue.w0_star = w0_star

    clearing_price = r   # initialise to reservation price

    for t in range(T):
        # Collect bids
        bids = {}
        for ue in ues:
            if strategy == 'truthful':
                bids[ue.ue_id] = bid_truthful(ue)
            elif strategy == 'heuristic':
                bids[ue.ue_id] = bid_heuristic(ue, clearing_price)
            elif strategy == 'opt_linear':
                bids[ue.ue_id] = bid_opt_linear(ue)
            # 'llm' would go here; we leave it as truthful placeholder
            # since LLM calls are external

        winners, payments, clearing_price = run_vcg_auction(ues, bids, K, r)

        for ue in ues:
            uid = ue.ue_id
            if uid in winners:
                pay_per_ch = payments[uid]
                total_pay  = ue.n_channels * pay_per_ch
                utility    = ue.n_channels * (ue.valuation - pay_per_ch)

                if static_budget and ue.remaining_budget < total_pay:
                    ue.consecutive_failures += 1
                    continue

                ue.accumulated_utility += utility
                ue.wins               += 1
                ue.consecutive_failures = 0

                if static_budget:
                    ue.remaining_budget -= total_pay
                    if ue.remaining_budget <= 0:
                        ue.active = False
            else:
                ue.consecutive_failures += 1

    acc_utilities   = np.array([ue.accumulated_utility for ue in ues])
    win_frequencies = np.array([ue.wins / T for ue in ues])

    return acc_utilities, win_frequencies


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

def plot_w0_search(
    ue_id:         int,
    w0_grid:       np.ndarray,
    utilities:     np.ndarray,
    total_payments:np.ndarray,
    budget:        float,
    w0_star:       float,
):
    """Visualise the w0 grid search for a single UE."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    feasible = ~np.isnan(utilities)

    ax1.plot(w0_grid[feasible],  utilities[feasible],
             color='steelblue', lw=2, label='Utility (feasible)')
    ax1.plot(w0_grid[~feasible], np.zeros(np.sum(~feasible)),
             'x', color='red', ms=4, label='Budget violated')
    ax1.axvline(w0_star, color='darkorange', ls='--', lw=1.5,
                label=f'$w_0^*={w0_star:.2f}$')
    ax1.set_ylabel('Total utility')
    ax1.legend(fontsize=8)
    ax1.set_title(f'OPT-Linear grid search — UE {ue_id}')

    ax2.plot(w0_grid, total_payments, color='seagreen', lw=2)
    ax2.axhline(budget, color='crimson', ls=':', lw=1.5,
                label=f'Budget $\\psi={budget}$')
    ax2.axvline(w0_star, color='darkorange', ls='--', lw=1.5)
    ax2.set_ylabel('Total payment')
    ax2.set_xlabel('Pacing parameter $w_0$')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'w0_search_ue{ue_id}.pdf', bbox_inches='tight')
    plt.show()


def plot_comparison(
    ue_ids:      np.ndarray,
    results:     dict,          # {'strategy': (acc_util, win_freq)}
    static:      bool,
):
    """Bar chart comparing accumulated utility and win frequency."""
    strategies = list(results.keys())
    colors     = {
        'truthful':   '#4878CF',
        'heuristic':  '#6ACC65',
        'opt_linear': '#D65F5F',
        'llm':        '#B47CC7',
    }
    labels = {
        'truthful':   'Truthful',
        'heuristic':  'Heuristic',
        'opt_linear': 'OPT-Linear',
        'llm':        'LLM',
    }

    n_ue     = len(ue_ids)
    n_strat  = len(strategies)
    bar_w    = 0.8 / n_strat
    x        = np.arange(n_ue)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    budget_str = 'Static budget' if static else 'Refilled budget'

    for k, strat in enumerate(strategies):
        acc_util, win_freq = results[strat]
        offset = (k - n_strat / 2 + 0.5) * bar_w
        ax1.bar(x + offset, acc_util,  bar_w,
                color=colors.get(strat, 'grey'),
                label=labels.get(strat, strat))
        ax2.bar(x + offset, win_freq, bar_w,
                color=colors.get(strat, 'grey'))

    ax1.set_xlabel('User ID')
    ax1.set_ylabel('Accumulated utility')
    ax1.set_title(f'UE utility — {budget_str}')
    ax1.set_xticks(x); ax1.set_xticklabels(ue_ids)
    ax1.legend(fontsize=8)

    ax2.set_xlabel('User ID')
    ax2.set_ylabel('Winning frequency')
    ax2.set_title(f'Win frequency — {budget_str}')
    ax2.set_xticks(x); ax2.set_xticklabels(ue_ids)

    plt.tight_layout()
    tag = 'static' if static else 'refill'
    plt.savefig(f'comparison_{tag}.pdf', bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------ #
#  Main experiment                                                      #
# ------------------------------------------------------------------ #

def main():
    # ── Simulation parameters (mapped from physical layer) ──────────
    # K sub-channels, T rounds, reservation price r = mu * P
    # Values calibrated as described in Section IV of the paper:
    #   W = 180 kHz, SINR in [5, 20] dB, alpha_i ~ U[0.8, 1.2]
    #   => v_i in [1.0, 3.5]; P = 200 mW, mu = 6 => r ~ 1.2
    K = 6           # available sub-channels
    T = 20          # auction rounds
    r = 1.2         # reservation price
    N_UE = 16       # number of UEs  (eta = K/sum(N_i) < 1 => scarcity)
    STATIC_BUDGET = True
    BUDGET = 15.0   # psi_i for static setting (~12.5 * r)

    rng = np.random.default_rng(42)

    # Generate UEs
    ues_template = []
    for i in range(N_UE):
        alpha_i = rng.uniform(0.8, 1.2)
        # Simplified: rate abstracted into scalar
        rate_i  = rng.uniform(1.0, 3.5) / alpha_i
        v_i     = alpha_i * rate_i          # valuation in [1.0, 3.5]
        n_i     = rng.integers(1, 3)        # 1 or 2 sub-channels needed
        ue = UE(
            ue_id     = i,
            valuation = round(float(v_i), 4),
            budget    = BUDGET,
            n_channels= int(n_i),
            strategy  = 'truthful',
        )
        ues_template.append(ue)

    print("=" * 55)
    print("  OPT-Linear Baseline — 6G Spectrum Auction Simulation")
    print("=" * 55)
    print(f"  UEs={N_UE}, K={K}, T={T}, r={r}, budget={BUDGET}")
    print(f"  eta ~ {K / sum(u.n_channels for u in ues_template):.2f}")
    print("-" * 55)

    # ── Step 1: Pilot run (truthful) to collect clearing prices ─────
    print("\n[1/3] Running pilot (truthful) to collect clearing prices...")
    pilot_clearing, pilot_payments, pilot_won = run_pilot(
        ues_template, T, K, r,
        static_budget=STATIC_BUDGET, seed=0,
    )
    print(f"      Mean clearing price: {pilot_clearing.mean():.3f}")
    print(f"      Min / Max:           {pilot_clearing.min():.3f} / "
          f"{pilot_clearing.max():.3f}")

    # ── Step 2: Compute w0* for each UE ─────────────────────────────
    print("\n[2/3] Computing OPT-Linear w0* for each UE...")
    w0_stars = {}
    for ue in ues_template:
        uid = ue.ue_id
        w0_star, w0_grid, utils, pays = compute_opt_linear_w0(
            pilot_clearing_prices = pilot_clearing,
            pilot_payments        = pilot_payments[uid],
            pilot_won             = pilot_won[uid],
            valuation             = ue.valuation,
            budget                = ue.budget,
            n_channels            = ue.n_channels,
        )
        w0_stars[uid] = w0_star
        print(f"      UE {uid:2d}: v={ue.valuation:.3f}  "
              f"w0*={w0_star:.2f}  "
              f"bid*={w0_star*ue.valuation:.3f}")

    # Plot w0 search for UE 0 as illustration
    uid = 0
    ue0 = ues_template[uid]
    _, w0_grid, utils, pays = compute_opt_linear_w0(
        pilot_clearing, pilot_payments[uid], pilot_won[uid],
        ue0.valuation, ue0.budget, ue0.n_channels,
    )
    plot_w0_search(uid, w0_grid, utils, pays, ue0.budget, w0_stars[uid])

    # ── Step 3: Simulate all strategies ─────────────────────────────
    print("\n[3/3] Simulating all strategies over T rounds...")

    # Inject w0* into template before passing to simulate_strategy
    for ue in ues_template:
        ue.w0_star = w0_stars[ue.ue_id]

    strategies_to_run = ['truthful', 'heuristic', 'opt_linear']
    results = {}

    for strat in strategies_to_run:
        acc_util, win_freq = simulate_strategy(
            ues_template   = ues_template,
            strategy       = strat,
            T              = T,
            K              = K,
            r              = r,
            static_budget  = STATIC_BUDGET,
            pilot_clearing = pilot_clearing,
            pilot_payments = pilot_payments,
            pilot_won      = pilot_won,
            seed           = 1,
        )
        results[strat] = (acc_util, win_freq)
        print(f"  {strat:12s}: total_utility={acc_util.sum():.3f}  "
              f"mean_win_rate={win_freq.mean():.3f}")

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Summary: accumulated utility per UE")
    print(f"  {'UE':>4}  {'Truthful':>10}  {'Heuristic':>10}  "
          f"{'OPT-Linear':>10}")
    print("-" * 55)
    for i, ue in enumerate(ues_template):
        uid = ue.ue_id
        print(f"  {uid:>4}  "
              f"{results['truthful'][0][i]:>10.3f}  "
              f"{results['heuristic'][0][i]:>10.3f}  "
              f"{results['opt_linear'][0][i]:>10.3f}")
    print("-" * 55)
    for strat in strategies_to_run:
        tot = results[strat][0].sum()
        print(f"  {'Total':>4}  " if strat == strategies_to_run[0] else "        ",
              end="")
        print(f"{tot:>10.3f}", end="  " if strat != strategies_to_run[-1] else "\n")
    print("=" * 55)

    ue_ids = np.array([ue.ue_id for ue in ues_template])
    plot_comparison(ue_ids, results, static=STATIC_BUDGET)


if __name__ == "__main__":
    main()
