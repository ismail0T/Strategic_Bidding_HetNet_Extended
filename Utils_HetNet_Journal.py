
from math import comb, log
import sys
from pydantic import BaseModel
from llama_index.core.prompts import PromptTemplate
from llama_index.core.output_parsers import PydanticOutputParser
import re
import ast
from openai import OpenAI
import openai
from typing import Dict, Tuple, List
import random
import numpy as np

api_key_1 = ""
api_key_2 = ""
api_key_3 = ""
api_key_4 = ""
api_key_5 = ""

api_keys =[api_key_1, api_key_2, api_key_3, api_key_4, api_key_5]


# Step 1: Define the JSON schema
class BSDecision(BaseModel):
    bs_id: int
    bid_value: float
   # reasoning: str

# Step 2: Create the parser
parser = PydanticOutputParser(output_cls=BSDecision)


def _knapsack_opt(bidders_items: List[Tuple[str, float, int]], M: int):
    """
    Solve 0-1 knapsack: items = (bidder_id, total_value, weight=demand)
    Returns (best_value, chosen_ids_set).
    DP complexity: O(N*M).
    """
    N = len(bidders_items)
    # dp[i][w] = best value using first i items with capacity w
    dp = [[0.0]*(M+1) for _ in range(N+1)]
    keep = [[False]*(M+1) for _ in range(N+1)]

    for i in range(1, N+1):
        bidder_id, tot_val, weight = bidders_items[i-1]
        for w in range(0, M+1):
            # don't take i
            dp[i][w] = dp[i-1][w]
            # try take i
            if weight <= w:
                cand = dp[i-1][w-weight] + tot_val
                if cand > dp[i][w]:
                    dp[i][w] = cand
                    keep[i][w] = True

    # reconstruct chosen set at w=M
    w = M
    chosen = set()
    for i in range(N, 0, -1):
        if keep[i][w]:
            bidder_id, tot_val, weight = bidders_items[i-1]
            chosen.add(bidder_id)
            w -= weight

    return dp[N][M], chosen

def vcg_all_or_nothing(bidders: Dict[str, Dict[str, float]], M: int):
    """
    VCG for identical items with all-or-nothing multi-unit demand.
    bidders: { bidder_id: {"bid_value": v_i (per-unit), "demand": q_i (int)} }
    M: total units.

    Returns:
      allocation: {bidder_id: allocated_units (either 0 or demand)}
      payments:   {bidder_id: VCG payment}
      welfare:    SW* (optimal welfare with all bidders)
    """
    # Build items: (id, total_value, weight)
    alloc_items = []
    value_items  = []

    for b_id, data in bidders.items():
        v = float(data["bid_value"])          # true valuation
        v_adj = float(data["bid_adjusted"])   # fairness-adjusted score
        q = int(data["demand"])
        alloc_items.append((b_id, v_adj * q, q))
        value_items .append((b_id, v * q, q))

    # 1) Optimal allocation with all bidders
    _, chosen = _knapsack_opt(alloc_items, M)

    

    # Build allocation dict (all-or-nothing)
    allocation = {b_id: 0 for b_id in bidders}
    for b_id in chosen:
        allocation[b_id] = int(bidders[b_id]["demand"])

    sw_star = sum(
        bidders[b_id]["bid_value"] * bidders[b_id]["demand"]
        for b_id in chosen
    )

    # print(chosen)
    # 2) VCG payments via leave-one-out knapsack
    payments = {b_id: 0.0 for b_id in bidders}
    for b_id in chosen:
        v_i = float(bidders[b_id]["bid_value"])
        q_i = int(bidders[b_id]["demand"])
        tot_i = v_i * q_i


        # Remove bidder i from BOTH lists
        alloc_items_minus_i = [tpl for tpl in alloc_items if tpl[0] != b_id]
        value_items_minus_i = [tpl for tpl in value_items if tpl[0] != b_id]

        # Recompute allocation WITHOUT i (still fairness-based)
        _, chosen_minus_i = _knapsack_opt(alloc_items_minus_i, M)

        # Welfare of others without i (TRUE valuations)
        sw_minus_i_value = sum(
            bidders[j]["bid_value"] * bidders[j]["demand"]
            for j in chosen_minus_i
        )

        # VCG payment
        payments[b_id] = sw_minus_i_value - (sw_star - tot_i)

        # Sum of others' value with i present = SW* - v_i q_i
        # payments[b_id] = sw_minus_i - (sw_star - tot_i)

        # if b_id == 1:
        #     print("sw_minus_i", sw_minus_i, "sw_star", sw_star, 'tot_i', tot_i, "payments[b_id]", payments[b_id])
        # Sanity: no overcharge versus total valuation
        # assert payments[b_id] <= tot_i + 1e-9, f"Overcharge for {b_id}"

        payments[b_id] = max(0.0, min(payments[b_id], tot_i))


        # Also ensure non-negativity up to numerical tolerance
        if payments[b_id] < 0 and payments[b_id] > -1e-9:
            payments[b_id] = 0.0

    # Losers pay zero
    return allocation, payments, sw_star

# Auction Mechanics
def vcg_auction(m_bids, bs, nb_sub_channels):
    bids = sorted(
        m_bids,
        key=lambda x: (-x["bid_adjusted"], -x["nb_succ_no_channels"], -x["blocks"])
    )
    # print(bids)
    # interference_penalty = 0
    # if bs.id == 0: # MBS
    #     interference_penalty = 0.4
    # else: # SBS
    #     interference_penalty = 0.2

    new_bidder_list = {
        bid['ue_id']: {
            "bid_value": bid['bid'],
            "bid_adjusted": bid['bid_adjusted'],
            "demand": bid['demand']
        }
        for bid in bids
    }

    M = nb_sub_channels # number of subcahnnels for current BS
    alloc, pays, welfare = vcg_all_or_nothing(new_bidder_list, M)
    bid_map = {bid['ue_id']: bid for bid in bids}

    # print("new_bidder_list", [bidder for bidder in new_bidder_list])
    # print("alloc", alloc)
    # print("pays", pays)
    # print("welfare", welfare)
    winners = []
    loosers = []
    payment = {}
    for i in alloc:
        payment[i] = pays[i]
        min_pay = bid_map[i]['bs_reservation_price'] * bid_map[i]['demand']

        # if bs.id == 2:
        #     print("pays[i]", pays[i], "min_pay", min_pay)
        if alloc[i] > 0 and bid_map[i]['valuation'] * bid_map[i]['demand'] >= min_pay and bid_map[i]['bid'] * bid_map[i]['demand'] >= min_pay: # winner
            winners.append(bid_map[i])
            
            # if bid_map[i]['ue_id'] == 4:
            #     print("min_pay", min_pay, 'pays', pays[i])
            payment[i] = max(min_pay, pays[i])
            # if len(alloc) < nb_sub_channels:
            #     payment[i] = max(2 * bid_map[i]['demand'], pays[i])
            # if bs.id == 2:
            #     print("len(alloc)", len(alloc),  payment[i], max(2 * bid_map[i]['demand'], pays[i]))
        else: #loser
            loosers.append(bid_map[i])
    # print("len winners", len(winners), len(loosers))
    return winners, payment, loosers # price is for the set of allocated sub-channels


def gsp_auction(bids, bs, nb_subchannels):
    sorted_bids = sorted(
        bids,
        key=lambda x: (-x["bid_adjusted"], -x["nb_succ_no_channels"], -x["blocks"])
    )
    # print(sorted_bids)
    # sorted_bids_adjusted = sorted(
    #     bids,
    #     key=lambda x: (-x["bid_adjusted"], -x["nb_succ_no_channels"], -x["blocks"])
    # )
    
    # sorted_bids = sorted(bids, key=lambda x: -x['bid'])
    winners = []
    loosers = []
    payments = {}
    capacity = bs.nb_subchannels

    for i, bidder in enumerate(sorted_bids):
        next_bid = sorted_bids[i+1]['bid'] if i+1 < len(sorted_bids) else bs.reservation_price
        
        
        if bidder['demand'] <= capacity and bidder['valuation'] >= bs.reservation_price and bidder['bid'] * bidder['demand'] >= bs.reservation_price:
            winners.append(bidder)
            capacity -= bidder['demand']
            payments[bidder['ue_id']] = next_bid * bidder['demand']
            if next_bid < bs.reservation_price:
                payments[bidder['ue_id']] = bs.reservation_price * bidder['demand']
            # if len(sorted_bids) < bs.nb_sub_channels:
            #     payments[bidder['ue_id']] = max(2 * next_bid['demand'], payments[bidder['ue_id']])
            # if bs.id == 2:
            #     print(bidder['ue_id'], "next_bid", next_bid, "bidder['valuation']", bidder['valuation'])
        else:
            loosers.append(bidder)

    return winners, payments, loosers


def llm_call(auction_type, list_BSs, ue_demand, ue_budget, historical_feedback, nb_ues, nb_episodes):
    auction_type_description = ""
    if auction_type == "VCG":
        auction_type_description = "VCG"
    elif auction_type == "GSP":
        auction_type_description = "GSP"

    response = []
    #     3. provide a brief explanation of your reasoning (e.g., competition, interference, budget).
# 6. do not output any other information except the JSON list.
    # 6. output a short paragraph to explain your reasoning.
    #  If the price you expect to pay is close to your true valuations, this means your utility will be less and hence, your benefit from the allocation is not high.
    #  bid close to the clearing price unless your remaining life episodes are very low [they do vary from 5 down to 0].
    
        # - You have a budget of {ue_budget} credits that must last {nb_episodes} rounds. Spending too quickly will disqualify you from later auctions.

    prompt = f"""
    You are an intelligent user equipment (UE) agent participating in a sub-channel auction using a {auction_type_description} winner selection and payment method.
    
    Given the following network and economic context:
    - Your valuations for a sub-channel for each BS : {list_BSs} 
    - You have a budget of {ue_budget} credits.
    - If you do not want to participate in this round, simply submit a zero bid value.
    - Number of sub-channels required for each BS: {ue_demand}
    - {nb_ues} UE are in the network looking to allocate sub-channels from the same set of BSs.
    - previous history of interaction with the system: {historical_feedback}.
    
    Please analyze and provide:

    1. Select the best BS to submit your bid to. The best BS is the one that maximizes your utility defined as the difference between the true valuation and the price you expect to pay. 
    2. The bid value that you choose should increase your chances of beeing amongst the winners but it should not be higher than your true valuation.
        - IMPORTANT: Your primary objective is to maximize cumulative utility while **never exhausting** your budget before round {nb_episodes}.

    3. Please output your bid and selected BS strictly in a json format.    
    4. Think step by step.
    5. The JSON must have the following keys:
    - "bs_id": integer
    - "bid_value": float
    6. output a short paragraph (letters and numbers only) to explain your reasoning.
    7. put the JSON list at then end of your output.
    """

    # print(prompt)

    client = openai.OpenAI(api_key=random.choice(api_keys))
    message= [{"role": "assistant", "content": prompt}]

    response = client.chat.completions.create(
    model="gpt-5-mini",
    #model="gpt-4o",
    # model="o1",
    # model="o1-mini",
    #model="gpt-4o-mini",
    messages=message
    )
    m_response_letter = response.choices[0].message.content
    print("\nresponse:\n", m_response_letter)

    matches = re.findall(r"\{[\s\S]*?\}", str(m_response_letter))
    json_str = matches[-1]  # last JSON block

    try:
        parsed = parser.parse(json_str)
        
    except Exception as e:
        print("Parsing Error:", str(e))
        print("Received JSON:", json_str)

    return parsed.bs_id, parsed.bid_value, ""


def shaded_bid(true_val, clearing_price, ue_nb_succ_no_channels, nb_max_no_channel):
    # np.random.seed(42)

    betta = log(ue_nb_succ_no_channels+1) / log(nb_max_no_channel) # so the last time before diying you bid equal to the true valuation
    bid_val = betta*true_val + (1-betta)*clearing_price
    bid_val += np.random.uniform(0.0, 0.001)
    bid_val = min(true_val, bid_val)
    return bid_val

def generate_bid_set(min_val, max_val):
    grid = np.linspace(min_val, max_val, 10) # 10 samples
    grid = np.unique(grid)
    grid = grid[grid > 0.0]
    # ensure ub included
    if max_val not in grid:
        grid = np.sort(np.append(grid, max_val))
    return grid

def win_prob_direct(b, empirical_samples, N_s, C_s):
    """
    Compute win probability directly using empirical CDF and binomial formula.
    b : float (UE's bid)
    empirical_samples : array-like (past clearing prices, used for empirical CDF)
    m : int (number of competitors)
    C_s : int (number of available channels at BS s)
    """
    # Empirical CDF at bid b
    m = N_s -1
    F_b = np.mean(empirical_samples <= b)
    # Binomial probability sum
    prob = 0.0
    for j in range(C_s):  # sum for j=0,...,C_s-1 competitors outbidding
        prob += comb(m, j) * (1 - F_b) ** j * (F_b) ** (m - j)
    return prob
    
def expected_utility(b, v_i_s, past_prices, N_s, C_s):
    pb_win = win_prob_direct(b, past_prices, N_s, C_s)
    expected_price = np.mean(past_prices)

    return pb_win * (v_i_s - expected_price)


def arg_max_expected_utility(BS_set, ue_true_valuation, N_s, C_s):
    best_bs = 1
    bid_value = 0
    best_utility = 0

    for s in BS_set:
        past_prices = np.array(s.clearing_prices)
        # past_prices = np.array([5.0, 5.5, 6.0, 6.2, 6.5, 7.0, 7.2, 7.5, 8.0, 8.5])
        Bid_set = generate_bid_set(past_prices[:-1], ue_true_valuation)
        best_u_hat = 0
        u_hat_bid_value = 0
        u_hat_best_bs = s.id
        u_hat = 0

        for b in Bid_set:
            u_hat = expected_utility(b, ue_true_valuation, past_prices, N_s, C_s)
            if u_hat > best_u_hat:
                best_u_hat = u_hat
                u_hat_bid_value = b
                u_hat_best_bs = s.id
        
        if u_hat > best_utility:
                best_utility = u_hat
                bid_value = u_hat_bid_value
                best_bs = u_hat_best_bs
        
        # print(u_hat_best_bs, "expected u", best_u_hat, "u_hat_bid_value", u_hat_bid_value, "\n")
    
    # if len(BS_set)>1 and best_bs == 2:
    #     print("MBS selected")
    
    bid_value = min(ue_true_valuation, bid_value)
    
    return best_bs, bid_value