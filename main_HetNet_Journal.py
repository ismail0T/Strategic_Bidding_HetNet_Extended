from posix import PRIO_PROCESS
import numpy as np
import pandas as pd
from collections import deque
import random
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
from pandas._libs.lib import item_from_zerodim
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import sys


import xlsxwriter
from Utils_HetNet_Journal import llm_call, vcg_auction, gsp_auction, shaded_bid, arg_max_expected_utility, generate_bid_set

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


# Parameters 
NUM_EPISODES = 30
NUM_BSs = 3
NUM_SUBCHANNELS = 4
WINDOW_SIZE = NUM_EPISODES  # Fairness tracking window
NB_MBS_UEs = 10
NB_SBS_UEs = 16
NM_UES = NB_SBS_UEs * 2 + NB_MBS_UEs # 16 for each SBS and 3 for the MBS
UE_ID_TRUTHFUL = 1000
BIDDING_MODE_TRUE_VAL ='true_val'
# BIDDING_MODE_SHADED_VAL ='shaded_val'
BIDDING_MODE_GREEDY ='greedy'
BIDDING_MODE_GREEDY_PACING ='greedy_pacing'
BIDDING_MODE_LLM ='LLM'
BIDDING_MODE_MIXED ='MIXED'
BIDDING_MODE_OPT_LINEAR ='OPT_LINEAR'
BIDDING_COST = 0.3
NB_MAX_NO_CHANNEL = 5
#NB_POINTS_FOR_A_WIN = 2
NB_NO_CHANNEL_TO_RESET = NB_MAX_NO_CHANNEL * 2

clearing_prices_bs_1 = [1.1, 1.7932916124473675, 1.2973178786058794, 1.2973178786058794, 1.2973178786058794, 1.2973178786058794, 1.2973178786058794, 1.2973178786058794, 
                        1.316665791019434, 1.316665791019434, 2.1820765846071275, 1.2743513095125962, 1.2743513095125962, 1.3192554715637268, 1.3192554715637268, 
                        1.3192554715637268, 1.3192554715637268, 2.157950290582754, 1.3113049222030257, 1.2630502952092915, 1.3113049222030257, 1.2630502952092915, 
                        1.263771502282408, 1.2, 1.2, 1.2, 1.2, 1.2]

clearing_prices_bs_2 = [1.1, 1.7599999999999998, 1.9327703582627667, 1.2743513095125962, 1.2743513095125962, 1.316665791019434, 1.316665791019434, 1.316665791019434, 
                        1.2743513095125962, 1.2743513095125962, 1.9327703582627667, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 
                        1.2, 1.2, 1.2, 1.2, 1.2, 1.2]

clearing_prices_bs_3 = [1.3, 1.7797277002924101, 1.9395329626408948, 1.4, 1.9395329626408957, 2.1073365387588647, 2.0604281942688436, 1.9395329626408953, 
                        1.9327703582627682, 2.2607344153798223, 2.1073365387588647, 1.9395329626408953, 1.9395329626408953, 1.9395329626408953, 1.9395329626408953, 
                        1.4, 1.4, 1.4, 1.4, 1.4]

# UE and BS Classes
class UE:
    def __init__(self, id, rate, bidding_mode):
        self.id = id
        self.budget = 15 # np.random.uniform(2, 2.3) #np.random.uniform(5, 6)
        self.demand = 1#np.random.randint(1, 3)
        #self.history_prices = {}
        self.alpha_0 = 0.6
        self.win_history = deque(maxlen=WINDOW_SIZE)
        self.loss_history = deque(maxlen=WINDOW_SIZE)
        self.utility_history = [0]
        self.utility = 0
        self.omega = 0.02
        self.bidding_history = []
        self.blocks = 0
        self.nb_succ_no_channels = 0
        self.rate_scaling_factor = 0.8
        self.lambda_poiss = np.random.uniform(1, 3)
        self.last_ep_win = 0
        self.class_QoS = 1.2 #np.random.uniform(1.0, 1.3)
        self.w0_star = float(1)

        self.bidding_mode = bidding_mode
        self.achievable_rate = np.random.uniform(2.1, 2.3) # rate
        self.true_valuation = self.rate_scaling_factor * self.achievable_rate

        if (self.id >=16 and self.id <= 19):# or self.id == 29 or self.id == 30:
            self.bidding_mode = BIDDING_MODE_GREEDY
            self.achievable_rate = 2.2 # rate
            self.true_valuation = self.rate_scaling_factor * self.achievable_rate

        if (self.id >=20 and self.id <= 23): # 
            self.bidding_mode = BIDDING_MODE_GREEDY_PACING
            self.achievable_rate = 2.2 # rate
            self.true_valuation = self.rate_scaling_factor * self.achievable_rate

        if (self.id >=24 and self.id <= 27): # 
            self.bidding_mode = BIDDING_MODE_OPT_LINEAR  
            self.achievable_rate = 2.2 # rate
            self.true_valuation = self.rate_scaling_factor * self.achievable_rate

        if (self.id >=28 and self.id <= 31): 
            self.bidding_mode = BIDDING_MODE_TRUE_VAL
            self.achievable_rate = 2.2 # rate
            self.true_valuation = self.rate_scaling_factor * self.achievable_rate

        # if self.id == 29:# or self.id == 29 or self.id == 30:
        #     self.bidding_mode = BIDDING_MODE_TRUE_VAL
        #     self.achievable_rate = 2.2 # rate
        #     self.true_valuation = self.rate_scaling_factor * self.achievable_rate

        # if self.id == 30:# or self.id == 29 or self.id == 30:
        #     self.bidding_mode = BIDDING_MODE_TRUE_VAL
        #     self.achievable_rate = 2.2 # rate
        #     self.true_valuation = self.rate_scaling_factor * self.achievable_rate

        # if self.id == 31:# or self.id == 29 or self.id == 30:
        #     self.bidding_mode = BIDDING_MODE_TRUE_VAL# BIDDING_MODE_LLM
        #     self.achievable_rate = 2.2 # rate
        #     self.true_valuation = self.rate_scaling_factor * self.achievable_rate

        

        # if self.bidding_mode == BIDDING_MODE_MIXED:
        #     if self.id == 16 or self.id == 17 or self.id == 18  or self.id == 19:
        #         self.bidding_mode = BIDDING_MODE_GREEDY
        #     elif self.id == 28 or self.id == 29 or self.id == 30  or self.id == 31:
        #         self.bidding_mode = BIDDING_MODE_LLM
        #     else:
        #         self.bidding_mode = BIDDING_MODE_TRUE_VAL

        #     self.true_valuation = self.rate_scaling_factor * 2.3

        
        # if self.id == UE_ID_SPECIAL:
        #     self.bidding_mode = BIDDING_MODE_LLM

    
    def update_valuation(self):
        #return
        #h = min(((self.nb_succ_no_channels)/NB_MAX_NO_CHANNEL), 1)
        #alpha_0 = 0.8
        self.achievable_rate = np.random.uniform(2.1, 2.3) # rate
        if self.id == 27 or self.id == 28:
            self.achievable_rate = 2.2 # rate
        epsilon = (self.achievable_rate - self.alpha_0)/NB_MAX_NO_CHANNEL
        alpha = self.alpha_0 + epsilon * self.nb_succ_no_channels
        self.rate_scaling_factor = alpha
        self.true_valuation = min(self.rate_scaling_factor * self.achievable_rate, self.achievable_rate)


class BS:
    def __init__(self, id):
        self.id = id
        self.clearing_prices = []
        if self.id == 2: #MBS
            self.reservation_price = 1.4
            self.clearing_prices = [1.3]
        else: #SBS
            self.reservation_price = 1.2
            self.clearing_prices = [1.1]

        self.nb_subchannels = NUM_SUBCHANNELS
        self.utility_history = [0]
        self.total_utility = 0
        
        # if self.id == 0: # Macro BS
        #     self.reservation_price += 0.75
            #self.subchannels = NUM_SUBCHANNELS + 2


def compute_opt_linear_w0(valuation, budget,
                           w0_min=0.5, w0_max=2.0, grid_step=0.01):

    pilot_clearing = np.array(clearing_prices_bs_2)

    T = len(pilot_clearing)
    if T == 0:
        return 1.0   # fallback: no history, bid truthfully

    w0_grid    = np.arange(w0_min, w0_max + grid_step / 2, grid_step)
    best_util  = -np.inf
    best_w0    = w0_min

    for w0 in w0_grid:
        bid    = w0 * valuation
        P_tot  = 0.0
        U_tot  = 0.0

        for t in range(T):
            if bid >= pilot_clearing[t]:
                pay  = pilot_clearing[t]
                P_tot += pay + BIDDING_COST        # per-channel payment
                U_tot += (valuation - pay)

                if P_tot > budget:    # budget exhausted mid-horizon
                    break

        if U_tot > best_util:
            best_util = U_tot
            best_w0   = w0
    
    print("best_w0", best_w0, U_tot)
    return best_w0


# Simulation standard
def run_simulation(num_ues, auction_type, ue_bidding_mode):
    np.random.seed(42)
    rate_init = 2.3 # np.random.uniform(2.1, 2.3)
    ues = [UE(i, rate_init, ue_bidding_mode) for i in range(num_ues)]
    bss = [BS(j) for j in range(NUM_BSs)]
    vcg_logs, gsp_logs = [], []
    fairness_logs = []

    glb_ue_util_hitsory_agv = []
    glb_ue_util_hitsory_acc = []
    glb_bs_util_hitsory_avg = []
    glb_bs_util_hitsory_acc = []

    stat_utility_all_episodes = []

    ue_statistics_history_per_episode_global = []
    # create historical_feedback
    historical_feedback_list = []
    # clearing_prices = [1.5, 1.5, 1.7]
    
    # valuations = {}
    # achievable_rates = {}   
    # achievable_rates[0] = 2.3 # MBS SINR is lower compared to that of the SBSs
    # achievable_rates[1] = 0
    # achievable_rates[2] = 0

    ratio_diff_list = []

    for ep in range(NUM_EPISODES):     
        if ep>0:
            for ue in ues:
                ues[ue.id].utility_history = []
                
            # ue.utility = 0

        # --- Generate bids ---
        all_bids = []

        ue_statistics_history_per_episode = []
        for ue in ues:  
            # if ue.nb_succ_no_channels > NB_NO_CHANNEL_TO_RESET:                
            #     ues[ue.id].nb_succ_no_channels = 0                
            # elif ue.nb_succ_no_channels > NB_MAX_NO_CHANNEL:
            #     ues[ue.id].nb_succ_no_channels += 1
            #     continue
            
            
            bs_prime = []          
            
            best_bs = 0
            bid_value = 0
            

            # if ue.id < NM_UES/3: #first 4 UEs are exactly between the two SBSs
            #     achievable_rates[1] = 2.2
            #     achievable_rates[2] = 2.2
            # elif ue.id < (NM_UES/3)*2:
            #     achievable_rates[1] = 2.3
            #     achievable_rates[2] = 2.1
            # elif ue.id < NM_UES:
            #     achievable_rates[1] = 2.1
            #     achievable_rates[2] = 2.3
            # else: 
            #     print("exit..", ue.id)
            #     sys.exit(0)
            
            # for bs in bss:
            #     valuations[bs.id] = achievable_rates[bs.id] * ue.rate_scaling_factor
            #     if achievable_rates[bs.id] >= ue.class_QoS:
                    # bs_prime.append({'bs_id': bs.id, 'ue_valuation': valuations[bs.id]})
            
            if ue.id < (NM_UES - NB_MBS_UEs)/2: # first SBS UEs
                bs_prime.append({'bs_id': 0, 'ue_valuation': ue.true_valuation})
                bs_prime.append({'bs_id': 2, 'ue_valuation': ue.true_valuation})
                best_bs = 0

            elif ue.id < (NM_UES - NB_MBS_UEs): # second SBS UEs
                bs_prime.append({'bs_id': 1, 'ue_valuation': ue.true_valuation})
                bs_prime.append({'bs_id': 2, 'ue_valuation': ue.true_valuation})
                best_bs = 1

            else: # MBS UEs
                bs_prime.append({'bs_id': 2, 'ue_valuation': ue.true_valuation})
                best_bs = 2
            
            # print("valuations", valuations)
            # bs_prime.append({'bs_id': 0, 'ue_valuation': ue.true_valuation})
            if ue.bidding_mode == BIDDING_MODE_LLM:
                historical_feedback = ""
                if ep>0:
                    for ue_episode in ues[ue.id].bidding_history[-5:]:
                        BS_type = ""
                        if ue_episode['bs_id'] == 0:
                            BS_type = " macro BS "
                        else:
                            BS_type = " small BS "
                        
                        if ue_episode['win']:
                            historical_feedback += ". Episode "+str(ue_episode['episode'])+': I won, I bidded:'+str(ue_episode['bid'])+' per channel at the '+BS_type+' and paid in total:'+\
                                str(ue_episode['price'])
                        else:
                            historical_feedback += ". Episode "+str(ue_episode['episode'])+': I lost, I bidded:'+str(ue_episode['bid'])+' per channel at the '+BS_type
                    
                    historical_feedback += ", number of times before beeing excluded from the market:"+str(NB_MAX_NO_CHANNEL-ues[ue.id].nb_succ_no_channels)

                
                for item in bs_prime:
                    bs_id = item['bs_id']
                    clearing_prices_txt = str(np.round(bss[bs_id].clearing_prices[-2:], 3))

                    historical_feedback += ". The clearing prices for BS "+str(bs_id)+ " so far: "+clearing_prices_txt
                # historical_feedback += ". The minimum clearing prices for current bs is:"+str(min(clearing_prices[-3:]))
                # print("budget", ue.budget)

                try:
                    # if ue.id == UE_ID_TRUTHFUL:
                    #     best_bs = 0
                    #     bid_value = ue.true_valuation# - 0.0000001
                    # else:
                    print(ep, "historical_feedback", historical_feedback)
                    best_bs, bid_value, _ = llm_call(auction_type, bs_prime, ue.demand, ue.budget, historical_feedback, NM_UES, NUM_EPISODES)
                    bid_value = min(bid_value, ue.true_valuation, ue.budget)
                    # best_bs = 1
                    # bid_value = 2.2999999999
                except:
                    print("An exception occurred in llm_call()\n\n")
                    best_bs = 2
                    bid_value = 0
                

            elif ue.bidding_mode == BIDDING_MODE_TRUE_VAL:
                bid_value = ue.true_valuation #+ np.random.uniform(-0.001, 0)

            elif ue.bidding_mode == BIDDING_MODE_GREEDY:
                
                BS_set = {bss[2]}
                if best_bs != 2: # best_bs is an SBS - default BS
                    BS_set.add(bss[best_bs])
                    # if ep == 0 or ep == 8:
                    #     print(ep, ue.id, bss[best_bs].clearing_prices)
                
                best_bs, bid_value = arg_max_expected_utility(BS_set, ue.true_valuation, NB_SBS_UEs, NUM_SUBCHANNELS)
                # if len(BS_set)>1 and best_bs == 2:
                    # print(ep, "MBS selected")
            
            elif ue.bidding_mode == BIDDING_MODE_GREEDY_PACING:
                
                BS_set = {bss[2]}
                if best_bs != 2: # best_bs is an SBS
                    BS_set.add(bss[best_bs])
                
                best_bs, bid_value = arg_max_expected_utility(BS_set, ue.true_valuation, NB_SBS_UEs, NUM_SUBCHANNELS)
                # if len(BS_set)>1 and best_bs == 2:
                    # print(ep, "MBS selected")
                
                if np.random.rand() < 0.5:
                # if (ep % 2) == 0: # DO not bid in this round
                    bid_value = 0

            elif ue.bidding_mode == BIDDING_MODE_OPT_LINEAR:
                original_budget = ue.budget  # must match UE.__init__ budget
                w0 = compute_opt_linear_w0(
                    valuation       = ue.true_valuation,
                    budget          = original_budget,
                )
                ue.w0_star = w0
                print(f"  UE {ue.id}: v={ue.true_valuation:.3f}  "
                    f"w0*={w0:.2f}  bid*={w0*ue.true_valuation:.3f}")

                bid_value = ue.w0_star * ue.true_valuation

            total_bid = 0    
            total_bid = ue.demand * bid_value

            # if total_bid > ue.budget or ue.nb_succ_no_channels >= NB_MAX_NO_CHANNEL:
            #     bid_value = -1 # equal to not participating
                

            if best_bs is not None and total_bid <= ue.budget:# and ue.nb_succ_no_channels <= NB_MAX_NO_CHANNEL:
                w_i_T = sum(ue.win_history)
                # f_i_T = w_i_T/(ep+0.0000001) # to elimanate devision by zero
                fairness_ratio = 1/(w_i_T +0.01)**0.04
                
                # fairness_ratio = 1/(1+0.1*(w_i_T/(ep+0.00000001)))
                
                # lam = 0.5
                # h = (1-lam)*
                # if ue.id == 16:
                #     ratio_diff_list.append(fairness_ratio)
                # ratio = (1/(wins+0.0001))**0.01

                # adjusted_bid_value = bid_value * fairness_ratio

                # adjusted_bid_value = bid_value - 4*w_i_T -4 works well for GSP
                adjusted_bid_value = bid_value * np.exp(1 * w_i_T)
                # adjusted_bid_value = bid_value - 0.04*w_i_T

                # if ep == 0 or ep == 3 or ep == 5 or ep == 8:
                #     print(ep, ue.id, "adjusted total value:", adjusted_bid_value)

                if ue.id == 16:
                    ratio_diff_list.append(adjusted_bid_value)

                all_bids.append({'ue_id': ue.id, 'bs_id': best_bs, 'bid': bid_value, 'bid_adjusted': adjusted_bid_value,
                                'valuation': ue.true_valuation, 'demand': ue.demand, 'budget': ue.budget,
                                'bs_reservation_price': bss[best_bs].reservation_price,
                                'blocks': ue.blocks, 'nb_succ_no_channels': ue.nb_succ_no_channels})
            else:
                ues[ue.id].nb_succ_no_channels += 1
                if ues[ue.id].nb_succ_no_channels > NB_NO_CHANNEL_TO_RESET:
                    # ues[bidder['ue_id']].nb_succ_no_channels = 0
                    pass
                elif ues[ue.id].nb_succ_no_channels == NB_MAX_NO_CHANNEL:
                    ues[ue.id].blocks += 1
                
                ues[ue.id].update_valuation()
                # if bidder['ue_id'] == 18:
                #     print(ep, "nb_succ_no_channels", ues[bidder['ue_id']].nb_succ_no_channels, "first") 

                ue_statistics_history_per_episode.append({
                    'ue_id': ue.id,
                    'bs_id': best_bs,
                    'bid': bid_value,
                    'valuation': ue.true_valuation,
                    'budget': ue.budget,
                    'demand': ue.demand,
                    'price': -2,
                    'ue_nb_succ_no_channels': ues[ue.id].nb_succ_no_channels,
                    'blocks': ues[ue.id].blocks,
                    'bs_reservation_price': bss[best_bs].reservation_price,
                    'win': -2
                })
            # else:# we did not find any BS that we can associate with due to budget or QoS constraints
            #     print("ERROR: we did not find any BS that we can associate with due to budget or QoS constraints")
            #     sys.exit(0)
            # END of BS selection and bid etimation for all UEs


        # --- Run Auctions per BS ---
        for auction_type_0, auction_func, logs in [('VCG', vcg_auction, vcg_logs), ('GSP', gsp_auction, gsp_logs)]:
            if auction_type != auction_type_0:
                continue
            bs_bids = {bs.id: [] for bs in bss}
            for bid in all_bids:
                bs_bids[bid['bs_id']].append(bid)

            winners_list = []
            first_round_loosers_list = []
            second_round_loosers_list = []
            bs_utilities = []

            for bs in bss:
                bids = bs_bids[bs.id]
                winners, payments, loosers = auction_func(bids, bs, bs.nb_subchannels)
                #print("winners", winners)
                #print("loosers", loosers)
                # print("\n")
                bs_revenue = 0
                for bidder0 in loosers:
                    first_round_loosers_list.append(bidder0)
                    ue = ues[bidder0['ue_id']]
                    ue.win_history.append(0)
                    ue.loss_history.append(1)

                for winner in winners:
                    # ue = ues[winner['ue_id']]
                    tmp_price = payments[winner['ue_id']] # TOTAL PRICE FOR ALL ITEMS
                    interference_penalty = 0
                    
                    # if bs.id == 0: # MBS
                    #     interference_penalty = 0.1
                    # else: # SBS
                    #     interference_penalty = 0.05

                    price = tmp_price + interference_penalty
                    # if bs.id != 2:
                    price = min(price, winner['valuation'] * winner['demand'])

                    util = 0
                    winner['price'] = price
                    
                    if price <= ues[winner['ue_id']].budget:# and price <= winner['valuation'] * winner['demand']:
                        util = winner['valuation'] * winner['demand'] - price
                        ues[winner['ue_id']].win_history.append(1)
                        ues[winner['ue_id']].loss_history.append(0)
                        bs_revenue_tmp = price - bs.reservation_price * winner['demand']
                        bs_revenue += bs_revenue_tmp
                        # print(auction_type, "bs.id", bs.id, "ue.id", ue.id, "util", util, "winner['valuation']", winner['valuation'], "winner['demand']", winner['demand'], "price", price, "bs.reservation_price", bs.reservation_price)
                        
                        winners_list.append(winner)
                        ues[winner['ue_id']].utility += util
                        ues[winner['ue_id']].utility_history.append(util)
                        # ue.update_price_estimate(bs.id, price)
                    else:
                        print(ep, 'price:', price, winner['valuation'] * winner['demand'], ues[winner['ue_id']].budget)
                        ues[winner['ue_id']].win_history.append(0)
                        ues[winner['ue_id']].loss_history.append(1)
                        second_round_loosers_list.append(winner)
                        # ue.utility += 0
                        ues[winner['ue_id']].utility_history.append(0)
                        #print("Loss..", auction_type,"bs.id", bs.id, "ue.id", ue.id, "winner['valuation']", winner['valuation'], "winner['demand']", winner['demand'], "price", price, "bs.reservation_price", bs.reservation_price, "ue.budget", ue.budget)

                 
                    
                
                #print(auction_type, "bs.id", bs.id, "bs_revenue", bs_revenue, "\n")
                bs.utility_history.append(bs_revenue)
                bs.total_utility += bs_revenue
                bs_utilities.append(bs_revenue)
            
            
        
        
            ### BEGIN STATS for current episode
            tmp_clearing_prices = {i: [] for i in range(3)}
            for bidder in winners_list:
                ues[bidder['ue_id']].budget -= bidder['price']
                ues[bidder['ue_id']].budget -= BIDDING_COST
                ues[bidder['ue_id']].nb_succ_no_channels = 0
                ues[bidder['ue_id']].update_valuation()
                tmp_clearing_prices[bidder['bs_id']].append(bidder['price']/bidder['demand'])
                
                ues[bidder['ue_id']].last_ep_win = ep
                ues[bidder['ue_id']].bidding_history.append({
                    'episode': ep,
                    'bs_id': bidder['bs_id'],
                    'bid': bidder['bid'],
                    'demand': bidder['demand'],
                    'price': bidder['price'],
                    'blocks': ues[bidder['ue_id']].blocks,
                    'win': 1
                })
                ue_statistics_history_per_episode.append({
                    'ue_id': bidder['ue_id'],
                    'bs_id': bidder['bs_id'],
                    'bid': bidder['bid'],
                    'valuation': bidder['valuation'],
                    'budget': ues[bidder['ue_id']].budget,
                    'demand': bidder['demand'],
                    'price': bidder['price'],
                    'ue_nb_succ_no_channels': ues[bidder['ue_id']].nb_succ_no_channels,
                    'blocks': ues[bidder['ue_id']].blocks,
                    'bs_reservation_price': bidder['bs_reservation_price'],
                    'win': 1
                })
            # UPDATE clearing price list
            for bs in bss:
                if tmp_clearing_prices[bs.id]:
                    bs.clearing_prices.append(min(tmp_clearing_prices[bs.id]))
            # tmp_clearing_prices.sort(reverse=True)
            # if len(tmp_clearing_prices)>0:
            #     clearing_prices.append(min(tmp_clearing_prices))
            #     clearing_prices.sort(reverse=True)
            #     clearing_prices = clearing_prices[-5:]
            
            for bidder in first_round_loosers_list:              
                ues[bidder['ue_id']].nb_succ_no_channels += 1                
                if ues[bidder['ue_id']].nb_succ_no_channels > NB_NO_CHANNEL_TO_RESET:
                    # ues[bidder['ue_id']].nb_succ_no_channels = 0
                    pass
                elif ues[bidder['ue_id']].nb_succ_no_channels == NB_MAX_NO_CHANNEL:
                    ues[bidder['ue_id']].blocks += 1

                
                if bidder['bid'] > 0:
                    ues[bidder['ue_id']].budget -= BIDDING_COST

                ues[bidder['ue_id']].update_valuation()

                ues[bidder['ue_id']].bidding_history.append({
                    'episode': ep,
                    'bs_id': bidder['bs_id'],
                    'bid': bidder['bid'],
                    'demand': bidder['demand'],
                    'price': 0,
                    'blocks': ues[bidder['ue_id']].blocks,
                    'win': 0
                })

                ue_statistics_history_per_episode.append({
                    'ue_id': bidder['ue_id'],
                    'bs_id': bidder['bs_id'],
                    'bid': bidder['bid'],
                    'valuation': bidder['valuation'],
                    'budget': ues[bidder['ue_id']].budget,
                    'demand': bidder['demand'],
                    'price': 0,
                    'ue_nb_succ_no_channels': ues[bidder['ue_id']].nb_succ_no_channels,
                    'blocks': ues[bidder['ue_id']].blocks,
                    'bs_reservation_price': bidder['bs_reservation_price'],
                    'win': 0
                })
                # print("first round looser", bidder['ue_id'], "points", ues[bidder['ue_id']].points, "succ no alloc", ues[bidder['ue_id']].nb_succ_no_channels)
            
            for bidder in second_round_loosers_list:
                ues[bidder['ue_id']].nb_succ_no_channels += 1
                if ues[bidder['ue_id']].nb_succ_no_channels > NB_NO_CHANNEL_TO_RESET:
                    # ues[bidder['ue_id']].nb_succ_no_channels = 0
                    pass
                elif ues[bidder['ue_id']].nb_succ_no_channels == NB_MAX_NO_CHANNEL:
                    ues[bidder['ue_id']].blocks += 1
                
                if bidder['bid'] > 0:
                    ues[bidder['ue_id']].budget -= BIDDING_COST

                ues[bidder['ue_id']].update_valuation()

                ues[bidder['ue_id']].bidding_history.append({
                    'episode': ep,
                    'bs_id': bidder['bs_id'],
                    'bid': bidder['bid'],
                    'demand': bidder['demand'],
                    'price': bidder['price'],
                    'blocks': ues[bidder['ue_id']].blocks,
                    'win': 0
                })

                ue_statistics_history_per_episode.append({
                    'ue_id': bidder['ue_id'],
                    'bs_id': bidder['bs_id'],
                    'bid': bidder['bid'],
                    'valuation': bidder['valuation'],
                    'budget': ues[bidder['ue_id']].budget,
                    'demand': bidder['demand'],
                    'price': bidder['price'],
                    'ue_nb_succ_no_channels': ues[bidder['ue_id']].nb_succ_no_channels,
                    'blocks': ues[bidder['ue_id']].blocks,
                    'bs_reservation_price': bidder['bs_reservation_price'],
                    'win': 0
                })

            win_prob = len(winners_list) / NM_UES
            total_price = sum(bidder['price'] for bidder in winners_list)
            alloc_efficency = sum(bidder['demand'] for bidder in winners_list) / (NUM_BSs * NUM_SUBCHANNELS)
            bs_costs = sum(bidder['bs_reservation_price'] * bidder['demand'] for bidder in winners_list)
            total_valuation = sum(bidder['valuation'] * bidder['demand'] for bidder in winners_list)
            m_bs_utility_currnt_ep = total_price - bs_costs
            m_ues_utility_current_ep = total_valuation - total_price

            # historical_feedback += 'Episode '+str(ep)+' outcome: '

            ue_statistics_history_per_episode.append({
                'win_prob': win_prob,
                'total_price': total_price,
                'alloc_efficency': alloc_efficency,
                'bs_costs': bs_costs,
                'm_bs_utility_currnt_ep': m_bs_utility_currnt_ep,
                'm_ues_utility_current_ep': m_ues_utility_current_ep
            })
            # print("episode", ep, auction_type, 'win_prob', win_prob, 'total_price', total_price, 'alloc_efficency', alloc_efficency, 'bs_costs', bs_costs)
            
            stat_utility_all_episodes.append({
                'bs_utility': m_bs_utility_currnt_ep,
                'ues_utility': m_ues_utility_current_ep
            })
            
            ue_statistics_history_per_episode_global.append(ue_statistics_history_per_episode)

        ### END STATS for current episode


            glb_bs_util_hitsory_avg.append(np.mean(bs_utilities))
            glb_bs_util_hitsory_acc.append(np.sum(bs_utilities))
            glb_ue_util_hitsory_agv.append(np.sum([u for ue in ues for u in ue.utility_history])/ (num_ues))
            glb_ue_util_hitsory_acc.append(np.sum([u for ue in ues for u in ue.utility_history]))
            # glb_ue_util_hitsory_agv.append(np.mean([u for ue in ues for u in ue.utility_history]))
            # glb_ue_util_hitsory_acc.append(np.sum([u for ue in ues for u in ue.utility_history]))
           

            logs.append({ # vcg_logs  and  gsp_logs
                'episode': ep,
                'avg_bs_util': np.mean(bs_utilities),
                'acc_bs_util': np.sum(bs_utilities),
                'avg_ue_util': np.mean([ue.utility for ue in ues]),
                'acc_ue_util': np.sum([ue.utility for ue in ues]),
                'glb_avg_bs_util': np.mean(glb_bs_util_hitsory_avg),
                'glb_acc_bs_util': np.sum(glb_bs_util_hitsory_avg),
                'glb_avg_ue_util': np.mean(glb_ue_util_hitsory_agv),
                'glb_acc_ue_util': np.sum(glb_ue_util_hitsory_acc)
            })

            # Update budget
            # for ue0 in ues:
            #     ues[ue0.id].budget += np.random.uniform(1.3, 1.5) # np.random.uniform(1.3, 1.5)
                # ues[ue0.id].budget += np.random.poisson(ues[ue0.id].lambda_poiss)
                # ues[ue0.id].demand = np.random.randint(1, 3)

            # print("\n")

        # --- Fairness logs ---
        for ue in ues:
            fairness_logs.append({
                'episode': ep,
                'ue_id': ue.id,
                'win_freq': sum(ue.win_history) / WINDOW_SIZE,
                'loss_freq': sum(ue.loss_history) / WINDOW_SIZE
            })

    # print(ratio_diff_list)
    
    utilities_file_name = "00_"+auction_type+"_logs_"+ue_bidding_mode+".csv"
    pd.DataFrame(stat_utility_all_episodes).to_csv(utilities_file_name, index=False)
    
    # print("gggggggggg", ue_statistics_history_per_episode_global)
    excel_file_name= "00_stat_"+auction_type+"_"+ue_bidding_mode+"_EP_"+str(ep+1)+".xlsx"
    writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter') # pylint: disable=abstract-class-instantiated
    for ep in range(0, NUM_EPISODES):
        # print("SAVING EPISODE", ep)
        df = pd.DataFrame(ue_statistics_history_per_episode_global[ep])
        df.to_excel(writer, sheet_name="EP_"+str(ep))
    
    writer._save()
    # pd.DataFrame(ue_statistics_history_per_episode_global[0]).to_csv("00_statistics_history_per_episode_"+auction_type+"_"+bidding_mode+".csv", index=False)


    # avg_win_all_UEs = 0
    stat_win_utility_all_UE= []
    for ue in ues:
        win_prob = sum(ue.win_history)/WINDOW_SIZE
        stat_win_utility_all_UE.append({
            'ue_id':ue.id,
            'utility': ue.utility,
            'win_prob': win_prob,
            'nb_wins': sum(ue.win_history)
        })
        print(f"UE {ue.id}, last_ep_win: {ue.last_ep_win} - Acc Utility: {ue.utility:.2f}, Win%: {win_prob:.2f}, Budget%: {ue.budget:.2f}, avg utility: {(ue.utility*win_prob):.2f}")
        # avg_win_all_UEs += sum(ue.win_history)/WINDOW_SIZE
    
    win_utilities_file_name = "01_"+auction_type+"_stat_win_utility_all_UEs_"+ue_bidding_mode+"_"+str(NUM_EPISODES)+".csv"
    pd.DataFrame(stat_win_utility_all_UE).to_csv(win_utilities_file_name, index=False)

    print(auction_type)

    # for bs in bss:
    #     print(bs.clearing_prices)


# Run
run_simulation(num_ues=NM_UES, auction_type="VCG", ue_bidding_mode=BIDDING_MODE_TRUE_VAL)
# run_simulation(num_ues=NM_UES, auction_type="VCG", ue_bidding_mode=BIDDING_MODE_GREEDY)

# run_simulation(num_ues=NM_UES, auction_type="VCG", ue_bidding_mode=BIDDING_MODE_SHADED_VAL)
# run_simulation(num_ues=NM_UES, auction_type="GSP", ue_bidding_mode=BIDDING_MODE_SHADED_VAL)

# run_simulation(num_ues=NM_UES, auction_type="VCG", ue_bidding_mode=BIDDING_MODE_LLM)




