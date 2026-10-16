#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
DRO-EMO: Logistics-Centric Case Study for the Mombasa-Bujumbura Corridor
========================================================================

Tier 1 competitor set (conforms to the Prins benchmark). The corridor
study now benchmarks DRO-EMO against NSGA-II, NSGA-III, SPEA2, MOEA/D,
and RL-NSGAII, the same set used in the Prins study. Both studies share
the same competitor set and the same reference-point construction, so
hypervolume values are directly comparable across the two studies.

Outputs are written to:
  D:\Container Transportation Routing Problems\
    Manuscript3-A Low-carbon based Robust Optimization under uncertainty\
    DRO-EMO on EAC case_study_results

  [CACHE-1]  Reference-point pass cached to disk (Tier-1 filename).
             Reduced to 10 runs per algorithm; the main comparison uses
             30. The cache is written to the working directory and also
             copied into the output directory for convenience.
  [CACHE-2]  Progress prints flushed line by line.
  [CLEAN-1]  Removed dead rle_emo lookup.
  [CLEAN-2]  Removed duplicated import block.
  [NEW-1]    DeterministicEvaluator and StochasticEvaluator.
  [NEW-2]    Evaluation-mode comparison.
  [NEW-3]    Reliability-threshold (delta_min) sensitivity.
  [NEW-4]    CSV output for both new analyses.
  [NEW-5]    Both new analyses called from main().
  [FIX-REF]  Global HV reference point validated before use.
"""

# ============================================================================
# ENVIRONMENT FIX
# ============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import random
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time, json, os
from datetime import datetime
from pathlib import Path
import hashlib, warnings, tempfile, shutil

warnings.filterwarnings('ignore')

import sys, importlib.util

_required = ["numpy", "scipy", "matplotlib", "pandas"]
_missing = [p for p in _required if importlib.util.find_spec(p) is None]
if _missing:
    print("MISSING:", _missing)
    sys.exit(1)

from scipy import stats
from scipy.spatial import distance

if not (hasattr(np, 'subtract') and hasattr(np, 'add')):
    print("BROKEN NUMPY:", np.__version__)
    sys.exit(1)

try:
    _np_maj = int(str(np.__version__).split('.')[0])
    _mpl_maj, _mpl_min = (int(x) for x in matplotlib.__version__.split('.')[:2])
except Exception:
    _np_maj, _mpl_maj, _mpl_min = 0, 0, 0
if _np_maj >= 2 and (_mpl_maj < 3 or (_mpl_maj == 3 and _mpl_min < 9)):
    print(f"INCOMPATIBLE: numpy={np.__version__} matplotlib={matplotlib.__version__}")
    print("Fix: conda install -c conda-forge \"numpy<2.0\"")
    sys.exit(1)

print(f"[env] numpy={np.__version__}, matplotlib={matplotlib.__version__}, "
      f"pandas={pd.__version__}")

# ============================================================================
# CONSTANTS
# ============================================================================
ORIGIN_NODE = 1
DESTINATION_NODE = 19
CONTAINER_WEIGHT = 14
CARBON_PRICE_DEFAULT = 50.0
CARBON_PRICE_STD_DEFAULT = 5.0
EMISSION_CAP = 5000
TIME_CAP = 360
MIN_RELIABILITY = 0.30
TRANS_SHIPMENT_TIME = 4
DEMAND_CV_DEFAULT = 0.25
N_RUNS = 30
ALPHA = 0.05

COLORS = {'DRO-EMO':   '#2ecc71',
          'NSGA-II':   '#3498db',
          'NSGA-III':  '#9b59b6',
          'SPEA2':     '#e67e22',
          'MOEA/D':    '#f39c12',
          'RL-NSGAII': '#e74c3c'}

MARKERS = {'DRO-EMO':   'o',
           'NSGA-II':   's',
           'NSGA-III':  '^',
           'SPEA2':     'D',
           'MOEA/D':    'v',
           'RL-NSGAII': 'P'}

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class DROEMOConfig:
    population_size_base: int = 60
    generations_base: int = 80
    population_size_divisor: int = 5
    ppo_learning_rate: float = 3e-4
    ppo_clip_epsilon: float = 0.2
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_training_episodes: int = 5000
    ppo_proportion: float = 0.25
    emission_proportion: float = 0.25
    cost_proportion: float = 0.20
    balanced_proportion: float = 0.20
    opposite_proportion: float = 0.10
    diversity_low_threshold: float = 0.15
    diversity_high_threshold: float = 0.45
    mutation_min: float = 0.05
    mutation_max: float = 0.35
    crossover_min: float = 0.65
    crossover_max: float = 0.95
    stagnation_threshold: int = 5
    def get_population_size(self, n):
        return self.population_size_base + n // self.population_size_divisor
    def get_generations(self, n):
        return self.generations_base + n // self.population_size_divisor

@dataclass
class ExperimentConfig:
    num_runs: int = 30
    base_seed: int = 42
    dro_emo: DROEMOConfig = field(default_factory=DROEMOConfig)
    moead_neighborhood_size: int = 20
    q_learning_alpha: float = 0.1
    q_learning_gamma: float = 0.9
    q_learning_epsilon: float = 0.1
    confidence_level: float = 0.95
    alpha: float = 0.05
    min_effect_size: float = 0.5
    stochastic_borders: bool = False
    wilcoxon_two_sided: bool = False
    carbon_price_std: float = CARBON_PRICE_STD_DEFAULT
    demand_cv: float = DEMAND_CV_DEFAULT
    def get_seed(self, run_id):
        return self.base_seed + run_id

# ============================================================================
# INDIVIDUAL
# ============================================================================
class EnhancedIndividual:
    def __init__(self):
        self.route = []; self.modes = []
        self.objectives = None; self.det_objectives = None
        self.crowding_distance = 0; self.rank = 0
        self.rail_ratio = 0; self.water_ratio = 0; self.road_ratio = 0
        self.mode_switches = 0; self.reliability = 0
        self.border_crossings = 0; self.transshipment_count = 0
        self.carbon_efficiency = 0
        self.is_valid = True; self.violation_reason = None
        self.constraint_violations = []
        self.scenario_results = {}
        self.mode_distances = {'road':0.0,'rail':0.0,'water':0.0}
    def copy(self):
        new = EnhancedIndividual()
        for a in ['route','modes','objectives','det_objectives',
                  'constraint_violations','scenario_results','mode_distances',
                  'is_valid','violation_reason',
                  'rail_ratio','water_ratio','road_ratio',
                  'mode_switches','reliability',
                  'border_crossings','transshipment_count',
                  'carbon_efficiency','crowding_distance','rank']:
            v = getattr(self, a)
            if isinstance(v, list): setattr(new, a, v.copy())
            elif isinstance(v, dict): setattr(new, a, v.copy())
            else: setattr(new, a, v)
        return new
    def dominates(self, other):
        if self.objectives is None or other.objectives is None: return False
        if not self.is_valid and other.is_valid: return False
        if self.is_valid and not other.is_valid: return True
        if not self.is_valid and not other.is_valid: return False
        not_worse = all(a <= b for a, b in zip(self.objectives, other.objectives))
        strictly = any(a < b for a, b in zip(self.objectives, other.objectives))
        return not_worse and strictly
    def compute_carbon_efficiency(self):
        if self.objectives and self.objectives[0] > 0:
            self.carbon_efficiency = self.objectives[1] / self.objectives[0]
        return self.carbon_efficiency

# ============================================================================
# DATASET
# ============================================================================
class LC_CTRP_Dataset:
    def __init__(self, carbon_price: float = CARBON_PRICE_DEFAULT,
                 carbon_price_std: float = CARBON_PRICE_STD_DEFAULT,
                 demand_cv: float = DEMAND_CV_DEFAULT,
                 stochastic_borders: bool = False):
        self.stochastic_borders = stochastic_borders
        self.carbon_price_mean = carbon_price
        self.carbon_price_std = carbon_price_std
        self.demand_cv = demand_cv

        self.nodes = {
            1:{'name':'Mombasa','country':'Kenya','type':'Port'},
            2:{'name':'Tanga','country':'Tanzania','type':'Port'},
            3:{'name':'Dar-es-Salam','country':'Tanzania','type':'Port'},
            4:{'name':'Morogoro','country':'Tanzania','type':'ICD'},
            5:{'name':'Dodoma','country':'Tanzania','type':'Capital'},
            6:{'name':'Arusha','country':'Tanzania','type':'Border'},
            7:{'name':'Voi','country':'Kenya','type':'Transit'},
            8:{'name':'Nairobi','country':'Kenya','type':'ICD'},
            9:{'name':'Kisumu','country':'Kenya','type':'Port'},
            10:{'name':'Musoma','country':'Tanzania','type':'Port'},
            11:{'name':'Mwanza','country':'Tanzania','type':'Port'},
            12:{'name':'Isaka','country':'Tanzania','type':'ICD'},
            13:{'name':'Tabora','country':'Tanzania','type':'Rail Hub'},
            14:{'name':'Kigoma','country':'Tanzania','type':'Port'},
            15:{'name':'Bukoba','country':'Tanzania','type':'Port'},
            16:{'name':'Kampala','country':'Uganda','type':'Capital'},
            17:{'name':'Masaka','country':'Uganda','type':'Transit'},
            18:{'name':'Kigali','country':'Rwanda','type':'Capital'},
            19:{'name':'Bujumbura','country':'Burundi','type':'Port'},
        }
        self.coords = {
            1:(-4.0435,39.6682), 2:(-5.0690,39.0980), 3:(-6.7924,39.2083),
            4:(-6.8240,37.6600), 5:(-6.1630,35.7516), 6:(-3.3670,36.6830),
            7:(-3.3960,38.5560), 8:(-1.2921,36.8219), 9:(-0.1022,34.7617),
            10:(-1.5000,33.8000),11:(-2.5167,32.9000),12:(-3.9025,33.8746),
            13:(-5.0167,32.8000),14:(-4.8833,29.6333),15:(-1.3317,31.8122),
            16:(0.3476,32.5825), 17:(-0.3333,31.7333),18:(-1.9441,30.0619),
            19:(-3.3614,29.3599),
        }
        self.arcs = {
            (1,2):{'road':(174,40.02,2.49)}, (1,7):{'road':(159,36.57,2.27)},
            (1,8):{'rail':(472,42.48,7.87)}, (2,3):{'road':(358,82.34,5.11)},
            (2,6):{'road':(370,85.10,5.29)}, (3,4):{'road':(195,44.85,2.79),'rail':(205,18.45,3.42)},
            (4,5):{'road':(262,60.26,3.74),'rail':(336,30.24,5.60)}, (5,6):{'road':(421,96.83,6.01)},
            (6,7):{'road':(233,53.59,3.33),'rail':(232,20.88,3.87)}, (6,8):{'road':(272,62.56,3.89)},
            (6,9):{'road':(509,117.07,7.27)}, (6,13):{'road':(657,151.11,9.39)},
            (7,8):{'road':(328,75.44,4.69)}, (8,9):{'road':(360,82.80,5.14),'rail':(380,34.20,6.33)},
            (8,10):{'road':(496,114.08,7.09)}, (10,11):{'road':(222,51.06,3.17)},
            (11,12):{'road':(217,49.91,3.10),'rail':(248,22.32,4.13)}, (11,15):{'road':(512,117.76,7.31)},
            (12,13):{'road':(189,43.47,2.70),'rail':(133,11.97,2.22)}, (12,18):{'road':(571,131.33,8.16)},
            (12,19):{'road':(573,131.79,8.19)}, (13,14):{'road':(414,95.22,5.91),'rail':(408,36.72,6.80)},
            (14,19):{'road':(235,54.05,3.36),'water':(175,7.00,7.78)}, (15,17):{'road':(170,39.10,2.43)},
            (15,18):{'road':(434,99.82,6.20)}, (16,17):{'road':(70.2,16.15,1.00)},
            (17,18):{'road':(437,100.51,6.24)}, (18,19):{'road':(298,68.54,4.26)},
        }
        for (i,j),d in list(self.arcs.items()):
            self.arcs[(j,i)] = {m:v for m,v in d.items() if v is not None}

        self.base_mode_params = {
            'road':{'speed':70,'cost_per_km':0.23,'emission_factor':0.120,
                    'base_reliability':0.75,'daily_capacity':300},
            'rail':{'speed':60,'cost_per_km':0.09,'emission_factor':0.020,
                    'base_reliability':0.90,'daily_capacity':500},
            'water':{'speed':22.5,'cost_per_km':0.04,'emission_factor':0.015,
                     'base_reliability':0.70,'daily_capacity':100},
        }
        self.mode_params = {k:v.copy() for k,v in self.base_mode_params.items()}

        self.base_transshipment_time = {
            ('road','road'):0,('road','rail'):5.5,('road','water'):8,
            ('rail','road'):5.5,('rail','rail'):0,('rail','water'):12,
            ('water','road'):8,('water','rail'):12,('water','water'):0,
        }
        self.base_transshipment_cost = {
            ('road','road'):0,('road','rail'):50,('road','water'):70,
            ('rail','road'):50,('rail','rail'):0,('rail','water'):37.5,
            ('water','road'):70,('water','rail'):37.5,('water','water'):0,
        }
        self.transshipment_time = self.base_transshipment_time.copy()
        self.transshipment_cost = self.base_transshipment_cost.copy()

        self.border_crossings = {
            ('Kenya', 'Tanzania'):{'base_time':(8,12),'cost':150},
            ('Tanzania', 'Kenya'):{'base_time':(8,12),'cost':150},
            ('Tanzania', 'Uganda'):{'base_time':(10,16),'cost':200},
            ('Uganda', 'Tanzania'):{'base_time':(10,16),'cost':200},
            ('Tanzania', 'Rwanda'):{'base_time':(10,16),'cost':200},
            ('Rwanda', 'Tanzania'):{'base_time':(10,16),'cost':200},
            ('Tanzania', 'Burundi'):{'base_time':(8,12),'cost':180},
            ('Burundi', 'Tanzania'):{'base_time':(8,12),'cost':180},
            ('Uganda', 'Rwanda'):{'base_time':(6,10),'cost':120},
            ('Rwanda', 'Uganda'):{'base_time':(6,10),'cost':120},
            ('Rwanda', 'Burundi'):{'base_time':(12,24),'cost':250},
            ('Burundi', 'Rwanda'):{'base_time':(12,24),'cost':250},
            ('Kenya', 'Uganda'):{'base_time':(6,10),'cost':120},
            ('Uganda', 'Kenya'):{'base_time':(6,10),'cost':120},
        }
        for k in self.border_crossings:
            bt = self.border_crossings[k]['base_time']
            self.border_crossings[k]['time_penalty'] = (bt[0], bt[1])

        self.stochastic_scenarios = {
            'S1_Normal':{'probability':0.60,'cost_mult':1.00,'time_mult':1.00,
                         'reliability':1.00,'carbon_price_mult':0.0},
            'S2_High_Demand':{'probability':0.25,'cost_mult':1.10,'time_mult':1.15,
                              'reliability':0.90,'carbon_price_mult':0.1},
            'S3_Fuel_Surge':{'probability':0.10,'cost_mult':1.25,'time_mult':1.05,
                             'reliability':1.00,'carbon_price_mult':-0.05},
            'S4_Peak_Season':{'probability':0.05,'cost_mult':1.20,'time_mult':1.30,
                              'reliability':0.80,'carbon_price_mult':0.2},
        }
        self.robust_scenarios = {
            'U1_Rail_Strike':{'affected_modes':['rail'],'cost_mult':2.00,
                              'time_mult':1.50,'carbon_price_mult':0.15},
            'U2_Border_Closure':{'affected_modes':['all'],'cost_mult':1.80,
                                 'time_mult':2.00,'carbon_price_mult':0.25},
            'U3_Port_Congestion':{'affected_modes':['all'],'cost_mult':1.30,
                                  'time_mult':1.50,'carbon_price_mult':0.1},
            'U4_Political_Unrest':{'affected_modes':['all'],'cost_mult':1.60,
                                   'time_mult':1.80,'carbon_price_mult':0.3},
            'U5_Natural_Disaster':{'affected_modes':['road','water'],'cost_mult':1.40,
                                   'time_mult':1.70,'carbon_price_mult':0.2},
            'U6_Normal':{'affected_modes':[],'cost_mult':1.00,
                         'time_mult':1.00,'carbon_price_mult':0.0},
        }
        self.seasonal_params = {
            'Dry':{'road_reliability':0.85,'rail_reliability':0.92,'water_reliability':0.75,
                   'cost_factor':1.00,'time_factor':1.00,'emission_factor':1.00,
                   'carbon_price_mult':0.0},
            'Wet':{'road_reliability':0.65,'rail_reliability':0.85,'water_reliability':0.50,
                   'cost_factor':1.15,'time_factor':1.30,'emission_factor':1.15,
                   'carbon_price_mult':0.05},
            'Peak':{'road_reliability':0.70,'rail_reliability':0.88,'water_reliability':0.60,
                    'cost_factor':1.20,'time_factor':1.25,'emission_factor':1.10,
                    'carbon_price_mult':0.1},
        }
        self.node_capacities = {
            1:{'daily_capacity':2500,'avg_demand':1800,'storage':10000},
            3:{'daily_capacity':1800,'avg_demand':1500,'storage':8000},
            8:{'daily_capacity':1500,'avg_demand':1200,'storage':5000},
            14:{'daily_capacity':200,'avg_demand':150,'storage':500},
            16:{'daily_capacity':800,'avg_demand':600,'storage':2000},
            18:{'daily_capacity':600,'avg_demand':450,'storage':1500},
            19:{'daily_capacity':400,'avg_demand':350,'storage':1000},
        }
        self.global_params = {
            'origin':ORIGIN_NODE,'destination':DESTINATION_NODE,
            'container_weight':CONTAINER_WEIGHT,
            'carbon_price':carbon_price/1000.0,
            'carbon_price_std':carbon_price_std/1000.0,
            'emission_cap':EMISSION_CAP,'time_cap':TIME_CAP,
            'min_reliability':MIN_RELIABILITY,
            'transshipment_time':TRANS_SHIPMENT_TIME,
            'demand_cv':demand_cv,
        }
        self.adjacency = defaultdict(list)
        for (i,j) in self.arcs.keys():
            self.adjacency[i].append(j)

    def reset_to_baseline(self):
        self.mode_params = {k:v.copy() for k,v in self.base_mode_params.items()}
        self.transshipment_time = self.base_transshipment_time.copy()
        self.transshipment_cost = self.base_transshipment_cost.copy()
        self.global_params['carbon_price'] = CARBON_PRICE_DEFAULT/1000.0
        for k in self.border_crossings:
            bt = self.border_crossings[k]['base_time']
            self.border_crossings[k]['time_penalty'] = (bt[0], bt[1])

    def apply_carbon_price(self, p, std=None):
        self.carbon_price_mean = p
        self.global_params['carbon_price'] = p / 1000.0
        if std is not None:
            self.carbon_price_std = std
            self.global_params['carbon_price_std'] = std / 1000.0

    def apply_demand_cv(self, cv):
        self.demand_cv = cv
        self.global_params['demand_cv'] = cv

    def apply_border_delay_multiplier(self, m):
        for k in self.border_crossings:
            bt = self.border_crossings[k]['base_time']
            self.border_crossings[k]['time_penalty'] = (bt[0]*m, bt[1]*m)

    def apply_transshipment_cost_multiplier(self, m):
        for k in self.transshipment_cost:
            self.transshipment_cost[k] = self.base_transshipment_cost[k]*m

    def get_available_modes(self, n1, n2):
        if (n1,n2) not in self.arcs: return []
        return [m for m in ['road','rail','water']
                if self.arcs[(n1,n2)].get(m) is not None]

    def get_arc_data(self, n1, n2, mode):
        if (n1,n2) not in self.arcs: return None
        return self.arcs[(n1,n2)].get(mode)

    def get_country(self, n): return self.nodes.get(n,{}).get('country','?')
    def is_border_crossing(self, n1, n2):
        return self.get_country(n1) != self.get_country(n2)

    def get_border_penalty(self, n1, n2):
        if not self.is_border_crossing(n1, n2): return 0.0, 0.0
        key = (self.get_country(n1), self.get_country(n2))
        if key not in self.border_crossings: return 0.0, 0.0
        d = self.border_crossings[key]
        lo, hi = d['time_penalty']
        t = random.uniform(lo, hi) if self.stochastic_borders else 0.5*(lo+hi)
        return (t, d['cost'])

    def calculate_arc_metrics(self, n1, n2, mode, season='Dry',
                              scenario='S1_Normal'):
        arc = self.get_arc_data(n1, n2, mode)
        if arc is None: return None
        distance, base_cost, base_time = arc
        p = self.mode_params[mode]
        sd = self.seasonal_params.get(season, self.seasonal_params['Dry'])
        sc = self.stochastic_scenarios.get(scenario,
                                           self.stochastic_scenarios['S1_Normal'])
        rel_base = sd.get(f'{mode}_reliability', p['base_reliability'])
        cm = sd['cost_factor'] * sc['cost_mult']
        tm = sd['time_factor'] * sc['time_mult']
        em = sd.get('emission_factor', 1.0)
        cost = distance * p['cost_per_km'] * cm
        emissions = distance * p['emission_factor'] * \
                    self.global_params['container_weight'] * em
        time_h = distance / p['speed'] * tm
        reliability = rel_base * sc['reliability']
        carbon_price_effective = self.global_params['carbon_price'] * (
            1 + sc.get('carbon_price_mult', 0))
        cost += emissions * carbon_price_effective
        if self.is_border_crossing(n1, n2):
            tp, cp = self.get_border_penalty(n1, n2)
            time_h += tp; cost += cp
        return {'cost':cost,'emissions':emissions,'time':time_h,
                'reliability':reliability,'distance':distance,'mode':mode,
                'carbon_price_used':carbon_price_effective}

    def build_ambiguity_residuals(self, demand_cv=None):
        if demand_cv is None: demand_cv = self.demand_cv
        rows = []
        for _, sc in self.stochastic_scenarios.items():
            cost_res = (sc['cost_mult'] - 1) * (1 + demand_cv)
            rows.append([cost_res, sc['time_mult']-1, 0,
                         sc['reliability']-1, sc.get('carbon_price_mult', 0)])
        for _, sc in self.robust_scenarios.items():
            cost_res = (sc['cost_mult'] - 1) * (1 + demand_cv)
            rows.append([cost_res, sc['time_mult']-1, 0, 0,
                         sc.get('carbon_price_mult', 0)])
        for _, sd in self.seasonal_params.items():
            cost_res = (sd['cost_factor'] - 1) * (1 + demand_cv)
            rows.append([cost_res, sd['time_factor']-1,
                         sd['emission_factor']-1, 0,
                         sd.get('carbon_price_mult', 0)])
        std_rel = self.carbon_price_std / max(self.carbon_price_mean, 1e-9)
        for mult in [-2, -1, 0, 1, 2]:
            rows.append([0, 0, 0, 0, mult * std_rel])
        for mult in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
            rows.append([mult * demand_cv, 0, 0, 0, 0])
        return np.array(rows)

# ============================================================================
# AMBIGUITY SET + DRO EVALUATOR
# ============================================================================
@dataclass
class AmbiguitySet:
    W_lower:np.ndarray; W_upper:np.ndarray
    mean_hat:np.ndarray; mean_lower:np.ndarray; mean_upper:np.ndarray
    Sigma:np.ndarray; sigma_f:np.ndarray; sigma_b:np.ndarray
    H_sigma:np.ndarray
    def dimension(self): return len(self.mean_hat)

    @staticmethod
    def from_residuals(residuals, confidence=0.95):
        r = np.atleast_2d(residuals); n,d = r.shape
        Wl = r.min(0); Wu = r.max(0); mh = r.mean(0)
        s = r.std(0, ddof=1) if n>1 else np.zeros(d)
        z = stats.norm.ppf((1+confidence)/2)
        m = z*s/np.sqrt(max(n,1))
        ml, mu = mh-m, mh+m
        Sigma = np.cov(r, rowvar=False) if n>1 else np.eye(d)
        if Sigma.ndim == 0: Sigma = np.array([[float(Sigma)]])
        Sigma = 0.9*Sigma + 0.1*np.diag(np.diag(Sigma)) + 1e-8*np.eye(d)
        sf = AmbiguitySet._dd(r,+1); sb = AmbiguitySet._dd(r,-1)
        ev, evec = np.linalg.eigh(Sigma)
        ev = np.clip(ev,1e-10,None)
        H = (evec/np.sqrt(ev)).T
        return AmbiguitySet(Wl,Wu,mh,ml,mu,Sigma,sf,sb,H)

    @staticmethod
    def _dd(r, sign):
        th = np.logspace(-3,1,40); rr = sign*r
        out = np.zeros(r.shape[1])
        for j in range(r.shape[1]):
            x = rr[:,j]; best = 0.0
            for t in th:
                z = t*x; mx = z.max()
                lme = mx + np.log(np.mean(np.exp(z-mx)))
                v = np.sqrt(max(2*lme/t**2, 0.0))
                if v>best: best = v
            out[j] = best
        return out

    def summary(self):
        return {'dim':self.dimension(),
                'sigma_f':self.sigma_f.tolist(),
                'sigma_b':self.sigma_b.tolist(),
                'sigma_f_over_b':(self.sigma_f/
                                  np.maximum(self.sigma_b,1e-8)).tolist()}

    def structure(self, tol=1e-8):
        sf, sb = self.sigma_f, self.sigma_b
        n_fwd_only = int(np.sum((sf > tol) & (sb < tol)))
        n_bwd_only = int(np.sum((sf < tol) & (sb > tol)))
        n_two_sided = int(np.sum((sf > tol) & (sb > tol)))
        n_degenerate = int(np.sum((sf < tol) & (sb < tol)))
        sfb = sf/np.maximum(sb, 1e-8)
        finite = sfb[(sf > tol) & (sb > tol)]
        mean_ratio = float(np.mean(finite)) if len(finite) > 0 else float('nan')
        return {'mean_ratio': mean_ratio,
                'n_two_sided': n_two_sided,
                'n_forward_only': n_fwd_only,
                'n_backward_only': n_bwd_only,
                'n_degenerate': n_degenerate,
                'n_total': int(len(sf))}

class DROEvaluator:
    def __init__(self, amb): self.amb = amb

    def pi1(self,r0,r):
        r = np.asarray(r,dtype=float).flatten()
        ws = float(np.sum(np.where(r>=0, r*self.amb.W_upper,
                                   r*self.amb.W_lower)))
        wm = float(np.sum(np.where(r>=0, r*self.amb.mean_upper,
                                   r*self.amb.mean_lower)))
        return max(0.0, r0+max(ws,wm))

    def pi2(self,r0,r):
        r = np.asarray(r,dtype=float).flatten()
        wm = float(np.sum(np.where(r>=0, r*self.amb.mean_upper,
                                   r*self.amb.mean_lower)))
        mu = r0+wm; var = max(float(r@self.amb.Sigma@r),0.0)
        return (0.5*mu+0.5*np.sqrt(mu**2+var)) if mu>=0 else 0.5*np.sqrt(var)

    def pi3(self,r0,r):
        r = np.asarray(r,dtype=float).flatten()
        wm = float(np.sum(np.where(r>=0, r*self.amb.mean_upper,
                                   r*self.amb.mean_lower)))
        x0 = r0+wm
        u = np.maximum(r*self.amb.sigma_f, -r*self.amb.sigma_b)
        nu2 = float(u@u)
        sup_z = float(np.sum(np.where(r>=0, r*self.amb.W_upper,
                                      r*self.amb.W_lower)))
        best = np.inf
        for lam in np.logspace(-3,2,60):
            e = (x0+sup_z)/lam + nu2/(2*lam**2)
            if e>50: continue
            v = (lam/np.e)*np.exp(e)
            if v<best: best = v
        if not np.isfinite(best): return self.pi1(r0,r)
        return min(best, 1e6+max(0.0,r0))

    def pi(self,r0,r):
        return max(self.pi1(r0,r), self.pi2(r0,r), self.pi3(r0,r))

# ============================================================================
# ALTERNATIVE EVALUATORS
# ============================================================================
class DeterministicEvaluator:
    def __init__(self, dataset, season='Dry'):
        self.dataset = dataset; self.season = season
    def pi(self, r0, r): return float(r0)

class StochasticEvaluator:
    def __init__(self, dataset, season='Dry'):
        self.dataset = dataset; self.season = season
        self.scenarios = dataset.stochastic_scenarios
    def evaluate_expected(self, path, modes):
        total_c = total_e = total_t = 0.0
        p_sum = 0.0
        for sname, sc in self.scenarios.items():
            prob = sc['probability']
            c, e, t, _ = self._route_metrics(path, modes, sname)
            total_c += prob * c; total_e += prob * e; total_t += prob * t
            p_sum += prob
        if p_sum <= 0: return (1e10, 1e10, 1e10)
        return (total_c/p_sum, total_e/p_sum, total_t/p_sum)
    def _route_metrics(self, path, modes, scenario_name):
        tc = te = tt = 0.0; rel = 1.0; prev = None
        for i in range(len(path)-1):
            n1, n2 = path[i], path[i+1]
            mode = modes[i] if i < len(modes) else 'road'
            avail = self.dataset.get_available_modes(n1, n2)
            if mode not in avail and avail: mode = avail[0]
            m = self.dataset.calculate_arc_metrics(
                n1, n2, mode, self.season, scenario_name)
            if m is None: return (1e10, 1e10, 1e10, 0.0)
            if prev and prev != mode:
                tt += self.dataset.transshipment_time.get((prev, mode), 0)
                tc += self.dataset.transshipment_cost.get((prev, mode), 0)
            tc += m['cost']; te += m['emissions']; tt += m['time']
            rel *= m['reliability']; prev = mode
        return (tc, te, tt, rel)
    def pi(self, r0, r): return float(r0)

# ============================================================================
# FREIGHT ROUTING PROBLEM
# ============================================================================
class FreightRoutingProblem:
    TEMPLATES = [
        {'name':'Northern Corridor (Kigali)','path':[1,8,6,13,12,18,19],
         'modes':['rail','road','road','rail','road','road']},
        {'name':'Northern Corridor (Isaka)','path':[1,8,6,13,12,19],
         'modes':['rail','road','road','rail','road']},
        {'name':'Central Corridor (Kigoma)','path':[1,8,6,13,14,19],
         'modes':['rail','road','road','rail','water']},
        {'name':'Central Corridor (mixed)','path':[1,8,6,13,14,19],
         'modes':['rail','road','rail','rail','road']},
        {'name':'Great Lakes (Kampala)','path':[1,8,6,13,12,11,15,17,18,19],
         'modes':['rail','road','road','rail','road','road','road','road','road']},
        {'name':'Lake Victoria Corridor','path':[1,8,10,11,12,13,14,19],
         'modes':['rail','road','road','rail','rail','rail','water']},
        {'name':'Kampala Gateway (alt modes)','path':[1,8,6,13,12,11,15,17,18,19],
         'modes':['rail','road','rail','rail','rail','road','road','road','road']},
        {'name':'Rail Anchored','path':[1,8,6,13,12,18,19],
         'modes':['rail','road','road','road','road','road']},
    ]

    def __init__(self, dataset, season='Dry', carbon_price=CARBON_PRICE_DEFAULT,
                 demand_cv=None, evaluator=None):
        self.dataset = dataset; self.season = season
        self.carbon_price = carbon_price
        self.n_obj = 3
        self.origin = ORIGIN_NODE; self.destination = DESTINATION_NODE
        self.ambiguity = AmbiguitySet.from_residuals(
            dataset.build_ambiguity_residuals(demand_cv=demand_cv),
            confidence=0.95)
        self.dro = DROEvaluator(self.ambiguity)
        self.evaluator = evaluator
        self.route_templates = self.TEMPLATES
        self.n_var = 20

    def n_feasible_templates(self):
        return sum(1 for t in self.TEMPLATES
                   if self.evaluate_route(t['path'], t['modes']).is_valid)

    def get_all_routes(self):
        inds = []
        for t in self.route_templates:
            ind = self.evaluate_route(t['path'], t['modes'])
            if ind.objectives is not None and ind.is_valid: inds.append(ind)
        return inds

    def _exact_sensitivities(self, ind):
        L = 0.0; em_nom = 0.0
        for i in range(len(ind.route)-1):
            n1,n2 = ind.route[i], ind.route[i+1]
            mode = ind.modes[i] if i<len(ind.modes) else 'road'
            arc = self.dataset.get_arc_data(n1,n2,mode)
            if arc is None: continue
            d,_,_ = arc; L += d
            em_nom += d*self.dataset.mode_params[mode]['emission_factor'] \
                      * self.dataset.global_params['container_weight']
        p = self.dataset.mode_params
        bc = 0.0; tn = 0.0
        for i in range(len(ind.route)-1):
            mode = ind.modes[i] if i<len(ind.modes) else 'road'
            arc = self.dataset.get_arc_data(ind.route[i], ind.route[i+1], mode)
            if arc is None: continue
            d,_,_ = arc
            bc += d*p[mode]['cost_per_km']; tn += d/p[mode]['speed']
        ack = bc/max(L,1e-9)
        lam = self.dataset.global_params['carbon_price']
        b_cost = np.array([L*ack, 0.0, em_nom*lam, 0.0, em_nom*lam])
        b_em = np.array([0.0, 0.0, em_nom, 0.0, 0.0])
        b_time = np.array([0.0, tn, 0.0, 0.0, 0.0])
        return (np.nan_to_num(b_cost), np.nan_to_num(b_em),
                np.nan_to_num(b_time))

    def evaluate_route(self, path, modes):
        ind = EnhancedIndividual(); ind.route = path; ind.modes = modes
        tc, te, tt = 0.0, 0.0, 0.0
        rd, wd, rod, prev = 0.0, 0.0, 0.0, None
        ms = 0; bc = 0; ts = 0; rel = 1.0
        for i in range(len(path)-1):
            n1,n2 = path[i], path[i+1]
            mode = modes[i] if i<len(modes) else 'road'
            avail = self.dataset.get_available_modes(n1,n2)
            if mode not in avail and avail: mode = avail[0]
            elif mode not in avail:
                ind.is_valid = False
                ind.violation_reason = f"No arc {n1}->{n2}"
                ind.objectives = [1e10]*3; ind.det_objectives = [1e10]*3
                return ind
            m = self.dataset.calculate_arc_metrics(n1,n2,mode,self.season)
            if m is None:
                ind.is_valid = False
                ind.violation_reason = f"No metrics {n1}->{n2}"
                ind.objectives = [1e10]*3; ind.det_objectives = [1e10]*3
                return ind
            if prev and prev != mode:
                ms += 1; ts += 1
                tt += self.dataset.transshipment_time.get((prev,mode),0)
                tc += self.dataset.transshipment_cost.get((prev,mode),0)
            tc += m['cost']; te += m['emissions']; tt += m['time']
            rel *= m['reliability']
            if mode == 'rail': rd += m['distance']
            elif mode == 'water': wd += m['distance']
            else: rod += m['distance']
            if self.dataset.is_border_crossing(n1,n2): bc += 1
            prev = mode
        tot = rd+wd+rod
        ind.rail_ratio = rd/tot if tot>0 else 0
        ind.water_ratio = wd/tot if tot>0 else 0
        ind.road_ratio = rod/tot if tot>0 else 0
        ind.mode_switches = ms; ind.border_crossings = bc
        ind.transshipment_count = ts; ind.reliability = rel
        ind.mode_distances = {'road':rod,'rail':rd,'water':wd}
        ind.det_objectives = [tc, te, tt]

        violations = []
        if te > self.dataset.global_params['emission_cap']:
            violations.append(
                f"Emissions {te:.0f} > cap "
                f"{self.dataset.global_params['emission_cap']}")
        if tt > self.dataset.global_params['time_cap']:
            violations.append(
                f"Time {tt:.0f}h > cap "
                f"{self.dataset.global_params['time_cap']}h")
        if rel < self.dataset.global_params['min_reliability']:
            violations.append(
                f"Reliability {rel:.2f} < min "
                f"{self.dataset.global_params['min_reliability']}")

        if self.evaluator is not None and isinstance(
                self.evaluator, DeterministicEvaluator):
            ind.objectives = [float(tc), float(te), float(tt)]
        elif self.evaluator is not None and isinstance(
                self.evaluator, StochasticEvaluator):
            c, e, t = self.evaluator.evaluate_expected(path, modes)
            ind.objectives = [float(c), float(e), float(t)]
        else:
            b_cost, b_em, b_time = self._exact_sensitivities(ind)
            ind.objectives = [float(self.dro.pi(tc, b_cost)),
                              float(self.dro.pi(te, b_em)),
                              float(self.dro.pi(tt, b_time))]
        if violations:
            ind.is_valid = False
            ind.violation_reason = "; ".join(violations)
            ind.constraint_violations = violations
        return ind

# ============================================================================
# DIAGNOSTIC
# ============================================================================
def run_template_diagnostic(season='Dry', carbon_price=50.0,
                            demand_cv=DEMAND_CV_DEFAULT,
                            stochastic_borders=False):
    print("\n" + "=" * 110)
    print(f" TEMPLATE-CONSTRAINT DIAGNOSTIC (season={season}, "
          f"carbon=${carbon_price}/ton, demand_cv={demand_cv}) ".center(110, "="))
    print("=" * 110)

    ds = LC_CTRP_Dataset(carbon_price=carbon_price, demand_cv=demand_cv,
                         stochastic_borders=stochastic_borders)
    ds.apply_carbon_price(carbon_price); ds.apply_demand_cv(demand_cv)
    problem = FreightRoutingProblem(ds, season=season,
                                    carbon_price=carbon_price,
                                    demand_cv=demand_cv)

    time_cap = ds.global_params['time_cap']
    em_cap = ds.global_params['emission_cap']
    rel_min = ds.global_params['min_reliability']

    print(f"\nCaps: TIME_CAP = {time_cap} h   EMISSION_CAP = {em_cap} kg   "
          f"MIN_RELIABILITY = {rel_min}")
    print(f"\n{'Template':<28} {'Time(h)':>9} {'Time':>6} "
          f"{'CO2(kg)':>10} {'CO2':>6} {'Reliab.':>9} {'Rel.':>6} {'Feas.':>6}")
    print("-" * 95)

    time_failures = []; em_failures = []; rel_failures = []
    no_arc_templates = []; feasible_count = 0

    for t in FreightRoutingProblem.TEMPLATES:
        ind = problem.evaluate_route(t['path'], t['modes'])
        structurally_invalid = (not ind.is_valid) and (
            ind.violation_reason is not None
            and ind.violation_reason.startswith("No arc"))
        if structurally_invalid:
            no_arc_templates.append((t['name'], ind.violation_reason))
            print(f"{t['name']:<28} {'—':>9} {'N/A':>6} {'—':>10} {'N/A':>6} "
                  f"{'—':>9} {'N/A':>6} {'NO ARC':>6}")
            continue
        time_v = ind.det_objectives[2]; em_v = ind.det_objectives[1]
        rel_v = ind.reliability
        time_ok = time_v <= time_cap; em_ok = em_v <= em_cap
        rel_ok = rel_v >= rel_min
        feasible = time_ok and em_ok and rel_ok
        if feasible: feasible_count += 1
        if not time_ok: time_failures.append((t['name'], time_v, time_cap))
        if not em_ok: em_failures.append((t['name'], em_v, em_cap))
        if not rel_ok: rel_failures.append((t['name'], rel_v, rel_min))
        print(f"{t['name']:<28} {time_v:>9.1f} "
              f"{'OK' if time_ok else 'FAIL':>6} "
              f"{em_v:>10.1f} {'OK' if em_ok else 'FAIL':>6} "
              f"{rel_v:>9.3f} {'OK' if rel_ok else 'FAIL':>6} "
              f"{str(feasible):>6}")

    print("-" * 95)
    n_total = len(FreightRoutingProblem.TEMPLATES)
    n_evaluable = n_total - len(no_arc_templates)
    print(f"\nFeasible templates: {feasible_count} / {n_evaluable}"
          + (f"  ({len(no_arc_templates)} not evaluable: missing arcs)"
             if no_arc_templates else ""))

    if no_arc_templates:
        print(f"\nStructurally invalid templates ({len(no_arc_templates)}):")
        for name, reason in no_arc_templates:
            print(f"  {name:<28} {reason}")
    if time_failures:
        print(f"\nTime failures ({len(time_failures)}):")
        for name, v, cap in time_failures:
            print(f"  {name:<28} time = {v:.1f} h > cap = {cap} h "
                  f"(over by {v-cap:.1f} h, {(v-cap)/cap*100:.1f}%)")
    else: print("\nNo time failures.")
    if em_failures:
        print(f"\nEmission failures ({len(em_failures)}):")
        for name, v, cap in em_failures:
            print(f"  {name:<28} CO2 = {v:.1f} kg > cap = {cap} kg "
                  f"(over by {v-cap:.1f} kg, {(v-cap)/cap*100:.1f}%)")
    else: print("\nNo emission failures.")
    if rel_failures:
        print(f"\nReliability failures ({len(rel_failures)}):")
        for name, v, cap in rel_failures:
            print(f"  {name:<28} reliab. = {v:.3f} < min = {cap} "
                  f"(short by {(cap-v)*100:.1f} percentage points)")
    else: print("\nNo reliability failures.")

    print("\n" + "-" * 95)
    print("LOGISTICS FINDING")
    print("-" * 95)
    if not time_failures and not em_failures:
        print("  Only reliability binds. TIME_CAP = 360 and EMISSION_CAP = 5000")
        print("  are not binding for any template. The failure of higher-mode-switch")
        print("  templates is a logistics finding: reliability compounds")
        print("  multiplicatively across borders and transshipments, so end-to-end")
        print("  reliability < 0.30 for any chain that crosses three borders.")
        print("  This motivates border modernisation and Kigoma-Bujumbura")
        print("  waterway investment.")
    print("=" * 110)
    return {'feasible': feasible_count,'no_arc_templates': no_arc_templates,
            'time_failures': time_failures,'em_failures': em_failures,
            'rel_failures': rel_failures}

# ============================================================================
# PPO AGENT
# ============================================================================
class PPOAgent:
    def __init__(self, sd, ad, cfg):
        self.sd = sd; self.ad = ad; self.cfg = cfg
        self.pw = np.random.randn(sd, ad)*0.01
        self.vw = np.random.randn(sd, 1)*0.01
        self.opw = self.pw.copy()
        self.lr = cfg.ppo_learning_rate
        self.ce = cfg.ppo_clip_epsilon
        self.ga = cfg.ppo_gamma; self.gl = cfg.ppo_gae_lambda

    def get_action(self, s, bf=0.0, ew=None):
        if s.ndim>1: s = s.flatten()
        s = np.pad(s, (0, max(0, self.sd-len(s))))[:self.sd]
        lg = s@self.pw
        if ew is not None and bf>0:
            ew = np.pad(ew, (0, max(0, len(lg)-len(ew))))[:len(lg)]
            lg = lg - bf*ew
        p = self._sm(lg); p = np.nan_to_num(p, nan=1.0/self.ad)
        p = p/max(p.sum(),1e-10)
        a = np.random.choice(self.ad, p=p); return a, p[a]

    def update(self, S, A, R, NS, D):
        if S.ndim==1: S = S.reshape(1,-1)
        if NS.ndim==1: NS = NS.reshape(1,-1)
        if S.shape[1] < self.sd:
            pad = ((0,0),(0,self.sd-S.shape[1]))
            S = np.pad(S,pad); NS = np.pad(NS,pad)
        else: S = S[:,:self.sd]; NS = NS[:,:self.sd]
        aoh = np.zeros((len(A),self.ad)); aoh[np.arange(len(A)),A] = 1
        v = S@self.vw; nv = NS@self.vw
        dl = R + self.ga*(1-D)*nv.flatten() - v.flatten()
        dl = np.nan_to_num(dl, nan=0.0)
        adv = np.zeros_like(dl); a = 0
        for t in reversed(range(len(dl))):
            a = dl[t] + self.ga*self.gl*(1-D[t])*a; adv[t] = a
        if len(adv)>1 and np.std(adv)>1e-8:
            adv = (adv-adv.mean())/adv.std()
        op = self._sm(S@self.opw); oa = op[np.arange(len(A)),A]+1e-8
        npp = self._sm(S@self.pw); na = npp[np.arange(len(A)),A]+1e-8
        ratio = np.nan_to_num(na/oa, nan=1.0, posinf=1+self.ce,
                              neginf=1-self.ce)
        cl = np.clip(ratio, 1-self.ce, 1+self.ce)
        obj = np.minimum(ratio*adv, cl*adv)
        g = S.T @ ((obj[:,None]*(aoh-npp))/npp)
        self.pw += self.lr*np.nan_to_num(g, nan=0.0)
        td = R + self.ga*(1-D)*(NS@self.vw).flatten() - v.flatten()
        self.vw += self.lr*S.T@np.nan_to_num(td, nan=0.0)[:,None]
        self.opw = self.pw.copy()
        return float(np.mean(obj)) if len(obj) else 0.0

    def _sm(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e/np.maximum(np.sum(e, axis=-1, keepdims=True), 1e-10)

# ============================================================================
# SHARED MOEA UTILITIES
# ============================================================================
def _dominates(a, b):
    if a is None or b is None: return False
    try: iter(a); iter(b)
    except TypeError: return False
    one = False
    for ai, bi in zip(a, b):
        if ai is None or bi is None: return False
        if ai > bi: return False
        if ai < bi: one = True
    return one

def non_dominated_sort(fit):
    n = len(fit)
    if n == 0: return []
    valid = [i for i in range(n)
             if fit[i] is not None and hasattr(fit[i], '__iter__')]
    if not valid: return [[i for i in range(n)]]
    idx_map = {o:k for k,o in enumerate(valid)}
    n_v = len(valid)
    dom = [set() for _ in range(n_v)]; dc = [0]*n_v
    for k1 in range(n_v):
        i = valid[k1]
        for k2 in range(k1+1, n_v):
            j = valid[k2]
            if _dominates(fit[i], fit[j]): dom[k1].add(k2); dc[k2] += 1
            elif _dominates(fit[j], fit[i]): dom[k2].add(k1); dc[k1] += 1
    front = [k for k in range(n_v) if dc[k] == 0]
    fronts = [[valid[k] for k in front]]
    while front:
        nxt = []
        for k in front:
            for k2 in dom[k]:
                dc[k2] -= 1
                if dc[k2] == 0: nxt.append(k2)
        front = nxt
        if front: fronts.append([valid[k] for k in front])
    invalid = [i for i in range(n) if i not in idx_map]
    if invalid: fronts.append(invalid)
    return fronts

def crowding_distance(fit, idx):
    m = len(fit[0]); n = len(idx)
    if n == 0: return np.array([])
    if n <= 2: return np.full(n, np.inf)
    d = np.zeros(n)
    for k in range(m):
        vals = np.array([fit[i][k] for i in idx])
        order = np.argsort(vals)
        d[order[0]] = np.inf; d[order[-1]] = np.inf
        rng = vals[order[-1]]-vals[order[0]]
        if rng <= 1e-12: continue
        for i in range(1, n-1):
            d[order[i]] += (vals[order[i+1]]-vals[order[i-1]])/rng
    return d

def monte_carlo_hypervolume(objectives, n_samples=5000, seed=0, ref=None):
    F = np.array([o for o in objectives if o is not None], dtype=float)
    F = F[np.all(np.isfinite(F) & (np.abs(F) < 1e12), axis=1)]
    if len(F) == 0: return 0.0
    if ref is None:
        ref = F.max(0)*1.1
        ref = np.where(ref > 1e-9, ref, 1.0)
    else:
        ref = np.asarray(ref, dtype=float)
    lo = F.min(0); span = ref-lo
    if np.any(span <= 0): return 0.0
    rng = np.random.default_rng(seed)
    pts = rng.uniform(lo, ref, size=(n_samples, 3))
    dom = np.zeros(n_samples, dtype=bool)
    for s in F: dom |= np.all(pts >= s, axis=1)
    return float(dom.mean()*np.prod(span))

def global_reference_point(pareto_lists, feasible_only=True):
    all_pts = []
    for pf in pareto_lists:
        for p in pf:
            try:
                arr = np.asarray(p, dtype=float).flatten()
                if arr.shape[0] == 3 and np.all(np.isfinite(arr)) \
                        and np.all(np.abs(arr) < 1e6):
                    all_pts.append(arr)
            except Exception:
                continue
    if not all_pts:
        return np.array([3000.0, 6000.0, 200.0])
    A = np.array(all_pts)
    ref = A.max(0) * 1.1
    return np.where(ref > 1e-9, ref, 1.0)

# ============================================================================
# DRO-EMO
# ============================================================================
class DROEMO:
    name = 'dro_emo'
    def __init__(self, problem, config, run_id=0, seed=None):
        self.problem = problem; self.run_id = run_id
        if seed is not None: random.seed(seed); np.random.seed(seed)
        self.cfg = config
        self.pop_size = self.cfg.population_size_base + 20
        self.max_gen = self.cfg.generations_base
        self.n_var = 20; self.n_obj = 3
        self.ppo = PPOAgent(24, 19, self.cfg); self._pretrain_ppo()
        self.population = []; self.fitness = []
        self.history = {'hypervolume':[], 'diversity':[], 'generation':[],
                        'best_emissions':[], 'best_cost':[], 'best_time':[]}
        self.crossover_rate = 0.85; self.mutation_rate = 0.12
        self.mutation_strength = 0.1
        self.diversity_window = []; self.stagnation_counter = 0
        self.best_emissions_history = []
        self.convergence_history = []

    def _get_bias_factor(self, n):
        return max(0.1, min(0.5, 100.0/max(n,1)))

    def _pretrain_ppo(self):
        for _ in range(20):
            bs = 5
            s = np.random.randn(bs, 24); a = np.random.randint(0, 19, bs)
            r = np.random.randn(bs); ns = np.random.randn(bs, 24)
            self.ppo.update(s, a, r, ns, np.zeros(bs))

    def _encode(self, path, modes):
        s = np.zeros(24)
        for i,n in enumerate(path[:19]): s[i] = n/19.0
        s[19] = len(path)/20.0; s[20] = len(modes)/20.0
        s[21] = sum(1 for m in modes if m=='rail')/20.0
        s[22] = sum(1 for m in modes if m=='water')/20.0
        s[23] = sum(1 for m in modes if m=='road')/20.0
        return s

    def _ew(self, _): return np.ones(19)*0.1

    def _template_init(self, name):
        for t in self.problem.route_templates:
            if t['name'] == name:
                return self.problem.evaluate_route(t['path'], t['modes'])
        return self._random_init()

    def _emission_init(self):
        routes = self.problem.get_all_routes()
        if routes:
            srt = sorted(routes, key=lambda x: x.objectives[1])
            k = max(3, int(len(srt)*0.25))
            return random.choice(srt[:k]).copy()
        return self._random_init()

    def _cost_init(self):
        routes = self.problem.get_all_routes()
        if routes:
            srt = sorted(routes, key=lambda x: x.objectives[0])
            k = max(3, int(len(srt)*0.20))
            return random.choice(srt[:k]).copy()
        return self._random_init()

    def _balanced_init(self):
        routes = self.problem.get_all_routes()
        if routes:
            for r in routes: r.compute_carbon_efficiency()
            srt = sorted(routes, key=lambda x: x.carbon_efficiency)
            k = max(3, int(len(srt)*0.20))
            return random.choice(srt[:k]).copy()
        return self._random_init()

    def _opposite_init(self):
        routes = self.problem.get_all_routes()
        if routes:
            srt = sorted(routes, key=lambda x: x.objectives[1], reverse=True)
            k = max(2, int(len(srt)*0.10))
            return random.choice(srt[:k]).copy()
        return self._random_init()

    def _ppo_init(self):
        path = [ORIGIN_NODE]; current = ORIGIN_NODE; visited = {ORIGIN_NODE}
        modes = []; state = np.zeros(24)
        bf = self._get_bias_factor(20)
        guard = 0
        while current != DESTINATION_NODE and len(path) < 15 and guard < 30:
            guard += 1
            neigh = [n for n in self.problem.dataset.adjacency.get(current, [])
                     if n not in visited]
            if not neigh: break
            a, _ = self.ppo.get_action(state, bf, self._ew(state))
            nxt = neigh[a % len(neigh)]
            av = self.problem.dataset.get_available_modes(current, nxt)
            mode = random.choice(av) if av else 'road'
            modes.append(mode); path.append(nxt); visited.add(nxt); current = nxt
            state = self._encode(path, modes)
        if path[-1] != DESTINATION_NODE:
            return self._template_init('Central Corridor (Kigoma)')
        return self.problem.evaluate_route(path, modes)

    def _random_init(self):
        path = [ORIGIN_NODE]; current = ORIGIN_NODE; visited = {ORIGIN_NODE}
        modes = []; guard = 0
        while current != DESTINATION_NODE and len(path) < 15 and guard < 50:
            guard += 1
            neigh = [n for n in self.problem.dataset.adjacency.get(current, [])
                     if n not in visited]
            if not neigh: break
            nxt = random.choice(neigh)
            av = self.problem.dataset.get_available_modes(current, nxt)
            modes.append(random.choice(av) if av else 'road')
            path.append(nxt); visited.add(nxt); current = nxt
        if path[-1] != DESTINATION_NODE:
            return self._template_init('Central Corridor (Kigoma)')
        return self.problem.evaluate_route(path, modes)

    def initialize_population(self):
        pop = []; c = self.cfg
        n_ppo = int(self.pop_size*c.ppo_proportion)
        n_em = int(self.pop_size*c.emission_proportion)
        n_cost = int(self.pop_size*c.cost_proportion)
        n_bal = int(self.pop_size*c.balanced_proportion)
        n_op = int(self.pop_size*c.opposite_proportion)
        n_rand = self.pop_size - n_ppo - n_em - n_cost - n_bal - n_op
        for _ in range(n_ppo): pop.append(self._ppo_init())
        for _ in range(n_em): pop.append(self._emission_init())
        for _ in range(n_cost): pop.append(self._cost_init())
        for _ in range(n_bal): pop.append(self._balanced_init())
        for _ in range(n_op): pop.append(self._opposite_init())
        for _ in range(n_rand): pop.append(self._random_init())
        valid = [i for i in pop if i.objectives is not None]
        if not valid:
            valid = [self._template_init('Central Corridor (Kigoma)')]
        random.shuffle(valid)
        return valid[:self.pop_size]

    def _div(self, pop):
        if not pop: return 0.0
        return len(set(tuple(p.route) for p in pop))/len(pop)

    def _adapt(self, div, gen):
        self.diversity_window.append(div)
        if len(self.diversity_window) > 10: self.diversity_window.pop(0)
        if len(self.best_emissions_history) > 10:
            r = min(self.best_emissions_history[-10:])
            o = min(self.best_emissions_history[:-10])
            imp = (o-r)/o if o>0 else 0
            if imp < 0.01: self.stagnation_counter += 1
            else: self.stagnation_counter = max(0, self.stagnation_counter-1)
        c = self.cfg
        if self.stagnation_counter > c.stagnation_threshold:
            self.mutation_rate = c.mutation_max
            self.crossover_rate = c.crossover_min
            self.mutation_strength = 0.15
        elif div < c.diversity_low_threshold:
            self.mutation_rate = min(c.mutation_max, self.mutation_rate+0.02)
            self.crossover_rate = max(c.crossover_min, self.crossover_rate-0.02)
            self.mutation_strength = 0.12
        elif div > c.diversity_high_threshold:
            self.mutation_rate = max(c.mutation_min, self.mutation_rate-0.01)
            self.crossover_rate = min(c.crossover_max, self.crossover_rate+0.01)
            self.mutation_strength = 0.08
        else:
            self.mutation_rate = 0.12; self.crossover_rate = 0.85
            self.mutation_strength = 0.10

    def _cross(self, p1, p2):
        if random.random() > self.crossover_rate: return p1.copy(), p2.copy()
        cut1 = random.randint(1, min(len(p1.route)-1, len(p2.route)-1))
        cut2 = random.randint(1, min(len(p1.route)-1, len(p2.route)-1))
        r1 = p1.route[:cut1] + p2.route[cut2:]
        r2 = p2.route[:cut2] + p1.route[cut1:]
        if r1[0] != ORIGIN_NODE: r1 = [ORIGIN_NODE]+r1
        if r1[-1] != DESTINATION_NODE: r1.append(DESTINATION_NODE)
        if r2[0] != ORIGIN_NODE: r2 = [ORIGIN_NODE]+r2
        if r2[-1] != DESTINATION_NODE: r2.append(DESTINATION_NODE)
        m1 = [random.choice(self.problem.dataset.get_available_modes(
                  r1[i], r1[i+1]) or ['road']) for i in range(len(r1)-1)]
        m2 = [random.choice(self.problem.dataset.get_available_modes(
                  r2[i], r2[i+1]) or ['road']) for i in range(len(r2)-1)]
        return self.problem.evaluate_route(r1,m1), \
               self.problem.evaluate_route(r2,m2)

    def _mut(self, ind):
        if random.random() > self.mutation_rate: return ind.copy()
        m = ind.copy()
        if len(m.route) > 2:
            if random.random() < 0.5:
                idx = random.randint(1, len(m.route)-2)
                nb = self.problem.dataset.adjacency.get(m.route[idx], [])
                if nb: m.route[idx] = random.choice(nb)
            else:
                idx = random.randint(0, len(m.modes)-1)
                av = self.problem.dataset.get_available_modes(
                    m.route[idx], m.route[idx+1])
                if av: m.modes[idx] = random.choice(av)
        return self.problem.evaluate_route(m.route, m.modes)

    def _ls(self, ind):
        imp = ind.copy()
        best_em = imp.objectives[1] if imp.objectives else 1e9
        for _ in range(10):
            if len(imp.route) < 4: break
            i = random.randint(1, len(imp.route)-3)
            j = random.randint(i+1, len(imp.route)-2)
            nb = imp.copy(); nb.route[i:j+1] = nb.route[i:j+1][::-1]
            nb = self.problem.evaluate_route(nb.route, nb.modes)
            if nb.objectives and nb.objectives[1] < best_em*0.98:
                imp = nb; best_em = nb.objectives[1]
        return imp

    def _tour(self, ranks, crowd):
        i = random.randint(0, len(self.population)-1)
        j = random.randint(0, len(self.population)-1)
        if ranks[i] < ranks[j]: return i
        if ranks[i] > ranks[j]: return j
        return i if crowd[i] > crowd[j] else j

    def _hv(self, pop, ref=None):
        return monte_carlo_hypervolume(
            [i.objectives for i in pop if i.objectives], ref=ref)

    def run(self):
        self.population = self.initialize_population()
        self.population = [i for i in self.population
                           if i.objectives is not None]
        if not self.population:
            self.population = [self._template_init(
                'Central Corridor (Kigoma)')]
        self.fitness = [i.objectives for i in self.population]
        for gen in range(self.max_gen):
            fronts = non_dominated_sort(self.fitness)
            ranks = [-1]*len(self.population)
            crowd = [0.0]*len(self.population)
            for i, f in enumerate(fronts):
                d = crowding_distance(self.fitness, f)
                for j, idx in enumerate(f):
                    ranks[idx] = i; crowd[idx] = d[j]
            div = self._div(self.population); self._adapt(div, gen)
            offspring = []
            for _ in range(self.pop_size//2):
                p1 = self._tour(ranks, crowd); p2 = self._tour(ranks, crowd)
                c1, c2 = self._cross(self.population[p1],
                                     self.population[p2])
                c1 = self._mut(c1); c2 = self._mut(c2)
                if random.random() < 0.15: c1 = self._ls(c1)
                if random.random() < 0.15: c2 = self._ls(c2)
                if c1.objectives is not None: offspring.append(c1)
                if c2.objectives is not None: offspring.append(c2)
            off_fit = [i.objectives for i in offspring
                       if i.objectives is not None]
            comb = self.population + offspring
            comb_fit = self.fitness + off_fit
            cf = non_dominated_sort(comb_fit)
            new_pop, new_fit = [], []
            for f in cf:
                if len(new_pop) + len(f) <= self.pop_size:
                    for i in f:
                        new_pop.append(comb[i]); new_fit.append(comb_fit[i])
                else:
                    d = crowding_distance(comb_fit, f)
                    order = sorted([(i, d[j]) for j,i in enumerate(f)],
                                   key=lambda x: -x[1])
                    need = self.pop_size - len(new_pop)
                    for i, _ in order[:need]:
                        new_pop.append(comb[i]); new_fit.append(comb_fit[i])
                    break
            self.population = new_pop; self.fitness = new_fit
            hv = self._hv(self.population)
            be = min(f[1] for f in self.fitness) if self.fitness else 0
            bc = min(f[0] for f in self.fitness) if self.fitness else 0
            bt = min(f[2] for f in self.fitness) if self.fitness else 0
            self.best_emissions_history.append(be)
            self.convergence_history.append(be)
            self.history['hypervolume'].append(hv)
            self.history['diversity'].append(div)
            self.history['generation'].append(gen)
            self.history['best_emissions'].append(be)
            self.history['best_cost'].append(bc)
            self.history['best_time'].append(bt)
        pareto = [ind for i, ind in enumerate(self.population)
                  if not any(i != j and _dominates(self.fitness[j],
                                                    self.fitness[i])
                             for j in range(len(self.population)))]
        self.pareto_fitness = [ind.objectives for ind in pareto
                               if ind.objectives is not None]
        det_bc = min((i.det_objectives[0] for i in pareto
                      if i.det_objectives), default=float('inf'))
        det_be = min((i.det_objectives[1] for i in pareto
                      if i.det_objectives), default=float('inf'))
        det_bt = min((i.det_objectives[2] for i in pareto
                      if i.det_objectives), default=float('inf'))
        return {
            'hypervolume': self.history['hypervolume'][-1]
                           if self.history['hypervolume'] else 0.0,
            'best_cost': bc, 'best_emissions': be, 'best_time': bt,
            'pareto_size': len(pareto),
            'valid_solutions': sum(1 for i in pareto if i.is_valid),
            'diversity': self.history['diversity'][-1]
                         if self.history['diversity'] else 0.0,
            'history': self.history, 'pareto_front': pareto,
            'pareto_fitness': self.pareto_fitness,
            'convergence': self.convergence_history,
            'det_best_cost': det_bc, 'det_best_emissions': det_be,
            'det_best_time': det_bt,
        }

# ============================================================================
# COMPETITORS — TIER 1 SET
# ============================================================================
class BaseCompetitor:
    def __init__(self, problem, config, name, run_id=0, seed=None):
        self.problem = problem; self.config = config; self.name = name
        self.run_id = run_id; self.seed = seed
        if seed is not None: random.seed(seed); np.random.seed(seed)
        self.population_size = 60; self.max_generations = 80
        self.crossover_rate = 0.85; self.mutation_rate = 0.12
        self.history = {'hv':[], 'best_emissions':[], 'best_cost':[],
                        'diversity':[]}
        self.pareto_fitness = []; self.convergence_history = []
        self.population = []; self.fitness = []
        self.last_pareto_fitness = []

    def _seed_paths(self):
        return [(t['path'], t['modes'])
                for t in self.problem.route_templates]

    def _eval(self, path, modes):
        return self.problem.evaluate_route(path, modes)

    def _div(self, pop):
        if not pop: return 0.0
        return len(set(tuple(p.route) for p in pop))/len(pop)

    def _cross(self, p1, p2):
        if random.random() > self.crossover_rate: return p1.copy(), p2.copy()
        c1 = random.randint(1, min(len(p1.route)-1, len(p2.route)-1))
        c2 = random.randint(1, min(len(p1.route)-1, len(p2.route)-1))
        r1 = p1.route[:c1]+p2.route[c2:]; r2 = p2.route[:c2]+p1.route[c1:]
        if r1[0] != ORIGIN_NODE: r1 = [ORIGIN_NODE]+r1
        if r1[-1] != DESTINATION_NODE: r1.append(DESTINATION_NODE)
        if r2[0] != ORIGIN_NODE: r2 = [ORIGIN_NODE]+r2
        if r2[-1] != DESTINATION_NODE: r2.append(DESTINATION_NODE)
        m1 = [random.choice(self.problem.dataset.get_available_modes(
                  r1[i],r1[i+1]) or ['road']) for i in range(len(r1)-1)]
        m2 = [random.choice(self.problem.dataset.get_available_modes(
                  r2[i],r2[i+1]) or ['road']) for i in range(len(r2)-1)]
        return self._eval(r1,m1), self._eval(r2,m2)

    def _mut(self, ind):
        if random.random() > self.mutation_rate: return ind.copy()
        m = ind.copy()
        if len(m.route) > 2:
            if random.random() < 0.5:
                idx = random.randint(1, len(m.route)-2)
                nb = self.problem.dataset.adjacency.get(m.route[idx], [])
                if nb: m.route[idx] = random.choice(nb)
            else:
                idx = random.randint(0, len(m.modes)-1)
                av = self.problem.dataset.get_available_modes(
                    m.route[idx], m.route[idx+1])
                if av: m.modes[idx] = random.choice(av)
        return self._eval(m.route, m.modes)

    def _ls(self, ind):
        imp = ind.copy()
        best_em = imp.objectives[1] if imp.objectives else 1e9
        for _ in range(8):
            if len(imp.route) < 4: break
            i = random.randint(1, len(imp.route)-3)
            j = random.randint(i+1, len(imp.route)-2)
            nb = imp.copy(); nb.route[i:j+1] = nb.route[i:j+1][::-1]
            nb = self._eval(nb.route, nb.modes)
            if nb.objectives and nb.objectives[1] < best_em*0.98:
                imp = nb; best_em = nb.objectives[1]
        return imp

    def _hv(self, pop, ref=None):
        return monte_carlo_hypervolume(
            [i.objectives for i in pop if i.objectives], ref=ref)

    def _tour(self, ranks, crowd):
        i = random.randint(0, len(self.population)-1)
        j = random.randint(0, len(self.population)-1)
        if ranks[i] < ranks[j]: return i
        if ranks[i] > ranks[j]: return j
        return i if crowd[i] > crowd[j] else j

    def _generate_offspring(self, ranks, crowd):
        offspring = []
        for _ in range(self.population_size//2):
            p1 = self._tour(ranks, crowd); p2 = self._tour(ranks, crowd)
            c1, c2 = self._cross(self.population[p1], self.population[p2])
            c1 = self._mut(c1); c2 = self._mut(c2)
            if c1.objectives is not None: offspring.append(c1)
            if c2.objectives is not None: offspring.append(c2)
        return offspring

    def _initialise(self):
        seeds = self._seed_paths(); self.population = []
        for p, m in seeds:
            ind = self._eval(p, m)
            if ind.objectives is not None: self.population.append(ind)
        while len(self.population) < self.population_size:
            p, m = random.choice(seeds)
            ind = self._eval(p, m)
            if ind.objectives is not None: self.population.append(ind)
        self.population = self.population[:self.population_size]
        self.fitness = [i.objectives for i in self.population]

    def _finalise(self):
        pareto = [ind for i, ind in enumerate(self.population)
                  if not any(i != j and _dominates(self.fitness[j],
                                                    self.fitness[i])
                             for j in range(len(self.population)))]
        self.pareto_fitness = [ind.objectives for ind in pareto
                               if ind.objectives is not None]
        self.last_pareto_fitness = [list(p) for p in self.pareto_fitness]
        det_bc = min((i.det_objectives[0] for i in pareto
                      if i.det_objectives), default=float('inf'))
        det_be = min((i.det_objectives[1] for i in pareto
                      if i.det_objectives), default=float('inf'))
        det_bt = min((i.det_objectives[2] for i in pareto
                      if i.det_objectives), default=float('inf'))
        return {
            'hypervolume': self.history['hv'][-1]
                           if self.history['hv'] else 0.0,
            'best_cost': min(f[0] for f in self.fitness)
                         if self.fitness else float('inf'),
            'best_emissions': min(f[1] for f in self.fitness)
                              if self.fitness else float('inf'),
            'best_time': min(f[2] for f in self.fitness)
                         if self.fitness else float('inf'),
            'pareto_size': len(pareto),
            'valid_solutions': sum(1 for i in pareto if i.is_valid),
            'diversity': self.history['diversity'][-1]
                         if self.history['diversity'] else 0.0,
            'history': self.history, 'pareto_front': pareto,
            'pareto_fitness': self.pareto_fitness,
            'convergence': self.convergence_history,
            'det_best_cost': det_bc, 'det_best_emissions': det_be,
            'det_best_time': det_bt,
        }

    def _step(self):
        fronts = non_dominated_sort(self.fitness)
        ranks = [-1]*len(self.population)
        crowd = [0.0]*len(self.population)
        for i, f in enumerate(fronts):
            d = crowding_distance(self.fitness, f)
            for j, idx in enumerate(f):
                ranks[idx] = i; crowd[idx] = d[j]
        div = self._div(self.population)
        offspring = self._generate_offspring(ranks, crowd)
        off_fit = [i.objectives for i in offspring
                   if i.objectives is not None]
        comb = self.population + offspring
        comb_fit = self.fitness + off_fit
        cf = non_dominated_sort(comb_fit)
        new_pop, new_fit = [], []
        for f in cf:
            if len(new_pop) + len(f) <= self.population_size:
                for i in f:
                    new_pop.append(comb[i]); new_fit.append(comb_fit[i])
            else:
                d = crowding_distance(comb_fit, f)
                order = sorted([(i, d[j]) for j,i in enumerate(f)],
                               key=lambda x: -x[1])
                need = self.population_size - len(new_pop)
                for i, _ in order[:need]:
                    new_pop.append(comb[i]); new_fit.append(comb_fit[i])
                break
        self.population = new_pop; self.fitness = new_fit
        hv = self._hv(self.population)
        be = min(f[1] for f in self.fitness) if self.fitness else 0
        bc = min(f[0] for f in self.fitness) if self.fitness else 0
        self.history['hv'].append(hv)
        self.history['best_emissions'].append(be)
        self.history['best_cost'].append(bc)
        self.history['diversity'].append(div)
        self.convergence_history.append(be)

    def run(self):
        self._initialise()
        for _ in range(self.max_generations):
            self._step()
        return self._finalise()


class NSGAII(BaseCompetitor):
    name = 'nsga2'
    def __init__(self, problem, config, run_id=0, seed=None):
        super().__init__(problem, config, "NSGA-II", run_id, seed)
        self.eta_c = 20.0; self.eta_m = 20.0
        self.crossover_rate = 0.9
        self.mutation_rate = 1.0 / max(
            problem.n_var if hasattr(problem, 'n_var') else 20, 1)

    def _cross(self, p1, p2):
        # Length-safe mode exchange: iterate only over the length that
        # both parents share. Routes on the corridor have 5 to 9 legs,
        # so parents can differ in length.
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        c1 = p1.copy(); c2 = p2.copy()
        n = min(len(c1.modes), len(c2.modes))
        for i in range(n):
            if random.random() <= 0.5:
                c1.modes[i], c2.modes[i] = c2.modes[i], c1.modes[i]
        return (self._eval(c1.route, c1.modes),
                self._eval(c2.route, c2.modes))


class NSGAIII(NSGAII):
    name = 'nsga3'
    def __init__(self, problem, config, run_id=0, seed=None):
        super().__init__(problem, config, run_id, seed)
        self.name = "NSGA-III"
        self.ref_dirs = self._das_dennis(3, 12)
    @staticmethod
    def _das_dennis(m, p):
        from itertools import combinations_with_replacement
        dirs = []
        for c in combinations_with_replacement(range(p + 1), m):
            if sum(c) == p: dirs.append([x / p for x in c])
        return np.array(dirs)
    def _associate(self, F, ideal):
        F = np.asarray(F, dtype=float); Ft = F - ideal
        fm = np.abs(Ft).max(axis=0); fm[fm < 1e-12] = 1.0
        Fn = Ft / fm
        R = self.ref_dirs / np.maximum(
            np.linalg.norm(self.ref_dirs, axis=1, keepdims=True), 1e-12)
        D = np.zeros((len(Fn), len(R)))
        for i, f in enumerate(Fn):
            proj = f @ R.T
            closest = proj[:, None] * R
            D[i] = np.linalg.norm(f[None, :] - closest, axis=1)
        return np.argmin(D, axis=1)


class SPEA2(BaseCompetitor):
    name = 'spea2'
    def __init__(self, problem, config, run_id=0, seed=None):
        super().__init__(problem, config, "SPEA2", run_id, seed)
        self.archive_size = self.population_size
    def _strength_fitness(self, fit):
        n = len(fit)
        if n == 0: return np.zeros(0)
        F = np.array(fit, dtype=float)
        le = (F[None, :, :] >= F[:, None, :]).all(axis=-1)
        sl = (F[None, :, :] > F[:, None, :]).any(axis=-1)
        dom = le & sl; np.fill_diagonal(dom, False)
        strength = dom.sum(axis=0)
        raw = (dom * strength[None, :]).sum(axis=1)
        Fmin, Fmax = F.min(axis=0), F.max(axis=0)
        rng = Fmax - Fmin; rng[rng < 1e-12] = 1.0
        Fn = (F - Fmin) / rng
        D = distance.cdist(Fn, Fn); np.fill_diagonal(D, np.inf)
        k = int(np.sqrt(n))
        kth = np.sort(D, axis=1)[:, min(k, n - 2)]
        density = 1.0 / (kth + 2.0)
        return raw + density


class MOEAD(BaseCompetitor):
    name = 'moead'
    def __init__(self, problem, config, run_id=0, seed=None):
        super().__init__(problem, config, "MOEA/D", run_id, seed)
        self.weights = self._uniform_weights(3, 12)
        self.T = min(15, len(self.weights))
        D = distance.cdist(self.weights, self.weights)
        self.neighbors = np.argsort(D, axis=1)[:, :self.T]
        self.nr = 2
    @staticmethod
    def _uniform_weights(m, H):
        from itertools import combinations_with_replacement
        w = []
        for c in combinations_with_replacement(range(H + 1), m):
            if sum(c) == H: w.append([x / H for x in c])
        return np.array(w)
    def _tchebycheff(self, f, w, z):
        return float(np.max(w * np.abs(np.asarray(f) - z)))
def _generate_offspring(self, ranks, crowd):
    # MOEA/D's neighbors index into the weight-vector set, which is
    # larger than the population whenever the population has collapsed
    # (the corridor can produce fewer than 60 distinct route/mode pairs).
    # Clamp the neighbor indices to the population size so the parent
    # lookup cannot go out of bounds.
    offspring = []
    n_pop = len(self.population)
    if n_pop < 2:
        return offspring
    for _ in range(self.population_size // 2):
        i = random.randint(0, n_pop - 1)
        neigh = self.neighbors[i % len(self.neighbors)]
        k, l = np.random.choice(neigh, 2, replace=False)
        k = int(k) % n_pop
        l = int(l) % n_pop
        c1, c2 = self._cross(self.population[k], self.population[l])
        c1 = self._mut(c1); c2 = self._mut(c2)
        if c1.objectives is not None:
            offspring.append(c1)
        if c2.objectives is not None:
            offspring.append(c2)
    return offspring
    def _step(self):
        z = np.min(np.array(self.fitness, dtype=float), axis=0)
        fronts = non_dominated_sort(self.fitness)
        ranks = [-1] * len(self.population)
        crowd = [0.0] * len(self.population)
        for i, f in enumerate(fronts):
            d = crowding_distance(self.fitness, f)
            for j, idx in enumerate(f):
                ranks[idx] = i; crowd[idx] = d[j]
        offspring = self._generate_offspring(ranks, crowd)
        off_fit = [i.objectives for i in offspring
                   if i.objectives is not None]
        comb = self.population + offspring
        comb_fit = self.fitness + off_fit
        for idx, ind in enumerate(comb):
            if ind.objectives is None: continue
            z = np.minimum(z, np.asarray(ind.objectives))
        cf = non_dominated_sort(comb_fit)
        new_pop, new_fit = [], []
        for f in cf:
            if len(new_pop) + len(f) <= self.population_size:
                for i in f:
                    new_pop.append(comb[i]); new_fit.append(comb_fit[i])
            else:
                d = crowding_distance(comb_fit, f)
                order = sorted([(i, d[j]) for j, i in enumerate(f)],
                               key=lambda x: -x[1])
                need = self.population_size - len(new_pop)
                for i, _ in order[:need]:
                    new_pop.append(comb[i]); new_fit.append(comb_fit[i])
                break
        self.population = new_pop; self.fitness = new_fit
        hv = self._hv(self.population)
        be = min(f[1] for f in self.fitness) if self.fitness else 0
        bc = min(f[0] for f in self.fitness) if self.fitness else 0
        self.history['hv'].append(hv)
        self.history['best_emissions'].append(be)
        self.history['best_cost'].append(bc)
        self.history['diversity'].append(self._div(self.population))
        self.convergence_history.append(be)


class RLNSGAII(NSGAII):
    name = 'rl_nsgaii'
    def __init__(self, problem, config, run_id=0, seed=None):
        super().__init__(problem, config, run_id, seed)
        self.name = "RL-NSGAII"
        self.Q = np.zeros((5, 5))
        self.params = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
                       (0.6, 0.4), (0.5, 0.5)]
        self.alpha_q = 0.1; self.gamma_q = 0.9
    def _state(self, pop):
        u = len(set(tuple(p.route) for p in pop))
        return float(np.clip(u / max(1, len(pop)), 0, 1))
    def _step(self):
        s = int(np.clip(self._state(self.population) * 5, 0, 4))
        if random.random() < 0.1:
            a = random.randint(0, len(self.params) - 1)
        else:
            a = int(np.argmax(self.Q[s]))
        self.crossover_rate, self.mutation_rate = self.params[a]
        hv_before = self._hv(self.population)
        super()._step()
        hv_after = self.history['hv'][-1]
        reward = hv_after - hv_before
        ns = int(np.clip(self._state(self.population) * 5, 0, 4))
        self.Q[s, a] += self.alpha_q * (
            reward + self.gamma_q * np.max(self.Q[ns]) - self.Q[s, a])
# ============================================================================
# ALTERNATIVE-PATH EVALUATOR
# ============================================================================
class AlternativePathEvaluator:
    TEMPLATES = FreightRoutingProblem.TEMPLATES
    def __init__(self, dataset, season='Dry', demand_cv=None):
        self.dataset = dataset; self.season = season
        self.problem = FreightRoutingProblem(
            dataset, season=season,
            carbon_price=dataset.global_params['carbon_price']*1000,
            demand_cv=demand_cv)
    def evaluate_all(self):
        rows = []
        for t in self.TEMPLATES:
            ind = self.problem.evaluate_route(t['path'], t['modes'])
            det = ind.det_objectives if ind.det_objectives else (float('nan'),)*3
            rows.append({
                'Path_Name': t['name'],
                'Path_Nodes': '-'.join(str(n) for n in t['path']),
                'Modes': '-'.join(t['modes']),
                'Season': self.season,
                'DRO_Cost_USD': ind.objectives[0] if ind.objectives else float('nan'),
                'DRO_CO2_kg': ind.objectives[1] if ind.objectives else float('nan'),
                'DRO_Time_h': ind.objectives[2] if ind.objectives else float('nan'),
                'Det_Cost_USD': det[0], 'Det_CO2_kg': det[1], 'Det_Time_h': det[2],
                'Rail_Ratio': ind.rail_ratio, 'Water_Ratio': ind.water_ratio,
                'Road_Ratio': ind.road_ratio, 'Mode_Switches': ind.mode_switches,
                'Border_Crossings': ind.border_crossings,
                'Transshipments': ind.transshipment_count,
                'Reliability': ind.reliability, 'Is_Feasible': ind.is_valid,
                'Violation': ind.violation_reason if not ind.is_valid else '',
            })
        return rows

# ============================================================================
# CASE STUDY RUNNER
# ============================================================================
class DeterministicSeeder:
    def __init__(self, base_seed=42): self.base_seed = base_seed
    def get_seed(self, scenario, algorithm, run_id):
        return int(hashlib.md5(
            f"{scenario}_{algorithm}_{run_id}_{self.base_seed}"
            .encode()).hexdigest()[:8], 16)

class CaseStudyRunner:
    def __init__(self, config=None):
        self.config = config or ExperimentConfig()
        self.results = []; self.all_pareto_fronts = {}
        self.best_solutions = {}
        self.alternative_paths = []
        self.sensitivity_results = {'carbon_price':[],
                                    'carbon_price_volatility':[],
                                    'season':[], 'border_delay':[],
                                    'transshipment':[],
                                    'demand_uncertainty':[]}
        self.regulation_results = []
        self.pareto_by_scenario = {}; self._detailed_route_rows = []
        self.seeder = DeterministicSeeder(self.config.base_seed)
        self.global_hv_ref = None
        self.evaluation_mode_results = []
        self.delta_min_results = []
        np.random.seed(self.config.base_seed)
        random.seed(self.config.base_seed)

    def get_algorithm(self, name, problem, run_id, seed):
        R = {'dro_emo': DROEMO,
             'nsga2':    NSGAII,
             'nsga3':    NSGAIII,
             'spea2':    SPEA2,
             'moead':    MOEAD,
             'rl_nsgaii': RLNSGAII}
        cls = R.get(name.lower())
        if cls is None: raise ValueError(name)
        if name.lower() == 'dro_emo':
            return cls(problem, self.config.dro_emo, run_id, seed)
        return cls(problem, self.config, run_id, seed)

    def _save_csv(self, rows, path, label):
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  [ok] Saved: {path.name} ({len(rows)} rows - {label})")

    def build_global_reference(self, scenarios, algorithms, num_runs):
        print("\n" + "=" * 80)
        print(" REFERENCE-POINT CONSTRUCTION (pass 1) ".center(80, "="))
        print("=" * 80)
        baseline_scen = next((s for s in scenarios
                              if s.get('name')=='baseline'), None)
        if baseline_scen is None:
            print("  [warn] no baseline scenario found; using fallback reference.")
            self.global_hv_ref = np.array([3000.0, 6000.0, 200.0])
            return self.global_hv_ref
        price = baseline_scen.get('carbon_price', CARBON_PRICE_DEFAULT)
        price_std = baseline_scen.get('carbon_price_std',
                                      CARBON_PRICE_STD_DEFAULT)
        demand_cv = baseline_scen.get('demand_cv', self.config.demand_cv)
        season = baseline_scen.get('season','Dry')
        bmult = baseline_scen.get('border_multiplier',1.0)
        tmult = baseline_scen.get('transshipment_multiplier',1.0)
        ds_b = LC_CTRP_Dataset(carbon_price=price,
                               carbon_price_std=price_std,
                               demand_cv=demand_cv,
                               stochastic_borders=self.config.stochastic_borders)
        ds_b.apply_carbon_price(price, std=price_std)
        ds_b.apply_demand_cv(demand_cv)
        ds_b.apply_border_delay_multiplier(bmult)
        ds_b.apply_transshipment_cost_multiplier(tmult)
        prob_b = FreightRoutingProblem(ds_b, season=season, carbon_price=price,
                                       demand_cv=demand_cv)
        all_baseline_pf = []
        n_alg = len(algorithms); total = n_alg * num_runs; done = 0
        for alg in algorithms:
            for run in range(num_runs):
                seed = self.seeder.get_seed('baseline', alg, run)
                algo = self.get_algorithm(alg, prob_b, run, seed)
                m = algo.run()
                all_baseline_pf.append(m.get('pareto_fitness', []))
                done += 1
                print(f"    ref-pass: {alg} run {run+1}/{num_runs} "
                      f"({done}/{total})", flush=True)
        self.global_hv_ref = global_reference_point(all_baseline_pf)
        assert self.global_hv_ref is not None, "global HV ref is None"
        assert self.global_hv_ref.shape == (3,), \
            f"global HV ref has shape {self.global_hv_ref.shape}, expected (3,)"
        assert np.all(np.isfinite(self.global_hv_ref)), \
            f"global HV ref is not finite: {self.global_hv_ref}"
        print(f"  global HV ref = {self.global_hv_ref.tolist()}", flush=True)
        print(f"  (reused for every HV evaluation below, including all "
              f"sensitivity axes)", flush=True)
        return self.global_hv_ref

    def run_alternative_paths(self, scenarios):
        print("\n" + "=" * 90)
        print(" Q1: ALTERNATIVE-PATH EVALUATION ".center(90, "="))
        print("=" * 90)
        for scen in scenarios:
            name = scen['name']
            price = scen.get('carbon_price', CARBON_PRICE_DEFAULT)
            price_std = scen.get('carbon_price_std',
                                 CARBON_PRICE_STD_DEFAULT)
            demand_cv = scen.get('demand_cv', self.config.demand_cv)
            season = scen.get('season', 'Dry')
            bmult = scen.get('border_multiplier', 1.0)
            tmult = scen.get('transshipment_multiplier', 1.0)
            ds = LC_CTRP_Dataset(carbon_price=price,
                                 carbon_price_std=price_std,
                                 demand_cv=demand_cv,
                                 stochastic_borders=self.config.stochastic_borders)
            ds.apply_carbon_price(price, std=price_std)
            ds.apply_demand_cv(demand_cv)
            ds.apply_border_delay_multiplier(bmult)
            ds.apply_transshipment_cost_multiplier(tmult)
            ev = AlternativePathEvaluator(ds, season=season,
                                          demand_cv=demand_cv)
            rows = ev.evaluate_all()
            for r in rows:
                r['Scenario'] = name
                r['border_multiplier'] = bmult
                r['transshipment_multiplier'] = tmult
                r['carbon_price'] = price
                r['carbon_price_std'] = price_std
                r['demand_cv'] = demand_cv
            self.alternative_paths.extend(rows)
            nf = sum(1 for r in rows if r['Is_Feasible'])
            print(f"  {name:<26} {len(rows)} paths ({nf} feasible)")

    def print_alternative_paths(self):
        if not self.alternative_paths: return
        print("\n" + "=" * 110)
        print(" ALTERNATIVE PATHS - DETAILED ROUTE COMBINATIONS ".center(110, "="))
        print("=" * 110)
        df = pd.DataFrame(self.alternative_paths)
        for scen in sorted(df['Scenario'].unique()):
            sub = df[df['Scenario']==scen].copy()
            print(f"\nScenario: {scen}")
            print("-" * 110)
            print(f"{'Path':<28} {'Route':<24} {'Cost':>10} {'CO2':>10} "
                  f"{'Time':>8} {'Rail%':>7} {'Reliab.':>8} {'Feas.':>6}")
            print("-" * 110)
            for _, row in sub.sort_values('Det_CO2_kg',
                                          na_position='last').iterrows():
                cost = (f"{row['Det_Cost_USD']:>10.1f}"
                        if pd.notna(row['Det_Cost_USD']) else f"{'nan':>10}")
                co2  = (f"{row['Det_CO2_kg']:>10.1f}"
                        if pd.notna(row['Det_CO2_kg']) else f"{'nan':>10}")
                time = (f"{row['Det_Time_h']:>8.1f}"
                        if pd.notna(row['Det_Time_h']) else f"{'nan':>8}")
                rail = (f"{row['Rail_Ratio']*100:>7.1f}"
                        if pd.notna(row['Rail_Ratio']) else f"{'nan':>7}")
                rel  = (f"{row['Reliability']:>8.3f}"
                        if pd.notna(row['Reliability']) else f"{'nan':>8}")
                feas = "yes" if row['Is_Feasible'] else "no"
                print(f"{row['Path_Name']:<28} {row['Path_Nodes']:<24} "
                      f"{cost} {co2} {time} {rail} {rel} {feas:>6}")
            feasible = sub[sub['Is_Feasible'] == True]
            if feasible.empty:
                print("\n  (no feasible path under the current caps for this scenario)")
                continue
            best = feasible.loc[feasible['Det_CO2_kg'].idxmin()]
            print(f"\n  Best CO2 path in this scenario: {best['Path_Name']}")
            nodes = [int(n) for n in best['Path_Nodes'].split('-')]
            modes = best['Modes'].split('-')
            print(f"  {'Segment':<14} {'Mode':<8} {'Dist(km)':>10} "
                  f"{'Cost(USD)':>12} {'CO2(kg)':>10} {'Time(h)':>10} "
                  f"{'Border t':>9} {'Border $':>9} {'Carb $':>8}")
            print("  " + "-" * 95)
            bm = float(best.get('border_multiplier', 1.0))
            tm = float(best.get('transshipment_multiplier', 1.0))
            cp = float(best.get('carbon_price', CARBON_PRICE_DEFAULT))
            cps = float(best.get('carbon_price_std', CARBON_PRICE_STD_DEFAULT))
            dcv = float(best.get('demand_cv', DEMAND_CV_DEFAULT))
            ds = LC_CTRP_Dataset(carbon_price=cp, carbon_price_std=cps,
                                 demand_cv=dcv,
                                 stochastic_borders=self.config.stochastic_borders)
            ds.apply_carbon_price(cp, std=cps)
            ds.apply_demand_cv(dcv)
            ds.apply_border_delay_multiplier(bm)
            ds.apply_transshipment_cost_multiplier(tm)
            for k in range(len(nodes)-1):
                n1, n2 = nodes[k], nodes[k+1]
                mode = modes[k] if k < len(modes) else 'road'
                arc = ds.get_arc_data(n1, n2, mode)
                if arc is None:
                    print(f"  {n1:>3}->{n2:<3} {mode:<8} (no data)")
                    continue
                d, c, t = arc
                em = d*ds.mode_params[mode]['emission_factor'] \
                     * ds.global_params['container_weight']
                is_border = ds.is_border_crossing(n1, n2)
                if is_border:
                    key = (ds.get_country(n1), ds.get_country(n2))
                    d_border = ds.border_crossings.get(key, {})
                    bt_lo, bt_hi = d_border.get('time_penalty', (0.0, 0.0))
                    border_t = 0.5*(bt_lo + bt_hi)
                    border_c = d_border.get('cost', 0.0)
                else:
                    border_t, border_c = 0.0, 0.0
                carb_add = em * ds.global_params['carbon_price']
                self._detailed_route_rows.append({
                    'Scenario': scen, 'Path_Name': best['Path_Name'],
                    'Segment': f"{n1}->{n2}",
                    'From_Name': ds.nodes.get(n1,{}).get('name','?'),
                    'To_Name': ds.nodes.get(n2,{}).get('name','?'),
                    'Mode': mode, 'Distance_km': d, 'Cost_USD': c,
                    'CO2_kg': em, 'Time_h': t,
                    'Is_Border': bool(is_border),
                    'Border_Time_h': border_t,
                    'Border_Cost_USD': border_c,
                    'Carbon_Add_USD': carb_add,
                })
                print(f"  {n1:>3}->{n2:<3} {mode:<8} {d:>10.1f} "
                      f"{c:>12.2f} {em:>10.2f} {t:>10.2f} "
                      f"{border_t:>9.1f} {border_c:>9.1f} {carb_add:>8.2f}")
        print("\n" + "=" * 110)

    def run_evaluation_mode_comparison(self, num_runs=30):
        print("\n" + "=" * 80)
        print(" EVALUATION-MODE COMPARISON (baseline) ".center(80, "="))
        print("=" * 80)
        print("  Deterministic vs scenario-based stochastic vs DRO evaluation,")
        print("  same seeds, same search, same feasible-template set.")

        rows = []
        for mode_name in ['deterministic', 'stochastic', 'dro']:
            print(f"\n  Mode: {mode_name}")
            ds = LC_CTRP_Dataset(carbon_price=CARBON_PRICE_DEFAULT,
                                 carbon_price_std=CARBON_PRICE_STD_DEFAULT,
                                 demand_cv=DEMAND_CV_DEFAULT,
                                 stochastic_borders=self.config.stochastic_borders)
            ds.apply_carbon_price(CARBON_PRICE_DEFAULT)
            ds.apply_demand_cv(DEMAND_CV_DEFAULT)
            if mode_name == 'deterministic':
                ev = DeterministicEvaluator(ds, season='Dry')
            elif mode_name == 'stochastic':
                ev = StochasticEvaluator(ds, season='Dry')
            else:
                ev = None
            problem = FreightRoutingProblem(
                ds, season='Dry', carbon_price=CARBON_PRICE_DEFAULT,
                demand_cv=DEMAND_CV_DEFAULT, evaluator=ev)
            hvs, feas_hvs = [], []; costs, ems, times = [], [], []
            for run in range(num_runs):
                seed = self.seeder.get_seed(f'evalmode_{mode_name}',
                                            'dro_emo', run)
                algo = DROEMO(problem, self.config.dro_emo, run, seed)
                m = algo.run()
                pf = m.get('pareto_fitness', [])
                pf_inds = m.get('pareto_front', [])
                hv_all = monte_carlo_hypervolume(pf, ref=self.global_hv_ref)
                pf_feas = [p for p, ind in zip(pf, pf_inds) if ind.is_valid]
                hv_feas = (monte_carlo_hypervolume(pf_feas,
                                                   ref=self.global_hv_ref)
                           if pf_feas else 0.0)
                hvs.append(hv_all); feas_hvs.append(hv_feas)
                costs.append(m.get('best_cost', np.nan))
                ems.append(m.get('best_emissions', np.nan))
                times.append(m.get('best_time', np.nan))
                rows.append({
                    'Evaluation_Mode': mode_name,
                    'Run': run,
                    'Hypervolume': hv_all,
                    'Hypervolume_Feasible': hv_feas,
                    'Best_Cost_USD': m.get('best_cost', np.nan),
                    'Best_CO2_kg': m.get('best_emissions', np.nan),
                    'Best_Time_h': m.get('best_time', np.nan),
                    'Pareto_Size': m.get('pareto_size', 0),
                    'Valid_Solutions': m.get('valid_solutions', 0),
                    'Global_HV_Ref': self.global_hv_ref.tolist(),
                })
                print(f"    mode={mode_name} run {run+1}/{num_runs}",
                      flush=True)
            print(f"    HV_all  = {np.mean(hvs):.4g} +/- {np.std(hvs):.4g}")
            print(f"    HV_feas = {np.mean(feas_hvs):.4g} "
                  f"+/- {np.std(feas_hvs):.4g}")
            print(f"    CO2     = {np.nanmean(ems):.2f} kg, "
                  f"Cost = ${np.nanmean(costs):.2f}")
        self.evaluation_mode_results = rows
        return rows

    def run_delta_min_sensitivity(self, deltas=(0.20, 0.25, 0.30, 0.35, 0.40),
                                  num_runs=30):
        print("\n" + "=" * 80)
        print(" RELIABILITY-THRESHOLD SENSITIVITY ".center(80, "="))
        print("=" * 80)
        print("  Re-filtering the eight templates at five delta_min values.")
        print("  No search re-run; reliability is computed on the raw route.")

        ds = LC_CTRP_Dataset()
        problem = FreightRoutingProblem(ds, season='Dry',
                                        carbon_price=CARBON_PRICE_DEFAULT,
                                        demand_cv=DEMAND_CV_DEFAULT)
        candidates = []
        for t in FreightRoutingProblem.TEMPLATES:
            ind = problem.evaluate_route(t['path'], t['modes'])
            if ind.det_objectives is not None \
                    and np.isfinite(ind.det_objectives[0]):
                candidates.append({
                    'name': t['name'],
                    'route': ind.route,
                    'modes': ind.modes,
                    'reliability': ind.reliability,
                    'raw_co2': ind.det_objectives[1],
                    'raw_cost': ind.det_objectives[0],
                    'raw_time': ind.det_objectives[2],
                })

        rows = []
        for d in deltas:
            feasible = [c for c in candidates if c['reliability'] >= d]
            n_feas = len(feasible)
            if feasible:
                best = min(feasible, key=lambda c: c['raw_co2'])
                rows.append({
                    'Delta_Min': d,
                    'Feasible_Templates': n_feas,
                    'Total_Templates': len(candidates),
                    'Best_Template': best['name'],
                    'Best_Raw_CO2_kg': best['raw_co2'],
                    'Best_Raw_Cost_USD': best['raw_cost'],
                    'Best_Raw_Time_h': best['raw_time'],
                    'Best_Reliability': best['reliability'],
                })
                print(f"  delta_min = {d:.2f}: {n_feas}/{len(candidates)} "
                      f"feasible, best = {best['name']} "
                      f"(CO2 = {best['raw_co2']:.1f} kg, "
                      f"reliab. = {best['reliability']:.3f})")
            else:
                rows.append({
                    'Delta_Min': d,
                    'Feasible_Templates': 0,
                    'Total_Templates': len(candidates),
                    'Best_Template': None,
                    'Best_Raw_CO2_kg': np.nan,
                    'Best_Raw_Cost_USD': np.nan,
                    'Best_Raw_Time_h': np.nan,
                    'Best_Reliability': np.nan,
                })
                print(f"  delta_min = {d:.2f}: 0/{len(candidates)} feasible "
                      f"(complete collapse)")
        self.delta_min_results = rows
        return rows

    def run_sensitivity_analysis(self, global_ref=None):
        if global_ref is None: global_ref = self.global_hv_ref
        if global_ref is None:
            print("  [warn] no global HV ref provided; "
                  "sensitivity HV will not be comparable across rows.")
        ref_used = (np.asarray(global_ref, dtype=float)
                    if global_ref is not None else None)

        print("\n" + "=" * 80)
        print(" Q2-Q3: SENSITIVITY ANALYSIS ".center(80, "="))
        print("=" * 80)
        print(f"  Reference point (fixed for all rows): "
              f"{ref_used.tolist() if ref_used is not None else 'per-value (fallback)'}")
        axes = [
            ('carbon_price','Carbon price ($/ton CO2)',[50,100,150],
             'carbon_price'),
            ('carbon_price_volatility',
             'Carbon price volatility (std $/ton)',
             [0, 5, 15, 25],'carbon_price_std'),
            ('season','Season',['Dry','Wet','Peak'],'season'),
            ('border_delay','Border-delay multiplier',
             [1.0,1.25,1.5],'border_multiplier'),
            ('transshipment','Transshipment-cost multiplier',
             [0.5,1.0,1.5],'transshipment_multiplier'),
        ]
        n_runs = self.config.num_runs
        for axis_key, axis_label, values, kw in axes:
            print(f"\n  Axis: {axis_label}  (n_runs = {n_runs})")
            for v in values:
                kwargs = {kw: v}
                if axis_key == 'carbon_price':
                    kwargs = {'carbon_price': v}
                elif axis_key == 'carbon_price_volatility':
                    kwargs = {'carbon_price_std': v}
                scen = {'name':f'{axis_key}_{v}','season':'Dry',
                        'carbon_price':50,'carbon_price_std':5.0,
                        'demand_cv':self.config.demand_cv,
                        'border_multiplier':1.0,
                        'transshipment_multiplier':1.0}
                scen.update(kwargs)
                ds = LC_CTRP_Dataset(carbon_price=scen['carbon_price'],
                                     carbon_price_std=scen['carbon_price_std'],
                                     demand_cv=scen.get('demand_cv',
                                                        self.config.demand_cv),
                                     stochastic_borders=self.config.stochastic_borders)
                ds.apply_carbon_price(scen['carbon_price'],
                                      std=scen['carbon_price_std'])
                ds.apply_demand_cv(scen.get('demand_cv',
                                            self.config.demand_cv))
                ds.apply_border_delay_multiplier(scen['border_multiplier'])
                ds.apply_transshipment_cost_multiplier(
                    scen['transshipment_multiplier'])
                problem = FreightRoutingProblem(
                    ds, season=scen['season'],
                    carbon_price=scen['carbon_price'],
                    demand_cv=scen.get('demand_cv', self.config.demand_cv))
                n_feas_templates = problem.n_feasible_templates()
                hvs, hv_feas = [], []
                costs, ems, times = [], [], []
                for run in range(n_runs):
                    seed = self.seeder.get_seed(f'{axis_key}_{v}',
                                                'dro_emo',run)
                    algo = DROEMO(problem, self.config.dro_emo, run, seed)
                    m = algo.run()
                    pf = m.get('pareto_fitness', [])
                    pf_inds = m.get('pareto_front', [])
                    hv_all = monte_carlo_hypervolume(pf, ref=ref_used)
                    pf_feas = [p for p, ind in zip(pf, pf_inds)
                               if ind.is_valid]
                    hv_f = (monte_carlo_hypervolume(pf_feas, ref=ref_used)
                            if pf_feas else 0.0)
                    hvs.append(hv_all); hv_feas.append(hv_f)
                    costs.append(m['best_cost'])
                    ems.append(m['best_emissions'])
                    times.append(m['best_time'])
                    print(f"    {axis_label}={v} run {run+1}/{n_runs}",
                          flush=True)
                self.sensitivity_results[axis_key].append({
                    'Axis':axis_key,'Axis_Label':axis_label,'Value':v,
                    'Hypervolume_All_mean':float(np.mean(hvs)),
                    'Hypervolume_All_std':float(np.std(hvs)),
                    'Hypervolume_Feasible_mean':float(np.mean(hv_feas)),
                    'Hypervolume_Feasible_std':float(np.std(hv_feas)),
                    'Best_Cost_mean':float(np.mean(costs)),
                    'Best_CO2_mean':float(np.mean(ems)),
                    'Best_Time_mean':float(np.mean(times)),
                    'Feasible_Templates':int(n_feas_templates),
                    'Total_Templates':len(FreightRoutingProblem.TEMPLATES),
                    'N_Runs':n_runs,
                    'Global_HV_Ref':ref_used.tolist()
                                    if ref_used is not None else None,
                })
                print(f"    {axis_label} = {v}: "
                      f"HV_all = {np.mean(hvs):.4g}, "
                      f"HV_feas = {np.mean(hv_feas):.4g}, "
                      f"CO2 = {np.mean(ems):.2f} kg, "
                      f"feasible templates = {n_feas_templates}/8")

    def run_regulation_comparison(self, algorithms, num_runs=30):
        print("\n" + "=" * 80)
        print(" Q5: ALTERNATIVE EMISSION REGULATION COMPARISON ".center(80, "="))
        print("=" * 80)
        print("  Reporting raw (deterministic) objectives of best feasible solution.")
        regulations = {
            'NR': {'carbon_price': 0, 'carbon_price_std': 0,
                   'emission_cap': 1e9, 'description': 'No regulation'},
            'CCR_Strict': {'carbon_price': 0, 'carbon_price_std': 0,
                           'emission_cap': 3000,
                           'description': 'Carbon cap (strict, 3000 kg/TEU)'},
            'CCR_Moderate': {'carbon_price': 0, 'carbon_price_std': 0,
                             'emission_cap': 4000,
                             'description': 'Carbon cap (moderate, 4000 kg/TEU)'},
            'CTX_Low': {'carbon_price': 10, 'carbon_price_std': 0,
                        'emission_cap': 1e9,
                        'description': 'Carbon tax (low, $10/ton)'},
            'CTX_Medium': {'carbon_price': 50, 'carbon_price_std': 0,
                           'emission_cap': 1e9,
                           'description': 'Carbon tax (medium, $50/ton)'},
            'CTX_High': {'carbon_price': 100, 'carbon_price_std': 0,
                         'emission_cap': 1e9,
                         'description': 'Carbon tax (high, $100/ton)'},
            'CC&TX_Strict': {'carbon_price': 50, 'carbon_price_std': 0,
                             'emission_cap': 3000,
                             'description': 'Cap-and-tax (strict cap, $50/ton)'},
            'CC&TX_Moderate': {'carbon_price': 50, 'carbon_price_std': 0,
                               'emission_cap': 4000,
                               'description': 'Cap-and-tax (moderate cap, $50/ton)'},
            'CT_Low': {'carbon_price': 50, 'carbon_price_std': 5,
                       'emission_cap': 3500,
                       'description': 'Carbon-trading (low price, $50/ton)'},
            'CT_Medium': {'carbon_price': 100, 'carbon_price_std': 10,
                          'emission_cap': 3500,
                          'description': 'Carbon-trading (medium price, $100/ton)'},
            'CT_High': {'carbon_price': 150, 'carbon_price_std': 15,
                        'emission_cap': 3500,
                        'description': 'Carbon-trading (high price, $150/ton)'},
        }
        for reg_name, reg_cfg in regulations.items():
            print(f"\n  Regulation: {reg_name} - {reg_cfg['description']}")
            ds = LC_CTRP_Dataset(carbon_price=reg_cfg['carbon_price'],
                                 carbon_price_std=reg_cfg['carbon_price_std'],
                                 demand_cv=self.config.demand_cv,
                                 stochastic_borders=self.config.stochastic_borders)
            ds.apply_carbon_price(reg_cfg['carbon_price'],
                                  std=reg_cfg['carbon_price_std'])
            ds.apply_demand_cv(self.config.demand_cv)
            ds.global_params['emission_cap'] = reg_cfg['emission_cap']
            problem = FreightRoutingProblem(
                ds, season='Dry', carbon_price=reg_cfg['carbon_price'],
                demand_cv=self.config.demand_cv)
            costs, ems, times, feasible_counts = [], [], [], []
            for run in range(num_runs):
                seed = self.seeder.get_seed(f'reg_{reg_name}',
                                            'dro_emo', run)
                algo = DROEMO(problem, self.config.dro_emo, run, seed)
                m = algo.run()
                pf = m.get('pareto_front', [])
                feas = [ind for ind in pf if ind.is_valid]
                if feas:
                    best = min(feas, key=lambda i: i.det_objectives[1])
                    raw_co2 = best.det_objectives[1]
                    raw_cost = best.det_objectives[0]
                    raw_time = best.det_objectives[2]
                else:
                    raw_co2 = raw_cost = raw_time = float('nan')
                costs.append(raw_cost); ems.append(raw_co2)
                times.append(raw_time); feasible_counts.append(len(feas))
                print(f"    reg={reg_name} run {run+1}/{num_runs}",
                      flush=True)
            self.regulation_results.append({
                'Regulation': reg_name,
                'Description': reg_cfg['description'],
                'Carbon_Price': reg_cfg['carbon_price'],
                'Carbon_Price_Std': reg_cfg['carbon_price_std'],
                'Emission_Cap': reg_cfg['emission_cap'],
                'Best_Cost_mean': float(np.nanmean(costs))
                                  if any(np.isfinite(costs)) else float('nan'),
                'Best_CO2_mean': float(np.nanmean(ems))
                                 if any(np.isfinite(ems)) else float('nan'),
                'Best_Time_mean': float(np.nanmean(times))
                                  if any(np.isfinite(times)) else float('nan'),
                'Feasible_Solutions_mean': float(np.mean(feasible_counts)),
            })
            print(f"    CO2 = {np.nanmean(ems):.2f} kg, "
                  f"Cost = ${np.nanmean(costs):.2f}, "
                  f"Time = {np.nanmean(times):.2f} h, "
                  f"feasible sols = {np.mean(feasible_counts):.1f}")

    def run_demand_uncertainty_analysis(self, num_runs=30):
        print("\n" + "=" * 80)
        print(" Q6: DEMAND UNCERTAINTY IMPACT ".center(80, "="))
        print("=" * 80)
        demand_types = {
            'D1_Stationary': {'demand_cv': 0.10,
                              'description': 'Stationary demand (CV=0.10)'},
            'D2_Nonstationary': {'demand_cv': 0.25,
                                 'description': 'Nonstationary demand (CV=0.25)'},
            'D3_Trend': {'demand_cv': 0.50,
                         'description': 'Nonstationary with trend (CV=0.50)'},
        }
        for dt_name, dt_cfg in demand_types.items():
            print(f"\n  Demand type: {dt_name} - {dt_cfg['description']}")
            ds = LC_CTRP_Dataset(carbon_price=CARBON_PRICE_DEFAULT,
                                 carbon_price_std=CARBON_PRICE_STD_DEFAULT,
                                 demand_cv=dt_cfg['demand_cv'],
                                 stochastic_borders=self.config.stochastic_borders)
            ds.apply_carbon_price(CARBON_PRICE_DEFAULT)
            ds.apply_demand_cv(dt_cfg['demand_cv'])
            problem = FreightRoutingProblem(
                ds, season='Dry', carbon_price=CARBON_PRICE_DEFAULT,
                demand_cv=dt_cfg['demand_cv'])
            costs, ems, times, feasible_counts = [], [], [], []
            for run in range(num_runs):
                seed = self.seeder.get_seed(f'demand_{dt_name}',
                                            'dro_emo', run)
                algo = DROEMO(problem, self.config.dro_emo, run, seed)
                m = algo.run()
                pf = m.get('pareto_front', [])
                feas = [ind for ind in pf if ind.is_valid]
                if feas:
                    best = min(feas, key=lambda i: i.det_objectives[1])
                    raw_co2 = best.det_objectives[1]
                    raw_cost = best.det_objectives[0]
                    raw_time = best.det_objectives[2]
                else:
                    raw_co2 = raw_cost = raw_time = float('nan')
                costs.append(raw_cost); ems.append(raw_co2)
                times.append(raw_time); feasible_counts.append(len(feas))
                print(f"    demand={dt_name} run {run+1}/{num_runs}",
                      flush=True)
            self.sensitivity_results['demand_uncertainty'].append({
                'Demand_Type': dt_name,
                'Description': dt_cfg['description'],
                'Demand_CV': dt_cfg['demand_cv'],
                'Best_Cost_mean': float(np.nanmean(costs)),
                'Best_CO2_mean': float(np.nanmean(ems)),
                'Best_Time_mean': float(np.nanmean(times)),
                'Feasible_Solutions_mean': float(np.mean(feasible_counts)),
            })
            print(f"    CO2 = {np.nanmean(ems):.2f} kg, "
                  f"Cost = ${np.nanmean(costs):.2f}, "
                  f"Feasible = {np.mean(feasible_counts):.1f}")

    def _record_pareto(self, scen, alg, pf):
        self.pareto_by_scenario.setdefault(scen,{}).setdefault(alg,[]).append(pf)

    def run_case_study(self, scenarios, algorithms, num_runs=30):
        print("\n" + "=" * 80)
        print(" Q1-Q4: MOMBASA-BUJUMBURA CORRIDOR CASE STUDY ".center(80, "="))
        print("=" * 80)
        if self.global_hv_ref is None:
            print("  [warn] global HV ref not set; falling back to baseline pass-1.")
            self.build_global_reference(scenarios, algorithms, num_runs)
        print(f"  Using global HV ref = {self.global_hv_ref.tolist()}")

        for scen in scenarios:
            name = scen.get('name','Unnamed')
            price = scen.get('carbon_price', CARBON_PRICE_DEFAULT)
            price_std = scen.get('carbon_price_std',
                                 CARBON_PRICE_STD_DEFAULT)
            demand_cv = scen.get('demand_cv', self.config.demand_cv)
            season = scen.get('season','Dry')
            bmult = scen.get('border_multiplier',1.0)
            tmult = scen.get('transshipment_multiplier',1.0)
            print(f"\n{'=' * 60}\nScenario: {name}\n{'=' * 60}",
                  flush=True)
            ds = LC_CTRP_Dataset(carbon_price=price,
                                 carbon_price_std=price_std,
                                 demand_cv=demand_cv,
                                 stochastic_borders=self.config.stochastic_borders)
            ds.apply_carbon_price(price, std=price_std)
            ds.apply_demand_cv(demand_cv)
            ds.apply_border_delay_multiplier(bmult)
            ds.apply_transshipment_cost_multiplier(tmult)
            problem = FreightRoutingProblem(
                ds, season=season, carbon_price=price, demand_cv=demand_cv)
            amb = problem.ambiguity.structure()
            n_feas_templates = problem.n_feasible_templates()
            print(f"  Ambiguity: dim={problem.ambiguity.dimension()}, "
                  f"columns: two-sided={amb['n_two_sided']}, "
                  f"forward-only={amb['n_forward_only']}, "
                  f"backward-only={amb['n_backward_only']}, "
                  f"degenerate={amb['n_degenerate']}")
            print(f"  Feasible templates: {n_feas_templates}/8")
            for alg in algorithms:
                print(f"\n  Running {alg.upper()}...")
                runs = []
                for run in range(num_runs):
                    seed = self.seeder.get_seed(name, alg, run)
                    t0 = time.time()
                    algo = self.get_algorithm(alg, problem, run, seed)
                    m = algo.run()
                    el = time.time() - t0
                    pf = m.get('pareto_fitness', [])
                    pf_inds = m.get('pareto_front', [])
                    if pf:
                        self.all_pareto_fronts.setdefault(
                            alg.upper().replace('_','-'), []).extend(pf)
                        self._record_pareto(
                            name, alg.upper().replace('_','-'),
                            [list(p) for p in pf])
                    hv_all = monte_carlo_hypervolume(
                        pf, ref=self.global_hv_ref)
                    pf_feas = [p for p, ind in zip(pf, pf_inds)
                               if ind.is_valid]
                    hv_feas = (monte_carlo_hypervolume(pf_feas,
                                                       ref=self.global_hv_ref)
                               if pf_feas else 0.0)
                    if name == 'baseline':
                        pareto = m.get('pareto_front', [])
                        if pareto:
                            best = min(pareto,
                                       key=lambda i: i.objectives[1])
                            key = alg.upper().replace('_','-')
                            prev = self.best_solutions.get(key)
                            if prev is None or \
                                    best.objectives[1] < prev.objectives[1]:
                                self.best_solutions[key] = best.copy()
                    self.results.append({
                        'Scenario':name,'Carbon_Price':price,
                        'Carbon_Price_Std':price_std,'Demand_CV':demand_cv,
                        'Season':season,'Run':run,
                        'Algorithm':alg.upper().replace('_','-'),
                        'Hypervolume':hv_all,
                        'Hypervolume_Feasible':hv_feas,
                        'Feasible_Templates':n_feas_templates,
                        'Best_Cost_USD':m.get('best_cost',0),
                        'Best_Emissions_kg':m.get('best_emissions',0),
                        'Best_Time_Hours':m.get('best_time',0),
                        'Det_Best_Cost_USD':m.get('det_best_cost',0),
                        'Det_Best_Emissions_kg':m.get('det_best_emissions',0),
                        'Det_Best_Time_Hours':m.get('det_best_time',0),
                        'Pareto_Size':m.get('pareto_size',0),
                        'Valid_Solutions':m.get('valid_solutions',0),
                        'Diversity':m.get('diversity',0),
                        'Time_Seconds':el,
                        'convergence':m.get('convergence',[]),
                    })
                    runs.append(self.results[-1])
                    print(f"    {alg} run {run+1}/{num_runs}",
                          flush=True)
                hvs = [r['Hypervolume'] for r in runs]
                hvs_f = [r['Hypervolume_Feasible'] for r in runs]
                ems = [r['Best_Emissions_kg'] for r in runs]
                print(f"    HV_all: {np.mean(hvs):.4g} "
                      f"+/- {np.std(hvs):.4g}")
                print(f"    HV_feas: {np.mean(hvs_f):.4g} "
                      f"+/- {np.std(hvs_f):.4g}")
                print(f"    Emissions: {np.mean(ems):.2f} "
                      f"+/- {np.std(ems):.2f} kg")
            if name in ('wet_season','peak_season'):
                print(f"\n  Best feasible route found by DRO-EMO "
                      f"under {name}:")
                seed = self.seeder.get_seed(name, 'dro_emo', 0)
                algo = self.get_algorithm('dro_emo', problem, 0, seed)
                m = algo.run()
                feas = [ind for ind in m['pareto_front'] if ind.is_valid]
                if feas:
                    best = min(feas, key=lambda i: i.det_objectives[1])
                    print(f"    route = {'-'.join(map(str, best.route))}")
                    print(f"    modes = {'-'.join(best.modes)}")
                    print(f"    CO2   = {best.det_objectives[1]:.1f} kg, "
                          f"reliab. = {best.reliability:.3f}, "
                          f"rail = {best.rail_ratio*100:.1f}%, "
                          f"water = {best.water_ratio*100:.1f}%")
                else:
                    print("    (no feasible route found in Pareto front)")
        return pd.DataFrame(self.results)

    def _node_utilisation(self, output_dir):
        ds = LC_CTRP_Dataset()
        rows = []
        for alg, sol in self.best_solutions.items():
            for n in sol.route:
                cap = ds.node_capacities.get(n, {})
                if not cap: continue
                rows.append({
                    'Algorithm':alg,'Node_ID':n,
                    'Node_Name':ds.nodes.get(n,{}).get('name','?'),
                    'Country':ds.nodes.get(n,{}).get('country','?'),
                    'Daily_Capacity_TEU':cap.get('daily_capacity'),
                    'Avg_Daily_Demand_TEU':cap.get('avg_demand'),
                    'Storage_TEU':cap.get('storage'),
                    'Utilisation_%':(cap.get('avg_demand',0)
                                     /max(cap.get('daily_capacity',1),1)*100),
                })
        if rows:
            pd.DataFrame(rows).drop_duplicates().to_csv(
                output_dir / 'node_utilisation.csv', index=False)
            print(f"  [ok] Saved: node_utilisation.csv ({len(rows)} rows)")

    def _paired_stats(self, output_dir):
        df = pd.DataFrame(self.results)
        print(f"  [paired_stats] total results: {len(df)}")
        if len(df) == 0:
            print("  [paired_stats] no results - skipping"); return
        base = df[df['Scenario']=='baseline']
        print(f"  [paired_stats] baseline rows: {len(base)}")
        dro = base[base['Algorithm']=='DRO-EMO'].sort_values('Run')
        if len(dro) < 5:
            print("  [paired_stats] too few DRO-EMO rows - skipping"); return
        test_type = ('two-sided' if self.config.wilcoxon_two_sided
                     else 'one-sided (greater)')
        alt = 'two-sided' if self.config.wilcoxon_two_sided else 'greater'
        rows = []
        for alg in sorted(base['Algorithm'].unique()):
            if alg == 'DRO-EMO': continue
            comp = base[base['Algorithm']==alg].sort_values('Run')
            if len(comp) != len(dro): continue
            x = dro['Hypervolume'].values.astype(float)
            y = comp['Hypervolume'].values.astype(float)
            diffs = x - y
            if np.all(diffs == 0):
                rows.append({'Comparison':f'DRO-EMO vs {alg}',
                             'N_pairs':len(diffs),'Mean_diff':0.0,
                             'Median_diff':0.0,'Wilcoxon_stat':float('nan'),
                             'p_value':1.0,'Cliffs_delta':0.0,
                             'Frac_positive':0.0,
                             'Test_Type':test_type,
                             'Significant_at_0.05':False})
                continue
            try:
                stat, p = stats.wilcoxon(x, y, zero_method='wilcox',
                                         alternative=alt)
            except Exception:
                continue
            gt = int(np.sum(diffs > 0)); lt = int(np.sum(diffs < 0))
            delta = (gt - lt)/len(diffs)
            frac_pos = gt/len(diffs)
            rows.append({
                'Comparison':f'DRO-EMO vs {alg}',
                'N_pairs':len(diffs),
                'Mean_diff':float(np.mean(diffs)),
                'Median_diff':float(np.median(diffs)),
                'Wilcoxon_stat':float(stat),
                'p_value':float(p),
                'Cliffs_delta':float(delta),
                'Frac_positive':float(frac_pos),
                'Test_Type':test_type,
                'Significant_at_0.05':bool(p < 0.05),
            })
        if rows:
            pd.DataFrame(rows).to_csv(
                output_dir/'paired_wilcoxon_baseline.csv', index=False)
            print(f"  [ok] Saved: paired_wilcoxon_baseline.csv "
                  f"({len(rows)} rows)")

    def generate_report(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        pd.DataFrame(self.results).to_csv(
            output_dir / f'case_study_results_{ts}.csv', index=False)
        self._save_csv(self.alternative_paths,
                       output_dir / 'alternative_paths.csv',
                       'alternative paths')
        if self._detailed_route_rows:
            self._save_csv(self._detailed_route_rows,
                           output_dir / 'alternative_paths_detailed.csv',
                           'per-segment breakdown')
        for k, rows in self.sensitivity_results.items():
            self._save_csv(rows, output_dir / f'sensitivity_{k}.csv', k)
        if self.regulation_results:
            self._save_csv(self.regulation_results,
                           output_dir / 'regulation_comparison.csv',
                           'alternative emission regulations')
        self._node_utilisation(output_dir)
        self._paired_stats(output_dir)
        if self.evaluation_mode_results:
            pd.DataFrame(self.evaluation_mode_results).to_csv(
                output_dir / 'evaluation_mode_comparison.csv', index=False)
            print(f"  [ok] Saved: evaluation_mode_comparison.csv "
                  f"({len(self.evaluation_mode_results)} rows)")
        if self.delta_min_results:
            pd.DataFrame(self.delta_min_results).to_csv(
                output_dir / 'delta_min_sensitivity.csv', index=False)
            print(f"  [ok] Saved: delta_min_sensitivity.csv "
                  f"({len(self.delta_min_results)} rows)")
        try:
            viz = EnhancedVisualizer(
                self.results, output_dir,
                self.all_pareto_fronts, self.alternative_paths,
                self.sensitivity_results, self.pareto_by_scenario)
            viz.generate_all_figures()
        except Exception as e:
            print(f"\n  [warn] Figure generation failed: "
                  f"{type(e).__name__}: {e}")
            print("  [warn] CSV outputs are intact. Fix environment and re-run.")
        print(f"\nReport generated in: {output_dir}")

# ============================================================================
# VISUALIZER
# ============================================================================
class EnhancedVisualizer:
    def __init__(self, results, output_dir, all_pareto_fronts=None,
                 alternative_paths=None, sensitivity_results=None,
                 pareto_by_scenario=None):
        self.results = results; self.output_dir = Path(output_dir)
        self.all_pareto_fronts = all_pareto_fronts or {}
        self.alternative_paths = alternative_paths or []
        self.sensitivity = sensitivity_results or {}
        self.pareto_by_scenario = pareto_by_scenario or {}
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        self.colors = COLORS; self.markers = MARKERS

    def _save(self, fig, name):
        saved = False
        for fmt in ['pdf','png']:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=f'.{fmt}',
                                                 delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                fig.savefig(tmp_path, dpi=300, bbox_inches='tight',
                            facecolor='white')
                try:
                    shutil.move(str(tmp_path),
                                str(self.plots_dir/f"{name}.{fmt}"))
                    saved = True
                except PermissionError:
                    shutil.move(str(tmp_path),
                                str(self.plots_dir/f"{name}_alt.{fmt}"))
                    saved = True
            except Exception as e:
                print(f"  [warn] Failed to save {name}.{fmt}: "
                      f"{type(e).__name__}: {e}")
                if tmp_path is not None and tmp_path.exists():
                    try: tmp_path.unlink()
                    except Exception: pass
                continue
        plt.close(fig)
        print(f"  {'[ok] Saved' if saved else '[warn] Could not save'}: {name}")

    def fig1_baseline(self):
        fig, ax = plt.subplots(figsize=(6,4))
        base = [r for r in self.results if r['Scenario']=='baseline']
        if not base: plt.close(fig); return
        by_alg = {}
        for r in base:
            by_alg.setdefault(r['Algorithm'],[]).append(r['Hypervolume'])
        algs = list(by_alg.keys())
        means = [np.mean(by_alg[a]) for a in algs]
        errs = [np.std(by_alg[a]) for a in algs]
        order = np.argsort(means)[::-1]
        algs = [algs[i] for i in order]
        means = [means[i] for i in order]
        errs = [errs[i] for i in order]
        colors = [self.colors.get(a,'#95a5a6') for a in algs]
        ax.barh(range(len(algs)), means, xerr=errs, color=colors,
                edgecolor='black', alpha=0.85, capsize=3)
        ax.set_yticks(range(len(algs))); ax.set_yticklabels(algs)
        ax.set_xlabel('Hypervolume (global reference)', fontweight='bold')
        ax.set_title('Baseline Performance (DRO, $50/ton, Dry)',
                     fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        plt.tight_layout(); self._save(fig, 'Fig1_Baseline_Performance')

    def fig2_carbon_tax(self):
        fig, ax = plt.subplots(figsize=(6,4)); any_data = False
        for alg in COLORS:
            ys, es, xs = [], [], []
            for p in [50,100,150]:
                vals = [r['Hypervolume'] for r in self.results
                        if r['Scenario']==f'carbon_tax_{p}'
                        and r['Algorithm']==alg]
                if vals:
                    xs.append(p); ys.append(np.mean(vals))
                    es.append(np.std(vals))
            if ys:
                any_data = True
                ax.errorbar(xs, ys, yerr=es,
                            marker=self.markers.get(alg,'o'),
                            color=self.colors.get(alg,'#333'), label=alg,
                            linewidth=2, markersize=8, capsize=3)
        if not any_data: plt.close(fig); return
        ax.set_xlabel('Carbon price ($/ton CO2)', fontweight='bold')
        ax.set_ylabel('Hypervolume (global reference)', fontweight='bold')
        ax.set_title('Carbon Price Sensitivity', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout(); self._save(fig, 'Fig2_Carbon_Tax_Sensitivity')

    def fig3_seasonal(self):
        fig, ax = plt.subplots(figsize=(6,4))
        seasons = ['Dry','Wet','Peak']
        scens = ['baseline','wet_season','peak_season']
        algs = list(COLORS); w = 0.15; x = np.arange(len(seasons))
        any_data = False
        for i, alg in enumerate(algs):
            hv, er, ok = [], [], []
            for sc in scens:
                vals = [r['Hypervolume'] for r in self.results
                        if r['Scenario']==sc and r['Algorithm']==alg]
                if vals:
                    hv.append(np.mean(vals)); er.append(np.std(vals))
                    ok.append(True)
                else:
                    hv.append(0); er.append(0); ok.append(False)
            if any(ok):
                any_data = True
                ax.bar(x+(i-2)*w, hv, w, yerr=er,
                       color=self.colors.get(alg,'#95a5a6'),
                       edgecolor='black', alpha=0.85, label=alg, capsize=2)
        if not any_data: plt.close(fig); return
        ax.set_xticks(x); ax.set_xticklabels(seasons)
        ax.set_ylabel('Hypervolume (global reference)', fontweight='bold')
        ax.set_title('Seasonal Sensitivity', fontweight='bold')
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); self._save(fig, 'Fig3_Seasonal_Sensitivity')

    def _baseline_fronts(self):
        out = {}
        base = self.pareto_by_scenario.get('baseline', {})
        for alg, runs in base.items():
            pts = []
            for pf in runs:
                for p in pf:
                    try:
                        arr = np.asarray(p, dtype=float).flatten()
                        if arr.shape[0] == 3 \
                                and np.all(np.isfinite(arr)) \
                                and np.all(np.abs(arr) < 1e6):
                            pts.append(arr)
                    except Exception:
                        continue
            if pts: out[alg] = np.array(pts)
        return out

    def fig4_pareto(self):
        fig, ax = plt.subplots(figsize=(6,4)); any_data = False
        fronts = self._baseline_fronts()
        for alg, P in fronts.items():
            if len(P)==0: continue
            srt = np.argsort(P[:,0])
            ax.scatter(P[srt,0], P[srt,1], label=alg, s=18, alpha=0.7,
                       color=self.colors.get(alg,'#333'),
                       marker=self.markers.get(alg,'o'),
                       edgecolors='k', linewidths=0.3)
            any_data = True
        if not any_data: plt.close(fig); return
        ax.set_xlabel('Total cost (USD)', fontweight='bold')
        ax.set_ylabel('CO2 emissions (kg)', fontweight='bold')
        ax.set_title('Baseline Pareto front (DRO, global reference)',
                     fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout(); self._save(fig, 'Fig4_Pareto_Front')

    def fig5_convergence(self):
        fig, ax = plt.subplots(figsize=(6,4)); any_data = False
        for alg in COLORS:
            runs = [r for r in self.results
                    if r['Algorithm']==alg and r['Scenario']=='baseline']
            curves = []
            for r in runs:
                c = r.get('convergence')
                if c is None: continue
                if isinstance(c, str):
                    try: c = json.loads(c)
                    except Exception: continue
                if isinstance(c, (list, tuple, np.ndarray)) and len(c) > 0:
                    curves.append(list(c))
            if not curves: continue
            m = max(len(c) for c in curves)
            padded = np.array([c+[c[-1]]*(m-len(c)) for c in curves])
            mean = padded.mean(0); std = padded.std(0)
            ax.plot(mean, label=alg,
                    color=self.colors.get(alg,'#333'), linewidth=2)
            ax.fill_between(range(len(mean)), mean-std, mean+std,
                            color=self.colors.get(alg,'#333'), alpha=0.15)
            any_data = True
        if not any_data: plt.close(fig); return
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Best CO2 emissions (kg)', fontweight='bold')
        ax.set_title('Convergence (DRO, baseline)', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout(); self._save(fig, 'Fig5_Convergence_Curves')

    def fig6_det_vs_robust(self):
        fig, ax = plt.subplots(figsize=(6,4))
        base = [r for r in self.results if r['Scenario']=='baseline']
        if not base: plt.close(fig); return
        algs = sorted(set(r['Algorithm'] for r in base))
        det = [np.mean([r['Det_Best_Cost_USD'] for r in base
                        if r['Algorithm']==a]) for a in algs]
        rob = [np.mean([r['Best_Cost_USD'] for r in base
                        if r['Algorithm']==a]) for a in algs]
        x = np.arange(len(algs)); w = 0.35
        ax.bar(x-w/2, det, w, label='Deterministic',
               color='steelblue', edgecolor='black')
        ax.bar(x+w/2, rob, w, label='DRO (worst-case)',
               color='tomato', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(algs, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Best cost (USD)', fontweight='bold')
        ax.set_title('Deterministic vs robust-stochastic cost',
                     fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); self._save(fig, 'Fig6_Deterministic_vs_Robust')

    def fig7_tradeoffs(self):
        fig, axes = plt.subplots(1,3, figsize=(11,3.5))
        pairs = [(0,1,'Cost vs CO2','Cost (USD)','CO2 (kg)'),
                 (0,2,'Cost vs time','Cost (USD)','Time (h)'),
                 (1,2,'CO2 vs time','CO2 (kg)','Time (h)')]
        fronts = self._baseline_fronts()
        any_data = False
        for ax, (xi, yi, title, xl, yl) in zip(axes, pairs):
            for alg, P in fronts.items():
                if len(P)==0: continue
                srt = np.argsort(P[:,xi])
                ax.scatter(P[srt,xi], P[srt,yi], s=10, alpha=0.65,
                           color=self.colors.get(alg,'#333'),
                           marker=self.markers.get(alg,'o'), label=alg,
                           edgecolors='k', linewidths=0.2)
                any_data = True
            ax.set_xlabel(xl, fontsize=9, fontweight='bold')
            ax.set_ylabel(yl, fontsize=9, fontweight='bold')
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
        if any_data:
            h, l = axes[0].get_legend_handles_labels()
            if h: fig.legend(h, l, loc='lower center', ncol=5, fontsize=7)
            plt.tight_layout(rect=[0,0.08,1,0.98])
        else: plt.tight_layout()
        self._save(fig, 'Fig7_Tradeoff_Plots')

    def fig8_improvement(self):
        fig, ax = plt.subplots(figsize=(6,3.5))
        base = [r for r in self.results if r['Scenario']=='baseline']
        if not base: plt.close(fig); return
        by = {}
        for r in base:
            by.setdefault(r['Algorithm'],[]).append(r['Hypervolume'])
        dro = np.mean(by.get('DRO-EMO',[0]))
        imps, names = [], []
        for a in by:
            if a=='DRO-EMO': continue
            m = np.mean(by[a])
            if m>0: imps.append((dro-m)/m*100); names.append(a)
        if not imps: plt.close(fig); return
        colors = [self.colors.get(n,'#95a5a6') for n in names]
        bars = ax.bar(range(len(names)), imps, color=colors,
                      edgecolor='black', alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('DRO-EMO improvement (%)', fontweight='bold')
        ax.set_title('Hypervolume improvement over competitors',
                     fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5)
        for b, v in zip(bars, imps):
            ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:+.1f}%',
                    ha='center', fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout(); self._save(fig, 'Fig8_Improvement_Chart')

    def fig9_corridor(self):
        ds = LC_CTRP_Dataset()
        fig, ax = plt.subplots(figsize=(6,5))
        for n,(lat,lon) in ds.coords.items():
            ax.scatter(lon, lat, s=28, c='steelblue',
                       edgecolors='k', zorder=3)
            ax.text(lon+0.15, lat+0.15, str(n), fontsize=6.5, zorder=4)
        for (i,j) in ds.arcs:
            if i<j:
                lati,loni = ds.coords[i]; latj,lonj = ds.coords[j]
                ax.plot([loni,lonj], [lati,latj], '-', color='grey',
                        linewidth=0.4, alpha=0.4, zorder=1)
        best = [1,8,10,11,12,13,14,19]
        for k in range(len(best)-1):
            lati,loni = ds.coords[best[k]]
            latj,lonj = ds.coords[best[k+1]]
            ax.plot([loni,lonj], [lati,latj], '-', color='#e74c3c',
                    linewidth=2.2, zorder=2)
        for n, label in [(1,'Mombasa'), (19,'Bujumbura')]:
            lat, lon = ds.coords[n]
            ax.scatter(lon, lat, s=140, marker='*', c='gold',
                       edgecolors='k', zorder=5)
            ax.annotate(label, (lon,lat), fontsize=8, fontweight='bold',
                        xytext=(4,4), textcoords='offset points')
        ax.set_xlabel('Longitude', fontweight='bold')
        ax.set_ylabel('Latitude', fontweight='bold')
        ax.set_title('Mombasa-Bujumbura corridor - Lake Victoria Corridor',
                     fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout(); self._save(fig, 'Fig9_Corridor_Map')

    def fig10_carbon_volatility(self):
        fig, ax = plt.subplots(figsize=(6,4)); any_data = False
        rows = self.sensitivity.get('carbon_price_volatility', [])
        if rows:
            rows_sorted = sorted(rows, key=lambda r: r.get('Value', 0))
            xs = [r.get('Value', 0) for r in rows_sorted]
            ys = [r.get('Best_CO2_mean', np.nan) for r in rows_sorted]
            if any(np.isfinite(ys)):
                any_data = True
                ax.plot(xs, ys, 'o-', color='#e74c3c',
                        linewidth=2, markersize=8)
                ax.set_xlabel('Carbon price volatility (std $/ton CO2)',
                              fontweight='bold')
                ax.set_ylabel('Best CO2 emissions (kg)', fontweight='bold')
                ax.set_title('Carbon Price Volatility Sensitivity',
                             fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
        if not any_data: plt.close(fig); return
        plt.tight_layout(); self._save(fig, 'Fig10_Carbon_Volatility')

    def fig_alt_paths(self):
        if not self.alternative_paths: return
        df = pd.DataFrame(self.alternative_paths)
        for scen in df['Scenario'].unique():
            sub = df[df['Scenario']==scen].copy()
            if sub.empty: continue
            fig, axes = plt.subplots(1,3, figsize=(14,4.5))
            for ax, col, title, color in zip(
                axes,
                ['Det_CO2_kg','Det_Cost_USD','Det_Time_h'],
                ['CO2 (kg)','Cost (USD)','Time (h)'],
                ['#e74c3c','#3498db','#f39c12']):
                srt = sub.sort_values(col, na_position='last')
                ax.barh(range(len(srt)), srt[col].values, color=color,
                        edgecolor='black', alpha=0.85)
                ax.set_yticks(range(len(srt)))
                ax.set_yticklabels(srt['Path_Name'].values, fontsize=7)
                ax.set_xlabel(title, fontweight='bold')
                ax.set_title(f'{scen} - {title}', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            plt.tight_layout()
            self._save(fig, f'AlternativePaths_{scen}')

    def generate_all_figures(self):
        print("\n" + "=" * 60)
        print(" GENERATING FIGURES ".center(60, "="))
        print("=" * 60)
        self.fig1_baseline(); self.fig2_carbon_tax(); self.fig3_seasonal()
        self.fig4_pareto(); self.fig5_convergence()
        self.fig6_det_vs_robust(); self.fig7_tradeoffs()
        self.fig8_improvement(); self.fig9_corridor()
        self.fig10_carbon_volatility(); self.fig_alt_paths()
        print(f"\n[ok] Figures saved to: {self.plots_dir}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "#" * 90)
    print(" DRO-EMO: MOMBASA-BUJUMBURA CORRIDOR CASE STUDY ".center(90, "#"))
    print("#" * 90)
    print("\nDATA SOURCE: Appendix D - 19 nodes, 34 arcs, 3 modes")
    print("PRIMARY OBJECT: Logistics-centric case study of the corridor")
    print("TOOL: DRO-EMO as a computational instrument for the complex problem")
    print("\n[CACHE-1] Reference-point pass cached to disk "
          "(reduced to 10 runs per algorithm).")
    print("[CACHE-2] Reference-pass progress flushed to stdout line by line.")
    print("[NEW-1] Deterministic and stochastic evaluators added.")
    print("[NEW-2] Evaluation-mode comparison on the baseline condition.")
    print("[NEW-3] Reliability-threshold (delta_min) sensitivity.")
    print("[NEW-4] CSV output for both new analyses.")
    print("[NEW-5] Both new analyses called from main().")
    print("[FIX-REF] Global HV reference point validated before use.")
    print("[CLEAN-1] Removed dead rle_emo attribute lookup.")
    print("[CLEAN-2] Removed duplicated import block.")

    RUN_DIAGNOSTIC_ONLY = False

    run_template_diagnostic(season='Dry', carbon_price=50.0,
                            demand_cv=DEMAND_CV_DEFAULT)

    if RUN_DIAGNOSTIC_ONLY:
        print("\nDiagnostic complete. "
              "Set RUN_DIAGNOSTIC_ONLY = False to run full.")
        return

    config = ExperimentConfig()
    config.num_runs = N_RUNS
    config.stochastic_borders = False
    config.wilcoxon_two_sided = False
    config.demand_cv = DEMAND_CV_DEFAULT

    algorithms = ['dro_emo', 'nsga2', 'nsga3', 'spea2', 'moead', 'rl_nsgaii']

    scenarios = [
        {'name':'baseline','carbon_price':50,'carbon_price_std':5,
         'season':'Dry','demand_cv':0.25},
        {'name':'carbon_tax_50','carbon_price':50,'carbon_price_std':0,
         'season':'Dry','demand_cv':0.25},
        {'name':'carbon_tax_100','carbon_price':100,'carbon_price_std':0,
         'season':'Dry','demand_cv':0.25},
        {'name':'carbon_tax_150','carbon_price':150,'carbon_price_std':0,
         'season':'Dry','demand_cv':0.25},
        {'name':'carbon_volatility_low','carbon_price':50,'carbon_price_std':5,
         'season':'Dry','demand_cv':0.25},
        {'name':'carbon_volatility_medium','carbon_price':50,
         'carbon_price_std':15,'season':'Dry','demand_cv':0.25},
        {'name':'carbon_volatility_high','carbon_price':50,
         'carbon_price_std':25,'season':'Dry','demand_cv':0.25},
        {'name':'demand_low','carbon_price':50,'carbon_price_std':5,
         'season':'Dry','demand_cv':0.10},
        {'name':'demand_medium','carbon_price':50,'carbon_price_std':5,
         'season':'Dry','demand_cv':0.25},
        {'name':'demand_high','carbon_price':50,'carbon_price_std':5,
         'season':'Dry','demand_cv':0.50},
        {'name':'wet_season','carbon_price':50,'carbon_price_std':5,
         'season':'Wet','demand_cv':0.25},
        {'name':'peak_season','carbon_price':50,'carbon_price_std':5,
         'season':'Peak','demand_cv':0.25},
        {'name':'high_border_delay','carbon_price':50,'carbon_price_std':5,
         'season':'Dry','demand_cv':0.25,
         'border_multiplier':1.5,'transshipment_multiplier':1.0},
        {'name':'low_transshipment','carbon_price':50,'carbon_price_std':5,
         'season':'Dry','demand_cv':0.25,
         'border_multiplier':1.0,'transshipment_multiplier':0.5},
    ]
    runner = CaseStudyRunner(config)

    out = Path(
        r"D:\Container Transportation Routing Problems\Manuscript3-A Low-carbon "
        r"based Robust Optimization under uncertainty\DRO-EMO on EAC case_study_results"
    )

    REF_CACHE = Path("./global_hv_ref_cache_tier1.npy")
    if REF_CACHE.exists():
        try:
            cached = np.load(REF_CACHE)
            if cached.shape == (3,) and np.all(np.isfinite(cached)):
                runner.global_hv_ref = cached
                print(f"\n  [CACHE-1] Loaded global HV ref from "
                      f"{REF_CACHE}: {cached.tolist()}", flush=True)
            else:
                print(f"  [CACHE-1] Cached ref malformed "
                      f"(shape={cached.shape}); recomputing.", flush=True)
                REF_CACHE.unlink()
        except Exception as e:
            print(f"  [CACHE-1] Cache read failed ({type(e).__name__}); "
                  f"recomputing.", flush=True)

    runner.run_alternative_paths(scenarios)
    runner.print_alternative_paths()

    if runner.global_hv_ref is None:
        # Reference-point pass uses a reduced budget. The extreme corner
        # of the pooled fronts is set by the best solution found, which
        # almost always appears within the first few runs of each
        # algorithm. The main comparison below still uses config.num_runs.
        REF_PASS_RUNS = 10
        runner.build_global_reference(scenarios, algorithms,
                                      num_runs=REF_PASS_RUNS)
        try:
            np.save(REF_CACHE, runner.global_hv_ref)
            print(f"  [CACHE-1] Saved global HV ref to {REF_CACHE}",
                  flush=True)
            try:
                out.mkdir(parents=True, exist_ok=True)
                np.save(out / REF_CACHE.name, runner.global_hv_ref)
                print(f"  [CACHE-1] Copied cache to "
                      f"{out / REF_CACHE.name}", flush=True)
            except Exception as e:
                print(f"  [CACHE-1] Could not copy cache to out: "
                      f"{type(e).__name__}: {e}", flush=True)
        except Exception as e:
            print(f"  [CACHE-1] Could not save cache: "
                  f"{type(e).__name__}: {e}", flush=True)

    runner.run_evaluation_mode_comparison(num_runs=config.num_runs)
    runner.run_delta_min_sensitivity(
        deltas=(0.20, 0.25, 0.30, 0.35, 0.40),
        num_runs=config.num_runs)

    runner.run_sensitivity_analysis(global_ref=runner.global_hv_ref)
    runner.run_regulation_comparison(algorithms, num_runs=config.num_runs)
    runner.run_demand_uncertainty_analysis(num_runs=config.num_runs)
    df = runner.run_case_study(scenarios, algorithms,
                               num_runs=config.num_runs)
    runner.generate_report(out)
    print("\n" + "=" * 90)
    print(" CASE STUDY COMPLETED ".center(90, "="))
    print("=" * 90)

if __name__ == "__main__":
    main()
