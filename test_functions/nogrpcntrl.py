#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
3-Objective Electrospun Oil Sorbent optimization problem

References

.. [Wang2020]
    B. Wang, J. Cai, C. Liu, J. Yang, X. Ding. Harnessing a Novel Machine-Learning-Assisted Evolutionary Algorithm to Co-optimize Three Characteristics of an Electrospun Oil Sorbent. ACS Applied Materials & Interfaces, 2020.
"""
from typing import List, Optional
import subprocess

import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.torch import BufferDict
from torch import Tensor
import os
import numpy as np
from opm.io.parser import Parser
from opm.io.ecl_state import EclipseState
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig
from opm.io.deck import DeckKeyword
from opm.io.ecl import ESmry
import matplotlib.pyplot as plt
import shutil
import string
import pickle
from .problem import DiscreteTestProblem


class GRRR(DiscreteTestProblem,MultiObjectiveTestProblem):
    decision_vars=3
    steps=160
    dim = 961  # total dimensions
    num_objectives = 4  # number of objectives
    _bounds = [(150,190)]*1 + [(30, 100.0)] * 3*steps + [(1,8)]*3*steps   # bounds for continuous and categorical variables
    _ref_point = [-0.1, -1e09,-1e-05,-1e09]


    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,

    ) -> None:
        MultiObjectiveTestProblem.__init__(
            self,
            noise_std=noise_std,
            negate=negate,
        )
        self._setup(integer_indices=list(range(0,1+6*160)))
        # Call the constructor of the parent class
        super().__init__(negate=negate, noise_std=noise_std,integer_indices=list(range(0,1+6*160)))

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_split = list(torch.split(X, 1, -1))

        parser = Parser()
        steps = 160
        ratio = []
        sequestration = []
        diff = []
        press = []
        for j in range(len(X_split[0])):
            deck = parser.parse('test_functions/deckoldbo.DATA')
            gas_inj = int(X_split[0][j].item())
            gconinje_string = f'''GCONINJE
                'FIELD' 'GAS' 'RATE' {gas_inj}E3 3* 'NO' /
            'Inj' 'GAS' 'RATE' {gas_inj}E3 3* 'NO' /
                 /
            WCONINJE
            'I1' 'GAS' 'OPEN' 'GRUP' 100000 1* 9000 /
            'I2' 'GAS' 'OPEN' 'GRUP' 100000 1* 9000 /
            'I3' 'GAS' 'OPEN' 'GRUP' 100000 1* 9000 /
            /
             '''
            deck = parser.parse_string(str(deck)+gconinje_string)

            for i in range(steps):
                prod_rate1 = int(X_split[i+1][j].item())
                prod_rate2 = int(X_split[i+1+steps][j].item())
                prod_rate3 = int(X_split[i+1+2*steps][j].item())

                gas_prod_rate1 = int(X_split[i+1+3*steps][j].item())
                gas_prod_rate2 = int(X_split[i+1+4*steps][j].item())
                gas_prod_rate3 = int(X_split[i+1+5*steps][j].item())
                wconprod_string = f"""
                WCONPROD
                'P1' 'OPEN' 'ORAT' {prod_rate1}E3 1* {gas_prod_rate1}E3 2* 3000 /
                'P2' 'OPEN' 'ORAT' {prod_rate2}E3 1* {gas_prod_rate2}E3 2* 3000 /
                'P3' 'OPEN' 'ORAT' {prod_rate3}E3 1* {gas_prod_rate3}E3 2* 3000 /

                /"""

                tstep_string = """
                TSTEP\n
                90\n
                /
                """

                deck = parser.parse_string(str(deck)+wconprod_string+tstep_string)
            deck.add(DeckKeyword(parser['END']))
            with open('deck.DATA', 'w') as file:
                file.write(str(deck))

            try:

                subprocess.run(["mpirun", "-np", "16", "--allow-run-as-root", "flow", "deck.DATA", "--output-dir=./i.1047"])
            except:
                pass


            try:

                summary = ESmry('i.1047/DECK.SMSPEC')

                ratio_val =  (summary['FWCD'][-1]+summary['FGCDI'][-1])/summary['FGCDM'][-1]
                obj1 = ratio_val if ratio_val <1 else 0
                obj2 = (summary['FGIT'][-1]-summary['FGPT'][-1])
                weights = 1 / (np.arange(1, len(np.diff(np.array(summary['FGPR']))) + 1))
                obj3 = 1/(np.sum((np.abs(np.diff(np.array(summary['FGIR']))))*weights)+1)
                co2_inflow = 30*0.0019/ 0.035 #30 euro/tonne * tonne/m^3 *m^3/Mscf
                co2_outflow = 6.2*0.0019/ 0.035 # 6.2 euro/tonne ...
                brine_treat = 4.83*1.2*0.159 #4.83 euro/tonne*tonne/m^3 *m^3/stb
                t = np.array([summary['TIME']]).flatten()
                t_int = np.round(t).astype(int)
                t_modified = np.insert(t_int, 0, t_int[-1] - 180)
                intervals = np.diff(t_modified)            
                FWPR = np.repeat(np.array(summary['FOPR']).flatten(), intervals)
                FGPR = np.repeat(np.array(summary['FGPR']).flatten(), intervals)
                FGIR = np.repeat(np.array(summary['FGIR']).flatten(), intervals)
                inflation_rate = 0.0142  # 1.42%
                days = np.arange(t[-1]-180+1,t[-1]+1)

                # Convert annual inflation rate to daily inflation rate
                daily_discount_rate = (1 + inflation_rate) ** (1/365) - 1

                # Calculate daily cash flows
                co2_inflow_cash_flows = (FGIR - FGPR) * co2_inflow
                co2_outflow_cash_flows = FGIR * co2_outflow
                brine_treatment_cash_flows = FWPR * brine_treat

                # Apply inflation adjustment to cash flows
                co2_inflow_present_values = co2_inflow_cash_flows/ (1 + daily_discount_rate) ** days
                co2_outflow_present_values = co2_outflow_cash_flows / (1 + daily_discount_rate) ** days
                brine_treatment_present_values = brine_treatment_cash_flows / (1 + daily_discount_rate) ** days

                # Calculate NPV
                npv = np.sum(co2_inflow_present_values) - np.sum(co2_outflow_present_values) - np.sum(brine_treatment_present_values)

                obj4 = npv

            except:
                obj1 = 0
                obj2 = 0
                obj3 = 0
                obj4 = 0
                ratio.append(obj1)
                sequestration.append(obj2)
                diff.append(obj3)
                press.append(obj4)
                print('hi')
                continue
            ratio.append(obj1)
            sequestration.append(obj2)
            diff.append(obj3)
            press.append(obj4)
            print(obj1,obj2,obj3,obj4)


            shutil.rmtree('./i.1047')
            os.remove('deck.DATA')
        ratio_tensor = torch.tensor(ratio).double().view(-1, 1)  # reshape to [15,1]
        sequestration_tensor = torch.tensor(sequestration).double().view(-1, 1)
        diff_tensor = torch.tensor(diff).double().view(-1, 1)
        press_tensor = torch.tensor(press).double().view(-1, 1)
        print(-torch.cat([ratio_tensor, sequestration_tensor,diff_tensor,press_tensor], dim=-1))

        return -torch.cat([ratio_tensor, sequestration_tensor,diff_tensor,press_tensor], dim=-1)


