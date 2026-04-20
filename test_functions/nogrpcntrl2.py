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


class GRRR2(DiscreteTestProblem,MultiObjectiveTestProblem):
    steps = 1
    dim = 16  # total dimensions
    num_objectives = 2  # number of objectives
    _bounds =   [(30, 100.0)] * 8  + [(1, 8)] * 8   # bounds for continuous and categorical variables
    _ref_point = [-1e-04,-1e-04]
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
        self._setup(integer_indices=list(range(0,16)))
        # Call the constructor of the parent class
        super().__init__(negate=negate, noise_std=noise_std,integer_indices=list(range(0,16)))

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_split = list(torch.split(X, 1, -1))

        parser = Parser()
        steps = 1
        sequestration = []
        diff = []

        for j in range(len(X_split[0])):
            deck = parser.parse('test_functions/deckoldbo2.DATA')
            gconinje_string = f'''GCONINJE
                'FIELD' 'GAS' 'RATE' 170E3 3* 'NO' /
            'Inj' 'GAS' 'RATE' 170E3 3* 'NO' /
                 /
            WCONINJE
            'I1' 'GAS' 'OPEN' 'GRUP' 100000 1* 9000 /
            'I2' 'GAS' 'OPEN' 'GRUP' 100000 1* 9000 /
            'I3' 'GAS' 'OPEN' 'GRUP' 100000 1* 9000 /
            /
             '''
            deck = parser.parse_string(str(deck)+gconinje_string)


            prod_rate = [int(X_split[i][j].item()) for i in range(8)]

            gas_prod_rate = [int(X_split[i+8][j].item()) for i in range(8)]
            wconprod_string = "WCONPROD\n"

            # Iterate over the lists prod_rate and gas_prod_rate
            for i in range(8):
                prod_rate_i = prod_rate[i]
                gas_prod_rate_i = gas_prod_rate[i]
                
                # Generate the string for each set of rates
                wconprod_string += f"'P{i+1}' 'OPEN' 'ORAT' {prod_rate_i}E3 1* {gas_prod_rate_i}E3 2* 3000 /\n"

            wconprod_string += "/"
            tstep_string = """
            TSTEP\n
            160*90\n
            /
            """

            deck = parser.parse_string(str(deck)+wconprod_string+tstep_string)
            deck.add(DeckKeyword(parser['END']))
            with open('deck.DATA', 'w') as file:
                file.write(str(deck))

            try:

                subprocess.run(["mpirun", "-np", "4", "--allow-run-as-root", "flow", "deck.DATA", "--output-dir=./i.1048"])
            except:
                pass


            try:

                summary = ESmry('i.1048/DECK.SMSPEC')


                obj2 = (summary['FGIT'][-1]-summary['FGPT'][-1])
                weights = 1 / (np.arange(1, len(np.diff(np.array(summary['FGPR']))) + 1))
                seq = [ np.abs(a-b) for a,b in zip(summary['FGIR'],summary['FGPR'])]
                obj3 = 1/(np.sum((np.abs(np.diff(np.array(seq))))*weights)+1)

            except:
                obj2 = 0
                obj3=0

                sequestration.append(obj2)
                diff.append(obj3)

                print('hi')
                continue
            sequestration.append(obj2)
            diff.append(obj3)



            shutil.rmtree('./i.1048')
            os.remove('deck.DATA')
        sequestration_tensor = torch.tensor(sequestration).double().view(-1, 1)
        diff_tensor = torch.tensor(diff).double().view(-1, 1)

        return -torch.cat([diff_tensor,sequestration_tensor], dim=-1)


