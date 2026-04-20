#!/usr/bin/env python3
from typing import Optional
import subprocess
import os
import shutil

import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.base import MultiObjectiveTestProblem
from opm.io.parser import Parser
from opm.io.deck import DeckKeyword
from opm.io.ecl import ESmry

from .problem import DiscreteTestProblem


class GRRR1(DiscreteTestProblem):
    steps = 1
    dim = 16
    num_objectives = 1
    _bounds = [(30, 100.0)] * 8 + [(1, 8)] * 8

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        integer_indices = list(range(16))

        # Pre-populate BoTorch's expected partition so validation passes
        self.continuous_inds = []
        self.discrete_inds = integer_indices
        self.categorical_inds = []

        # Now run the parent init
        super().__init__(
            negate=negate,
            noise_std=noise_std,
            integer_indices=integer_indices,
        )

        # Now that Module.__init__ and bounds exist, run your legacy setup
        self._setup(integer_indices=integer_indices)
        super().__init__(
            negate=negate,
            noise_std=noise_std,
            integer_indices=list(range(16)),
        )

    def _evaluate_true(self, X: Tensor) -> Tensor:
        X_split = list(torch.split(X, 1, -1))
        parser = Parser()
        objective_values = []

        for j in range(len(X_split[0])):
            deck = parser.parse("test_functions/deckoldbo2.DATA")

            gconinje_string = """GCONINJE
                'FIELD' 'GAS' 'RATE' 170E3 3* 'NO' /
                'Inj'   'GAS' 'RATE' 170E3 3* 'NO' /
            /
            WCONINJE
                'I1' 'GAS' 'OPEN' 'GRUP' 100000 1* 9000 /
                'I2' 'GAS' 'OPEN' 'GRUP' 100000 1* 9000 /
                'I3' 'GAS' 'OPEN' 'GRUP' 100000 1* 9000 /
            /
            """
            deck = parser.parse_string(str(deck) + gconinje_string)

            prod_rate = [int(X_split[i][j].item()) for i in range(8)]
            gas_prod_rate = [int(X_split[i + 8][j].item()) for i in range(8)]

            wconprod_string = "WCONPROD\n"
            for i in range(8):
                wconprod_string += (
                    f"'P{i+1}' 'OPEN' 'ORAT' {prod_rate[i]}E3 1* {gas_prod_rate[i]}E3 2* 3000 /\n"
                )
            wconprod_string += "/\n"

            tstep_string = """
            TSTEP
            160*90
            /
            """

            deck = parser.parse_string(str(deck) + wconprod_string + tstep_string)
            deck.add(DeckKeyword(parser["END"]))

            with open("deck.DATA", "w") as file:
                file.write(str(deck))

            try:
                subprocess.run(
                    [
                        "mpirun", "-np", "16", "--allow-run-as-root",
                        "flow", "deck.DATA", "--output-dir=./i.1048"
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,  # <--- disable stdout
                    stderr=subprocess.DEVNULL   # <--- disable stderr
                )

                summary = ESmry("i.1048/DECK.SMSPEC")
                fgir = np.asarray(summary["FGIR"], dtype=float)
                fgpr = np.asarray(summary["FGPR"], dtype=float)

                seq = np.abs(fgir - fgpr)
                if len(seq) <= 1:
                    obj = 0.0
                else:
                    weights = 1.0 / np.arange(1, len(seq), dtype=float)
                    obj = 1.0 / (np.sum(np.abs(np.diff(seq)) * weights) + 1.0)
            except Exception:
                obj = 0.0
            finally:
                if os.path.isdir("./i.1048"):
                    shutil.rmtree("./i.1048", ignore_errors=True)
                if os.path.exists("deck.DATA"):
                    os.remove("deck.DATA")

            objective_values.append(obj)

        return torch.tensor(objective_values, dtype=torch.float64)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self._evaluate_true(X)