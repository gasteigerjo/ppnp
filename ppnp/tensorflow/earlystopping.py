from typing import List
import copy
import operator
from enum import Enum, auto
import numpy as np

from .model import Model


class StopVariable(Enum):
    LOSS = auto()
    ACCURACY = auto()
    F1_SCORE = auto()
    NONE = auto()


class Best(Enum):
    RANKED = auto()
    ALL = auto()


stopping_args = dict(
        stop_varnames=[StopVariable.ACCURACY, StopVariable.LOSS],
        patience=100, max_steps=10000, remember=Best.RANKED)


class EarlyStopping:
    def __init__(
            self, model: Model, stop_varnames: List[StopVariable],
            patience: int = 10, max_steps: int = 200, remember: Best = Best.ALL):
        self.model = model
        self.comp_ops = []
        self.stop_vars = []
        self.best_vals = []
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.LOSS:
                self.stop_vars.append(model.loss)
                self.comp_ops.append(operator.le)
                self.best_vals.append(np.inf)
            elif stop_varname is StopVariable.ACCURACY:
                self.stop_vars.append(model.accuracy)
                self.comp_ops.append(operator.ge)
                self.best_vals.append(-np.inf)
            elif stop_varname is StopVariable.F1_SCORE:
                self.stop_vars.append(model.f1_score)
                self.comp_ops.append(operator.ge)
                self.best_vals.append(-np.inf)
        self.remember = remember
        self.remembered_vals = copy.copy(self.best_vals)
        self.max_patience = patience
        self.patience = self.max_patience
        self.max_steps = max_steps
        self.best_step = None
        self.best_trainables = None

    def check(self, values: List[np.floating], step: int) -> bool:
        checks = [self.comp_ops[i](val, self.best_vals[i])
                  for i, val in enumerate(values)]
        if any(checks):
            self.best_vals = np.choose(checks, [self.best_vals, values])
            self.patience = self.max_patience

            comp_remembered = [
                    self.comp_ops[i](val, self.remembered_vals[i])
                    for i, val in enumerate(values)]
            if self.remember is Best.ALL:
                if all(comp_remembered):
                    self.best_step = step
                    self.remembered_vals = copy.copy(values)
                    self.best_trainables = self.model.get_vars()
            elif self.remember is Best.RANKED:
                for i, comp in enumerate(comp_remembered):
                    if comp:
                        if not(self.remembered_vals[i] == values[i]):
                            self.best_step = step
                            self.remembered_vals = copy.copy(values)
                            self.best_trainables = self.model.get_vars()
                            break
                    else:
                        break
        else:
            self.patience -= 1
        return self.patience == 0
