import itertools
import pathlib
import random
from typing import Dict, Set, Text

import numpy as np

_State = int
_Label = Text
_StateTransitions = Dict[_Label, _State]

END_OF_SEQUENCE = "#"


class DFA:
    def __init__(
        self,
        transitions: Dict[_State, _StateTransitions],
        accepting_states=Set[_State],
    ):
        self._states: Set[_State] = set(transitions.keys()) | set(
            itertools.chain(*[tuple(x.values()) for x in transitions.values()])
        )
        self._transitions: Dict[_State, _StateTransitions] = transitions
        self._accepting_states: Set[_State] = accepting_states

    def generate_string(self):
        curr_state = 0
        string = ""
        while curr_state not in self._accepting_states:
            char, curr_state = random.choice(
                tuple(self._transitions[curr_state].items())
            )
            string += char
        return string

    def visualize(self, name: Text):
        dot = "digraph G {\n" "colorscheme=X11\n"
        # inputs and outputs
        for state in sorted(self._states):
            if state in self._accepting_states:
                style = "peripheries=2"
            else:
                style = ""
            description = f'[label="q{state}" {style}]'

            dot += f"{state} {description}\n"

        for state, transitions in self._transitions.items():
            for label, neighbor in transitions.items():
                dot += f'{state} -> {neighbor} [ label="{label}" ];\n'

        dot += "}"

        path = pathlib.Path(f"dot_files/dfa_{name}.dot")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(dot)

    def get_optimal_data_given_grammar_for_dfa(
        self, input_sequence: np.ndarray
    ) -> float:
        total_d_g = 0.0
        curr_state = 0

        for b in range(input_sequence.shape[0]):
            for i in range(input_sequence.shape[1]):
                curr_vec = input_sequence[b, i]
                if np.all(np.isnan(curr_vec)):
                    # Sequence is masked until its end.
                    break
                if curr_vec.shape[0] == 1:
                    curr_val = curr_vec[0]
                else:
                    curr_val = curr_vec.argmax()

                curr_transitions = self._transitions[curr_state]
                total_d_g += -np.log2(1 / len(curr_transitions))

                curr_char = {0: "0", 1: "1", 2: END_OF_SEQUENCE}[curr_val]
                curr_state = curr_transitions[curr_char]

                if curr_state in self._accepting_states:
                    curr_state = 0

        return total_d_g
