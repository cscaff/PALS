"""A tiny explicit-graph Environment for unit-testing the shielding modules."""

from pals.envs.base import Action, Environment, Player


class GraphEnv(Environment):
    """An environment defined by explicit tables.

    ``players[s]`` is the player to move; ``edges[s][a]`` is the successor;
    ``terminals`` are leaves. ``p1_alphabet`` is unused by the safety modules.
    """

    def __init__(self, start, players, edges, terminals):
        self.start = start
        self.players = players
        self.edges = edges
        self.terminals = set(terminals)

    @property
    def p1_alphabet(self) -> list[Action]:
        actions: set[Action] = set()
        for table in self.edges.values():
            actions.update(table)
        return sorted(actions)

    def initial_state(self):
        return self.start

    def current_player(self, state) -> Player:
        return self.players[state]

    def legal_actions(self, state) -> list[Action]:
        if state in self.terminals:
            return []
        return list(self.edges.get(state, {}))

    def step(self, state, action):
        return self.edges[state][action]

    def is_terminal(self, state) -> bool:
        return state in self.terminals

    def reward(self, state) -> float:
        return 0.0
