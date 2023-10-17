from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests

import tkinter as tk
import time
from tkinter import messagebox

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""

    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""

    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3


class LogType(Enum):
    SelectEmpty = 0
    NotAdjacent = 1
    HealAtMax = 2
    TrivialHeal = 3
    EngagedInCombat = 4
    IllegalMove = 5
    OthersTurn = 6
    SelfDestruct = 7
    Heal = 8
    Attack = 9
    Move = 10
    GameEnd = 11


##############################################################################################################


@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################


@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""

    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = "?"
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = "?"
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over the 4adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    def iter__diagonal(self) -> Iterable[Coord]:
        """Iterates over the 4 diagonal Coords."""
        yield Coord(self.row - 1, self.col - 1)
        yield Coord(self.row + 1, self.col - 1)
        yield Coord(self.row - 1, self.col + 1)
        yield Coord(self.row + 1, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 2:
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################


@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""

    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 4:
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None


##############################################################################################################


@dataclass(slots=True)
class Options:
    """Representation of the game options."""

    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None
    attacker_heuristic: str | None = 0
    defender_heuristic: str | None = 0


##############################################################################################################


@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""

    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    branching_factor: list[float] = field(default_factory=list)


##############################################################################################################


@dataclass(slots=True)
class Game:
    """Representation of the game state."""

    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True
    turn_start_time: float = 0.0
    after_half: bool = False
    depth_penalty: int = 0
    def_ai_position: Coord = field(default_factory=Coord)

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        self.reset_board()
        for i in range(self.options.max_depth):
            self.stats.evaluations_per_depth[i] = 0

    def reset_board(self):
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(
            Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall)
        )

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit
            if unit == UnitType.AI and unit.player==Player.Defender:
                self.def_ai_position = coord

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair) -> (bool, LogType):
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            # print("Coords outside board dimensions.")
            return False, None

        src_unit = self.get(coords.src)
        dst_unit = self.get(coords.dst)

        if src_unit is None:
            return False, None

        if src_unit.player != self.next_player:
            return False, None

        if coords.src == coords.dst and src_unit.player == self.next_player:
            # print("Self destruct Valid")
            # print(self.next_player)
            # print(src_unit.player)
            return True, LogType.SelfDestruct

        # Check if the source and destination coordinates are adjacent
        adjacent_coords = list(coords.src.iter_adjacent())
        # print(adjacent_coords)
        if coords.dst not in adjacent_coords:
            # print("Src and dst coords are not adjacent.")
            return False, LogType.NotAdjacent

        # Seems unnecessary
        # if unit is None or unit.player != self.next_player:
        #     print("No unit to move or wrong player's turn")
        #     return False

        if (
            dst_unit is not None
            and self.get(coords.src).player == self.get(coords.dst).player
        ):
            if dst_unit.health == 9:
                # print(f"{dst_unit.type.name} is at maximum health.")
                return False, LogType.HealAtMax
            elif src_unit.repair_table[src_unit.type.value][dst_unit.type.value] <= 0:
                # print(f"{src_unit.type.name} cannot heal {dst_unit.type.name}.")
                return False, LogType.TrivialHeal
            else:
                # print("Healing Valid")
                return True, LogType.Heal

        if (
            dst_unit is not None
            and self.get(coords.src).player != self.get(coords.dst).player
        ):
            # print("Attacking Enemy Valid")
            return True, LogType.Attack

        src_is_ai_firewall_program = (
            src_unit.type.value == 0
            or src_unit.type.value == 3
            or src_unit.type.value == 4
        )

        # Need to verify that src_unit is not engaged in combat for relevant units
        if src_is_ai_firewall_program:
            for adj in adjacent_coords:
                adjacent_unit = self.get(adj)
                if (
                    adjacent_unit is not None
                    and adjacent_unit.player != self.next_player
                ):
                    # print(
                    #     f"{src_unit.type.name} is already engaged in combat. Cannot move."
                    # )
                    return False, LogType.EngagedInCombat

        if dst_unit is None:
            if src_is_ai_firewall_program:
                if src_unit.player.value == 0:
                    if (
                        coords.dst.row > coords.src.row
                        or coords.dst.col > coords.src.col
                    ):
                        # print(
                        #     f"{src_unit.player.name}'s {src_unit.type.name} cannot move that way."
                        # )
                        return False, LogType.IllegalMove
                    else:
                        # print("Move Valid")
                        return True, LogType.Move
                else:
                    if (
                        coords.dst.row < coords.src.row
                        or coords.dst.col < coords.src.col
                    ):
                        # print(
                        #     f"{src_unit.player.name}'s {src_unit.type.name} cannot move that way."
                        # )
                        return False, LogType.IllegalMove
                    else:
                        # print("Move Valid")
                        return True, LogType.Move
            else:
                # print("Move Valid")
                return True, LogType.Move
        else:
            print("Something is wrong")
            return False, None

    def perform_move(self, coords: CoordPair) -> (bool, LogType):
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        is_valid, log_type = self.is_valid_move(coords)
        if is_valid:
            if self.get(coords.dst) is not None:
                if coords.src == coords.dst:
                    # print("Self destruct Action")
                    # Self destroy Code here, self destruction should AOE everything around itself for 2 hp
                    self.mod_health(coords.src, -9)
                    affected_coords = list(coords.src.iter_adjacent()) + list(
                        coords.src.iter__diagonal()
                    )
                    for coord in affected_coords:
                        self.mod_health(coord, -2)

                elif self.get(coords.src).player == self.get(coords.dst).player:
                    # print("Healing Ally Action")
                    # Heal Ally code
                    self.mod_health(
                        coords.dst,
                        self.get(coords.src).repair_amount(self.get(coords.dst)),
                    )

                elif self.get(coords.src).player != self.get(coords.dst).player:
                    # print("Attacking Enemy Action")
                    dmg = -self.get(coords.dst).damage_amount(self.get(coords.src))
                    self.mod_health(
                        coords.dst,
                        -self.get(coords.src).damage_amount(self.get(coords.dst)),
                    )
                    self.mod_health(coords.src, dmg)
                    # Attack and enemy code here
                    # if dead = set coord src to None
                    # self.set(coords.src, None)

            else:
                self.set(coords.dst, self.get(coords.src))
                self.set(coords.src, None)
            return True, log_type
        return False, log_type

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        output += "\n   "
        output += self.get_board_config()
        return output

    def get_board_config(self) -> str:
        dim = self.options.dim
        coord = Coord()
        output = "   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if (
            coord is None
            or coord.row < 0
            or coord.row >= dim
            or coord.col < 0
            or coord.col >= dim
        ):
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(f"Player {self.next_player.name}, enter your move: ")
            coords = CoordPair.from_string(s)
            if (
                coords is not None
                and self.is_valid_coord(coords.src)
                and self.is_valid_coord(coords.dst)
            ):
                return coords
            else:
                print("Invalid coordinates! Try again.")

    def human_turn(self, coord: Coord) -> (bool, LogType):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    success, result = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ", end="")
                    print(result)
                    if success:
                        self.next_turn()
                        return True, result
                        break
                sleep(0.1)
        else:
            while True:
                # mv = self.read_move()
                mv = coord
                success, result = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ", end="")
                    print(result)
                    self.next_turn()
                    return True, result
                    break
                else:
                    print("The move is not valid! Try again.")
                    return False, result

    def computer_turn(self) -> (bool, LogType, CoordPair, float, int | None):
        """Computer plays a move."""
        (mv, time, score) = self.suggest_move()
        print(f"Suggested Move: {mv} + {self.next_player}")
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ", end="")
                print(result)

                self.next_turn()
                print(f"TURN to PLAY: {self.next_player.name}")
        return True, result, mv, time, score

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if (
            self.options.max_turns is not None
            and self.turns_played >= self.options.max_turns
        ):
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates_bad(
        self,
    ) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for src, _ in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def move_candidates(self, current_player) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        # Initialize an empty list to store valid move candidates
        valid_moves = []

        # Iterate through the board and check each cell for valid moves
        for row in range(self.options.dim):
            for col in range(self.options.dim):
                unit = self.board[row][col]
                # Check if the cell contains a unit owned by the current player
                if unit is not None and unit.player == current_player:
                    # Iterate through possible destination coordinates for the unit
                    for dst_coord in Coord(row, col).iter_adjacent():
                        if self.is_valid_coord(dst_coord):
                            # Create a CoordPair representing the move and add it to the list
                            move = CoordPair(src=Coord(row, col), dst=dst_coord)
                            (is_valid, _) = self.is_valid_move(move)
                            if is_valid:
                                valid_moves.append(move)
        # Return the list of valid move candidates
        return valid_moves

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def heuristic_e0(self):
        # Initialize counts for each unit type for both players
        vp1, tp1, fp1, pp1, aip1 = 0, 0, 0, 0, 0
        vp2, tp2, fp2, pp2, aip2 = 0, 0, 0, 0, 0

        # Loop through the game board
        for row in self.board:
            for unit in row:
                if unit is not None and unit.is_alive():
                    if unit.player == Player.Attacker:
                        if unit.type == UnitType.Virus:
                            vp1 += 1
                        elif unit.type == UnitType.Tech:
                            tp1 += 1
                        elif unit.type == UnitType.Firewall:
                            fp1 += 1
                        elif unit.type == UnitType.Program:
                            pp1 += 1
                        elif unit.type == UnitType.AI:
                            aip1 += 1
                    elif unit.player == Player.Defender:
                        if unit.type == UnitType.Virus:
                            vp2 += 1
                        elif unit.type == UnitType.Tech:
                            tp2 += 1
                        elif unit.type == UnitType.Firewall:
                            fp2 += 1
                        elif unit.type == UnitType.Program:
                            pp2 += 1
                        elif unit.type == UnitType.AI:
                            aip2 += 1

            # Calculate the heuristic score formula
        # if current_player == Player.Attacker:
        score = (
            3 * (vp1 + tp1 + fp1 + pp1)
            + 9999 * aip1
            - 3 * (vp2 + tp2 + fp2 + pp2)
            - 9999 * aip2
        )
        # else:
        #     score = (
        #         3 * (vp2 + tp2 + fp2 + pp2)
        #         + 9999 * aip2
        #         - 3 * (vp1 + tp1 + fp1 + pp1)
        #         - 9999 * aip1
        #     )

        # print(f"Next player: {next_player}  heuristic cost at node: {score}")
        return score

    def heuristic_e1(self):
        # Initialize health counts for each unit type for both players
        vp1, tp1, fp1, pp1, aip1 = 0, 0, 0, 0, 0
        vp2, tp2, fp2, pp2, aip2 = 0, 0, 0, 0, 0

        # Loop through the game board
        for row in self.board:
            for unit in row:
                if unit is not None and unit.is_alive():
                    if unit.player == Player.Attacker:
                        if unit.type == UnitType.Virus:
                            vp1 += unit.health
                        elif unit.type == UnitType.Tech:
                            tp1 += unit.health
                        elif unit.type == UnitType.Firewall:
                            fp1 += unit.health
                        elif unit.type == UnitType.Program:
                            pp1 += unit.health
                        elif unit.type == UnitType.AI:
                            aip1 += unit.health
                    elif unit.player == Player.Defender:
                        if unit.type == UnitType.Virus:
                            vp2 += unit.health
                        elif unit.type == UnitType.Tech:
                            tp2 += unit.health
                        elif unit.type == UnitType.Firewall:
                            fp2 += unit.health
                        elif unit.type == UnitType.Program:
                            pp2 += unit.health
                        elif unit.type == UnitType.AI:
                            aip2 += unit.health
        # Calculate the heuristic score formula
        score = (
            (100*vp1 + 100*tp1 + 20*fp1 + 20*pp1 + 1000 * aip1)
            - (100*vp2 + 100*tp2 + 20*fp2 + 20*pp2 + 1000 * aip2)
            + self.get_virus_ai_factor()
        )

        return score

    # Calculates a factor for e1 that tries to make virus closer to opponent's AI
    # and the opponent's AI keep more adjacent units
    def get_virus_ai_factor(self) -> int:
        distances_to_ai = []
        i=0
        for row in self.board:
            j=0
            for unit in row:
                if unit is not None and unit.type == UnitType.Virus:
                    distances_to_ai.append(abs(self.def_ai_position.col-j)+abs(self.def_ai_position.row-i)-1)
                j += 1
            i += 1
        adjacent_units = 0
        adjacent_coords = list(self.def_ai_position.iter_adjacent())
        for coord in adjacent_coords:
            if self.get(coord) is not None and self.get(coord).player == Player.Defender:
                adjacent_units += 1

        return 90*adjacent_units-15*sum(distances_to_ai)

    def heuristic_e2(self, current_player):
        # Define unit values based on game-specific knowledge
        unit_values = {
            UnitType.AI: 10,
            UnitType.Tech: 5,
            UnitType.Firewall: 3,
            UnitType.Program: 2,
            UnitType.Virus: 1,
        }

        # Define positional advantage values for specific positions on the board
        positional_values = [
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 3, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ]

        # Initialize scores for each player
        attacker_score = 0
        defender_score = 0

        # Loop through the game board
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                unit = self.board[row][col]
                if unit is not None and unit.is_alive():
                    if unit.player == Player.Attacker:
                        # Add unit value and positional advantage
                        attacker_score += (
                            unit_values[unit.type] * positional_values[row][col]
                        )
                    elif unit.player == Player.Defender:
                        # Add unit value and positional advantage
                        defender_score += (
                            unit_values[unit.type] * positional_values[row][col]
                        )

        # Consider other factors, such as objectives or long-term goals
        objectives_score = self.evaluate_objectives(current_player)

        # Calculate the heuristic score, considering objectives
        score = attacker_score - defender_score + objectives_score
        # print(f"---------------------------")
        # print(f"Score: {score}")
        # print(f"attacker_score: {score}")
        # print(f"defender_score: {score}")
        # print(f"objectives_score: {score}")

        return score

    def evaluate_objectives(self, current_player):
        # Define your objectives evaluation logic here
        # Calculate the score based on the progress toward achieving objectives
        # Consider factors like capturing key positions, reaching specific conditions, or avoiding enemy AI units
        # Return an objective score, which can be positive or negative based on how well the player is doing

        # Example: Calculate an objective score based on the proximity to enemy AI units
        objective_score = 0

        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                unit = self.board[row][col]
                if unit is not None and unit.is_alive():
                    if unit.player != current_player and unit.type == UnitType.AI:
                        # Calculate the Manhattan distance to this enemy AI unit
                        distance = abs(
                            row - self.get_closest_enemy_ai_row(current_player)
                        )
                        distance += abs(
                            col - self.get_closest_enemy_ai_col(current_player)
                        )

                        # Penalize the player for being closer to an enemy AI unit
                        # Adjust the factor (-5) as needed to control the penalty strength
                        objective_score -= 5 * distance

        return objective_score

    def get_closest_enemy_ai_row(self, current_player):
        # Find the row of the closest enemy AI unit to the current player
        closest_row = None
        closest_distance = float("inf")

        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                unit = self.board[row][col]
                if (
                    unit is not None
                    and unit.is_alive()
                    and unit.player != current_player
                    and unit.type == UnitType.AI
                ):
                    # Calculate the Manhattan distance to this enemy AI unit
                    distance = abs(row - self.get_ai_row(unit))
                    distance += abs(col - self.get_ai_col(unit))

                    # Update the closest enemy AI unit
                    if distance < closest_distance:
                        closest_row = row
                        closest_distance = distance

        return closest_row

    def get_closest_enemy_ai_col(self, current_player):
        # Find the column of the closest enemy AI unit to the current player
        closest_col = None
        closest_distance = float("inf")

        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                unit = self.board[row][col]
                if (
                    unit is not None
                    and unit.is_alive()
                    and unit.player != current_player
                    and unit.type == UnitType.AI
                ):
                    # Calculate the Manhattan distance to this enemy AI unit
                    distance = abs(row - self.get_ai_row(unit))
                    distance += abs(col - self.get_ai_col(unit))

                    # Update the closest enemy AI unit
                    if distance < closest_distance:
                        closest_col = col
                        closest_distance = distance

        return closest_col

    def get_ai_row(self, ai_unit):
        # Get the row of the given AI unit
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                if self.board[row][col] == ai_unit:
                    return row

    def get_ai_col(self, ai_unit):
        # Get the column of the given AI unit
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                if self.board[row][col] == ai_unit:
                    return col

    def minmax(
        self, depth, current_player, next_player
    ) -> Tuple[int, CoordPair | None, float]:
        if (
            depth == 0
            or self._attacker_has_ai == False
            or self._defender_has_ai == False
        ):
            heuristic = (
                self.options.attacker_heuristic
                if current_player.value == 0
                else self.options.defender_heuristic
            )
            # Calculate and return the heuristic value for this node
            if heuristic == 0:
                heuristic_value = self.heuristic_e0()
                return heuristic_value, None
            elif heuristic == 1:
                heuristic_value = self.heuristic_e1()
                return heuristic_value, None
            elif heuristic == 2:
                heuristic_value = self.heuristic_e2(current_player)
                return heuristic_value, None

        move_candidates = self.move_candidates(next_player)
        self.stats.evaluations_per_depth[self.options.max_depth - depth] += sum(
            1 for i in move_candidates
        )
        if not self.after_half:
            if (time.time() - self.turn_start_time) / self.options.max_time >= 0.90:
                self.after_half = True

        if current_player == Player.Attacker:  # (Maximizer)
            max_eval = MIN_HEURISTIC_SCORE
            best_move = None

            for move in move_candidates:
                game_clone = self.clone()
                # Make the move in the copied game state
                (success, result) = game_clone.perform_move(move)
                if success:
                    game_clone.next_turn()

                    # Recursively evaluate the move in the copied game state
                    eval, _ = game_clone.minmax(
                        depth - 1 if depth <= 1 else depth - 1 - self.depth_penalty,
                        current_player,
                        game_clone.next_player,
                    )

                    # Update max_eval and best_move if needed
                    if eval >= max_eval:
                        max_eval = eval
                        best_move = move
                    if self.after_half:
                        if (
                            time.time() - self.turn_start_time
                        ) / self.options.max_time >= 0.98 and best_move != None:
                            break

            return max_eval, best_move

        else:  # (Minimizer)
            min_eval = MAX_HEURISTIC_SCORE
            best_move = None

            for move in move_candidates:
                game_clone = self.clone()
                # Make the move in the copied game state
                (success, result) = game_clone.perform_move(move)
                # print(f"Min: {depth}")
                if success:
                    # Recursively evaluate the move in the copied game state
                    game_clone.next_turn()

                    eval, _ = game_clone.minmax(
                        depth - 1 if depth <= 1 else depth - 1 - self.depth_penalty,
                        current_player,
                        game_clone.next_player,
                    )

                    # Update min_eval and best_move if needed
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    if self.after_half:
                        if (
                            time.time() - self.turn_start_time
                        ) / self.options.max_time >= 0.98 and best_move != None:
                            break

            return min_eval, best_move

    def minmax_alphabeta(
        self, depth, current_player, next_player, alpha, beta
    ) -> Tuple[int, CoordPair | None, float]:
        if (
            depth == 0
            or self._attacker_has_ai == False
            or self._defender_has_ai == False
        ):
            heuristic = (
                self.options.attacker_heuristic
                if current_player.value == 0
                else self.options.defender_heuristic
            )
            # Calculate and return the heuristic value for this node
            if heuristic == 0:
                heuristic_value = self.heuristic_e0()
                return heuristic_value, None
            elif heuristic == 1:
                heuristic_value = self.heuristic_e1()
                return heuristic_value, None
            elif heuristic == 2:
                heuristic_value = self.heuristic_e2(current_player)
                return heuristic_value, None

        move_candidates = self.move_candidates(next_player)
        self.stats.evaluations_per_depth[self.options.max_depth - depth] = sum(
            1 for i in move_candidates
        )
        if not self.after_half:
            if (time.time() - self.turn_start_time) / self.options.max_time >= 0.90:
                self.after_half = True

        if current_player == next_player:  # Maximizer
            max_eval = MIN_HEURISTIC_SCORE
            best_move = None
            for move in move_candidates:
                game_clone = self.clone()
                (success, result) = game_clone.perform_move(move)
                if success:
                    game_clone.next_turn()

                    # Recursively evaluate the move in the copied game state
                    eval, _ = game_clone.minmax_alphabeta(
                        depth - 1 if depth <= 1 else depth - 1 - self.depth_penalty,
                        current_player,
                        game_clone.next_player,
                        alpha,
                        beta,
                    )

                    # Update max_eval and best_move if needed
                    if eval >= max_eval:
                        max_eval = eval
                        best_move = move

                    # Update alpha
                    alpha = max(alpha, eval)

                    # Perform alpha-beta pruning
                    if beta <= alpha:
                        break
                    if self.after_half:
                        if (
                            time.time() - self.turn_start_time
                        ) / self.options.max_time >= 0.98 and best_move != None:
                            break

            return max_eval, best_move

        else:  # Minimizer
            min_eval = MAX_HEURISTIC_SCORE
            best_move = None

            for move in move_candidates:
                game_clone = self.clone()
                (success, result) = game_clone.perform_move(move)
                if success:
                    game_clone.next_turn()

                    # Recursively evaluate the move in the copied game state
                    eval, _ = game_clone.minmax_alphabeta(
                        depth - 1 if depth <= 1 else depth - 1 - self.depth_penalty,
                        current_player,
                        game_clone.next_player,
                        alpha,
                        beta,
                    )

                    # Update min_eval and best_move if needed
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move

                    # Update beta
                    beta = min(beta, eval)

                    # Perform alpha-beta pruning
                    if beta <= alpha:
                        break
                    if self.after_half:
                        if (
                            time.time() - self.turn_start_time
                        ) / self.options.max_time >= 0.98 and best_move != None:
                            break

            return min_eval, best_move

    def suggest_move(self) -> (CoordPair, float, int) | None:
        """Suggest the next move using minimax alpha beta."""
        if self.options.alpha_beta:
            self.turn_start_time = time.time()
            (score, move) = self.minmax_alphabeta(
                self.options.max_depth,
                self.next_player,
                self.next_player,
                MIN_HEURISTIC_SCORE,
                MAX_HEURISTIC_SCORE,
            )
        else:
            self.turn_start_time = time.time()
            (score, move) = self.minmax(
                self.options.max_depth, self.next_player, self.next_player
            )
        elapsed_time = time.time() - self.turn_start_time  # need for log
        self.after_half = False
        self.depth_penalty = 0
        move_candidates = self.move_candidates(self.next_player)
        self.stats.branching_factor.append(sum(1 for i in move_candidates))

        return (move, elapsed_time, score)

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played,
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if (
                r.status_code == 200
                and r.json()["success"]
                and r.json()["data"] == data
            ):
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(
                    f"Broker error: status code: {r.status_code}, response: {r.json()}"
                )
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {"Accept": "application/json"}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()["success"]:
                data = r.json()["data"]
                if data is not None:
                    if data["turn"] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data["from"]["row"], data["from"]["col"]),
                            Coord(data["to"]["row"], data["to"]["col"]),
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(
                    f"Broker error: status code: {r.status_code}, response: {r.json()}"
                )
        except Exception as error:
            print(f"Broker error: {error}")
        return None


##############################################################################################################
class GameGUI:
    def __init__(self, root, game):
        self.root = root
        self.game = game
        self.console = Console(root, self)
        self.buttons = []
        self.selected_coord = None  # Initialize selected_coord attribute
        self.turn_count = 0  # Initialize turn_count
        self.game_label = tk.Label(root, text="AI WARGAME", font=("Arial", 18))
        self.game_label.grid(
            row=0, column=3, columnspan=2, pady=10
        )  # Place the label at the top
        self.turn_label = tk.Label(root, text="Turn: 1\n(Attacker's Turn)")
        self.turn_label.grid(
            row=0, column=6, columnspan=1
        )  # Place the label at the top

        param_font = ("Arial", 10)

        self.init_params_row(root, param_font)

        # Create a variable to store the selected game mode
        self.game_mode = tk.StringVar()
        self.game_mode.set("H-H")  # Initialize with "H-H" selected
        self.init_game_mode_row(root, param_font)

        # Create a variable to store the selected heuristic
        self.attacker_heuristic = tk.StringVar()
        self.attacker_heuristic.set("e0")  # Initialize with "e0" selected
        self.defender_heuristic = tk.StringVar()
        self.defender_heuristic.set("e0")  # Initialize with "e0" selected
        self.init_heuristic_row(root, param_font)

        self.init_board(root, param_font)

    def init_params_row(self, root, param_font):
        max_time_frame = tk.Frame(root)
        max_time_label = tk.Label(
            max_time_frame, text="Max Time (seconds):", font=param_font
        )
        max_time_label.grid(row=0, column=0)
        self.max_time_entry = tk.Entry(max_time_frame)
        self.max_time_entry.insert(
            0, self.game.options.max_time
        )  # Set the default value default of options
        self.max_time_entry.grid(row=0, column=1)
        max_time_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=2)

        max_turns_frame = tk.Frame(root)
        max_turns_label = tk.Label(max_turns_frame, text="Max Turns:", font=param_font)
        max_turns_label.grid(row=0, column=0)
        self.max_turns_entry = tk.Entry(max_turns_frame)
        self.max_turns_entry.insert(
            0, self.game.options.max_turns
        )  # Set the default value default of options
        self.max_turns_entry.grid(row=0, column=1)
        max_turns_frame.grid(row=1, column=2, columnspan=2, padx=5, pady=2)

        max_depth_frame = tk.Frame(root)
        max_depth_label = tk.Label(max_depth_frame, text="Max Depth:", font=param_font)
        max_depth_label.grid(row=0, column=0)
        self.max_depth_entry = tk.Entry(max_depth_frame)
        self.max_depth_entry.insert(
            0, self.game.options.max_depth
        )  # Set the default value default of options
        self.max_depth_entry.grid(row=0, column=1)
        max_depth_frame.grid(row=1, column=3, columnspan=2, pady=2)

        min_depth_frame = tk.Frame(root)
        min_depth_label = tk.Label(min_depth_frame, text="Min Depth:", font=param_font)
        min_depth_label.grid(row=0, column=0)
        self.min_depth_entry = tk.Entry(min_depth_frame)
        self.min_depth_entry.insert(
            0, self.game.options.min_depth
        )  # Set the default value default of options
        self.min_depth_entry.grid(row=0, column=1)
        min_depth_frame.grid(row=1, column=4, columnspan=2, pady=2)

        self.alpha_beta_var = tk.BooleanVar()
        alpha_beta_checkbox = tk.Checkbutton(
            root, text="Alpha-Beta", font=param_font, variable=self.alpha_beta_var
        )
        alpha_beta_checkbox.grid(row=1, column=5, pady=5)
        self.alpha_beta_var.set(self.game.options.alpha_beta)  # Default to Alpha-Beta

    def init_game_mode_row(self, root, param_font):
        # Create radio buttons for game mode selection
        game_mode_frame = tk.Frame(root)
        h_h_radio = tk.Radiobutton(
            game_mode_frame,
            text="H-H",
            variable=self.game_mode,
            value="H-H",
            font=param_font,
        )
        h_h_radio.grid(row=0, column=0, padx=5)

        h_ai_radio = tk.Radiobutton(
            game_mode_frame,
            text="H-AI",
            variable=self.game_mode,
            value="H-AI",
            font=param_font,
        )
        h_ai_radio.grid(row=0, column=1, padx=5)

        h_ai_radio = tk.Radiobutton(
            game_mode_frame,
            text="AI-H",
            variable=self.game_mode,
            value="AI-H",
            font=param_font,
        )
        h_ai_radio.grid(row=0, column=2, padx=5)

        ai_ai_radio = tk.Radiobutton(
            game_mode_frame,
            text="AI-AI",
            variable=self.game_mode,
            value="AI-AI",
            font=param_font,
        )
        ai_ai_radio.grid(row=0, column=3, padx=5)

        game_mode_frame.grid(row=2, column=0, columnspan=3)

    def init_heuristic_row(self, root, param_font):
        # Create radio buttons for game mode selection
        heuristic_frame = tk.Frame(root)
        for i in range(2):
            label = tk.Label(
                heuristic_frame,
                text="Attacker:" if i == 0 else "Defender:",
                font=param_font,
            )
            label.grid(row=i, column=0, padx=5)
            e_0_radio = tk.Radiobutton(
                heuristic_frame,
                text="e0",
                variable=self.attacker_heuristic if i == 0 else self.defender_heuristic,
                value="e0",
                font=param_font,
            )
            e_0_radio.grid(row=i, column=1, padx=5)

            e_1_radio = tk.Radiobutton(
                heuristic_frame,
                text="e1",
                variable=self.attacker_heuristic if i == 0 else self.defender_heuristic,
                value="e1",
                font=param_font,
            )
            e_1_radio.grid(row=i, column=2, padx=5)

            e_2_radio = tk.Radiobutton(
                heuristic_frame,
                text="e2",
                variable=self.attacker_heuristic if i == 0 else self.defender_heuristic,
                value="e2",
                font=param_font,
            )
            e_2_radio.grid(row=i, column=3, padx=5)

        heuristic_frame.grid(row=2, column=4, columnspan=3, pady=5)

    def init_board(self, root, param_font):
        restart_button = tk.Button(
            root, text="Restart Game", font=param_font, command=self.restart_game
        )
        restart_button.grid(row=3, column=0)

        board_font = ("Arial", 12)
        header_color = "lightblue"
        row_shift = 5
        col_shift = 2

        # Create board column header
        for col in range(col_shift, 5 + col_shift):
            button = tk.Label(
                root,
                width=15,
                height=3,
                text=col - col_shift,
                font=board_font,
                bg=header_color,
            )
            button.grid(row=3, column=col)

        # Create board row header
        alphabet = ["A", "B", "C", "D", "E"]
        for row in range(row_shift, 5 + row_shift):
            button = tk.Label(
                root,
                width=14,
                height=4,
                text=alphabet[row - row_shift],
                font=board_font,
                bg=header_color,
            )
            button.grid(row=row, column=0)

        # Create a 5x5 grid of buttons
        for row in range(row_shift, 5 + row_shift):
            button_row = []
            for col in range(col_shift, 5 + col_shift):
                button = tk.Button(
                    root,
                    width=14,
                    height=3,
                    font=board_font,
                    command=lambda row=row, col=col: self.on_button_click(
                        row - row_shift, col - col_shift
                    ),
                )
                button.grid(row=row, column=col)
                button_row.append(button)
            self.buttons.append(button_row)

        self.update_buttons()

    def restart_game(self):
        self.selected_coord = None
        self.game.turns_played = 0
        self.turn_count = 0
        self.game.next_player = Player.Attacker
        self.game._attacker_has_ai = True
        self.game._defender_has_ai = True

        # Read values from entry widgets
        max_time = float(self.max_time_entry.get())
        max_turns = int(self.max_turns_entry.get())
        max_depth = int(self.max_depth_entry.get())
        min_depth = int(self.min_depth_entry.get())
        alpha_beta = bool(self.alpha_beta_var.get())  # Retrieve alpha_beta_var value

        # Set options in the game.options object
        self.game.options.max_time = max_time
        self.game.options.max_turns = max_turns
        self.game.options.max_depth = max_depth
        self.game.options.min_depth = min_depth
        self.game.options.alpha_beta = alpha_beta  # Set alpha_beta

        print(self.game.options.max_turns)

        self.game.reset_board()
        self.update_turn_label()
        self.update_buttons()
        self.console.logs.config(state=tk.NORMAL)
        self.console.logs.delete("1.0", tk.END)
        self.console.logs.config(state=tk.DISABLED)
        self.computer_options()
        self.heuristic_options()
        self.console.create_initial_log()
        for i in range(self.game.options.max_depth):
            self.game.stats.evaluations_per_depth[i] = 0
        self.game.stats.branching_factor.clear()

    def computer_options(self):
        selected_mode = self.game_mode.get()
        if selected_mode == "H-H":
            # Start the game in H-H mode
            msg = "Starting Game in: " + selected_mode
            self.manual_entry()
        elif selected_mode == "H-AI":
            # Start the game in H-AI mode
            msg = "Starting Game in: " + selected_mode
            print(msg)
            # self.insert_in_log(msg)
            self.manual_vs_ai()
        elif selected_mode == "AI-H":
            # Start the game in H-AI mode
            msg = "Starting Game in: " + selected_mode
            print(msg)
            # self.insert_in_log(msg)
            self.ai_vs_manual()
        elif selected_mode == "AI-AI":
            # Start the game in AI-AI mode
            msg = "Starting Game in: " + selected_mode
            print(msg)
            # self.insert_in_log(msg)
            self.ai_vs_ai()

    def heuristic_options(self):
        for i in range(2):
            selected_heuristic = (
                self.attacker_heuristic.get()
                if i == 0
                else self.defender_heuristic.get()
            )
            if selected_heuristic == "e0":
                msg = "Heuristic: " + selected_heuristic
                if i == 0:
                    self.game.options.attacker_heuristic = 0
                else:
                    self.game.options.defender_heuristic = 0
            elif selected_heuristic == "e1":
                msg = "Heuristic: " + selected_heuristic
                print(msg)
                if i == 0:
                    self.game.options.attacker_heuristic = 1
                else:
                    self.game.options.defender_heuristic = 1
            elif selected_heuristic == "e2":
                # Start the game in H-AI mode
                msg = "Heuristic: " + selected_heuristic
                print(msg)
                if i == 0:
                    self.game.options.attacker_heuristic = 2
                else:
                    self.game.options.defender_heuristic = 2

    def manual_entry(self):
        # Implement manual entry logic here
        self.game.options.game_type = GameType.AttackerVsDefender
        # print(self.game.options.game_type)

    def manual_vs_ai(self):
        # Implement manual vs AI logic here
        self.game.options.game_type = GameType.AttackerVsComp
        # print(self.game.options.game_type)

    def ai_vs_manual(self):
        # Implement manual vs AI logic here
        self.game.options.game_type = GameType.CompVsDefender
        success2, result2, coord2, time, score = self.game.computer_turn()
        self.game_AI_turn_function(success2, result2, coord2, time, score)
        print(self.game.options.game_type)

    def ai_vs_ai(self):
        # Initialize the game for AI vs AI
        self.game.options.game_type = GameType.CompVsComp
        print("Starting Game in AI-AI mode")

        # Schedule the first AI turn
        self.root.after(100, self.ai_turn)  # 100ms delay, you can adjust this

    def ai_turn(self):
        if (
            not self.game.is_finished()
            and self.turn_count < self.game.options.max_turns
        ):
            success, result, coord, time, score = self.game.computer_turn()
            if success:
                self.turn_count += 1  # Increment turn count
                self.update_buttons()
                self.update_turn_label()  # Update the turn label
                self.console.create_log(result, coord.dst, coord.src)
                self.console.create_ai_log(time, score)
                self.root.after(100, self.ai_turn)  # Schedule the next turn
            else:
                self.console.create_log(result, coord.dst, coord.src)
                self.console.create_ai_log(time, score)
                print("Computer doesn't know what to do")
        elif self.game.is_finished():
            winner = self.game.has_winner()
            self.console.create_log(LogType.GameEnd, None, None)
            messagebox.showinfo("Game Over", f"{winner.name} wins!")

    def update_buttons(self):
        for row in range(5):
            for col in range(5):
                unit = self.game.get(Coord(row, col))
                text = ""
                color = "SystemButtonFace"
                if unit:
                    text = f"{unit.type.name}\n{unit.health}"
                    if unit.player == Player.Defender:
                        color = "green"
                    if unit.player == Player.Attacker:
                        color = "red"
                self.buttons[row][col].config(text=text)
                self.buttons[row][col].config(bg=color)

    def on_button_click(self, row, col):
        coord = Coord(row, col)
        unit = self.game.get(coord)

        if unit is not None and unit.player == self.game.next_player:
            if self.selected_coord is None:  # Selecting unit
                self.selected_coord = coord
                self.buttons[row][col].config(bg="yellow")
            else:  # Healing or self-destruct unit
                move = CoordPair(self.selected_coord, coord)
                success, result = self.game.human_turn(move)
                self.game_manual_turn_function(success, result, coord)
                if success and (
                    self.game.options.game_type == GameType.AttackerVsComp
                    or self.game.options.game_type == GameType.CompVsDefender
                ):
                    success2, result2, coord2, time, score = self.game.computer_turn()
                    self.game_AI_turn_function(success2, result2, coord2, time, score)
        elif self.selected_coord is not None:  # Attacking enemy unit or moving
            move = CoordPair(self.selected_coord, coord)
            success, result = self.game.human_turn(move)
            self.game_manual_turn_function(success, result, coord)
            if success and (
                self.game.options.game_type == GameType.AttackerVsComp
                or self.game.options.game_type == GameType.CompVsDefender
            ):
                success2, result2, coord2, time, score = self.game.computer_turn()
                self.game_AI_turn_function(success2, result2, coord2, time, score)
        else:
            if unit is not None:
                self.console.create_log(LogType.OthersTurn, coord, self.selected_coord)
            else:
                self.console.create_log(LogType.SelectEmpty, coord, self.selected_coord)
            self.reset_turn(LogType.SelectEmpty)

    def game_manual_turn_function(self, success, result, coord):
        if success:
            self.update_buttons()
            self.turn_count += 1  # Increment turn count
            self.update_turn_label()  # Update the turn label
            self.console.create_log(result, coord, self.selected_coord)
            self.selected_coord = None
            if self.game.has_winner() is not None:
                winner = self.game.has_winner()
                self.console.create_log(LogType.GameEnd, coord, self.selected_coord)
                messagebox.showinfo("Game Over", f"{winner.name} wins!")
        else:
            self.console.create_log(result, coord, self.selected_coord)
            self.reset_turn(result)

    def game_AI_turn_function(self, success, result, coord, time, score):
        if success:
            self.update_buttons()
            self.turn_count += 1  # Increment turn count
            self.update_turn_label()  # Update the turn label
            self.console.create_log(result, coord.dst, coord.src)
            self.console.create_ai_log(time, score)
        else:
            self.console.create_log(result, coord.dst, coord.src)
            print("Computer doesnt know what to do")

    def update_turn_label(self):
        # Update the turn label with the current turn count and player's turn
        player_turn = (
            "Attacker's Turn"
            if self.game.next_player == Player.Attacker
            else "Defender's Turn"
        )
        self.turn_label.config(text=f"Turn: {self.turn_count+1}\n({player_turn})")

    def reset_turn(self, log_type):
        if log_type.value < 6:
            if self.game.get(self.selected_coord).player.value == 0:
                self.buttons[self.selected_coord.row][self.selected_coord.col].config(
                    bg="red"
                )
            else:
                self.buttons[self.selected_coord.row][self.selected_coord.col].config(
                    bg="green"
                )
        self.selected_coord = None


##############################################################################################################


##############################################################################################################
class Console:
    game_gui: GameGUI
    logs: tk.Text
    log_console_frame: tk.Frame

    def __init__(self, root, game_gui):
        self.game_gui = game_gui

        download_logs_button = tk.Button(
            root,
            text="Download Logs",
            font=("Arial", 10),
            command=self.download_logs,
        )
        download_logs_button.grid(row=0, column=7)

        self.log_console_frame = tk.Frame(root, width=25)
        scrollbar = tk.Scrollbar(self.log_console_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        self.logs = tk.Text(
            self.log_console_frame,
            width=50,
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED,
        )
        self.logs.pack()
        scrollbar.config(command=self.logs.yview)
        self.create_initial_log()
        self.log_console_frame.grid(row=0, column=7, rowspan=9)

    def create_initial_log(self):
        # TODO for D2: add if/else to add AI-specific params
        attacker = ""
        defender = ""
        match self.game_gui.game.options.game_type:
            case GameType.AttackerVsDefender:
                attacker = "H"
                defender = "H"
            case GameType.CompVsComp:
                attacker = "AI"
                defender = "AI"
            case GameType.AttackerVsComp:
                attacker = "H"
                defender = "AI"
            case GameType.CompVsDefender:
                attacker = "AI"
                defender = "H"

        msg = f"Timeout: {self.game_gui.game.options.max_time} s\nMax Turns: {self.game_gui.game.options.max_turns}\nGame Type: {self.game_gui.game.options.game_type.name}\nMax Depth: {self.game_gui.game.options.max_depth}\nMin Depth: {self.game_gui.game.options.min_depth}\nAlpha-Beta: {self.game_gui.game.options.alpha_beta}\nHeuristics: e{self.game_gui.game.options.attacker_heuristic}(Attacker) e{self.game_gui.game.options.defender_heuristic}(Defender) \n\n Attacker: {attacker} Defender: {defender}\n\n{self.game_gui.game.get_board_config()}"
        self.insert_in_log(msg)

    def create_log(self, log_type, coord_dst, coord_src):
        msg = ""
        selected_unit = ""
        if log_type.value != 11:
            selected_unit = self.game_gui.game.get(coord_src)
            if log_type.value <= 6:
                msg += f"(Turn #{self.game_gui.turn_count+1}) {self.game_gui.game.next_player.name}: Invalid Move - "
            else:
                msg += f"-----------------------------------\n(Turn #{self.game_gui.turn_count}) {self.game_gui.game.next_player.next().name}: "
        affected_unit = self.game_gui.game.get(coord_dst)

        match log_type.value:
            case 0:
                msg += f"There is no unit on {coord_dst}."
            case 1:
                msg += f"{coord_dst} is not adjacent to {coord_src}."
            case 2:
                msg += (
                    f"{affected_unit.type.name} ({coord_dst}) is already at max health."
                )
            case 3:
                msg += f"A {affected_unit.type.name} cannot be healed by a {selected_unit.type.name}."
            case 4:
                msg += f"{selected_unit.type.name} ({coord_src}) is already engaged in combat."
            case 5:
                msg += f"{selected_unit.player.name}'s {selected_unit.type.name} cannot move in that direction."
            case 6:
                msg += f"You can not select the other's player unit."
            case 7:
                msg += f"Unit on {coord_dst} self_desctructed."
            case 8:
                msg += f"{selected_unit.type.name} ({coord_src}) healed {affected_unit} ({coord_dst})."
            case 9:
                msg += f"Attack from {coord_src} to {coord_dst}."
            case 10:
                msg += (
                    f"{affected_unit.type.name} moved from {coord_src} to {coord_dst}."
                )
            case 11:
                msg += f"{self.game_gui.game.has_winner().name} wins in {self.game_gui.turn_count} turns!"
            case _:
                msg += f"Wrong log type was passed."

        if log_type.value <= 6:
            messagebox.showerror("Invalid Move", msg)
        else:
            msg += f"\n{self.game_gui.game.get_board_config()}"
        self.insert_in_log(msg)
        if log_type.value == 11:
            self.download_logs()

    def create_ai_log(self, elapsed_time, score):
        evals_per_depth = self.game_gui.game.stats.evaluations_per_depth
        msg = f"Action Time={round(elapsed_time,2)}s, Heuristic Score={self.game_gui.game.heuristic_e0()}(e0)/{self.game_gui.game.heuristic_e1()}(e1)\n"
        msg += f"Cumul. Eval.: {sum(evals_per_depth.values())}\n"
        msg += "Cumul. Eval. by depth: "
        for k in sorted(evals_per_depth.keys()):
            msg += f"[{k+1}]={evals_per_depth[k]}; "
        msg += "\nCumul. Percent. Eval. by depth: "
        for k in sorted(evals_per_depth.keys()):
            msg += f"[{k+1}]={round((100*evals_per_depth[k]/float(sum(evals_per_depth.values()))), 2)}%; "
        msg += f"\nAvg. Branching Factor: {round((sum(self.game_gui.game.stats.branching_factor)/float(self.game_gui.game.turns_played)), 2)}\n-----------------------------------"

        print(msg)
        self.insert_in_log(msg)

    def insert_in_log(self, msg):
        self.logs.config(state=tk.NORMAL)
        self.logs.insert(tk.END, f"{msg}\n\n")
        self.logs.see(tk.END)
        self.logs.config(state=tk.DISABLED)

    def download_logs(self):
        file = open(
            f"gameTrace-{self.game_gui.game.options.alpha_beta}-{self.game_gui.game.options.max_time}-{self.game_gui.game.options.max_turns}.txt",
            "w",
        )
        file.write(self.logs.get("1.0", tk.END))
        file.close()
        messagebox.showinfo("Info", "Game trace downloaded successfully.")


##############################################################################################################


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog="ai_wargame", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--max_depth", type=int, help="maximum search depth")
    parser.add_argument("--max_time", type=float, help="maximum search time")
    parser.add_argument(
        "--game_type",
        type=str,
        default="manual",
        help="game type: auto|attacker|defender|manual",
    )
    parser.add_argument("--broker", type=str, help="play via a game broker")
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker

    # create a new game
    root = tk.Tk()
    root.title("Game GUI")
    game = Game(options=options)
    game_gui = GameGUI(root, game)
    root.mainloop()

    # the skeleton main game loop
    # while True:
    #     print()
    #     print(game)
    #     winner = game.has_winner()
    #     if winner is not None:
    #         print(f"{winner.name} wins!")
    #         break
    #     if game.options.game_type == GameType.AttackerVsDefender:
    #         game.human_turn()
    #     elif (
    #         game.options.game_type == GameType.AttackerVsComp
    #         and game.next_player == Player.Attacker
    #     ):
    #         game.human_turn()
    #     elif (
    #         game.options.game_type == GameType.CompVsDefender
    #         and game.next_player == Player.Defender
    #     ):
    #         game.human_turn()
    #     else:
    #         player = game.next_player
    #         move = game.computer_turn()
    #         if move is not None:
    #             game.post_move_to_broker(move)
    #         else:
    #             print("Computer doesn't know what to do!!!")
    #             exit(1)


##############################################################################################################

if __name__ == "__main__":
    main()
