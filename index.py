from tkinter import *
from tkinter import messagebox as tkMessageBox
from tkinter import ttk
from collections import deque
import random
import platform
import time
from datetime import datetime
import json
import os
from abc import ABC, abstractmethod

# Difficulty settings
DIFFICULTIES = {
    "Beginner": {"size_x": 9, "size_y": 9, "mines": 10},
    "Intermediate": {"size_x": 16, "size_y": 16, "mines": 40},
    "Expert": {"size_x": 16, "size_y": 30, "mines": 99},
    "Custom": {"size_x": 10, "size_y": 10, "mines": 15}
}
class AIStrategy(ABC):
    @abstractmethod
    def next_move(self, game):
        # return ('click', 0, 0)  # Default action, should be overridden by subclasses
        pass

class MinesweeperAI:
    def __init__(self, game, strategy: AIStrategy):
        self.game = game
        self.strategy = strategy
        self.actions_taken = []

    def set_strategy(self, strategy):
        """Set a new strategy for the AI."""
        self.strategy = strategy
        self.actions_taken.clear()

    def play(self):
        """Execute the next move based on the current strategy."""
        if not self.strategy:
            return False
            
        move = self.strategy.next_move(self.game)
        if move:
            action, x, y = move
            if action == 'click':
                self.game.on_click(self.game.tiles[x][y])
            elif action == 'flag':
                self.game.on_right_click(self.game.tiles[x][y])
            self.actions_taken.append(move)
            return True
        return False        

    def reset(self):
        self.actions_taken = []

class RandomStrategy(AIStrategy):
    def next_move(self, game):
        # Randomly select a tile to click
        unclicked_tiles = [(x, y) for x in range(game.size_x) for y in range(game.size_y)
                           if game.tiles[x][y]["state"] == STATE_DEFAULT]
        if unclicked_tiles:
            x, y = random.choice(unclicked_tiles)
            return ('click', x, y)
        return None

class AutoOpenStrategy(AIStrategy):
    def next_move(self, game):
        # Tìm ô số đã mở, đủ flag quanh nó
        for x in range(game.size_x):
            for y in range(game.size_y):
                t = game.tiles[x][y]
                if t["state"] == STATE_CLICKED and t["mines"] > 0:
                    neigh = game.get_neighbors(x, y)
                    flag_cnt = sum(1 for n in neigh if n["state"] == STATE_FLAGGED)
                    if flag_cnt == t["mines"]:
                        # mở các ô chưa mở quanh nó
                        for n in neigh:
                            if n["state"] == STATE_DEFAULT:
                                return ("click", n["coords"]["x"], n["coords"]["y"])
        
        # If no auto-open moves available, try to flag obvious mines
        for x in range(game.size_x):
            for y in range(game.size_y):
                t = game.tiles[x][y]
                if t["state"] == STATE_CLICKED and t["mines"] > 0:
                    neigh = game.get_neighbors(x, y)
                    unflagged_default = [n for n in neigh if n["state"] == STATE_DEFAULT]
                    flagged_count = sum(1 for n in neigh if n["state"] == STATE_FLAGGED)
                    
                    # If remaining unflagged tiles equals remaining mines, flag them all
                    if len(unflagged_default) > 0 and len(unflagged_default) == (t["mines"] - flagged_count):
                        n = unflagged_default[0]
                        return ("flag", n["coords"]["x"], n["coords"]["y"])
        
        # If no strategic moves, make a random safe click
        unclicked_tiles = [(x, y) for x in range(game.size_x) for y in range(game.size_y)
                           if game.tiles[x][y]["state"] == STATE_DEFAULT]
        if unclicked_tiles:
            x, y = random.choice(unclicked_tiles)
            return ('click', x, y)
        
        return None

class ProbabilisticStrategy(AIStrategy):
    def next_move(self, game):
        prob_map = {}  # (x,y) -> estimated mine probability

        # 1. Duyệt qua tất cả ô số đã mở
        for x in range(game.size_x):
            for y in range(game.size_y):
                t = game.tiles[x][y]
                if t["state"] == STATE_CLICKED and t["mines"] > 0:
                    neigh = game.get_neighbors(x, y)
                    unflagged_default = [n for n in neigh if n["state"] == STATE_DEFAULT]
                    flagged_count = sum(1 for n in neigh if n["state"] == STATE_FLAGGED)

                    remaining_mines = t["mines"] - flagged_count
                    if len(unflagged_default) > 0 and remaining_mines >= 0:
                        p = remaining_mines / len(unflagged_default)

                        # cập nhật xác suất cho từng ô
                        for n in unflagged_default:
                            coord = (n["coords"]["x"], n["coords"]["y"])
                            if coord not in prob_map:
                                prob_map[coord] = []
                            prob_map[coord].append(p)

        # 2. Gom xác suất lại (lấy max hoặc trung bình)
        final_probs = {}
        for coord, probs in prob_map.items():
            # final_probs[coord] = max(probs)   # conservative
            final_probs[coord] = sum(probs)/len(probs)  # average

        if not final_probs:
            # Nếu không có constraint nào, random
            unclicked = [(x, y) for x in range(game.size_x) for y in range(game.size_y)
                         if game.tiles[x][y]["state"] == STATE_DEFAULT]
            if unclicked:
                return ("click", *random.choice(unclicked))
            return None

        # 3. Tìm nước đi tốt nhất
        safe_moves = [c for c, p in final_probs.items() if p == 0]
        if safe_moves:
            x, y = safe_moves[0]
            return ("click", x, y)

        sure_flags = [c for c, p in final_probs.items() if p == 1]
        if sure_flags:
            x, y = sure_flags[0]
            return ("flag", x, y)

        # 4. Nếu không có chắc chắn: chọn ô ít nguy hiểm nhất
        best_coord = min(final_probs.items(), key=lambda kv: kv[1])[0]
        return ("click", best_coord[0], best_coord[1])

class CSPStrategy(AIStrategy):
    def next_move(self, game):
        # Collect all constraints from numbered tiles
        constraints = []
        variables = set()  # All unknown tiles
        
        # Find all unknown tiles
        for x in range(game.size_x):
            for y in range(game.size_y):
                if game.tiles[x][y]["state"] == STATE_DEFAULT:
                    variables.add((x, y))
        
        if not variables:
            return None
        
        # Build constraints from numbered tiles
        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]
                if tile["state"] == STATE_CLICKED and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    unknown_neighbors = []
                    flagged_count = 0
                    
                    for n in neighbors:
                        if n["state"] == STATE_DEFAULT:
                            unknown_neighbors.append((n["coords"]["x"], n["coords"]["y"]))
                        elif n["state"] == STATE_FLAGGED:
                            flagged_count += 1
                    
                    if unknown_neighbors:
                        remaining_mines = tile["mines"] - flagged_count
                        constraints.append((unknown_neighbors, remaining_mines))
        
        if not constraints:
            # No constraints, random move
            return ("click", *random.choice(list(variables)))
        
        # Solve CSP using constraint propagation and backtracking
        solution = self.solve_csp(variables, constraints)
        
        if solution:
            # Find definite mines and safe tiles
            mines = [var for var, is_mine in solution.items() if is_mine]
            safe_tiles = [var for var, is_mine in solution.items() if not is_mine]
            
            # Flag a mine if found
            if mines:
                x, y = mines[0]
                return ("flag", x, y)
            
            # Click a safe tile if found
            if safe_tiles:
                x, y = safe_tiles[0]
                return ("click", x, y)
        
        # If CSP can't determine anything definitive, use probability
        return self.fallback_probabilistic(game, constraints, variables)
    
    def solve_csp(self, variables, constraints):
        """Solve CSP using constraint propagation and backtracking"""
        assignment = {}
        
        # Try constraint propagation first
        if self.constraint_propagation(variables, constraints, assignment):
            return assignment
        
        # If propagation isn't enough, try backtracking on a subset
        # (limit to prevent too much computation)
        if len(variables) <= 20:  # Only for small problems
            return self.backtrack_search(variables, constraints, assignment)
        
        return None
    
    def constraint_propagation(self, variables, constraints, assignment):
        """Apply constraint propagation to find forced assignments"""
        changed = True
        while changed:
            changed = False
            
            for constraint_vars, target_mines in constraints:
                # Filter to unassigned variables in this constraint
                unassigned = [v for v in constraint_vars if v not in assignment]
                assigned_mines = sum(1 for v in constraint_vars 
                                   if v in assignment and assignment[v])
                
                remaining_mines = target_mines - assigned_mines
                
                # If remaining mines == 0, all unassigned must be safe
                if remaining_mines == 0:
                    for var in unassigned:
                        if var not in assignment:
                            assignment[var] = False
                            changed = True
                
                # If remaining mines == unassigned count, all must be mines
                elif remaining_mines == len(unassigned):
                    for var in unassigned:
                        if var not in assignment:
                            assignment[var] = True
                            changed = True
        
        return len(assignment) > 0
    
    def backtrack_search(self, variables, constraints, assignment):
        """Backtracking search for CSP solution"""
        if len(assignment) == len(variables):
            return assignment if self.is_consistent(constraints, assignment) else None
        
        # Choose next variable (simple heuristic: first unassigned)
        var = next(v for v in variables if v not in assignment)
        
        # Try both values: mine and safe
        for value in [True, False]:
            assignment[var] = value
            if self.is_consistent_partial(constraints, assignment):
                result = self.backtrack_search(variables, constraints, assignment)
                if result is not None:
                    return result
            del assignment[var]
        
        return None
    
    def is_consistent_partial(self, constraints, assignment):
        """Check if partial assignment is consistent with constraints"""
        for constraint_vars, target_mines in constraints:
            assigned_mines = sum(1 for v in constraint_vars 
                               if v in assignment and assignment[v])
            unassigned_count = sum(1 for v in constraint_vars if v not in assignment)
            
            # Too many mines already assigned
            if assigned_mines > target_mines:
                return False
            
            # Not enough unassigned variables to reach target
            if assigned_mines + unassigned_count < target_mines:
                return False
        
        return True
    
    def is_consistent(self, constraints, assignment):
        """Check if complete assignment satisfies all constraints"""
        for constraint_vars, target_mines in constraints:
            actual_mines = sum(1 for v in constraint_vars 
                             if assignment.get(v, False))
            if actual_mines != target_mines:
                return False
        return True
    
    def fallback_probabilistic(self, game, constraints, variables):
        """Fallback to probabilistic reasoning when CSP is inconclusive"""
        prob_map = {}
        
        for constraint_vars, target_mines in constraints:
            unassigned = [v for v in constraint_vars 
                         if game.tiles[v[0]][v[1]]["state"] == STATE_DEFAULT]
            
            if len(unassigned) > 0:
                prob = target_mines / len(unassigned)
                for var in unassigned:
                    if var not in prob_map:
                        prob_map[var] = []
                    prob_map[var].append(prob)
        
        if prob_map:
            # Average probabilities
            final_probs = {var: sum(probs)/len(probs) 
                          for var, probs in prob_map.items()}
            
            # Choose lowest probability tile
            best_var = min(final_probs.items(), key=lambda x: x[1])[0]
            return ("click", best_var[0], best_var[1])
        
        # Complete fallback to random
        return ("click", *random.choice(list(variables)))

class HybridStrategy(AIStrategy):
    def __init__(self):
        self.move_count = 0
        self.last_progress = 0
        self.stuck_count = 0
    
    def next_move(self, game):
        self.move_count += 1
        
        # Phase 1: Deterministic moves (highest priority)
        deterministic_move = self.find_deterministic_move(game)
        if deterministic_move:
            self.stuck_count = 0
            return deterministic_move
        
        # Phase 2: CSP solving for complex patterns
        csp_move = self.solve_with_csp(game)
        if csp_move:
            self.stuck_count = 0
            return csp_move
        
        # Phase 3: Advanced probabilistic reasoning
        prob_move = self.advanced_probabilistic_move(game)
        if prob_move:
            return prob_move
        
        # Phase 4: Pattern recognition for common scenarios
        pattern_move = self.pattern_recognition_move(game)
        if pattern_move:
            return pattern_move
        
        # Phase 5: Strategic random (avoid obvious bad moves)
        return self.strategic_random_move(game)
    
    def find_deterministic_move(self, game):
        """Find 100% certain moves using basic logical rules"""
        # Check for auto-open opportunities (flagged neighbors match mine count)
        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]
                if tile["state"] == STATE_CLICKED and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    flagged_count = sum(1 for n in neighbors if n["state"] == STATE_FLAGGED)
                    unflagged = [n for n in neighbors if n["state"] == STATE_DEFAULT]
                    
                    # Safe click: all mines are already flagged
                    if flagged_count == tile["mines"] and unflagged:
                        n = unflagged[0]
                        return ("click", n["coords"]["x"], n["coords"]["y"])
                    
                    # Sure flag: remaining unflagged tiles must all be mines
                    remaining_mines = tile["mines"] - flagged_count
                    if remaining_mines > 0 and len(unflagged) == remaining_mines:
                        n = unflagged[0]
                        return ("flag", n["coords"]["x"], n["coords"]["y"])
        
        return None
    
    def solve_with_csp(self, game):
        """Advanced CSP solving with multiple techniques"""
        constraints = self.build_constraints(game)
        variables = self.get_unknown_variables(game)
        
        if not constraints or not variables:
            return None
        
        # Try constraint propagation first
        definite_assignments = self.advanced_constraint_propagation(constraints, variables)
        
        if definite_assignments:
            for var, is_mine in definite_assignments.items():
                if is_mine:
                    return ("flag", var[0], var[1])
                else:
                    return ("click", var[0], var[1])
        
        # Try smaller CSP problems with backtracking
        if len(variables) <= 25:
            solution = self.solve_csp_subset(variables, constraints)
            if solution:
                mines = [var for var, is_mine in solution.items() if is_mine]
                safe = [var for var, is_mine in solution.items() if not is_mine]
                
                if mines:
                    return ("flag", mines[0][0], mines[0][1])
                if safe:
                    return ("click", safe[0][0], safe[0][1])
        
        return None
    
    def build_constraints(self, game):
        """Build constraint system from game state"""
        constraints = []
        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]
                if tile["state"] == STATE_CLICKED and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    unknown_neighbors = []
                    flagged_count = 0
                    
                    for n in neighbors:
                        if n["state"] == STATE_DEFAULT:
                            unknown_neighbors.append((n["coords"]["x"], n["coords"]["y"]))
                        elif n["state"] == STATE_FLAGGED:
                            flagged_count += 1
                    
                    if unknown_neighbors:
                        remaining_mines = tile["mines"] - flagged_count
                        constraints.append((unknown_neighbors, remaining_mines))
        
        return constraints
    
    def get_unknown_variables(self, game):
        """Get all unknown tile positions"""
        variables = set()
        for x in range(game.size_x):
            for y in range(game.size_y):
                if game.tiles[x][y]["state"] == STATE_DEFAULT:
                    variables.add((x, y))
        return variables
    
    def advanced_constraint_propagation(self, constraints, variables):
        """Advanced constraint propagation with conflict detection"""
        assignment = {}
        changed = True
        
        while changed:
            changed = False
            
            for constraint_vars, target_mines in constraints:
                unassigned = [v for v in constraint_vars if v not in assignment]
                assigned_mines = sum(1 for v in constraint_vars 
                                   if v in assignment and assignment[v])
                
                remaining_mines = target_mines - assigned_mines
                
                # All remaining must be safe
                if remaining_mines == 0:
                    for var in unassigned:
                        if var not in assignment:
                            assignment[var] = False
                            changed = True
                
                # All remaining must be mines
                elif remaining_mines == len(unassigned) and remaining_mines > 0:
                    for var in unassigned:
                        if var not in assignment:
                            assignment[var] = True
                            changed = True
            
            # Cross-constraint analysis
            if not changed:
                changed = self.cross_constraint_analysis(constraints, assignment)
        
        return assignment
    
    def cross_constraint_analysis(self, constraints, assignment):
        """Analyze interactions between multiple constraints"""
        # Find overlapping constraints
        for i, (vars1, mines1) in enumerate(constraints):
            for j, (vars2, mines2) in enumerate(constraints[i+1:], i+1):
                overlap = set(vars1) & set(vars2)
                if overlap:
                    # Try subset reasoning
                    if set(vars1) <= set(vars2):
                        # vars1 is subset of vars2
                        remaining_vars = [v for v in vars2 if v not in vars1]
                        if remaining_vars:
                            mines_in_remaining = mines2 - mines1
                            if mines_in_remaining == 0:
                                for var in remaining_vars:
                                    if var not in assignment:
                                        assignment[var] = False
                                        return True
                            elif mines_in_remaining == len(remaining_vars):
                                for var in remaining_vars:
                                    if var not in assignment:
                                        assignment[var] = True
                                        return True
        return False
    
    def solve_csp_subset(self, variables, constraints):
        """Solve smaller CSP problems with backtracking"""
        if len(variables) > 25:
            # Focus on most constrained variables
            variable_scores = {}
            for var in variables:
                score = sum(1 for constraint_vars, _ in constraints if var in constraint_vars)
                variable_scores[var] = score
            
            # Take top constrained variables
            sorted_vars = sorted(variable_scores.items(), key=lambda x: x[1], reverse=True)
            variables = set([var for var, _ in sorted_vars[:20]])
        
        assignment = {}
        if self.backtrack_csp(list(variables), constraints, assignment, 0):
            return assignment
        return None
    
    def backtrack_csp(self, variables, constraints, assignment, var_index):
        """Backtracking search with pruning"""
        if var_index == len(variables):
            return self.satisfies_all_constraints(assignment, constraints)
        
        var = variables[var_index]
        
        for value in [False, True]:  # Try safe first, then mine
            assignment[var] = value
            
            if self.is_consistent_assignment(assignment, constraints):
                if self.backtrack_csp(variables, constraints, assignment, var_index + 1):
                    return True
            
            del assignment[var]
        
        return False
    
    def is_consistent_assignment(self, assignment, constraints):
        """Check if current assignment is consistent"""
        for constraint_vars, target_mines in constraints:
            assigned_mines = sum(1 for v in constraint_vars 
                               if v in assignment and assignment[v])
            unassigned = [v for v in constraint_vars if v not in assignment]
            
            if assigned_mines > target_mines:
                return False
            if assigned_mines + len(unassigned) < target_mines:
                return False
        
        return True
    
    def satisfies_all_constraints(self, assignment, constraints):
        """Check if complete assignment satisfies all constraints"""
        for constraint_vars, target_mines in constraints:
            actual_mines = sum(1 for v in constraint_vars 
                             if assignment.get(v, False))
            if actual_mines != target_mines:
                return False
        return True
    
    def advanced_probabilistic_move(self, game):
        """Advanced probabilistic analysis with multiple probability sources"""
        prob_map = {}
        
        # Basic constraint-based probabilities
        constraints = self.build_constraints(game)
        for constraint_vars, target_mines in constraints:
            if len(constraint_vars) > 0:
                base_prob = target_mines / len(constraint_vars)
                for var in constraint_vars:
                    if var not in prob_map:
                        prob_map[var] = []
                    prob_map[var].append(base_prob)
        
        # Global mine density consideration
        total_tiles = game.size_x * game.size_y
        revealed_tiles = sum(1 for x in range(game.size_x) for y in range(game.size_y) 
                           if game.tiles[x][y]["state"] != STATE_DEFAULT)
        flagged_mines = sum(1 for x in range(game.size_x) for y in range(game.size_y) 
                          if game.tiles[x][y]["state"] == STATE_FLAGGED)
        
        remaining_tiles = total_tiles - revealed_tiles
        remaining_mines = game.total_mines - flagged_mines
        
        if remaining_tiles > 0:
            global_prob = remaining_mines / remaining_tiles
            
            # Apply global probability to unconstrained tiles
            for x in range(game.size_x):
                for y in range(game.size_y):
                    if game.tiles[x][y]["state"] == STATE_DEFAULT:
                        var = (x, y)
                        if var not in prob_map:
                            prob_map[var] = [global_prob]
        
        if not prob_map:
            return None
        
        # Combine probabilities (weighted average favoring constraint-based)
        final_probs = {}
        for var, probs in prob_map.items():
            if len(probs) > 1:
                # Weight constraint-based probabilities more heavily
                constraint_probs = probs[:-1] if len(probs) > 1 else probs
                global_prob = probs[-1] if len(probs) > 1 else 0
                
                if constraint_probs:
                    avg_constraint = sum(constraint_probs) / len(constraint_probs)
                    final_probs[var] = 0.8 * avg_constraint + 0.2 * global_prob
                else:
                    final_probs[var] = global_prob
            else:
                final_probs[var] = probs[0]
        
        # Find best move
        safe_tiles = [var for var, prob in final_probs.items() if prob < 0.001]
        if safe_tiles:
            return ("click", safe_tiles[0][0], safe_tiles[0][1])
        
        sure_mines = [var for var, prob in final_probs.items() if prob > 0.999]
        if sure_mines:
            return ("flag", sure_mines[0][0], sure_mines[0][1])
        
        # Choose lowest probability
        if final_probs:
            best_var = min(final_probs.items(), key=lambda x: x[1])[0]
            return ("click", best_var[0], best_var[1])
        
        return None
    
    def pattern_recognition_move(self, game):
        """Recognize common minesweeper patterns"""
        # Pattern 1: 1-2-1 pattern
        for x in range(game.size_x - 2):
            for y in range(game.size_y):
                if (game.tiles[x][y]["state"] == STATE_CLICKED and game.tiles[x][y]["mines"] == 1 and
                    game.tiles[x+1][y]["state"] == STATE_CLICKED and game.tiles[x+1][y]["mines"] == 2 and
                    game.tiles[x+2][y]["state"] == STATE_CLICKED and game.tiles[x+2][y]["mines"] == 1):
                    
                    # Check for specific 1-2-1 pattern solutions
                    pattern_move = self.solve_121_pattern(game, x, y)
                    if pattern_move:
                        return pattern_move
        
        # Pattern 2: Corner patterns
        corner_move = self.solve_corner_patterns(game)
        if corner_move:
            return corner_move
        
        return None
    
    def solve_121_pattern(self, game, x, y):
        """Solve 1-2-1 patterns"""
        # This is a simplified version - would need more complex analysis
        # for real 1-2-1 pattern recognition
        return None
    
    def solve_corner_patterns(self, game):
        """Solve corner and edge patterns"""
        # Check corners for special cases
        corners = [(0, 0), (0, game.size_y-1), (game.size_x-1, 0), (game.size_x-1, game.size_y-1)]
        
        for cx, cy in corners:
            if game.tiles[cx][cy]["state"] == STATE_CLICKED:
                neighbors = game.get_neighbors(cx, cy)
                unknown = [n for n in neighbors if n["state"] == STATE_DEFAULT]
                if len(unknown) == 1 and game.tiles[cx][cy]["mines"] == 1:
                    # Single unknown neighbor of a 1 in corner
                    flagged_neighbors = sum(1 for n in neighbors if n["state"] == STATE_FLAGGED)
                    if flagged_neighbors == 0:
                        n = unknown[0]
                        return ("flag", n["coords"]["x"], n["coords"]["y"])
        
        return None
    
    def strategic_random_move(self, game):
        """Smart random move that avoids obviously bad choices"""
        unknown_tiles = []
        for x in range(game.size_x):
            for y in range(game.size_y):
                if game.tiles[x][y]["state"] == STATE_DEFAULT:
                    unknown_tiles.append((x, y))
        
        if not unknown_tiles:
            return None
        
        # Prefer tiles that are:
        # 1. Not adjacent to many numbered tiles (lower constraint density)
        # 2. In areas with more space (less cramped)
        # 3. Away from high-number tiles when possible
        
        tile_scores = {}
        for x, y in unknown_tiles:
            score = 0
            neighbors = game.get_neighbors(x, y)
            
            # Prefer tiles with fewer numbered neighbors (less constrained)
            numbered_neighbors = sum(1 for n in neighbors 
                                   if n["state"] == STATE_CLICKED and n["mines"] > 0)
            score -= numbered_neighbors * 2
            
            # Avoid tiles near high numbers
            high_number_neighbors = sum(1 for n in neighbors 
                                      if n["state"] == STATE_CLICKED and n["mines"] >= 4)
            score -= high_number_neighbors * 3
            
            # Prefer tiles with more unknown neighbors (more options later)
            unknown_neighbors = sum(1 for n in neighbors if n["state"] == STATE_DEFAULT)
            score += unknown_neighbors
            
            # Slight preference for center tiles
            center_x, center_y = game.size_x // 2, game.size_y // 2
            distance_from_center = abs(x - center_x) + abs(y - center_y)
            score -= distance_from_center * 0.1
            
            tile_scores[(x, y)] = score
        
        # Choose randomly from top 25% of tiles
        sorted_tiles = sorted(tile_scores.items(), key=lambda x: x[1], reverse=True)
        top_quarter = max(1, len(sorted_tiles) // 4)
        best_tiles = [tile for tile, _ in sorted_tiles[:top_quarter]]
        
        chosen_tile = random.choice(best_tiles)
        return ("click", chosen_tile[0], chosen_tile[1])

STRATEGIES = {
    "Random": RandomStrategy,
    "AutoOpen": AutoOpenStrategy,
    "Probabilistic": ProbabilisticStrategy,
    "CSP": CSPStrategy,
    "Hybrid": HybridStrategy
}

# Game states
STATE_DEFAULT = 0
STATE_CLICKED = 1
STATE_FLAGGED = 2

# Button bindings
BTN_CLICK = "<Button-1>"
BTN_FLAG = "<Button-2>" if platform.system() == 'Darwin' else "<Button-3>"

# Themes
THEMES = {
    "Classic": {
        "bg": "#c0c0c0",
        "button_bg": "#c0c0c0",
        "text_color": "#000000",
        "mine_color": "#ff0000",
        "flag_color": "#ff0000"
    },
    "Dark": {
        "bg": "#2b2b2b",
        "button_bg": "#404040",
        "text_color": "#ffffff",
        "mine_color": "#ff4444",
        "flag_color": "#ffaa00"
    },
    "Blue": {
        "bg": "#e6f3ff",
        "button_bg": "#cce7ff",
        "text_color": "#003366",
        "mine_color": "#cc0000",
        "flag_color": "#0066cc"
    }
}

class Settings:
    def __init__(self):
        self.difficulty = "Beginner"
        self.theme = "Classic"
        self.safe_first_click = True
        self.auto_flag = False
        self.show_timer = True
        self.play_sounds = False
        self.save_stats = True
        self.load_settings()
    
    def save_settings(self):
        try:
            settings_data = {
                "difficulty": self.difficulty,
                "theme": self.theme,
                "safe_first_click": self.safe_first_click,
                "auto_flag": self.auto_flag,
                "show_timer": self.show_timer,
                "play_sounds": self.play_sounds,
                "save_stats": self.save_stats
            }
            with open("settings.json", "w") as f:
                json.dump(settings_data, f)
        except:
            pass
    
    def load_settings(self):
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r") as f:
                    settings_data = json.load(f)
                    self.difficulty = settings_data.get("difficulty", "Beginner")
                    self.theme = settings_data.get("theme", "Classic")
                    self.safe_first_click = settings_data.get("safe_first_click", True)
                    self.auto_flag = settings_data.get("auto_flag", False)
                    self.show_timer = settings_data.get("show_timer", True)
                    self.play_sounds = settings_data.get("play_sounds", False)
                    self.save_stats = settings_data.get("save_stats", True)
        except:
            pass

class Statistics:
    def __init__(self):
        self.stats = {
            "Beginner": {"games_played": 0, "games_won": 0, "best_time": None},
            "Intermediate": {"games_played": 0, "games_won": 0, "best_time": None},
            "Expert": {"games_played": 0, "games_won": 0, "best_time": None},
            "Custom": {"games_played": 0, "games_won": 0, "best_time": None}
        }
        self.load_stats()
    
    def save_stats(self):
        try:
            with open("stats.json", "w") as f:
                json.dump(self.stats, f)
        except:
            pass
    
    def load_stats(self):
        try:
            if os.path.exists("stats.json"):
                with open("stats.json", "r") as f:
                    self.stats = json.load(f)
        except:
            pass
    
    def add_game(self, difficulty, won, time_taken=None):
        self.stats[difficulty]["games_played"] += 1
        if won:
            self.stats[difficulty]["games_won"] += 1
            if time_taken and (not self.stats[difficulty]["best_time"] or time_taken < self.stats[difficulty]["best_time"]):
                self.stats[difficulty]["best_time"] = time_taken

class SettingsWindow:
    def __init__(self, parent, settings, callback):
        self.parent = parent
        self.settings = settings
        self.callback = callback
        self.window = Toplevel(parent)
        self.window.title("Settings")
        self.window.geometry("400x500")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()
        
        self.create_widgets()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_widgets(self):
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Game Settings Tab
        game_frame = ttk.Frame(notebook)
        notebook.add(game_frame, text="Game")
        
        # Difficulty
        ttk.Label(game_frame, text="Difficulty:").grid(row=0, column=0, sticky=W, padx=5, pady=5)
        self.difficulty_var = StringVar(value=self.settings.difficulty)
        difficulty_combo = ttk.Combobox(game_frame, textvariable=self.difficulty_var, 
                                       values=list(DIFFICULTIES.keys()), state="readonly")
        difficulty_combo.grid(row=0, column=1, sticky=EW, padx=5, pady=5)
        difficulty_combo.bind("<<ComboboxSelected>>", self.on_difficulty_change)
        
        # Custom difficulty settings
        self.custom_frame = ttk.LabelFrame(game_frame, text="Custom Settings")
        self.custom_frame.grid(row=1, column=0, columnspan=2, sticky=EW, padx=5, pady=5)
        
        ttk.Label(self.custom_frame, text="Width:").grid(row=0, column=0, padx=5, pady=2)
        self.custom_width = IntVar(value=DIFFICULTIES["Custom"]["size_y"])
        ttk.Spinbox(self.custom_frame, from_=5, to=50, textvariable=self.custom_width, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.custom_frame, text="Height:").grid(row=1, column=0, padx=5, pady=2)
        self.custom_height = IntVar(value=DIFFICULTIES["Custom"]["size_x"])
        ttk.Spinbox(self.custom_frame, from_=5, to=30, textvariable=self.custom_height, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(self.custom_frame, text="Mines:").grid(row=2, column=0, padx=5, pady=2)
        self.custom_mines = IntVar(value=DIFFICULTIES["Custom"]["mines"])
        ttk.Spinbox(self.custom_frame, from_=1, to=500, textvariable=self.custom_mines, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # Game options
        self.safe_first_var = BooleanVar(value=self.settings.safe_first_click)
        ttk.Checkbutton(game_frame, text="Safe first click", variable=self.safe_first_var).grid(row=2, column=0, columnspan=2, sticky=W, padx=5, pady=5)
        
        self.auto_flag_var = BooleanVar(value=self.settings.auto_flag)
        ttk.Checkbutton(game_frame, text="Auto flag when all mines found", variable=self.auto_flag_var).grid(row=3, column=0, columnspan=2, sticky=W, padx=5, pady=5)
        
        # Appearance Tab
        appear_frame = ttk.Frame(notebook)
        notebook.add(appear_frame, text="Appearance")
        
        ttk.Label(appear_frame, text="Theme:").grid(row=0, column=0, sticky=W, padx=5, pady=5)
        self.theme_var = StringVar(value=self.settings.theme)
        theme_combo = ttk.Combobox(appear_frame, textvariable=self.theme_var, 
                                  values=list(THEMES.keys()), state="readonly")
        theme_combo.grid(row=0, column=1, sticky=EW, padx=5, pady=5)
        
        self.show_timer_var = BooleanVar(value=self.settings.show_timer)
        ttk.Checkbutton(appear_frame, text="Show timer", variable=self.show_timer_var).grid(row=1, column=0, columnspan=2, sticky=W, padx=5, pady=5)
        
        # Other Tab
        other_frame = ttk.Frame(notebook)
        notebook.add(other_frame, text="Other")
        
        self.play_sounds_var = BooleanVar(value=self.settings.play_sounds)
        ttk.Checkbutton(other_frame, text="Play sounds", variable=self.play_sounds_var).grid(row=0, column=0, columnspan=2, sticky=W, padx=5, pady=5)
        
        self.save_stats_var = BooleanVar(value=self.settings.save_stats)
        ttk.Checkbutton(other_frame, text="Save statistics", variable=self.save_stats_var).grid(row=1, column=0, columnspan=2, sticky=W, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="OK", command=self.on_ok).pack(side=RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side=RIGHT)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.on_reset).pack(side=LEFT)
        
        self.on_difficulty_change()
    
    def on_difficulty_change(self, event=None):
        is_custom = self.difficulty_var.get() == "Custom"
        for widget in self.custom_frame.winfo_children():
            widget.configure(state=NORMAL if is_custom else DISABLED)
    
    def on_ok(self):
        # Update custom difficulty if selected
        if self.difficulty_var.get() == "Custom":
            DIFFICULTIES["Custom"] = {
                "size_x": self.custom_height.get(),
                "size_y": self.custom_width.get(),
                "mines": self.custom_mines.get()
            }
        
        # Update settings
        self.settings.difficulty = self.difficulty_var.get()
        self.settings.theme = self.theme_var.get()
        self.settings.safe_first_click = self.safe_first_var.get()
        self.settings.auto_flag = self.auto_flag_var.get()
        self.settings.show_timer = self.show_timer_var.get()
        self.settings.play_sounds = self.play_sounds_var.get()
        self.settings.save_stats = self.save_stats_var.get()
        
        self.settings.save_settings()
        self.callback()
        self.window.destroy()
    
    def on_cancel(self):
        self.window.destroy()
    
    def on_reset(self):
        self.difficulty_var.set("Beginner")
        self.theme_var.set("Classic")
        self.safe_first_var.set(True)
        self.auto_flag_var.set(False)
        self.show_timer_var.set(True)
        self.play_sounds_var.set(False)
        self.save_stats_var.set(True)
    
    def on_close(self):
        self.window.destroy()

class StatsWindow:
    def __init__(self, parent, stats):
        self.window = Toplevel(parent)
        self.window.title("Statistics")
        self.window.geometry("400x300")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()
        
        # Create treeview for stats
        columns = ("Difficulty", "Games Played", "Games Won", "Win Rate", "Best Time")
        tree = ttk.Treeview(self.window, columns=columns, show="headings", height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor=CENTER)
        
        # Populate with data
        for difficulty, data in stats.stats.items():
            games_played = data["games_played"]
            games_won = data["games_won"]
            win_rate = f"{(games_won/games_played*100):.1f}%" if games_played > 0 else "0%"
            best_time = f"{data['best_time']:.2f}s" if data["best_time"] else "N/A"
            
            tree.insert("", END, values=(difficulty, games_played, games_won, win_rate, best_time))
        
        tree.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Close button
        ttk.Button(self.window, text="Close", command=self.window.destroy).pack(pady=5)

class Minesweeper:
    def __init__(self, tk):
        self.tk = tk
        self.settings = Settings()
        self.stats = Statistics()
        self.strategy_name = StringVar(value="AutoOpen")
        
        # Initialize game variables
        self.size_x = DIFFICULTIES[self.settings.difficulty]["size_x"]
        self.size_y = DIFFICULTIES[self.settings.difficulty]["size_y"]
        self.total_mines = DIFFICULTIES[self.settings.difficulty]["mines"]
        
        # Initialize AI with proper strategy instance
        self.ai = MinesweeperAI(self, AutoOpenStrategy())
        
        self.create_menu()
        self.setup_ui()
        self.apply_theme()
        self.restart()
        self.update_timer()
    
    def create_menu(self):
        menubar = Menu(self.tk)
        self.tk.config(menu=menubar)
        
        # Game menu
        game_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="New Game", command=self.restart, accelerator="F2")
        game_menu.add_separator()
        game_menu.add_command(label="Settings", command=self.open_settings, accelerator="F3")
        game_menu.add_command(label="Statistics", command=self.open_stats, accelerator="F4")
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.tk.quit)
        
        # AI menu
        ai_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)
        
        for strategy_name in STRATEGIES.keys():
            ai_menu.add_radiobutton(label=strategy_name, 
                                  variable=self.strategy_name, 
                                  value=strategy_name,
                                  command=self.change_strategy)
        
        # Bind keyboard shortcuts
        self.tk.bind("<F2>", lambda e: self.restart())
        self.tk.bind("<F3>", lambda e: self.open_settings())
        self.tk.bind("<F4>", lambda e: self.open_stats())
    
    def change_strategy(self):
        """Change the AI strategy based on the selected option."""
        strategy_class = STRATEGIES[self.strategy_name.get()]
        self.ai.set_strategy(strategy_class())
        # Update button text
        if hasattr(self, 'strategy_btn'):
            self.strategy_btn.config(text=f"Strategy: {self.strategy_name.get()}")
    
    def cycle_strategy(self):
        """Cycle through available strategies"""
        strategy_names = list(STRATEGIES.keys())
        current_index = strategy_names.index(self.strategy_name.get())
        next_index = (current_index + 1) % len(strategy_names)
        self.strategy_name.set(strategy_names[next_index])
        self.change_strategy()
    
    def setup_ui(self):
        # Main container frame
        self.main_frame = Frame(self.tk)
        self.main_frame.pack(padx=10, pady=10)
        
        # Top frame for status labels
        self.top_frame = Frame(self.main_frame)
        self.top_frame.pack(fill=X, pady=(0, 10))
        
        # Status labels
        self.labels = {
            "mines": Label(self.top_frame, text=f"Mines: {self.total_mines}", font=("Arial", 12, "bold")),
            "flags": Label(self.top_frame, text="Flags: 0", font=("Arial", 12, "bold")),
            "time": Label(self.top_frame, text="00:00:00", font=("Arial", 12, "bold"))
        }
        
        self.labels["mines"].pack(side=LEFT)
        self.labels["flags"].pack(side=LEFT, padx=(20, 0))
        self.labels["time"].pack(side=RIGHT)
        
        # Game frame for the minefield
        self.game_frame = Frame(self.main_frame)
        self.game_frame.pack()
        
        # Bottom frame for controls
        self.bottom_frame = Frame(self.main_frame)
        self.bottom_frame.pack(fill=X, pady=(10, 0))
        
        # Control buttons
        restart_btn = Button(self.bottom_frame, text="New Game (F2)", command=self.restart, 
                           font=("Arial", 10), padx=10)
        restart_btn.pack(side=LEFT)
        
        settings_btn = Button(self.bottom_frame, text="Settings (F3)", command=self.open_settings,
                            font=("Arial", 10), padx=10)
        settings_btn.pack(side=LEFT, padx=(10, 0))
        
        stats_btn = Button(self.bottom_frame, text="Stats (F4)", command=self.open_stats,
                         font=("Arial", 10), padx=10)
        stats_btn.pack(side=LEFT, padx=(10, 0))

        # Strategy selection button
        self.strategy_btn = Button(self.bottom_frame, text=f"Strategy: {self.strategy_name.get()}", 
                                 command=self.cycle_strategy, font=("Arial", 10), padx=10)
        self.strategy_btn.pack(side=LEFT, padx=(10, 0))

        self.auto_play_btn = Button(self.bottom_frame, text="Auto Play", command=self.toggle_auto_play,
                               font=("Arial", 10), padx=10)
        self.auto_play_btn.pack(side=LEFT, padx=(10, 0))
        
        # AI status
        self.ai_running = False
        self.ai_status_label = Label(self.bottom_frame, text="AI: Stopped", font=("Arial", 9))
        self.ai_status_label.pack(side=LEFT, padx=(10, 0))
        
        # Difficulty label
        difficulty_text = f"Difficulty: {self.settings.difficulty}"
        if self.settings.difficulty == "Custom":
            difficulty_text += f" ({self.size_y}x{self.size_x}, {self.total_mines} mines)"
        
        self.difficulty_label = Label(self.bottom_frame, text=difficulty_text, font=("Arial", 9))
        self.difficulty_label.pack(side=RIGHT)
    
    def toggle_auto_play(self):
        """Toggle auto play on/off."""
        if self.game_over_flag:
            return
            
        if not self.ai_running:
            self.ai_running = True
            self.auto_play_btn.config(text="Stop AI")
            self.ai_status_label.config(text=f"AI: Running ({self.strategy_name.get()})")
            self.auto_play()
        else:
            self.ai_running = False
            self.auto_play_btn.config(text="Auto Play")
            self.ai_status_label.config(text="AI: Stopped")
    
    def auto_play(self):
        if self.game_over_flag or not self.ai_running:
            self.ai_running = False
            self.auto_play_btn.config(text="Auto Play")
            self.ai_status_label.config(text="AI: Stopped")
            return
            
        if self.ai.play():
            self.tk.after(50, self.auto_play) 
        else:
            # No more moves available
            self.ai_running = False
            self.auto_play_btn.config(text="Auto Play")
            self.ai_status_label.config(text="AI: No moves")

    def apply_theme(self):
        theme = THEMES[self.settings.theme]
        self.tk.configure(bg=theme["bg"])
        self.main_frame.configure(bg=theme["bg"])
        self.top_frame.configure(bg=theme["bg"])
        self.game_frame.configure(bg=theme["bg"])
        self.bottom_frame.configure(bg=theme["bg"])
        
        for label in self.labels.values():
            label.configure(bg=theme["bg"], fg=theme["text_color"])
        
        self.difficulty_label.configure(bg=theme["bg"], fg=theme["text_color"])
        self.ai_status_label.configure(bg=theme["bg"], fg=theme["text_color"])
    
    def create_images(self):
        """Load images from the images folder, falling back to simple colors if files don't exist"""
        self.images = {
            "plain": None,
            "clicked": None,
            "mine": None,
            "flag": None,
            "wrong": None,
            "numbers": []
        }
        
        # Try to load actual image files first
        image_path = "images"
        image_files = {
            "plain": "tile_plain.gif",
            "clicked": "tile_clicked.gif",
            "mine": "tile_mine.gif",
            "flag": "tile_flag.gif",
            "wrong": "tile_wrong.gif"
        }
        
        # Load basic tiles
        for key, filename in image_files.items():
            try:
                filepath = os.path.join(image_path, filename)
                if os.path.exists(filepath):
                    self.images[key] = PhotoImage(file=filepath)
                else:
                    # Fallback to colored rectangles
                    self.images[key] = self.create_fallback_image(key)
            except:
                self.images[key] = self.create_fallback_image(key)
        
        # Load number tiles (1-8)
        for i in range(1, 9):
            try:
                filepath = os.path.join(image_path, f"tile_{i}.gif")
                if os.path.exists(filepath):
                    img = PhotoImage(file=filepath)
                else:
                    img = self.create_fallback_number_image(i)
                self.images["numbers"].append(img)
            except:
                self.images["numbers"].append(self.create_fallback_number_image(i))
    
    def create_fallback_image(self, image_type):
        """Create fallback colored rectangle images"""
        theme = THEMES[self.settings.theme]
        img = PhotoImage(width=20, height=20)
        
        if image_type == "plain":
            img.put(theme["button_bg"], (0, 0, 20, 20))
        elif image_type == "clicked":
            img.put("#ffffff", (0, 0, 20, 20))
        elif image_type == "mine":
            img.put(theme["mine_color"], (0, 0, 20, 20))
        elif image_type == "flag":
            img.put(theme["flag_color"], (0, 0, 20, 20))
        elif image_type == "wrong":
            img.put("#888888", (0, 0, 20, 20))
        
        return img
    
    def create_fallback_number_image(self, number):
        """Create fallback number images"""
        img = PhotoImage(width=20, height=20)
        img.put("#ffffff", (0, 0, 20, 20))
        return img
    
    def setup(self):
        # Clear existing game tiles
        for widget in self.game_frame.winfo_children():
            widget.destroy()
        
        # Initialize game state
        self.flag_count = 0
        self.correct_flag_count = 0
        self.clicked_count = 0
        self.start_time = None
        self.first_click = True
        self.game_over_flag = False
        
        # Reset AI
        self.ai.reset()
        self.ai_running = False
        if hasattr(self, 'auto_play_btn'):
            self.auto_play_btn.config(text="Auto Play")
        if hasattr(self, 'ai_status_label'):
            self.ai_status_label.config(text="AI: Stopped")
        
        self.create_images()
        
        # Create tiles using grid layout in the game_frame
        self.tiles = {}
        
        for x in range(self.size_x):
            self.tiles[x] = {}
            for y in range(self.size_y):
                tile = {
                    "id": f"{x}_{y}",
                    "is_mine": False,
                    "state": STATE_DEFAULT,
                    "coords": {"x": x, "y": y},
                    "mines": 0,
                    "button": Button(self.game_frame, image=self.images["plain"], 
                                   width=25, height=25, bd=1, relief="raised")
                }
                
                tile["button"].bind(BTN_CLICK, self.on_click_wrapper(x, y))
                tile["button"].bind(BTN_FLAG, self.on_right_click_wrapper(x, y))
                tile["button"].bind("<Double-Button-1>", self.on_double_click_wrapper(x, y))  # Double-click for auto-open
                tile["button"].grid(row=x, column=y, padx=1, pady=1)
                
                self.tiles[x][y] = tile
        
        # Place mines (will be done on first click if safe_first_click is enabled)
        if not self.settings.safe_first_click:
            self.place_mines()
    
    def place_mines(self, avoid_x=None, avoid_y=None):
        mine_positions = set()
        avoid_positions = set()
        
        # If safe first click, avoid the clicked cell and its neighbors
        if avoid_x is not None and avoid_y is not None:
            avoid_positions.add((avoid_x, avoid_y))
            for neighbor in self.get_neighbors(avoid_x, avoid_y):
                avoid_positions.add((neighbor["coords"]["x"], neighbor["coords"]["y"]))
        
        # Place mines randomly
        while len(mine_positions) < self.total_mines:
            x = random.randint(0, self.size_x - 1)
            y = random.randint(0, self.size_y - 1)
            
            if (x, y) not in mine_positions and (x, y) not in avoid_positions:
                mine_positions.add((x, y))
                self.tiles[x][y]["is_mine"] = True
        
        # Calculate numbers for each tile
        for x in range(self.size_x):
            for y in range(self.size_y):
                if not self.tiles[x][y]["is_mine"]:
                    count = sum(1 for neighbor in self.get_neighbors(x, y) if neighbor["is_mine"])
                    self.tiles[x][y]["mines"] = count
    
    def restart(self):
        # Update difficulty settings
        difficulty_data = DIFFICULTIES[self.settings.difficulty]
        self.size_x = difficulty_data["size_x"]
        self.size_y = difficulty_data["size_y"]
        self.total_mines = difficulty_data["mines"]
        
        self.setup()
        self.refresh_labels()
        self.apply_theme()
        
        # Update difficulty label
        difficulty_text = f"Difficulty: {self.settings.difficulty}"
        if self.settings.difficulty == "Custom":
            difficulty_text += f" ({self.size_y}x{self.size_x}, {self.total_mines} mines)"
        self.difficulty_label.config(text=difficulty_text)
        
        # Adjust window size to fit the grid
        self.tk.update_idletasks()
        self.tk.geometry("")  # Let tkinter calculate the size
    
    def refresh_labels(self):
        self.labels["flags"].config(text=f"Flags: {self.flag_count}")
        self.labels["mines"].config(text=f"Mines: {self.total_mines}")
    
    def game_over(self, won):
        self.game_over_flag = True
        
        # Stop AI if running
        if self.ai_running:
            self.ai_running = False
            self.auto_play_btn.config(text="Auto Play")
            self.ai_status_label.config(text="AI: Stopped")
        
        # Record statistics
        if self.settings.save_stats and self.start_time:
            time_taken = (datetime.now() - self.start_time).total_seconds()
            self.stats.add_game(self.settings.difficulty, won, time_taken if won else None)
            self.stats.save_stats()
        
        if not won:
            # Show only mines and wrong flags, highlight the triggered mine
            for x in range(self.size_x):
                for y in range(self.size_y):
                    tile = self.tiles[x][y]
                    if tile["is_mine"] and tile["state"] != STATE_FLAGGED:
                        if tile["state"] == STATE_CLICKED:  # This is the mine that was clicked
                            tile["button"].config(image=self.images["mine"], bg="red")
                        else:
                            tile["button"].config(image=self.images["mine"])
                    elif not tile["is_mine"] and tile["state"] == STATE_FLAGGED:
                        tile["button"].config(image=self.images["wrong"], bg="red")
        else:
            # For winning, just show unflagged mines normally
            for x in range(self.size_x):
                for y in range(self.size_y):
                    tile = self.tiles[x][y]
                    if tile["is_mine"] and tile["state"] != STATE_FLAGGED:
                        tile["button"].config(image=self.images["mine"])
        
        self.tk.update()
        
        # Show result dialog
        msg = "Congratulations! You won!\n\nPlay again?" if won else "Game Over! You hit a mine!\n\nPlay again?"
        if tkMessageBox.askyesno("Game Over", msg):
            self.restart()
    
    def update_timer(self):
        if self.settings.show_timer:
            ts = "00:00:00"
            if self.start_time and not self.game_over_flag:
                delta = datetime.now() - self.start_time
                total_seconds = int(delta.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                ts = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.labels["time"].config(text=ts)
        else:
            self.labels["time"].config(text="")
        
        self.tk.after(100, self.update_timer)
    
    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size_x and 0 <= ny < self.size_y:
                    neighbors.append(self.tiles[nx][ny])
        return neighbors
    
    def on_click_wrapper(self, x, y):
        return lambda event: self.on_click(self.tiles[x][y])
    
    def on_right_click_wrapper(self, x, y):
        return lambda event: self.on_right_click(self.tiles[x][y])
    
    def on_double_click_wrapper(self, x, y):
        return lambda event: self.on_double_click(self.tiles[x][y])
    
    def on_double_click(self, tile):
        """Handle double-click for auto-opening neighbors (like middle-click in traditional minesweeper)"""
        if self.game_over_flag or tile["state"] != STATE_CLICKED:
            return
        
        x, y = tile["coords"]["x"], tile["coords"]["y"]
        neighbors = self.get_neighbors(x, y)
        flag_count = sum(1 for n in neighbors if n["state"] == STATE_FLAGGED)
        
        # Only auto-open if the number of flags matches the mine count
        if flag_count == tile["mines"]:
            for neighbor in neighbors:
                if neighbor["state"] == STATE_DEFAULT:
                    self.on_click(neighbor)
    
    def on_click(self, tile):
        if self.game_over_flag or tile["state"] != STATE_DEFAULT:
            return
        
        # Start timer on first click
        if self.start_time is None:
            self.start_time = datetime.now()
        
        # Handle safe first click
        if self.first_click and self.settings.safe_first_click:
            self.place_mines(tile["coords"]["x"], tile["coords"]["y"])
            self.first_click = False
        
        # Check if mine
        if tile["is_mine"]:
            self.game_over(False)
            return
        
        # Reveal tile
        if tile["mines"] == 0:
            tile["button"].config(image=self.images["clicked"], relief="sunken")
            self.clear_surrounding_tiles(tile["id"])
        else:
            tile["button"].config(image=self.images["numbers"][tile["mines"] - 1], relief="sunken")
        
        if tile["state"] != STATE_CLICKED:
            tile["state"] = STATE_CLICKED
            self.clicked_count += 1
        
        # Check win condition
        if self.clicked_count == (self.size_x * self.size_y) - self.total_mines:
            self.game_over(True)

    
    def on_right_click(self, tile):
        if self.game_over_flag or tile["state"] == STATE_CLICKED:
            return
        
        # Start timer on first click
        if self.start_time is None:
            self.start_time = datetime.now()
        
        if tile["state"] == STATE_DEFAULT:
            # Flag tile
            tile["button"].config(image=self.images["flag"])
            tile["state"] = STATE_FLAGGED
            tile["button"].unbind(BTN_CLICK)
            self.flag_count += 1
            if tile["is_mine"]:
                self.correct_flag_count += 1
        else:  # STATE_FLAGGED
            # Unflag tile
            tile["button"].config(image=self.images["plain"])
            tile["state"] = STATE_DEFAULT
            tile["button"].bind(BTN_CLICK, self.on_click_wrapper(tile["coords"]["x"], tile["coords"]["y"]))
            self.flag_count -= 1
            if tile["is_mine"]:
                self.correct_flag_count -= 1
        
        self.refresh_labels()
    
    def clear_surrounding_tiles(self, tile_id):
        queue = deque([tile_id])
        
        while queue:
            current_id = queue.popleft()
            parts = current_id.split("_")
            x, y = int(parts[0]), int(parts[1])
            
            for neighbor in self.get_neighbors(x, y):
                self.clear_tile(neighbor, queue)
    
    def clear_tile(self, tile, queue):
        if tile["state"] != STATE_DEFAULT:
            return
        
        if tile["mines"] == 0:
            tile["button"].config(image=self.images["clicked"], relief="sunken")
            queue.append(tile["id"])
        else:
            tile["button"].config(image=self.images["numbers"][tile["mines"] - 1], relief="sunken")
        
        tile["state"] = STATE_CLICKED
        self.clicked_count += 1
    
    def open_settings(self):
        SettingsWindow(self.tk, self.settings, self.on_settings_changed)
    
    def on_settings_changed(self):
        self.apply_theme()
        self.restart()
    
    def open_stats(self):
        StatsWindow(self.tk, self.stats)

def main():
    # Create main window
    window = Tk()
    window.title("Enhanced Minesweeper")
    window.resizable(False, False)
    
    # Center window on screen
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f"+{x}+{y}")
    
    # Create game instance
    minesweeper = Minesweeper(window)
    
    # Run event loop
    window.mainloop()

if __name__ == "__main__":
    main()