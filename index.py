from tkinter import *
from tkinter import messagebox as tkMessageBox
from tkinter import ttk
from collections import deque
import random
import platform
from datetime import datetime
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Tuple

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
            final_probs[coord] = sum(probs) / len(probs)  # average

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


class EnhancedProbabilisticStrategy(AIStrategy):
    def __init__(self):
        self.probability_cache = {}  # Cache probability calculations
        self.constraint_groups = []  # Track related constraint groups
        self.global_mine_density = 0.0
        self.local_densities = {}  # Track local mine densities

    def next_move(self, game):
        # Phase 1: Deterministic moves first
        deterministic_move = self.find_deterministic_moves(game)
        if deterministic_move:
            return deterministic_move

        # Phase 2: Advanced probability analysis
        prob_move = self.advanced_probability_analysis(game)
        if prob_move:
            return prob_move

        # Phase 3: Bayesian probability with global constraints
        bayesian_move = self.bayesian_probability_analysis(game)
        if bayesian_move:
            return bayesian_move

        # Phase 4: Monte Carlo estimation for complex scenarios
        monte_carlo_move = self.monte_carlo_probability_analysis(game)
        if monte_carlo_move:
            return monte_carlo_move

        # Phase 5: Smart random with density consideration
        return self.density_aware_random(game)

    def find_deterministic_moves(self, game):
        """Quick deterministic moves - same as CSP but faster"""
        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]
                if tile["state"] == STATE_CLICKED and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    flagged_count = sum(1 for n in neighbors if n["state"] == STATE_FLAGGED)
                    unflagged = [n for n in neighbors if n["state"] == STATE_DEFAULT]

                    # Auto-open: all mines flagged
                    if flagged_count == tile["mines"] and unflagged:
                        n = unflagged[0]
                        return ("click", n["coords"]["x"], n["coords"]["y"])

                    # Auto-flag: remaining tiles must all be mines
                    remaining_mines = tile["mines"] - flagged_count
                    if remaining_mines > 0 and len(unflagged) == remaining_mines:
                        n = unflagged[0]
                        return ("flag", n["coords"]["x"], n["coords"]["y"])
        return None

    def advanced_probability_analysis(self, game):
        """Multi-level probability analysis with constraint interaction"""
        # Build constraint system
        constraints = self.build_constraint_system(game)
        if not constraints:
            return None

        # Group related constraints
        constraint_groups = self.group_related_constraints(constraints)

        # Analyze each group independently for better accuracy
        all_probabilities = {}

        for group in constraint_groups:
            group_probs = self.analyze_constraint_group(group)
            all_probabilities.update(group_probs)

        # Apply global mine constraint
        corrected_probs = self.apply_global_mine_constraint(game, all_probabilities, constraints)

        # Find best move with confidence analysis
        return self.select_best_probabilistic_move(corrected_probs)

    def build_constraint_system(self, game):
        """Build comprehensive constraint system"""
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
                        constraints.append({
                            'variables': unknown_neighbors,
                            'mines': remaining_mines,
                            'center': (x, y),
                            'total_neighbors': len(neighbors)
                        })

        return constraints

    def group_related_constraints(self, constraints):
        """Group constraints that share variables for joint analysis"""
        if not constraints:
            return []

        # Build adjacency graph of constraints
        constraint_graph = defaultdict(set)

        for i, c1 in enumerate(constraints):
            for j, c2 in enumerate(constraints[i + 1:], i + 1):
                vars1 = set(c1['variables'])
                vars2 = set(c2['variables'])

                if vars1 & vars2:  # Share at least one variable
                    constraint_graph[i].add(j)
                    constraint_graph[j].add(i)

        # Find connected components (constraint groups)
        visited = set()
        groups = []

        for i in range(len(constraints)):
            if i not in visited:
                group = []
                stack = [i]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.append(constraints[current])
                        stack.extend(constraint_graph[current] - visited)

                groups.append(group)

        return groups

    def analyze_constraint_group(self, group):
        """Analyze a group of related constraints using advanced techniques"""
        if not group:
            return {}

        # Collect all variables in this group
        all_variables = set()
        for constraint in group:
            all_variables.update(constraint['variables'])

        all_variables = list(all_variables)

        # For small groups, use exact enumeration
        if len(all_variables) <= 12:
            return self.exact_probability_enumeration(group, all_variables)

        # For larger groups, use approximation methods
        return self.approximate_probability_analysis(group, all_variables)

    def exact_probability_enumeration(self, group, variables):
        """Exact probability calculation by enumerating all valid assignments"""
        if len(variables) > 15:  # Safety limit
            return self.approximate_probability_analysis(group, variables)

        variable_mine_counts = defaultdict(int)
        total_valid_assignments = 0

        # Generate all possible mine assignments
        for assignment_bits in range(2 ** len(variables)):
            assignment = {}
            mine_count = 0

            for i, var in enumerate(variables):
                is_mine = bool(assignment_bits & (1 << i))
                assignment[var] = is_mine
                if is_mine:
                    mine_count += 1

            # Check if assignment satisfies all constraints
            if self.assignment_satisfies_constraints(assignment, group):
                total_valid_assignments += 1
                for var, is_mine in assignment.items():
                    if is_mine:
                        variable_mine_counts[var] += 1

        # Calculate probabilities
        probabilities = {}
        if total_valid_assignments > 0:
            for var in variables:
                probabilities[var] = variable_mine_counts[var] / total_valid_assignments

        return probabilities

    def assignment_satisfies_constraints(self, assignment, constraints):
        """Check if an assignment satisfies all constraints in the group"""
        for constraint in constraints:
            mine_count = sum(1 for var in constraint['variables'] if assignment.get(var, False))
            if mine_count != constraint['mines']:
                return False
        return True

    def approximate_probability_analysis(self, group, variables):
        """Approximate probability analysis for large constraint groups"""
        probabilities = {}

        # Method 1: Linear constraint approximation
        linear_probs = self.linear_constraint_approximation(group, variables)

        # Method 2: Iterative constraint satisfaction
        iterative_probs = self.iterative_probability_refinement(group, variables)

        # Method 3: Sampling-based estimation
        sampling_probs = self.sampling_based_probability(group, variables)

        # Combine methods with weighted average
        for var in variables:
            probs = []
            if var in linear_probs:
                probs.append(linear_probs[var])
            if var in iterative_probs:
                probs.append(iterative_probs[var])
            if var in sampling_probs:
                probs.append(sampling_probs[var])

            if probs:
                # Weight recent methods more heavily
                weights = [1.0, 1.5, 2.0][:len(probs)]
                probabilities[var] = sum(p * w for p, w in zip(probs, weights)) / sum(weights)

        return probabilities

    def linear_constraint_approximation(self, group, variables):
        """Linear approximation assuming independence"""
        probabilities = {}

        for var in variables:
            prob_estimates = []

            for constraint in group:
                if var in constraint['variables']:
                    # Basic probability from this constraint
                    basic_prob = constraint['mines'] / len(constraint['variables'])
                    prob_estimates.append(basic_prob)

            if prob_estimates:
                # Average with bias toward extreme values
                avg_prob = sum(prob_estimates) / len(prob_estimates)

                # Apply non-linear transformation to emphasize certainty
                if avg_prob > 0.5:
                    probabilities[var] = 0.5 + (avg_prob - 0.5) * 1.2
                else:
                    probabilities[var] = avg_prob * 0.8

                # Clamp to valid range
                probabilities[var] = max(0.0, min(1.0, probabilities[var]))

        return probabilities

    def iterative_probability_refinement(self, group, variables):
        """Iteratively refine probabilities considering constraint interactions"""
        probabilities = {var: 0.5 for var in variables}  # Start with uniform prior

        max_iterations = 20
        convergence_threshold = 0.001

        for iteration in range(max_iterations):
            old_probabilities = probabilities.copy()

            for constraint in group:
                constraint_vars = constraint['variables']
                target_mines = constraint['mines']

                # Calculate expected mines with current probabilities
                expected_mines = sum(probabilities[var] for var in constraint_vars)

                if expected_mines > 0:
                    # Adjust probabilities proportionally
                    adjustment_factor = target_mines / expected_mines

                    for var in constraint_vars:
                        # Adjust probability with dampening
                        old_prob = probabilities[var]
                        new_prob = old_prob * adjustment_factor

                        # Apply dampening to prevent oscillation
                        dampening = 0.3
                        probabilities[var] = old_prob * (1 - dampening) + new_prob * dampening

                        # Keep in valid range
                        probabilities[var] = max(0.01, min(0.99, probabilities[var]))

            # Check convergence
            max_change = max(abs(probabilities[var] - old_probabilities[var])
                             for var in variables)

            if max_change < convergence_threshold:
                break

        return probabilities

    def sampling_based_probability(self, group, variables, num_samples=1000):
        """Monte Carlo sampling to estimate probabilities"""
        if len(variables) > 20:  # Limit for performance
            num_samples = min(500, num_samples)

        variable_mine_counts = defaultdict(int)
        valid_samples = 0
        max_attempts = num_samples * 10  # Prevent infinite loops

        for _ in range(max_attempts):
            if valid_samples >= num_samples:
                break

            # Generate random assignment
            assignment = {}
            for var in variables:
                assignment[var] = random.random() < 0.5

            # Check if it satisfies constraints
            if self.assignment_satisfies_constraints(assignment, group):
                valid_samples += 1
                for var, is_mine in assignment.items():
                    if is_mine:
                        variable_mine_counts[var] += 1

        # Calculate probabilities
        probabilities = {}
        if valid_samples > 0:
            for var in variables:
                probabilities[var] = variable_mine_counts[var] / valid_samples

        return probabilities

    def apply_global_mine_constraint(self, game, local_probabilities, constraints):
        """Apply global mine count constraint to adjust local probabilities"""
        if not local_probabilities:
            return local_probabilities

        # Calculate current mine situation
        total_mines = game.total_mines
        flagged_mines = sum(1 for x in range(game.size_x) for y in range(game.size_y)
                            if game.tiles[x][y]["state"] == STATE_FLAGGED)
        remaining_mines = total_mines - flagged_mines

        # Calculate total unknown tiles
        all_unknown = set()
        constrained_unknown = set(local_probabilities.keys())

        for x in range(game.size_x):
            for y in range(game.size_y):
                if game.tiles[x][y]["state"] == STATE_DEFAULT:
                    all_unknown.add((x, y))

        unconstrained_unknown = all_unknown - constrained_unknown
        total_unknown = len(all_unknown)

        if total_unknown == 0:
            return local_probabilities

        # Expected mines in constrained region
        expected_constrained_mines = sum(local_probabilities.values())

        # Expected mines in unconstrained region
        if len(unconstrained_unknown) > 0:
            expected_unconstrained_mines = remaining_mines - expected_constrained_mines
            unconstrained_probability = max(0.0, min(1.0,
                                                     expected_unconstrained_mines / len(unconstrained_unknown)))
        else:
            unconstrained_probability = 0.0
            expected_unconstrained_mines = 0.0

        # Adjust probabilities to match global constraint
        total_expected = expected_constrained_mines + expected_unconstrained_mines

        if abs(total_expected - remaining_mines) > 0.1:  # Significant discrepancy
            if total_expected > 0:
                adjustment_factor = remaining_mines / total_expected

                # Apply adjustment to constrained probabilities
                adjusted_probabilities = {}
                for var, prob in local_probabilities.items():
                    adjusted_prob = prob * adjustment_factor
                    adjusted_probabilities[var] = max(0.0, min(1.0, adjusted_prob))

                # Add unconstrained probabilities
                adjusted_unconstrained_prob = unconstrained_probability * adjustment_factor
                adjusted_unconstrained_prob = max(0.0, min(1.0, adjusted_unconstrained_prob))

                for var in unconstrained_unknown:
                    adjusted_probabilities[var] = adjusted_unconstrained_prob

                return adjusted_probabilities

        # If no significant adjustment needed, add unconstrained probabilities
        result_probabilities = local_probabilities.copy()
        for var in unconstrained_unknown:
            result_probabilities[var] = unconstrained_probability

        return result_probabilities

    def bayesian_probability_analysis(self, game):
        """Bayesian analysis with prior knowledge and evidence updating"""
        constraints = self.build_constraint_system(game)
        if not constraints:
            return None

        # Calculate prior probabilities based on game state
        priors = self.calculate_bayesian_priors(game)

        # Update probabilities with constraint evidence
        posteriors = self.update_with_constraint_evidence(constraints, priors)

        # Apply spatial correlation analysis
        correlated_posteriors = self.apply_spatial_correlation(game, posteriors)

        return self.select_best_probabilistic_move(correlated_posteriors)

    def calculate_bayesian_priors(self, game):
        """Calculate prior probabilities based on game state and patterns"""
        priors = {}

        # Global mine density
        total_tiles = game.size_x * game.size_y
        revealed_tiles = sum(1 for x in range(game.size_x) for y in range(game.size_y)
                             if game.tiles[x][y]["state"] != STATE_DEFAULT)
        flagged_mines = sum(1 for x in range(game.size_x) for y in range(game.size_y)
                            if game.tiles[x][y]["state"] == STATE_FLAGGED)

        remaining_tiles = total_tiles - revealed_tiles
        remaining_mines = game.total_mines - flagged_mines

        if remaining_tiles > 0:
            global_density = remaining_mines / remaining_tiles
        else:
            global_density = 0.0

        # Calculate local density adjustments
        for x in range(game.size_x):
            for y in range(game.size_y):
                if game.tiles[x][y]["state"] == STATE_DEFAULT:
                    local_adjustment = self.calculate_local_density_adjustment(game, x, y)
                    priors[(x, y)] = max(0.01, min(0.99, global_density * local_adjustment))

        return priors

    def calculate_local_density_adjustment(self, game, x, y):
        """Calculate local density adjustment based on nearby revealed tiles"""
        neighbors = game.get_neighbors(x, y)

        # Factor 1: Nearby mine density
        nearby_mine_density = 1.0
        revealed_neighbors = [n for n in neighbors if n["state"] == STATE_CLICKED]

        if revealed_neighbors:
            total_nearby_mines = sum(n["mines"] for n in revealed_neighbors)
            max_possible_mines = len(revealed_neighbors) * 8  # Max mines per tile

            if max_possible_mines > 0:
                nearby_density = total_nearby_mines / max_possible_mines
                nearby_mine_density = 0.5 + nearby_density  # Adjust around neutral

        # Factor 2: Edge/corner adjustment
        edge_adjustment = 1.0
        is_edge = (x == 0 or x == game.size_x - 1 or y == 0 or y == game.size_y - 1)
        is_corner = ((x == 0 or x == game.size_x - 1) and (y == 0 or y == game.size_y - 1))

        if is_corner:
            edge_adjustment = 0.8  # Corners slightly less likely
        elif is_edge:
            edge_adjustment = 0.9  # Edges slightly less likely

        # Factor 3: Distance from revealed areas (exploration bonus)
        distance_adjustment = 1.0
        min_distance_to_revealed = float('inf')

        for rx in range(game.size_x):
            for ry in range(game.size_y):
                if game.tiles[rx][ry]["state"] != STATE_DEFAULT:
                    distance = abs(x - rx) + abs(y - ry)  # Manhattan distance
                    min_distance_to_revealed = min(min_distance_to_revealed, distance)

        if min_distance_to_revealed != float('inf') and min_distance_to_revealed > 2:
            distance_adjustment = 1.1  # Slightly higher probability for isolated areas

        return nearby_mine_density * edge_adjustment * distance_adjustment

    def update_with_constraint_evidence(self, constraints, priors):
        """Update prior probabilities with constraint evidence using Bayesian updating"""
        posteriors = priors.copy()

        # Iterative Bayesian updating
        for iteration in range(10):  # Multiple rounds for convergence
            old_posteriors = posteriors.copy()

            for constraint in constraints:
                constraint_vars = constraint['variables']
                target_mines = constraint['mines']

                # Calculate likelihood of current posterior assignment
                current_expected = sum(posteriors.get(var, 0.5) for var in constraint_vars)

                if current_expected > 0:
                    # Bayesian update factor
                    likelihood_ratio = target_mines / current_expected

                    # Apply update with dampening
                    for var in constraint_vars:
                        if var in posteriors:
                            prior = posteriors[var]
                            # Bayesian update: posterior ∝ prior × likelihood
                            raw_posterior = prior * likelihood_ratio

                            # Apply dampening and normalization
                            dampening = 0.2
                            posteriors[var] = prior * (1 - dampening) + raw_posterior * dampening
                            posteriors[var] = max(0.01, min(0.99, posteriors[var]))

            # Check convergence
            max_change = max(abs(posteriors.get(var, 0.5) - old_posteriors.get(var, 0.5))
                             for constraint in constraints for var in constraint['variables'])

            if max_change < 0.01:
                break

        return posteriors

    def apply_spatial_correlation(self, game, probabilities):
        """Apply spatial correlation to smooth probability estimates"""
        if not probabilities:
            return probabilities

        correlated_probs = probabilities.copy()

        # Apply Gaussian smoothing with constraint awareness
        for (x, y), prob in probabilities.items():
            if game.tiles[x][y]["state"] != STATE_DEFAULT:
                continue

            neighbors = game.get_neighbors(x, y)
            neighbor_probs = []

            for n in neighbors:
                n_pos = (n["coords"]["x"], n["coords"]["y"])
                if n_pos in probabilities and n["state"] == STATE_DEFAULT:
                    neighbor_probs.append(probabilities[n_pos])

            if neighbor_probs:
                # Weighted combination of own probability and neighbor average
                neighbor_avg = sum(neighbor_probs) / len(neighbor_probs)
                smoothing_factor = 0.15  # How much to smooth

                smoothed_prob = prob * (1 - smoothing_factor) + neighbor_avg * smoothing_factor
                correlated_probs[(x, y)] = max(0.01, min(0.99, smoothed_prob))

        return correlated_probs

    def monte_carlo_probability_analysis(self, game):
        """Monte Carlo simulation for complex probability scenarios"""
        constraints = self.build_constraint_system(game)
        if not constraints:
            return None

        # Group constraints for efficiency
        constraint_groups = self.group_related_constraints(constraints)
        large_groups = [g for g in constraint_groups if len(set().union(*[c['variables'] for c in g])) > 10]

        if not large_groups:
            return None

        # Run Monte Carlo simulation on largest group
        largest_group = max(large_groups, key=lambda g: len(set().union(*[c['variables'] for c in g])))
        mc_probabilities = self.run_monte_carlo_simulation(largest_group, num_simulations=2000)

        return self.select_best_probabilistic_move(mc_probabilities)

    def run_monte_carlo_simulation(self, constraint_group, num_simulations=1000):
        """Run Monte Carlo simulation on a constraint group"""
        all_variables = list(set().union(*[c['variables'] for c in constraint_group]))

        if len(all_variables) > 25:  # Limit for performance
            all_variables = all_variables[:25]
            num_simulations = min(num_simulations, 500)

        variable_mine_counts = defaultdict(int)
        successful_simulations = 0
        max_attempts = num_simulations * 20

        for attempt in range(max_attempts):
            if successful_simulations >= num_simulations:
                break

            # Generate weighted random assignment
            assignment = {}
            for var in all_variables:
                # Use slight bias toward safer assignments
                mine_probability = 0.4 + random.random() * 0.2  # 0.4 to 0.6
                assignment[var] = random.random() < mine_probability

            # Check constraint satisfaction with tolerance
            if self.assignment_approximately_satisfies_constraints(assignment, constraint_group):
                successful_simulations += 1
                for var, is_mine in assignment.items():
                    if is_mine:
                        variable_mine_counts[var] += 1

        # Calculate probabilities with confidence intervals
        probabilities = {}
        if successful_simulations > 0:
            for var in all_variables:
                prob = variable_mine_counts[var] / successful_simulations

                # Apply confidence-based adjustment
                confidence = min(1.0, successful_simulations / 500.0)
                adjusted_prob = prob * confidence + 0.5 * (1 - confidence)

                probabilities[var] = max(0.01, min(0.99, adjusted_prob))

        return probabilities

    def assignment_approximately_satisfies_constraints(self, assignment, constraints, tolerance=0.1):
        """Check if assignment approximately satisfies constraints"""
        for constraint in constraints:
            mine_count = sum(1 for var in constraint['variables'] if assignment.get(var, False))
            target = constraint['mines']

            # Allow small tolerance for Monte Carlo
            if abs(mine_count - target) > tolerance:
                return False

        return True

    def select_best_probabilistic_move(self, probabilities):
        """Select the best move based on probability analysis"""
        if not probabilities:
            return None

        # Separate certain and uncertain moves
        certain_safe = [(var, prob) for var, prob in probabilities.items() if prob < 0.05]
        certain_mines = [(var, prob) for var, prob in probabilities.items() if prob > 0.95]
        uncertain = [(var, prob) for var, prob in probabilities.items() if 0.05 <= prob <= 0.95]

        # Priority 1: Certain safe moves
        if certain_safe:
            best_safe = min(certain_safe, key=lambda x: x[1])
            return ("click", best_safe[0][0], best_safe[0][1])

        # Priority 2: Certain mines
        if certain_mines:
            best_mine = max(certain_mines, key=lambda x: x[1])
            return ("flag", best_mine[0][0], best_mine[0][1])

        # Priority 3: Best uncertain move
        if uncertain:
            # Choose based on risk-reward analysis
            best_uncertain = min(uncertain, key=lambda x: x[1])

            # Additional safety check: avoid very high probability tiles
            if best_uncertain[1] < 0.6:  # Reasonable risk threshold
                return ("click", best_uncertain[0][0], best_uncertain[0][1])
            else:
                # If all remaining moves are risky, choose the least risky
                safest = min(probabilities.items(), key=lambda x: x[1])
                return ("click", safest[0][0], safest[0][1])

        return None

    def density_aware_random(self, game):
        """Smart random selection considering local and global density"""
        unknown_tiles = [(x, y) for x in range(game.size_x) for y in range(game.size_y)
                         if game.tiles[x][y]["state"] == STATE_DEFAULT]

        if not unknown_tiles:
            return None

        # Score tiles based on multiple density factors
        tile_scores = {}

        for x, y in unknown_tiles:
            score = 1.0  # Base score

            # Factor 1: Local constraint density (lower is better)
            neighbors = game.get_neighbors(x, y)
            constraint_neighbors = sum(1 for n in neighbors
                                       if n["state"] == STATE_CLICKED and n["mines"] > 0)
            score *= (0.9 ** constraint_neighbors)  # Exponential penalty for constraints

            # Factor 2: High-number avoidance
            high_numbers = sum(1 for n in neighbors
                               if n["state"] == STATE_CLICKED and n["mines"] >= 4)
            score *= (0.7 ** high_numbers)

            # Factor 3: Edge/corner preference in early game
            total_revealed = sum(1 for gx in range(game.size_x) for gy in range(game.size_y)
                                 if game.tiles[gx][gy]["state"] != STATE_DEFAULT)
            game_progress = total_revealed / (game.size_x * game.size_y)

            if game_progress < 0.3:  # Early game
                is_edge = (x == 0 or x == game.size_x - 1 or y == 0 or y == game.size_y - 1)
                if is_edge:
                    score *= 1.2

            # Factor 4: Distance from existing constraints (exploration bonus)
            min_constraint_distance = float('inf')
            for gx in range(game.size_x):
                for gy in range(game.size_y):
                    if (game.tiles[gx][gy]["state"] == STATE_CLICKED and
                            game.tiles[gx][gy]["mines"] > 0):
                        distance = abs(x - gx) + abs(y - gy)
                        min_constraint_distance = min(min_constraint_distance, distance)

            if min_constraint_distance > 2:
                score *= 1.1  # Bonus for exploring new areas

            tile_scores[(x, y)] = score

        # Choose probabilistically based on scores
        total_score = sum(tile_scores.values())
        if total_score > 0:
            # Weighted random selection
            r = random.random() * total_score
            cumulative = 0

            for tile, score in sorted(tile_scores.items(), key=lambda x: x[1], reverse=True):
                cumulative += score
                if r <= cumulative:
                    return ("click", tile[0], tile[1])

        # Fallback to simple random
        return ("click", *random.choice(unknown_tiles))


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
            final_probs = {var: sum(probs) / len(probs)
                           for var, probs in prob_map.items()}

            # Choose the lowest probability tile
            best_var = min(final_probs.items(), key=lambda x: x[1])[0]
            return ("click", best_var[0], best_var[1])

        # Complete fallback to random
        return ("click", *random.choice(list(variables)))


class EnhancedCSPStrategy(AIStrategy):
    def __init__(self):
        self.solution_cache = {}  # Cache solutions for similar patterns
        self.constraint_history = []  # Track constraint evolution

    def next_move(self, game):
        # Phase 1: Quick deterministic moves (highest priority)
        deterministic_move = self.find_deterministic_moves(game)
        if deterministic_move:
            return deterministic_move

        # Phase 2: Enhanced CSP solving with multiple techniques
        csp_move = self.solve_enhanced_csp(game)
        if csp_move:
            return csp_move

        # Phase 3: Advanced probability with CSP insights
        prob_move = self.csp_guided_probability(game)
        if prob_move:
            return prob_move

        # Phase 4: Fallback to strategic random
        return self.strategic_fallback(game)

    def find_deterministic_moves(self, game):
        """Find 100% certain moves using advanced logical rules"""
        constraints = self.build_enhanced_constraints(game)

        # Apply multiple rounds of constraint propagation
        definite_assignments = self.multi_round_propagation(constraints)

        if definite_assignments:
            # Prioritize flagging mines over safe clicks for better AI flow
            mines = [var for var, is_mine in definite_assignments.items() if is_mine]
            safe = [var for var, is_mine in definite_assignments.items() if not is_mine]

            if mines:
                return "flag", mines[0][0], mines[0][1]
            if safe:
                return "click", safe[0][0], safe[0][1]

        return None

    def build_enhanced_constraints(self, game):
        """Build comprehensive constraint system with optimization"""
        constraints = []
        constraint_map = {}  # Track which tiles appear in which constraints

        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]
                if tile["state"] == STATE_CLICKED and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    unknown_neighbors = []
                    flagged_count = 0

                    for n in neighbors:
                        if n["state"] == STATE_DEFAULT:
                            pos = (n["coords"]["x"], n["coords"]["y"])
                            unknown_neighbors.append(pos)
                            # Track constraint membership
                            if pos not in constraint_map:
                                constraint_map[pos] = []
                            constraint_map[pos].append(len(constraints))
                        elif n["state"] == STATE_FLAGGED:
                            flagged_count += 1

                    if unknown_neighbors:
                        remaining_mines = tile["mines"] - flagged_count
                        constraints.append({
                            'vars': unknown_neighbors,
                            'mines': remaining_mines,
                            'source': (x, y),
                            'satisfied': False
                        })

        # Simplify constraints by removing redundancies
        simplified_constraints = self.simplify_constraints(constraints)
        return simplified_constraints, constraint_map

    def simplify_constraints(self, constraints):
        """Remove redundant and dominated constraints"""
        simplified = []

        for i, c1 in enumerate(constraints):
            is_redundant = False

            # Check if this constraint is dominated by others
            for j, c2 in enumerate(constraints):
                if i != j and set(c1['vars']) <= set(c2['vars']):
                    # c1 is subset of c2, check if it's redundant
                    if len(c1['vars']) < len(c2['vars']):
                        # Not redundant, might provide useful info
                        continue
                    elif c1['mines'] == c2['mines']:
                        # Identical constraint, remove duplicate
                        is_redundant = True
                        break

            if not is_redundant:
                simplified.append(c1)

        return simplified

    def multi_round_propagation(self, constraint_data):
        """Advanced constraint propagation with multiple techniques"""
        constraints, constraint_map = constraint_data
        assignment = {}

        max_rounds = 10
        for round_num in range(max_rounds):
            old_assignment_size = len(assignment)

            # Round 1: Basic constraint propagation
            self.basic_propagation_round(constraints, assignment)

            # Round 2: Cross-constraint analysis
            self.cross_constraint_round(constraints, assignment)

            # Round 3: Subset constraint reasoning
            self.subset_constraint_round(constraints, assignment)

            # Round 4: Advanced pattern detection
            self.pattern_detection_round(constraints, assignment)

            # If no progress, break
            if len(assignment) == old_assignment_size:
                break

        return assignment

    def basic_propagation_round(self, constraints, assignment):
        """Basic constraint propagation - faster than original"""
        changed = True
        iterations = 0
        max_iterations = 50  # Prevent infinite loops

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for constraint in constraints:
                if constraint['satisfied']:
                    continue

                vars_list = constraint['vars']
                target_mines = constraint['mines']

                # Filter to unassigned variables
                unassigned = [v for v in vars_list if v not in assignment]
                assigned_mines = sum(1 for v in vars_list if assignment.get(v, False))

                remaining_mines = target_mines - assigned_mines

                # All remaining must be safe
                if remaining_mines == 0 and unassigned:
                    for var in unassigned:
                        assignment[var] = False
                        changed = True

                # All remaining must be mines
                elif remaining_mines == len(unassigned) and remaining_mines > 0:
                    for var in unassigned:
                        assignment[var] = True
                        changed = True

                # Check if constraint is satisfied
                if len(unassigned) == 0:
                    constraint['satisfied'] = True

    def cross_constraint_round(self, constraints, assignment):
        """Analyze interactions between overlapping constraints"""
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                c1, c2 = constraints[i], constraints[j]

                if c1['satisfied'] or c2['satisfied']:
                    continue

                vars1_set = set(c1['vars'])
                vars2_set = set(c2['vars'])
                overlap = vars1_set & vars2_set

                if not overlap:
                    continue

                # Subset reasoning
                if vars1_set <= vars2_set:
                    # c1 is subset of c2
                    remaining_vars = list(vars2_set - vars1_set)
                    if remaining_vars:
                        mines_in_remaining = c2['mines'] - c1['mines']

                        if mines_in_remaining == 0:
                            # All remaining variables are safe
                            for var in remaining_vars:
                                if var not in assignment:
                                    assignment[var] = False
                        elif mines_in_remaining == len(remaining_vars):
                            # All remaining variables are mines
                            for var in remaining_vars:
                                if var not in assignment:
                                    assignment[var] = True

                elif vars2_set <= vars1_set:
                    # c2 is subset of c1
                    remaining_vars = list(vars1_set - vars2_set)
                    if remaining_vars:
                        mines_in_remaining = c1['mines'] - c2['mines']

                        if mines_in_remaining == 0:
                            for var in remaining_vars:
                                if var not in assignment:
                                    assignment[var] = False
                        elif mines_in_remaining == len(remaining_vars):
                            for var in remaining_vars:
                                if var not in assignment:
                                    assignment[var] = True

    def subset_constraint_round(self, constraints, assignment):
        """Advanced subset constraint reasoning"""
        # Look for constraint combinations that create forced moves
        for i in range(len(constraints)):
            c1 = constraints[i]
            if c1['satisfied']:
                continue

            unassigned1 = [v for v in c1['vars'] if v not in assignment]
            if len(unassigned1) <= 1:
                continue

            # Find constraints that share exactly some variables with c1
            for j in range(len(constraints)):
                if i == j:
                    continue

                c2 = constraints[j]
                if c2['satisfied']:
                    continue

                unassigned2 = [v for v in c2['vars'] if v not in assignment]

                shared = set(unassigned1) & set(unassigned2)
                if len(shared) == 0 or len(shared) == len(unassigned1) or len(shared) == len(unassigned2):
                    continue

                # Complex subset analysis
                c1_only = set(unassigned1) - shared
                c2_only = set(unassigned2) - shared

                if len(c1_only) > 0 and len(c2_only) > 0:
                    # Try to derive information about exclusive regions
                    assigned_mines1 = sum(1 for v in c1['vars'] if assignment.get(v, False))
                    assigned_mines2 = sum(1 for v in c2['vars'] if assignment.get(v, False))

                    remaining_mines1 = c1['mines'] - assigned_mines1
                    remaining_mines2 = c2['mines'] - assigned_mines2

                    # If we can determine bounds on the shared region
                    min_shared = max(0, remaining_mines1 - len(c1_only), remaining_mines2 - len(c2_only))
                    max_shared = min(len(shared), remaining_mines1, remaining_mines2)

                    if min_shared == max_shared:
                        # Exact number of mines in shared region is determined
                        mines_in_c1_only = remaining_mines1 - min_shared
                        mines_in_c2_only = remaining_mines2 - min_shared

                        # Apply this knowledge
                        if mines_in_c1_only == 0:
                            for var in c1_only:
                                if var not in assignment:
                                    assignment[var] = False
                        elif mines_in_c1_only == len(c1_only):
                            for var in c1_only:
                                if var not in assignment:
                                    assignment[var] = True

                        if mines_in_c2_only == 0:
                            for var in c2_only:
                                if var not in assignment:
                                    assignment[var] = False
                        elif mines_in_c2_only == len(c2_only):
                            for var in c2_only:
                                if var not in assignment:
                                    assignment[var] = True

    def pattern_detection_round(self, constraints, assignment):
        """Detect and solve common minesweeper patterns"""
        # Look for 1-2-1 patterns and similar
        for constraint in constraints:
            if constraint['satisfied'] or constraint['mines'] != 2:
                continue

            vars_list = [v for v in constraint['vars'] if v not in assignment]
            if len(vars_list) != 3:
                continue

            # Check if this forms a line (common 1-2-1 pattern signature)
            positions = sorted(vars_list)

            # Check for horizontal line
            if (len(set(pos[0] for pos in positions)) == 1 and
                    positions[1][1] == positions[0][1] + 1 and
                    positions[2][1] == positions[1][1] + 1):

                # This might be part of a 1-2-1 pattern
                # Look for neighboring constraints that might complete the pattern
                x = positions[0][0]
                y_start = positions[0][1]

                # Look for 1-constraints on either side
                side_constraints = [c for c in constraints if c['mines'] == 1 and not c['satisfied']]

                for side_c in side_constraints:
                    side_vars = [v for v in side_c['vars'] if v not in assignment]
                    if len(side_vars) == 2:
                        # Check if this could complete a 1-2-1 pattern
                        # This is a simplified version - could be expanded significantly
                        pass

    def solve_enhanced_csp(self, game):
        """Enhanced CSP solving with intelligent search"""
        constraints, constraint_map = self.build_enhanced_constraints(game)
        variables = set()
        for constraint in constraints:
            variables.update(constraint['vars'])

        if not constraints or not variables:
            return None

        # Try guided backtracking for smaller problems
        if len(variables) <= 30:  # Increased from 25
            solution = self.guided_backtrack_search(variables, constraints)
            if solution:
                mines = [var for var, is_mine in solution.items() if is_mine]
                safe = [var for var, is_mine in solution.items() if not is_mine]

                if mines:
                    return ("flag", mines[0][0], mines[0][1])
                if safe:
                    return ("click", safe[0][0], safe[0][1])

        return None

    def guided_backtrack_search(self, variables, constraints):
        """Intelligent backtracking with heuristics"""
        variables_list = self.order_variables_intelligently(variables, constraints)
        constraints_list = self.order_constraints_by_tightness(constraints)

        assignment = {}

        # Pre-process with constraint propagation
        self.multi_round_propagation((constraints_list, {}))

        if self.backtrack_with_heuristics(variables_list, constraints_list, assignment, 0):
            return assignment

        return None

    def order_variables_intelligently(self, variables, constraints):
        """Order variables using Most Constraining Variable heuristic"""
        variable_scores = {}

        for var in variables:
            score = 0

            # Count how many constraints this variable appears in
            constraint_count = sum(1 for c in constraints if var in c['vars'])
            score += constraint_count * 10

            # Prefer variables in tighter constraints
            for constraint in constraints:
                if var in constraint['vars']:
                    tightness = constraint['mines'] / len(constraint['vars'])
                    score += tightness * 5

            # Prefer variables with fewer unassigned neighbors (domain size)
            unassigned_neighbors = len([c for c in constraints
                                        if var in c['vars'] and not c.get('satisfied', False)])
            score += unassigned_neighbors * 3

            variable_scores[var] = score

        # Sort by score descending (most constraining first)
        return sorted(variables, key=lambda v: variable_scores.get(v, 0), reverse=True)

    def order_constraints_by_tightness(self, constraints):
        """Order constraints by tightness for better pruning"""

        def constraint_tightness(c):
            if len(c['vars']) == 0:
                return 0
            return abs(c['mines'] / len(c['vars']) - 0.5)  # Distance from 0.5 probability

        return sorted(constraints, key=constraint_tightness, reverse=True)

    def backtrack_with_heuristics(self, variables, constraints, assignment, var_index):
        """Enhanced backtracking with intelligent pruning"""
        if var_index == len(variables):
            return self.verify_solution(assignment, constraints)

        var = variables[var_index]

        # Most Constraining Value heuristic: try False first (safe), then True (mine)
        # This often leads to faster solutions in minesweeper
        for value in [False, True]:
            assignment[var] = value

            # Early constraint checking for efficiency
            if self.is_consistent_fast(assignment, constraints, var):
                # Forward checking: see if this assignment makes future assignments impossible
                if self.forward_check_viable(assignment, constraints, variables, var_index):
                    if self.backtrack_with_heuristics(variables, constraints, assignment, var_index + 1):
                        return True

            del assignment[var]

        return False

    def is_consistent_fast(self, assignment, constraints, changed_var):
        """Fast consistency check focusing on constraints involving changed variable"""
        for constraint in constraints:
            if changed_var not in constraint['vars']:
                continue  # Skip constraints not involving the changed variable

            assigned_mines = sum(1 for v in constraint['vars']
                                 if v in assignment and assignment[v])
            unassigned = [v for v in constraint['vars'] if v not in assignment]

            # Early pruning conditions
            if assigned_mines > constraint['mines']:
                return False
            if assigned_mines + len(unassigned) < constraint['mines']:
                return False

        return True

    def forward_check_viable(self, assignment, constraints, variables, current_index):
        """Check if remaining variables can satisfy constraints"""
        remaining_vars = set(variables[current_index + 1:])

        for constraint in constraints:
            constraint_vars = set(constraint['vars'])
            remaining_in_constraint = constraint_vars & remaining_vars

            if not remaining_in_constraint:
                continue  # No remaining variables in this constraint

            assigned_mines = sum(1 for v in constraint['vars']
                                 if v in assignment and assignment[v])
            remaining_mines = constraint['mines'] - assigned_mines

            # Check if it's possible to satisfy this constraint
            if remaining_mines < 0:
                return False
            if remaining_mines > len(remaining_in_constraint):
                return False

        return True

    def verify_solution(self, assignment, constraints):
        """Verify that the complete assignment satisfies all constraints"""
        for constraint in constraints:
            actual_mines = sum(1 for v in constraint['vars']
                               if assignment.get(v, False))
            if actual_mines != constraint['mines']:
                return False
        return True

    def csp_guided_probability(self, game):
        """Probability analysis enhanced with CSP insights"""
        constraints, constraint_map = self.build_enhanced_constraints(game)

        if not constraints:
            return self.basic_probability_fallback(game)

        prob_map = {}
        confidence_map = {}  # Track confidence in probability estimates

        # Analyze each variable's probability across multiple constraint contexts
        all_variables = set()
        for constraint in constraints:
            all_variables.update(constraint['vars'])

        for var in all_variables:
            var_probs = []
            var_confidences = []

            # Get probability from each constraint involving this variable
            for constraint in constraints:
                if var in constraint['vars'] and len(constraint['vars']) > 0:
                    basic_prob = constraint['mines'] / len(constraint['vars'])
                    var_probs.append(basic_prob)

                    # Confidence based on constraint tightness and size
                    confidence = min(1.0, len(constraint['vars']) / 8.0)  # More variables = more confidence
                    confidence *= abs(basic_prob - 0.5) * 2  # Extreme probabilities more confident
                    var_confidences.append(confidence)

            if var_probs:
                # Weighted average based on confidence
                if sum(var_confidences) > 0:
                    weighted_prob = sum(p * c for p, c in zip(var_probs, var_confidences)) / sum(var_confidences)
                else:
                    weighted_prob = sum(var_probs) / len(var_probs)

                prob_map[var] = weighted_prob
                confidence_map[var] = sum(var_confidences) / len(var_confidences) if var_confidences else 0

        # Find best move based on probability and confidence
        if prob_map:
            # Prefer high-confidence low-probability tiles
            best_var = min(prob_map.items(),
                           key=lambda x: x[1] - confidence_map.get(x[0], 0) * 0.1)[0]

            if prob_map[best_var] < 0.1:  # Very likely safe
                return ("click", best_var[0], best_var[1])
            elif prob_map[best_var] > 0.9:  # Very likely mine
                return ("flag", best_var[0], best_var[1])
            else:
                # Choose the safest option
                safest = min(prob_map.items(), key=lambda x: x[1])[0]
                return ("click", safest[0], safest[1])

        return None

    def basic_probability_fallback(self, game):
        """Basic probability calculation when no constraints available"""
        # Count remaining mines and tiles
        total_tiles = game.size_x * game.size_y
        revealed_tiles = sum(1 for x in range(game.size_x) for y in range(game.size_y)
                             if game.tiles[x][y]["state"] != STATE_DEFAULT)
        flagged_mines = sum(1 for x in range(game.size_x) for y in range(game.size_y)
                            if game.tiles[x][y]["state"] == STATE_FLAGGED)

        remaining_tiles = total_tiles - revealed_tiles
        remaining_mines = game.total_mines - flagged_mines

        if remaining_tiles > 0:
            global_prob = remaining_mines / remaining_tiles

            # Choose a random tile with global probability consideration
            unknown_tiles = [(x, y) for x in range(game.size_x) for y in range(game.size_y)
                             if game.tiles[x][y]["state"] == STATE_DEFAULT]

            if unknown_tiles:
                # Prefer corner/edge tiles in early game (lower probability)
                if len(unknown_tiles) > (total_tiles * 0.7):
                    corners_edges = [(x, y) for x, y in unknown_tiles
                                     if x == 0 or x == game.size_x - 1 or y == 0 or y == game.size_y - 1]
                    if corners_edges:
                        return ("click", *random.choice(corners_edges))

                return ("click", *random.choice(unknown_tiles))

        return None

    def strategic_fallback(self, game):
        """Intelligent fallback when CSP can't determine moves"""
        unknown_tiles = [(x, y) for x in range(game.size_x) for y in range(game.size_y)
                         if game.tiles[x][y]["state"] == STATE_DEFAULT]

        if not unknown_tiles:
            return None

        # Score tiles based on multiple factors
        tile_scores = {}

        for x, y in unknown_tiles:
            score = 0
            neighbors = game.get_neighbors(x, y)

            # Factor 1: Prefer tiles with fewer numbered neighbors (less constrained)
            numbered_neighbors = sum(1 for n in neighbors
                                     if n["state"] == STATE_CLICKED and n["mines"] > 0)
            score -= numbered_neighbors * 3

            # Factor 2: Avoid tiles near high numbers
            high_numbers = sum(1 for n in neighbors
                               if n["state"] == STATE_CLICKED and n["mines"] >= 4)
            score -= high_numbers * 5

            # Factor 3: Prefer tiles with more unknown neighbors (preservation of options)
            unknown_neighbors = sum(1 for n in neighbors if n["state"] == STATE_DEFAULT)
            score += unknown_neighbors * 2

            # Factor 4: Slight preference for edge/corner tiles in mid-game
            total_revealed = sum(1 for x in range(game.size_x) for y in range(game.size_y)
                                 if game.tiles[x][y]["state"] != STATE_DEFAULT)
            total_tiles = game.size_x * game.size_y

            if total_revealed < total_tiles * 0.6:  # Mid-game
                if x == 0 or x == game.size_x - 1 or y == 0 or y == game.size_y - 1:
                    score += 1

            tile_scores[(x, y)] = score

        # Choose from top 30% of tiles randomly
        sorted_tiles = sorted(tile_scores.items(), key=lambda x: x[1], reverse=True)
        top_portion = max(1, len(sorted_tiles) // 3)
        best_tiles = [tile for tile, _ in sorted_tiles[:top_portion]]

        chosen = random.choice(best_tiles)
        return ("click", chosen[0], chosen[1])


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
            for j, (vars2, mines2) in enumerate(constraints[i + 1:], i + 1):
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

        # Choose the lowest probability
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
                        game.tiles[x + 1][y]["state"] == STATE_CLICKED and game.tiles[x + 1][y]["mines"] == 2 and
                        game.tiles[x + 2][y]["state"] == STATE_CLICKED and game.tiles[x + 2][y]["mines"] == 1):

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
        corners = [(0, 0), (0, game.size_y - 1), (game.size_x - 1, 0), (game.size_x - 1, game.size_y - 1)]

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

            # Factor 1: Prefer tiles with fewer numbered neighbors (less constrained)
            numbered_neighbors = sum(1 for n in neighbors
                                     if n["state"] == STATE_CLICKED and n["mines"] > 0)
            score -= numbered_neighbors * 2

            # Factor 2: Avoid tiles near high numbers
            high_number_neighbors = sum(1 for n in neighbors
                                        if n["state"] == STATE_CLICKED and n["mines"] >= 4)
            score -= high_number_neighbors * 3

            # Factor 3: Prefer tiles with more unknown neighbors (more options later)
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


class BasicPatternStrategy(AIStrategy):
    """
    Class xử lý các pattern cơ bản trong Minesweeper: 1-1, 1-2, 1-2-1, 1-2-2-1.
    Kế thừa AIStrategy để tích hợp vào MinesweeperAI.
    """

    def next_move(self, game) -> Optional[Tuple[str, int, int]]:
        """
        Tìm nước đi dựa trên pattern. Ưu tiên flag mìn trước, sau đó click safe.
        Quét horizontal và vertical.
        """
        # First try basic deterministic moves (like AutoOpenStrategy)
        basic_move = self._find_basic_moves(game)
        if basic_move:
            return basic_move

        # Ưu tiên 1-2-1 vì phổ biến
        move = self._find_121_pattern(game)
        if move:
            return move

        # Tiếp theo 1-2-2-1
        move = self._find_1221_pattern(game)
        if move:
            return move

        # 1-2
        move = self._find_12_pattern(game)
        if move:
            return move

        # 1-1
        move = self._find_11_pattern(game)
        if move:
            return move

        # Fallback to random move
        unclicked_tiles = [(x, y) for x in range(game.size_x) for y in range(game.size_y)
                           if game.tiles[x][y]["state"] == STATE_DEFAULT]
        if unclicked_tiles:
            x, y = random.choice(unclicked_tiles)
            return ('click', x, y)

        return None

    def _find_basic_moves(self, game):
        """Find basic deterministic moves first"""
        for x in range(game.size_x):
            for y in range(game.size_y):
                t = game.tiles[x][y]
                if t["state"] == STATE_CLICKED and t["mines"] > 0:
                    neigh = game.get_neighbors(x, y)
                    flag_cnt = sum(1 for n in neigh if n["state"] == STATE_FLAGGED)
                    unflagged = [n for n in neigh if n["state"] == STATE_DEFAULT]

                    # Auto-open: all mines already flagged
                    if flag_cnt == t["mines"] and unflagged:
                        n = unflagged[0]
                        return ("click", n["coords"]["x"], n["coords"]["y"])

                    # Auto-flag: remaining tiles must all be mines
                    remaining_mines = t["mines"] - flag_cnt
                    if remaining_mines > 0 and len(unflagged) == remaining_mines:
                        n = unflagged[0]
                        return ("flag", n["coords"]["x"], n["coords"]["y"])
        return None

    def _check_tiles_in_line(self, game, tiles_coords):
        """Helper: Check if all tiles in given coordinates exist and are unknown"""
        for x, y in tiles_coords:
            if x < 0 or x >= game.size_x or y < 0 or y >= game.size_y:
                return False
            if game.tiles[x][y]["state"] != STATE_DEFAULT:
                return False
        return True

    def _find_121_pattern(self, game):
        """
        Tìm 1-2-1 horizontal và vertical pattern.
        Pattern: khi có 1-2-1 và một hàng unknown bên trên/dưới,
        thì ô đối diện với các số 1 là mìn, ô đối diện với số 2 là safe.
        """
        # Horizontal 1-2-1 pattern
        for x in range(game.size_x):
            for y in range(game.size_y - 2):
                t1 = game.tiles[x][y]
                t2 = game.tiles[x][y + 1]
                t3 = game.tiles[x][y + 2]

                if (t1["state"] == STATE_CLICKED and t1["mines"] == 1 and
                        t2["state"] == STATE_CLICKED and t2["mines"] == 2 and
                        t3["state"] == STATE_CLICKED and t3["mines"] == 1):

                    # Check row above
                    if x > 0:
                        above_coords = [(x - 1, y), (x - 1, y + 1), (x - 1, y + 2)]
                        if self._check_tiles_in_line(game, above_coords):
                            # Flag mines (opposite to 1s), click safe (opposite to 2)
                            return ('flag', x - 1, y)  # Flag first mine

                    # Check row below
                    if x < game.size_x - 1:
                        below_coords = [(x + 1, y), (x + 1, y + 1), (x + 1, y + 2)]
                        if self._check_tiles_in_line(game, below_coords):
                            return ('flag', x + 1, y)  # Flag first mine

        # Vertical 1-2-1 pattern
        for y in range(game.size_y):
            for x in range(game.size_x - 2):
                t1 = game.tiles[x][y]
                t2 = game.tiles[x + 1][y]
                t3 = game.tiles[x + 2][y]

                if (t1["state"] == STATE_CLICKED and t1["mines"] == 1 and
                        t2["state"] == STATE_CLICKED and t2["mines"] == 2 and
                        t3["state"] == STATE_CLICKED and t3["mines"] == 1):

                    # Check column left
                    if y > 0:
                        left_coords = [(x, y - 1), (x + 1, y - 1), (x + 2, y - 1)]
                        if self._check_tiles_in_line(game, left_coords):
                            return ('flag', x, y - 1)  # Flag first mine

                    # Check column right
                    if y < game.size_y - 1:
                        right_coords = [(x, y + 1), (x + 1, y + 1), (x + 2, y + 1)]
                        if self._check_tiles_in_line(game, right_coords):
                            return ('flag', x, y + 1)  # Flag first mine

        return None

    def _find_1221_pattern(self, game):
        """
        Tìm 1-2-2-1 pattern.
        Similar to 1-2-1 but with 4 tiles: 1-2-2-1
        Mines are opposite to 1s, safe tiles are opposite to 2s
        """
        # Horizontal 1-2-2-1 pattern
        for x in range(game.size_x):
            for y in range(game.size_y - 3):
                t1 = game.tiles[x][y]
                t2 = game.tiles[x][y + 1]
                t3 = game.tiles[x][y + 2]
                t4 = game.tiles[x][y + 3]

                if (t1["state"] == STATE_CLICKED and t1["mines"] == 1 and
                        t2["state"] == STATE_CLICKED and t2["mines"] == 2 and
                        t3["state"] == STATE_CLICKED and t3["mines"] == 2 and
                        t4["state"] == STATE_CLICKED and t4["mines"] == 1):

                    # Check row above
                    if x > 0:
                        above_coords = [(x - 1, y), (x - 1, y + 1), (x - 1, y + 2), (x - 1, y + 3)]
                        if self._check_tiles_in_line(game, above_coords):
                            return ('flag', x - 1, y)  # Flag first mine

                    # Check row below
                    if x < game.size_x - 1:
                        below_coords = [(x + 1, y), (x + 1, y + 1), (x + 1, y + 2), (x + 1, y + 3)]
                        if self._check_tiles_in_line(game, below_coords):
                            return ('flag', x + 1, y)  # Flag first mine

        # Vertical 1-2-2-1 pattern
        for y in range(game.size_y):
            for x in range(game.size_x - 3):
                t1 = game.tiles[x][y]
                t2 = game.tiles[x + 1][y]
                t3 = game.tiles[x + 2][y]
                t4 = game.tiles[x + 3][y]

                if (t1["state"] == STATE_CLICKED and t1["mines"] == 1 and
                        t2["state"] == STATE_CLICKED and t2["mines"] == 2 and
                        t3["state"] == STATE_CLICKED and t3["mines"] == 2 and
                        t4["state"] == STATE_CLICKED and t4["mines"] == 1):

                    # Check column left
                    if y > 0:
                        left_coords = [(x, y - 1), (x + 1, y - 1), (x + 2, y - 1), (x + 3, y - 1)]
                        if self._check_tiles_in_line(game, left_coords):
                            return ('flag', x, y - 1)  # Flag first mine

                    # Check column right
                    if y < game.size_y - 1:
                        right_coords = [(x, y + 1), (x + 1, y + 1), (x + 2, y + 1), (x + 3, y + 1)]
                        if self._check_tiles_in_line(game, right_coords):
                            return ('flag', x, y + 1)  # Flag first mine

        return None

    def _find_12_pattern(self, game):
        """
        Tìm 1-2 pattern followed by unknown tile.
        In a 1-2-X pattern along edges, X is often a mine.
        """
        # Horizontal 1-2 pattern
        for x in range(game.size_x):
            for y in range(game.size_y - 2):
                t1 = game.tiles[x][y]
                t2 = game.tiles[x][y + 1]
                t_unknown = game.tiles[x][y + 2]

                if (t1["state"] == STATE_CLICKED and t1["mines"] == 1 and
                        t2["state"] == STATE_CLICKED and t2["mines"] == 2 and
                        t_unknown["state"] == STATE_DEFAULT):

                    # Check if this is near an edge or has specific neighbor patterns
                    # For simplicity, flag the unknown tile in 1-2-X pattern
                    if x == 0 or x == game.size_x - 1:  # Near top/bottom edge
                        return ('flag', x, y + 2)

        # Vertical 1-2 pattern
        for y in range(game.size_y):
            for x in range(game.size_x - 2):
                t1 = game.tiles[x][y]
                t2 = game.tiles[x + 1][y]
                t_unknown = game.tiles[x + 2][y]

                if (t1["state"] == STATE_CLICKED and t1["mines"] == 1 and
                        t2["state"] == STATE_CLICKED and t2["mines"] == 2 and
                        t_unknown["state"] == STATE_DEFAULT):

                    if y == 0 or y == game.size_y - 1:  # Near left/right edge
                        return ('flag', x + 2, y)

        return None

    def _find_11_pattern(self, game):
        """
        Tìm 1-1 pattern near borders.
        In a 1-1-X pattern near edges, X is often safe.
        """
        # Horizontal 1-1 pattern near edges
        for x in range(game.size_x):
            for y in range(game.size_y - 2):
                t1 = game.tiles[x][y]
                t2 = game.tiles[x][y + 1]
                t_unknown = game.tiles[x][y + 2]

                if (t1["state"] == STATE_CLICKED and t1["mines"] == 1 and
                        t2["state"] == STATE_CLICKED and t2["mines"] == 1 and
                        t_unknown["state"] == STATE_DEFAULT):

                    # Near edge, the third tile is often safe
                    if x == 0 or x == game.size_x - 1:  # Near top/bottom edge
                        return ('click', x, y + 2)

        # Vertical 1-1 pattern near edges
        for y in range(game.size_y):
            for x in range(game.size_x - 2):
                t1 = game.tiles[x][y]
                t2 = game.tiles[x + 1][y]
                t_unknown = game.tiles[x + 2][y]

                if (t1["state"] == STATE_CLICKED and t1["mines"] == 1 and
                        t2["state"] == STATE_CLICKED and t2["mines"] == 1 and
                        t_unknown["state"] == STATE_DEFAULT):

                    if y == 0 or y == game.size_y - 1:  # Near left/right edge
                        return ('click', x + 2, y)

        return None


class AdvancedPatternStrategy(AIStrategy):
    """
    Advanced Pattern Recognition Strategy for Minesweeper
    Recognizes complex patterns including:
    - 1-1-2-X corner patterns
    - 1-2-2-1 wall patterns
    - 3-2-1 sequences
    - L-shaped patterns
    - T-junction patterns
    - And many more advanced configurations
    """

    def __init__(self):
        self.pattern_cache = {}  # Cache successful pattern matches
        self.debug_mode = False  # Set to True for debugging output

    def next_move(self, game):
        """Main pattern recognition logic with prioritized pattern matching"""

        # Phase 1: Basic deterministic moves (highest priority)
        basic_move = self._find_basic_moves(game)
        if basic_move:
            return basic_move

        # Phase 2: Advanced corner patterns (high success rate)
        corner_move = self._find_corner_patterns(game)
        if corner_move:
            return corner_move

        # Phase 3: Wall and edge patterns
        wall_move = self._find_wall_patterns(game)
        if wall_move:
            return wall_move

        # Phase 4: Sequential number patterns
        sequence_move = self._find_sequence_patterns(game)
        if sequence_move:
            return sequence_move

        # Phase 5: L-shaped and T-junction patterns
        junction_move = self._find_junction_patterns(game)
        if junction_move:
            return junction_move

        # Phase 6: Complex geometric patterns
        geometric_move = self._find_geometric_patterns(game)
        if geometric_move:
            return geometric_move

        # Phase 7: Probabilistic pattern matching
        prob_move = self._find_probabilistic_patterns(game)
        if prob_move:
            return prob_move

        # Fallback to smart random
        return self._smart_random_fallback(game)

    def _find_basic_moves(self, game):
        """Find basic deterministic moves (same as before but optimized)"""
        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]
                if tile["state"] == STATE_CLICKED and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    flagged_count = sum(1 for n in neighbors if n["state"] == STATE_FLAGGED)
                    unflagged = [n for n in neighbors if n["state"] == STATE_DEFAULT]

                    # Auto-open: all mines already flagged
                    if flagged_count == tile["mines"] and unflagged:
                        return ("click", unflagged[0]["coords"]["x"], unflagged[0]["coords"]["y"])

                    # Auto-flag: remaining tiles must all be mines
                    remaining_mines = tile["mines"] - flagged_count
                    if remaining_mines > 0 and len(unflagged) == remaining_mines:
                        return ("flag", unflagged[0]["coords"]["x"], unflagged[0]["coords"]["y"])
        return None

    def _find_corner_patterns(self, game):
        """Advanced corner pattern recognition including 1-1-2-X patterns"""

        # Check all corner positions and near-corner areas
        corner_regions = [
            # True corners
            [(0, 0, 3, 3)],  # Top-left
            [(0, game.size_y - 3, 3, 3)],  # Top-right
            [(game.size_x - 3, 0, 3, 3)],  # Bottom-left
            [(game.size_x - 3, game.size_y - 3, 3, 3)]  # Bottom-right
        ]

        for region in corner_regions:
            move = self._analyze_corner_region(game, region[0])
            if move:
                return move

        # Check edge corners (not true corners but near edges)
        for x in range(min(2, game.size_x)):
            for y in range(min(2, game.size_y)):
                move = self._check_112x_pattern(game, x, y)
                if move:
                    return move

        return None

    def _check_112x_pattern(self, game, start_x, start_y):
        """Check for 1-1-2-X corner patterns in various orientations"""

        # Pattern configurations to check
        patterns = [
            # Horizontal then vertical
            [(0, 0, 1), (0, 1, 1), (0, 2, 2), (1, 0), (1, 1), (1, 2)],  # L-shape right
            [(0, 0, 1), (1, 0, 1), (2, 0, 2), (0, 1), (1, 1), (2, 1)],  # L-shape down

            # Diagonal patterns
            [(0, 0, 1), (1, 1, 1), (2, 2, 2), (0, 1), (1, 0), (1, 2), (2, 1)],  # Diagonal

            # More complex corner arrangements
            [(0, 0, 1), (0, 1, 2), (1, 0, 1), (1, 1), (0, 2), (1, 2)],  # Compact L
        ]

        for pattern in patterns:
            move = self._match_corner_pattern(game, start_x, start_y, pattern)
            if move:
                return move

        return None

    def _match_corner_pattern(self, game, base_x, base_y, pattern):
        try:
            known_positions = []
            unknown_positions = []
            for item in pattern:
                if len(item) == 3:
                    dx, dy, expected_mines = item
                    known_positions.append((base_x + dx, base_y + dy, expected_mines))
                elif len(item) == 2:
                    dx, dy = item
                    unknown_positions.append((base_x + dx, base_y + dy))
                else:
                    # unexpected pattern element — skip or log
                    continue

            # bounds check
            all_positions = [(x, y) for x, y, _ in known_positions] + unknown_positions
            for x, y in all_positions:
                if not (0 <= x < game.size_x and 0 <= y < game.size_y):
                    return None

            # verify known positions are revealed and match expected mine counts
            for x, y, expected_mines in known_positions:
                tile = game.tiles[x][y]
                if tile["state"] != STATE_CLICKED or tile["mines"] != expected_mines:
                    return None

            # unknowns must be default
            for x, y in unknown_positions:
                if game.tiles[x][y]["state"] != STATE_DEFAULT:
                    return None

            if len(known_positions) >= 3:
                return self._deduce_corner_move(game, known_positions, unknown_positions)
        except (IndexError, KeyError):
            pass

        return None

    def _deduce_corner_move(self, game, known_positions, unknown_positions):
        """Deduce moves from corner pattern analysis"""

        # For 1-1-2 patterns, common deductions:
        # - If two 1's share a corner with a 2, the corner is usually a mine
        # - If a 2 is between two 1's, middle positions are often safe

        if len(known_positions) == 3 and len(unknown_positions) >= 2:
            mines_counts = [mines for _, _, mines in known_positions]

            # Classic 1-1-2 corner pattern
            if sorted(mines_counts) == [1, 1, 2]:
                # Find the position that's adjacent to both 1's and the 2
                shared_positions = []

                for ux, uy in unknown_positions:
                    adjacent_to_all = True
                    adjacent_count = 0

                    for kx, ky, mines in known_positions:
                        if abs(ux - kx) <= 1 and abs(uy - ky) <= 1 and (ux != kx or uy != ky):
                            adjacent_count += 1

                    if adjacent_count >= 2:  # Adjacent to multiple known tiles
                        shared_positions.append((ux, uy))

                # In 1-1-2 patterns, shared positions are often mines
                if shared_positions:
                    return ("flag", shared_positions[0][0], shared_positions[0][1])

        return None

    def _analyze_corner_region(self, game, region):
        """Analyze a corner region for complex patterns"""
        start_x, start_y, width, height = region

        # Extract all tiles in the region
        region_tiles = []
        for x in range(start_x, min(start_x + width, game.size_x)):
            for y in range(start_y, min(start_y + height, game.size_y)):
                region_tiles.append((x, y, game.tiles[x][y]))

        # Look for specific corner arrangements
        numbered_tiles = [(x, y, tile) for x, y, tile in region_tiles
                          if tile["state"] == STATE_CLICKED and tile["mines"] > 0]
        unknown_tiles = [(x, y, tile) for x, y, tile in region_tiles
                         if tile["state"] == STATE_DEFAULT]

        if len(numbered_tiles) >= 2 and len(unknown_tiles) >= 1:
            return self._analyze_numbered_cluster(game, numbered_tiles, unknown_tiles)

        return None

    def _find_wall_patterns(self, game):
        """Find 1-2-2-1 wall patterns and similar edge configurations"""

        # Check horizontal walls (top and bottom edges)
        for edge_x in [0, game.size_x - 1]:
            for y in range(game.size_y - 3):
                move = self._check_wall_pattern_horizontal(game, edge_x, y)
                if move:
                    return move

        # Check vertical walls (left and right edges)
        for edge_y in [0, game.size_y - 1]:
            for x in range(game.size_x - 3):
                move = self._check_wall_pattern_vertical(game, x, edge_y)
                if move:
                    return move

        # Check internal wall patterns (one row/column from edge)
        for x in [1, game.size_x - 2]:
            for y in range(game.size_y - 3):
                move = self._check_wall_pattern_horizontal(game, x, y)
                if move:
                    return move

        for y in [1, game.size_y - 2]:
            for x in range(game.size_x - 3):
                move = self._check_wall_pattern_vertical(game, x, y)
                if move:
                    return move

        return None

    def _check_wall_pattern_horizontal(self, game, x, start_y):
        """Check for horizontal wall patterns like 1-2-2-1"""

        # Try different pattern lengths
        for length in [4, 3]:  # 1-2-2-1 or 1-2-1
            if start_y + length > game.size_y:
                continue

            tiles = [game.tiles[x][start_y + i] for i in range(length)]

            # Check if this matches known wall patterns
            move = self._match_wall_pattern(game, x, start_y, tiles, "horizontal")
            if move:
                return move

        return None

    def _check_wall_pattern_vertical(self, game, start_x, y):
        """Check for vertical wall patterns like 1-2-2-1"""

        for length in [4, 3]:  # 1-2-2-1 or 1-2-1
            if start_x + length > game.size_x:
                continue

            tiles = [game.tiles[start_x + i][y] for i in range(length)]

            move = self._match_wall_pattern(game, start_x, y, tiles, "vertical")
            if move:
                return move

        return None

    def _match_wall_pattern(self, game, base_x, base_y, tiles, orientation):
        """Match specific wall patterns and deduce moves"""

        # Extract mine counts for revealed tiles
        mine_counts = []
        all_revealed = True

        for tile in tiles:
            if tile["state"] == STATE_CLICKED:
                mine_counts.append(tile["mines"])
            else:
                all_revealed = False

        if not all_revealed or len(mine_counts) < 3:
            return None

        # Check for 1-2-2-1 pattern
        if len(mine_counts) == 4 and mine_counts == [1, 2, 2, 1]:
            return self._solve_1221_wall_pattern(game, base_x, base_y, orientation)

        # Check for 1-2-1 pattern
        if len(mine_counts) == 3 and mine_counts == [1, 2, 1]:
            return self._solve_121_wall_pattern(game, base_x, base_y, orientation)

        # Check for 2-3-2 pattern (higher numbers)
        if len(mine_counts) == 3 and mine_counts == [2, 3, 2]:
            return self._solve_232_wall_pattern(game, base_x, base_y, orientation)

        return None

    def _solve_1221_wall_pattern(self, game, base_x, base_y, orientation):
        """Solve 1-2-2-1 wall patterns"""

        # In 1-2-2-1 patterns along walls, typical solutions:
        # - Positions opposite to 1's are mines
        # - Positions opposite to 2's are safe

        perpendicular_positions = []

        if orientation == "horizontal":
            # Check row above and below
            for offset in [-1, 1]:
                new_x = base_x + offset
                if 0 <= new_x < game.size_x:
                    row_positions = [(new_x, base_y + i) for i in range(4)]
                    if all(0 <= y < game.size_y for _, y in row_positions):
                        perpendicular_positions.extend(row_positions)
        else:  # vertical
            # Check columns left and right
            for offset in [-1, 1]:
                new_y = base_y + offset
                if 0 <= new_y < game.size_y:
                    col_positions = [(base_x + i, new_y) for i in range(4)]
                    if all(0 <= x < game.size_x for x, _ in col_positions):
                        perpendicular_positions.extend(col_positions)

        # Apply 1-2-2-1 logic: flag positions opposite to 1's
        for i, (x, y) in enumerate(perpendicular_positions[:4]):
            if game.tiles[x][y]["state"] == STATE_DEFAULT:
                # Position 0 and 3 correspond to 1's, so they're mines
                if i % 4 in [0, 3]:
                    return ("flag", x, y)
                # Position 1 and 2 correspond to 2's, so they're safe
                elif i % 4 in [1, 2]:
                    return ("click", x, y)

        return None

    def _solve_121_wall_pattern(self, game, base_x, base_y, orientation):
        """Solve 1-2-1 wall patterns"""

        perpendicular_positions = []

        if orientation == "horizontal":
            for offset in [-1, 1]:
                new_x = base_x + offset
                if 0 <= new_x < game.size_x:
                    row_positions = [(new_x, base_y + i) for i in range(3)]
                    if all(0 <= y < game.size_y for _, y in row_positions):
                        perpendicular_positions.extend(row_positions)
        else:
            for offset in [-1, 1]:
                new_y = base_y + offset
                if 0 <= new_y < game.size_y:
                    col_positions = [(base_x + i, new_y) for i in range(3)]
                    if all(0 <= x < game.size_x for x, _ in col_positions):
                        perpendicular_positions.extend(col_positions)

        # Apply 1-2-1 logic: positions 0 and 2 are mines, position 1 is safe
        for i, (x, y) in enumerate(perpendicular_positions[:3]):
            if game.tiles[x][y]["state"] == STATE_DEFAULT:
                if i % 3 in [0, 2]:  # Positions opposite to 1's
                    return ("flag", x, y)
                elif i % 3 == 1:  # Position opposite to 2
                    return ("click", x, y)

        return None

    def _solve_232_wall_pattern(self, game, base_x, base_y, orientation):
        """Solve 2-3-2 wall patterns"""

        # 2-3-2 patterns typically indicate dense mine fields
        # Middle position (opposite to 3) often has mines on both sides
        # Side positions need careful analysis

        perpendicular_positions = []

        if orientation == "horizontal":
            for offset in [-1, 1]:
                new_x = base_x + offset
                if 0 <= new_x < game.size_x:
                    row_positions = [(new_x, base_y + i) for i in range(3)]
                    if all(0 <= y < game.size_y for _, y in row_positions):
                        perpendicular_positions.extend(row_positions)
        else:
            for offset in [-1, 1]:
                new_y = base_y + offset
                if 0 <= new_y < game.size_y:
                    col_positions = [(base_x + i, new_y) for i in range(3)]
                    if all(0 <= x < game.size_x for x, _ in col_positions):
                        perpendicular_positions.extend(col_positions)

        # For 2-3-2, typically all positions opposite are mines
        for x, y in perpendicular_positions[:3]:
            if game.tiles[x][y]["state"] == STATE_DEFAULT:
                return ("flag", x, y)

        return None

    def _find_sequence_patterns(self, game):
        """Find sequential number patterns like 3-2-1"""

        # Check horizontal sequences
        for x in range(game.size_x):
            for y in range(game.size_y - 2):
                move = self._check_sequence_horizontal(game, x, y)
                if move:
                    return move

        # Check vertical sequences
        for x in range(game.size_x - 2):
            for y in range(game.size_y):
                move = self._check_sequence_vertical(game, x, y)
                if move:
                    return move

        # Check diagonal sequences
        for x in range(game.size_x - 2):
            for y in range(game.size_y - 2):
                move = self._check_sequence_diagonal(game, x, y)
                if move:
                    return move

        return None

    def _check_sequence_horizontal(self, game, x, y):
        """Check for horizontal sequential patterns"""

        # Get three consecutive tiles
        tiles = [game.tiles[x][y + i] for i in range(3)]

        # Check if all are revealed
        if not all(tile["state"] == STATE_CLICKED for tile in tiles):
            return None

        mine_counts = [tile["mines"] for tile in tiles]

        # Check for various sequence patterns
        return self._analyze_sequence_pattern(game, x, y, mine_counts, "horizontal")

    def _check_sequence_vertical(self, game, x, y):
        """Check for vertical sequential patterns"""

        tiles = [game.tiles[x + i][y] for i in range(3)]

        if not all(tile["state"] == STATE_CLICKED for tile in tiles):
            return None

        mine_counts = [tile["mines"] for tile in tiles]

        return self._analyze_sequence_pattern(game, x, y, mine_counts, "vertical")

    def _check_sequence_diagonal(self, game, x, y):
        """Check for diagonal sequential patterns"""

        # Check both diagonal directions
        diag1_tiles = [game.tiles[x + i][y + i] for i in range(3)]
        diag2_tiles = [game.tiles[x + i][y + 2 - i] for i in range(3)]

        for tiles, direction in [(diag1_tiles, "diag1"), (diag2_tiles, "diag2")]:
            if all(tile["state"] == STATE_CLICKED for tile in tiles):
                mine_counts = [tile["mines"] for tile in tiles]
                move = self._analyze_sequence_pattern(game, x, y, mine_counts, direction)
                if move:
                    return move

        return None

    def _analyze_sequence_pattern(self, game, base_x, base_y, mine_counts, direction):
        """Analyze sequential patterns and deduce moves"""

        # 3-2-1 descending sequence
        if mine_counts == [3, 2, 1]:
            return self._solve_321_sequence(game, base_x, base_y, direction)

        # 1-2-3 ascending sequence
        if mine_counts == [1, 2, 3]:
            return self._solve_123_sequence(game, base_x, base_y, direction)

        # 2-1-2 pattern
        if mine_counts == [2, 1, 2]:
            return self._solve_212_sequence(game, base_x, base_y, direction)

        # 3-1-3 pattern
        if mine_counts == [3, 1, 3]:
            return self._solve_313_sequence(game, base_x, base_y, direction)

        return None

    def _solve_321_sequence(self, game, base_x, base_y, direction):
        """Solve 3-2-1 sequence patterns"""

        # In 3-2-1 sequences, mines typically concentrate near the 3
        # and become sparser near the 1

        perpendicular_positions = self._get_perpendicular_positions(game, base_x, base_y, 3, direction)

        # Flag positions near the 3 (first position)
        for i, (x, y) in enumerate(perpendicular_positions[:3]):
            if game.tiles[x][y]["state"] == STATE_DEFAULT:
                if i == 0:  # Position opposite to 3
                    # High probability of mines near 3
                    return ("flag", x, y)
                elif i == 2:  # Position opposite to 1
                    # Lower probability near 1, often safe
                    return ("click", x, y)

        return None

    def _solve_123_sequence(self, game, base_x, base_y, direction):
        """Solve 1-2-3 sequence patterns"""

        perpendicular_positions = self._get_perpendicular_positions(game, base_x, base_y, 3, direction)

        # In 1-2-3 sequences, mines concentrate near the 3
        for i, (x, y) in enumerate(perpendicular_positions[:3]):
            if game.tiles[x][y]["state"] == STATE_DEFAULT:
                if i == 2:  # Position opposite to 3
                    return ("flag", x, y)
                elif i == 0:  # Position opposite to 1
                    return ("click", x, y)

        return None

    def _solve_212_sequence(self, game, base_x, base_y, direction):
        """Solve 2-1-2 sequence patterns"""

        perpendicular_positions = self._get_perpendicular_positions(game, base_x, base_y, 3, direction)

        # In 2-1-2 patterns, the middle (opposite to 1) is often safe
        # The sides (opposite to 2's) often have mines
        for i, (x, y) in enumerate(perpendicular_positions[:3]):
            if game.tiles[x][y]["state"] == STATE_DEFAULT:
                if i == 1:  # Position opposite to 1 (middle)
                    return ("click", x, y)
                elif i in [0, 2]:  # Positions opposite to 2's
                    return ("flag", x, y)

        return None

    def _solve_313_sequence(self, game, base_x, base_y, direction):
        """Solve 3-1-3 sequence patterns"""

        perpendicular_positions = self._get_perpendicular_positions(game, base_x, base_y, 3, direction)

        # 3-1-3 patterns usually have mines on the sides and safe in middle
        for i, (x, y) in enumerate(perpendicular_positions[:3]):
            if game.tiles[x][y]["state"] == STATE_DEFAULT:
                if i in [0, 2]:  # Positions opposite to 3's
                    return ("flag", x, y)
                elif i == 1:  # Position opposite to 1
                    return ("click", x, y)

        return None

    def _get_perpendicular_positions(self, game, base_x, base_y, length, direction):
        """Get positions perpendicular to a sequence"""

        positions = []

        if direction == "horizontal":
            for offset in [-1, 1]:
                new_x = base_x + offset
                if 0 <= new_x < game.size_x:
                    for i in range(length):
                        pos = (new_x, base_y + i)
                        if 0 <= pos[1] < game.size_y:
                            positions.append(pos)

        elif direction == "vertical":
            for offset in [-1, 1]:
                new_y = base_y + offset
                if 0 <= new_y < game.size_y:
                    for i in range(length):
                        pos = (base_x + i, new_y)
                        if 0 <= pos[0] < game.size_x:
                            positions.append(pos)

        elif direction in ["diag1", "diag2"]:
            # For diagonal sequences, get adjacent positions
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    for i in range(length):
                        if direction == "diag1":
                            pos = (base_x + i + dx, base_y + i + dy)
                        else:
                            pos = (base_x + i + dx, base_y + 2 - i + dy)

                        if (0 <= pos[0] < game.size_x and 0 <= pos[1] < game.size_y):
                            positions.append(pos)

        return positions

    def _find_junction_patterns(self, game):
        """Find L-shaped and T-junction patterns"""

        # Look for T-junction patterns (three-way intersections)
        for x in range(1, game.size_x - 1):
            for y in range(1, game.size_y - 1):
                move = self._check_t_junction(game, x, y)
                if move:
                    return move

        # Look for L-shaped patterns
        for x in range(game.size_x - 1):
            for y in range(game.size_y - 1):
                move = self._check_l_patterns(game, x, y)
                if move:
                    return move

        return None

    def _check_t_junction(self, game, center_x, center_y):
        """Check for T-junction patterns centered at a position"""

        center_tile = game.tiles[center_x][center_y]

        # Only analyze if center is revealed and has mines
        if center_tile["state"] != STATE_CLICKED or center_tile["mines"] == 0:
            return None

        # Get cross-shaped neighbors (T-junction arms)
        cross_positions = [
            (center_x - 1, center_y),  # Up
            (center_x + 1, center_y),  # Down
            (center_x, center_y - 1),  # Left
            (center_x, center_y + 1)  # Right
        ]

        # Check various T-junction configurations
        for i in range(4):
            # Try each rotation of T-junction
            t_positions = [cross_positions[i], cross_positions[(i + 2) % 4], cross_positions[(i + 3) % 4]]

            # Verify positions are valid
            if all(0 <= x < game.size_x and 0 <= y < game.size_y for x, y in t_positions):
                move = self._analyze_t_configuration(game, center_x, center_y, t_positions)
                if move:
                    return move

        return None

    def _analyze_t_configuration(self, game, center_x, center_y, t_positions):
        """Analyze a specific T-junction configuration"""

        center_mines = game.tiles[center_x][center_y]["mines"]

        # Check if we can determine anything about the T-arms
        revealed_arms = []
        unknown_arms = []

        for x, y in t_positions:
            tile = game.tiles[x][y]
            if tile["state"] == STATE_CLICKED:
                revealed_arms.append((x, y, tile["mines"]))
            elif tile["state"] == STATE_DEFAULT:
                unknown_arms.append((x, y))

        # If we have enough information, make deductions
        if len(revealed_arms) >= 2 and len(unknown_arms) >= 1:
            return self._deduce_from_t_junction(game, center_mines, revealed_arms, unknown_arms)

        return None

    def _deduce_from_t_junction(self, game, center_mines, revealed_arms, unknown_arms):
        """Make deductions from T-junction analysis"""

        # Analyze the relationship between center mines and arm mines
        arm_mine_counts = [mines for _, _, mines in revealed_arms]

        # If center has high mine count and arms have low counts,
        # unknown positions likely have mines
        if center_mines >= 3 and all(count <= 2 for count in arm_mine_counts):
            for x, y in unknown_arms:
                if game.tiles[x][y]["state"] == STATE_DEFAULT:
                    return ("flag", x, y)

        # If center has low mine count and arms have high counts,
        # unknown positions are likely safe
        elif center_mines <= 2 and any(count >= 3 for count in arm_mine_counts):
            for x, y in unknown_arms:
                if game.tiles[x][y]["state"] == STATE_DEFAULT:
                    return ("click", x, y)

        return None

    def _check_l_patterns(self, game, start_x, start_y):
        """Check for L-shaped patterns"""

        # Define L-shape configurations (corner positions)
        l_shapes = [
            # L-shape orientations: [(dx1,dy1), (dx2,dy2), (corner_dx,corner_dy)]
            [(0, 1), (1, 0), (1, 1)],  # Top-left L
            [(0, 1), (-1, 0), (-1, 1)],  # Top-right L
            [(0, -1), (1, 0), (1, -1)],  # Bottom-left L
            [(0, -1), (-1, 0), (-1, -1)]  # Bottom-right L
        ]

        for shape in l_shapes:
            move = self._analyze_l_shape(game, start_x, start_y, shape)
            if move:
                return move

        return None

    def _analyze_l_shape(self, game, base_x, base_y, l_shape):
        """Analyze a specific L-shaped configuration"""

        try:
            # Get the three positions that form the L
            pos1 = (base_x + l_shape[0][0], base_y + l_shape[0][1])
            pos2 = (base_x + l_shape[1][0], base_y + l_shape[1][1])
            corner = (base_x + l_shape[2][0], base_y + l_shape[2][1])

            # Check bounds
            positions = [pos1, pos2, corner]
            if not all(0 <= x < game.size_x and 0 <= y < game.size_y for x, y in positions):
                return None

            # Get tiles
            tile1 = game.tiles[pos1[0]][pos1[1]]
            tile2 = game.tiles[pos2[0]][pos2[1]]
            corner_tile = game.tiles[corner[0]][corner[1]]

            # Check if we have revealed L-arms and unknown corner
            if (tile1["state"] == STATE_CLICKED and tile2["state"] == STATE_CLICKED and
                    corner_tile["state"] == STATE_DEFAULT):

                return self._deduce_from_l_shape(game, tile1["mines"], tile2["mines"], corner)

            # Check reverse: known corner, unknown arms
            elif (corner_tile["state"] == STATE_CLICKED and
                  tile1["state"] == STATE_DEFAULT and tile2["state"] == STATE_DEFAULT):

                # For high corner values, arms likely have mines
                if corner_tile["mines"] >= 4:
                    return ("flag", pos1[0], pos1[1])
                # For low corner values, arms likely safe
                elif corner_tile["mines"] <= 1:
                    return ("click", pos1[0], pos1[1])

        except (IndexError, KeyError):
            pass

        return None

    def _deduce_from_l_shape(self, game, mines1, mines2, corner_pos):
        """Deduce corner tile from L-arm information"""

        # L-shape corner logic:
        # - If both arms have high mine counts, corner likely has mine
        # - If both arms have low mine counts, corner likely safe
        # - Mixed cases require more analysis

        if mines1 >= 3 and mines2 >= 3:
            # Both arms have many mines, corner likely mined
            return ("flag", corner_pos[0], corner_pos[1])
        elif mines1 <= 1 and mines2 <= 1:
            # Both arms have few mines, corner likely safe
            return ("click", corner_pos[0], corner_pos[1])
        elif (mines1 >= 3 and mines2 <= 1) or (mines1 <= 1 and mines2 >= 3):
            # Mixed case - corner probability depends on specific pattern
            # Conservative approach: if one arm has many mines, corner might be safe
            return ("click", corner_pos[0], corner_pos[1])

        return None

    def _find_geometric_patterns(self, game):
        """Find complex geometric patterns like boxes, diamonds, etc."""

        # Check 2x2 box patterns
        for x in range(game.size_x - 1):
            for y in range(game.size_y - 1):
                move = self._check_box_pattern(game, x, y)
                if move:
                    return move

        # Check diamond patterns (3x3 with center focus)
        for x in range(1, game.size_x - 1):
            for y in range(1, game.size_y - 1):
                move = self._check_diamond_pattern(game, x, y)
                if move:
                    return move

        # Check cross patterns
        for x in range(1, game.size_x - 1):
            for y in range(1, game.size_y - 1):
                move = self._check_cross_pattern(game, x, y)
                if move:
                    return move

        return None

    def _check_box_pattern(self, game, x, y):
        """Check 2x2 box patterns"""

        # Get 2x2 box tiles
        box_tiles = [
            game.tiles[x][y], game.tiles[x][y + 1],
            game.tiles[x + 1][y], game.tiles[x + 1][y + 1]
        ]

        # Count revealed and unknown tiles
        revealed = [tile for tile in box_tiles if tile["state"] == STATE_CLICKED]
        unknown = [tile for tile in box_tiles if tile["state"] == STATE_DEFAULT]

        # Need at least 3 revealed tiles for pattern analysis
        if len(revealed) >= 3 and len(unknown) >= 1:
            return self._analyze_box_pattern(game, revealed, unknown)

        return None

    def _analyze_box_pattern(self, game, revealed_tiles, unknown_tiles):
        """Analyze 2x2 box pattern"""

        mine_counts = [tile["mines"] for tile in revealed_tiles]

        # High-density box (many high numbers)
        if sum(mine_counts) >= 9:  # Average > 3
            # Unknown tile likely has mine
            unknown_tile = unknown_tiles[0]
            return ("flag", unknown_tile["coords"]["x"], unknown_tile["coords"]["y"])

        # Low-density box (many low numbers)
        elif sum(mine_counts) <= 3:  # Average <= 1
            # Unknown tile likely safe
            unknown_tile = unknown_tiles[0]
            return ("click", unknown_tile["coords"]["x"], unknown_tile["coords"]["y"])

        return None

    def _check_diamond_pattern(self, game, center_x, center_y):
        """Check diamond/cross patterns centered at position"""

        center_tile = game.tiles[center_x][center_y]

        # Diamond positions (4-directional)
        diamond_positions = [
            (center_x - 1, center_y), (center_x + 1, center_y),
            (center_x, center_y - 1), (center_x, center_y + 1)
        ]

        # Extended diamond (diagonal positions)
        extended_positions = [
            (center_x - 1, center_y - 1), (center_x - 1, center_y + 1),
            (center_x + 1, center_y - 1), (center_x + 1, center_y + 1)
        ]

        return self._analyze_diamond_configuration(game, center_tile,
                                                   diamond_positions, extended_positions)

    def _analyze_diamond_configuration(self, game, center_tile, diamond_pos, extended_pos):
        """Analyze diamond pattern configuration"""

        if center_tile["state"] != STATE_CLICKED:
            return None

        center_mines = center_tile["mines"]

        # Check primary diamond positions
        diamond_tiles = []
        diamond_unknown = []

        for x, y in diamond_pos:
            if 0 <= x < game.size_x and 0 <= y < game.size_y:
                tile = game.tiles[x][y]
                if tile["state"] == STATE_CLICKED:
                    diamond_tiles.append(tile["mines"])
                elif tile["state"] == STATE_DEFAULT:
                    diamond_unknown.append((x, y))

        # Analyze based on center mine count and surrounding pattern
        if center_mines >= 5 and len(diamond_unknown) > 0:
            # High center count suggests dense mine field
            return ("flag", diamond_unknown[0][0], diamond_unknown[0][1])
        elif center_mines <= 2 and len(diamond_unknown) > 0:
            # Low center count suggests sparse mine field
            return ("click", diamond_unknown[0][0], diamond_unknown[0][1])

        return None

    def _check_cross_pattern(self, game, center_x, center_y):
        """Check cross (+) patterns"""

        center_tile = game.tiles[center_x][center_y]

        if center_tile["state"] != STATE_CLICKED:
            return None

        # Cross arms (further out)
        cross_arms = [
            (center_x - 2, center_y), (center_x + 2, center_y),
            (center_x, center_y - 2), (center_x, center_y + 2)
        ]

        valid_arms = []
        unknown_arms = []

        for x, y in cross_arms:
            if 0 <= x < game.size_x and 0 <= y < game.size_y:
                tile = game.tiles[x][y]
                if tile["state"] == STATE_CLICKED:
                    valid_arms.append(tile["mines"])
                elif tile["state"] == STATE_DEFAULT:
                    unknown_arms.append((x, y))

        # Cross pattern deductions
        if len(valid_arms) >= 2 and len(unknown_arms) >= 1:
            avg_arm_mines = sum(valid_arms) / len(valid_arms)
            center_mines = center_tile["mines"]

            # If center is much higher than arms, unknown arms likely safe
            if center_mines >= avg_arm_mines + 2:
                return ("click", unknown_arms[0][0], unknown_arms[0][1])
            # If arms are high and center is high, unknown arms likely mined
            elif center_mines >= 4 and avg_arm_mines >= 3:
                return ("flag", unknown_arms[0][0], unknown_arms[0][1])

        return None

    def _find_probabilistic_patterns(self, game):
        """Advanced probabilistic pattern matching"""

        # This combines multiple weak patterns for probabilistic decisions
        probability_map = {}

        # Collect probability estimates from various sources
        for x in range(game.size_x):
            for y in range(game.size_y):
                if game.tiles[x][y]["state"] == STATE_DEFAULT:
                    prob = self._calculate_position_probability(game, x, y)
                    if prob is not None:
                        probability_map[(x, y)] = prob

        if not probability_map:
            return None

        # Find positions with extreme probabilities
        min_prob_pos = min(probability_map.items(), key=lambda x: x[1])
        max_prob_pos = max(probability_map.items(), key=lambda x: x[1])

        # Very safe positions (probability < 0.1)
        if min_prob_pos[1] < 0.1:
            return ("click", min_prob_pos[0][0], min_prob_pos[0][1])

        # Very dangerous positions (probability > 0.9)
        if max_prob_pos[1] > 0.9:
            return ("flag", max_prob_pos[0][0], max_prob_pos[0][1])

        # Moderately safe positions (probability < 0.3)
        safe_positions = [(pos, prob) for pos, prob in probability_map.items() if prob < 0.3]
        if safe_positions:
            best_safe = min(safe_positions, key=lambda x: x[1])
            return ("click", best_safe[0][0], best_safe[0][1])

        return None

    def _calculate_position_probability(self, game, x, y):
        """Calculate mine probability for a position using multiple factors"""

        # Get neighboring revealed tiles
        neighbors = game.get_neighbors(x, y)
        revealed_neighbors = [n for n in neighbors if n["state"] == STATE_CLICKED and n["mines"] > 0]

        if not revealed_neighbors:
            return None

        # Factor 1: Basic constraint probability
        constraint_probs = []
        for neighbor in revealed_neighbors:
            n_neighbors = game.get_neighbors(neighbor["coords"]["x"], neighbor["coords"]["y"])
            unknown_count = sum(1 for n in n_neighbors if n["state"] == STATE_DEFAULT)
            flagged_count = sum(1 for n in n_neighbors if n["state"] == STATE_FLAGGED)

            if unknown_count > 0:
                remaining_mines = neighbor["mines"] - flagged_count
                basic_prob = max(0, remaining_mines) / unknown_count
                constraint_probs.append(basic_prob)

        if not constraint_probs:
            return None

        base_probability = sum(constraint_probs) / len(constraint_probs)

        # Factor 2: Positional adjustments
        positional_multiplier = 1.0

        # Edge/corner adjustment
        if x == 0 or x == game.size_x - 1 or y == 0 or y == game.size_y - 1:
            positional_multiplier *= 0.9  # Slightly lower probability on edges

        # High-number neighbor adjustment
        high_number_neighbors = sum(1 for n in revealed_neighbors if n["mines"] >= 4)
        if high_number_neighbors > 0:
            positional_multiplier *= (1.0 + 0.2 * high_number_neighbors)

        # Low-number neighbor adjustment
        low_number_neighbors = sum(1 for n in revealed_neighbors if n["mines"] <= 1)
        if low_number_neighbors > 0:
            positional_multiplier *= (1.0 - 0.1 * low_number_neighbors)

        # Factor 3: Pattern-based adjustment
        pattern_multiplier = self._get_pattern_probability_adjustment(game, x, y)

        final_probability = base_probability * positional_multiplier * pattern_multiplier
        return max(0.0, min(1.0, final_probability))

    def _get_pattern_probability_adjustment(self, game, x, y):
        """Get probability adjustment based on local patterns"""

        # Look for local patterns that affect probability
        adjustment = 1.0

        # Check for nearby number sequences
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # Different directions
            sequence = []
            for i in range(-2, 3):  # Check 5 positions in each direction
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < game.size_x and 0 <= ny < game.size_y:
                    tile = game.tiles[nx][ny]
                    if tile["state"] == STATE_CLICKED:
                        sequence.append(tile["mines"])
                    else:
                        sequence.append(None)

            # Analyze sequence for patterns
            if len([s for s in sequence if s is not None]) >= 3:
                pattern_adj = self._analyze_sequence_for_probability(sequence, 2)  # Position 2 is our target
                adjustment *= pattern_adj

        return adjustment

    def _analyze_sequence_for_probability(self, sequence, target_index):
        """Analyze a sequence to adjust probability for target position"""

        if target_index >= len(sequence) or sequence[target_index] is not None:
            return 1.0

        # Look for ascending/descending patterns
        before_values = [s for s in sequence[:target_index] if s is not None]
        after_values = [s for s in sequence[target_index + 1:] if s is not None]

        if len(before_values) >= 2:
            # Check if ascending
            if all(before_values[i] < before_values[i + 1] for i in range(len(before_values) - 1)):
                return 1.3  # Higher probability in ascending sequence
            # Check if descending
            elif all(before_values[i] > before_values[i + 1] for i in range(len(before_values) - 1)):
                return 0.7  # Lower probability in descending sequence

        return 1.0

    def _smart_random_fallback(self, game):
        """Intelligent random selection when no patterns are found"""

        unknown_tiles = [(x, y) for x in range(game.size_x) for y in range(game.size_y)
                         if game.tiles[x][y]["state"] == STATE_DEFAULT]

        if not unknown_tiles:
            return None

        # Score each unknown tile
        tile_scores = {}
        for x, y in unknown_tiles:
            score = self._calculate_fallback_score(game, x, y)
            tile_scores[(x, y)] = score

        # Choose from top 20% of tiles
        sorted_tiles = sorted(tile_scores.items(), key=lambda x: x[1], reverse=True)
        top_count = max(1, len(sorted_tiles) // 5)
        best_tiles = [tile for tile, _ in sorted_tiles[:top_count]]

        chosen = random.choice(best_tiles)
        return ("click", chosen[0], chosen[1])

    def _calculate_fallback_score(self, game, x, y):
        """Calculate a score for fallback tile selection"""

        score = 0
        neighbors = game.get_neighbors(x, y)

        # Prefer tiles with fewer high-number neighbors
        high_numbers = sum(1 for n in neighbors if n["state"] == STATE_CLICKED and n["mines"] >= 4)
        score -= high_numbers * 3

        # Prefer tiles with more unknown neighbors (keeps options open)
        unknown_neighbors = sum(1 for n in neighbors if n["state"] == STATE_DEFAULT)
        score += unknown_neighbors

        # Prefer tiles away from edges in late game
        total_unknown = sum(1 for gx in range(game.size_x) for gy in range(game.size_y)
                            if game.tiles[gx][gy]["state"] == STATE_DEFAULT)
        total_tiles = game.size_x * game.size_y

        if total_unknown < total_tiles * 0.3:  # Late game
            # Prefer center tiles
            center_x, center_y = game.size_x // 2, game.size_y // 2
            distance_from_center = abs(x - center_x) + abs(y - center_y)
            score -= distance_from_center * 0.5
        else:  # Early/mid game
            # Prefer edge tiles
            is_edge = (x == 0 or x == game.size_x - 1 or y == 0 or y == game.size_y - 1)
            if is_edge:
                score += 2

        return score

    def _analyze_numbered_cluster(self, game, numbered_tiles, unknown_tiles):
        """Analyze clusters of numbered tiles for pattern deduction"""

        if len(numbered_tiles) < 2 or len(unknown_tiles) < 1:
            return None

        # Extract mine counts and positions
        mine_counts = [tile["mines"] for _, _, tile in numbered_tiles]
        positions = [(x, y) for x, y, _ in numbered_tiles]

        # Look for specific cluster patterns
        total_mines = sum(mine_counts)
        avg_mines = total_mines / len(mine_counts)

        # High-density cluster
        if avg_mines >= 3.5:
            # Unknown tiles likely have mines
            unknown_pos = unknown_tiles[0][:2]
            return ("flag", unknown_pos[0], unknown_pos[1])

        # Low-density cluster
        elif avg_mines <= 1.5:
            # Unknown tiles likely safe
            unknown_pos = unknown_tiles[0][:2]
            return ("click", unknown_pos[0], unknown_pos[1])

        # Check for specific number combinations
        if len(numbered_tiles) == 3:
            sorted_mines = sorted(mine_counts)

            # 1-1-3 pattern
            if sorted_mines == [1, 1, 3]:
                unknown_pos = unknown_tiles[0][:2]
                # Position near the 3 is likely a mine
                return ("flag", unknown_pos[0], unknown_pos[1])

            # 1-2-2 pattern
            elif sorted_mines == [1, 2, 2]:
                unknown_pos = unknown_tiles[0][:2]
                # Moderate probability, lean toward safe
                return ("click", unknown_pos[0], unknown_pos[1])

        return None


STRATEGIES = {
    "Random": RandomStrategy,
    "AutoOpen": AutoOpenStrategy,
    "Probabilistic": EnhancedProbabilisticStrategy,
    "Pattern": AdvancedPatternStrategy,
    "CSP": EnhancedCSPStrategy,
    "Hybrid": HybridStrategy,
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
            if time_taken and (
                    not self.stats[difficulty]["best_time"] or time_taken < self.stats[difficulty]["best_time"]):
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
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

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
        ttk.Spinbox(self.custom_frame, from_=5, to=50, textvariable=self.custom_width, width=10).grid(row=0, column=1,
                                                                                                      padx=5, pady=2)

        ttk.Label(self.custom_frame, text="Height:").grid(row=1, column=0, padx=5, pady=2)
        self.custom_height = IntVar(value=DIFFICULTIES["Custom"]["size_x"])
        ttk.Spinbox(self.custom_frame, from_=5, to=30, textvariable=self.custom_height, width=10).grid(row=1, column=1,
                                                                                                       padx=5, pady=2)

        ttk.Label(self.custom_frame, text="Mines:").grid(row=2, column=0, padx=5, pady=2)
        self.custom_mines = IntVar(value=DIFFICULTIES["Custom"]["mines"])
        ttk.Spinbox(self.custom_frame, from_=1, to=500, textvariable=self.custom_mines, width=10).grid(row=2, column=1,
                                                                                                       padx=5, pady=2)

        # Game options
        self.safe_first_var = BooleanVar(value=self.settings.safe_first_click)
        ttk.Checkbutton(game_frame, text="Safe first click", variable=self.safe_first_var).grid(row=2, column=0,
                                                                                                columnspan=2, sticky=W,
                                                                                                padx=5, pady=5)

        self.auto_flag_var = BooleanVar(value=self.settings.auto_flag)
        ttk.Checkbutton(game_frame, text="Auto flag when all mines found", variable=self.auto_flag_var).grid(row=3,
                                                                                                             column=0,
                                                                                                             columnspan=2,
                                                                                                             sticky=W,
                                                                                                             padx=5,
                                                                                                             pady=5)

        # Appearance Tab
        appear_frame = ttk.Frame(notebook)
        notebook.add(appear_frame, text="Appearance")

        ttk.Label(appear_frame, text="Theme:").grid(row=0, column=0, sticky=W, padx=5, pady=5)
        self.theme_var = StringVar(value=self.settings.theme)
        theme_combo = ttk.Combobox(appear_frame, textvariable=self.theme_var,
                                   values=list(THEMES.keys()), state="readonly")
        theme_combo.grid(row=0, column=1, sticky=EW, padx=5, pady=5)

        self.show_timer_var = BooleanVar(value=self.settings.show_timer)
        ttk.Checkbutton(appear_frame, text="Show timer", variable=self.show_timer_var).grid(row=1, column=0,
                                                                                            columnspan=2, sticky=W,
                                                                                            padx=5, pady=5)

        # Other Tab
        other_frame = ttk.Frame(notebook)
        notebook.add(other_frame, text="Other")

        self.play_sounds_var = BooleanVar(value=self.settings.play_sounds)
        ttk.Checkbutton(other_frame, text="Play sounds", variable=self.play_sounds_var).grid(row=0, column=0,
                                                                                             columnspan=2, sticky=W,
                                                                                             padx=5, pady=5)

        self.save_stats_var = BooleanVar(value=self.settings.save_stats)
        ttk.Checkbutton(other_frame, text="Save statistics", variable=self.save_stats_var).grid(row=1, column=0,
                                                                                                columnspan=2, sticky=W,
                                                                                                padx=5, pady=5)

        # Buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(button_frame, text="OK", command=self.on_ok).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side="right")
        ttk.Button(button_frame, text="Reset to Defaults", command=self.on_reset).pack(side="left")

        self.on_difficulty_change()

    def on_difficulty_change(self, event=None):
        is_custom = self.difficulty_var.get() == "Custom"
        for widget in self.custom_frame.winfo_children():
            widget.configure(state="normal" if is_custom else "disabled")

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
            tree.column(col, width=80, anchor="center")

        # Populate with data
        for difficulty, data in stats.stats.items():
            games_played = data["games_played"]
            games_won = data["games_won"]
            win_rate = f"{(games_won / games_played * 100):.1f}%" if games_played > 0 else "0%"
            best_time = f"{data['best_time']:.2f}s" if data["best_time"] else "N/A"

            tree.insert("", "end", values=(difficulty, games_played, games_won, win_rate, best_time))

        tree.pack(fill="both", expand=True, padx=10, pady=10)

        # Close button
        ttk.Button(self.window, text="Close", command=self.window.destroy).pack(pady=5)


class Minesweeper:
    def __init__(self, tk):
        self.start_time = None
        self.tk = tk
        self.settings = Settings()
        self.stats = Statistics()
        self.strategy_name = StringVar(value="Random")

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
        game_menu.add_command(label="Change Strategy", command=self.change_strategy, accelerator="F5")
        game_menu.add_command(label="Auto Play", command=self.auto_play, accelerator="F6")

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
        self.tk.bind("<F6>", lambda e: self.change_strategy())
        self.tk.bind("<F5>", lambda e: self.auto_play())


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
        self.top_frame.pack(fill="x", pady=(0, 10))

        # Status labels
        self.labels = {
            "mines": Label(self.top_frame, text=f"Mines: {self.total_mines}", font=("Arial", 12, "bold")),
            "flags": Label(self.top_frame, text="Flags: 0", font=("Arial", 12, "bold")),
            "time": Label(self.top_frame, text="00:00:00", font=("Arial", 12, "bold"))
        }

        self.labels["mines"].pack(side="left")
        self.labels["flags"].pack(side="left", padx=(20, 0))
        self.labels["time"].pack(side="right")

        # Game frame for the minefield
        self.game_frame = Frame(self.main_frame)
        self.game_frame.pack()

        # Bottom frame for controls
        self.bottom_frame = Frame(self.main_frame)
        self.bottom_frame.pack(fill="x", pady=(10, 0))

        # Control buttons
        restart_btn = Button(self.bottom_frame, text="New Game (F2)", command=self.restart,
                             font=("Arial", 10), padx=10)
        restart_btn.pack(side="left")

        settings_btn = Button(self.bottom_frame, text="Settings (F3)", command=self.open_settings,
                              font=("Arial", 10), padx=10)
        settings_btn.pack(side="left", padx=(10, 0))

        stats_btn = Button(self.bottom_frame, text="Stats (F4)", command=self.open_stats,
                           font=("Arial", 10), padx=10)
        stats_btn.pack(side="left", padx=(10, 0))

        # Strategy selection button
        self.strategy_btn = Button(self.bottom_frame, text=f"Strategy: {self.strategy_name.get()}",
                                   command=self.cycle_strategy, font=("Arial", 10), padx=10, width=25)
        self.strategy_btn.pack(side="left", padx=(10, 0))

        self.auto_play_btn = Button(self.bottom_frame, text="Auto Play", command=self.toggle_auto_play,
                            font=("Arial", 10), padx=10, width=20)  # Add fixed width
        self.auto_play_btn.pack(side="left", padx=(10, 0))

        # AI status
        self.ai_running = False
        self.ai_status_label = Label(self.bottom_frame, text="AI: Stopped", font=("Arial", 10), width=15)  # Add fixed width
        self.ai_status_label.pack(side="left", padx=(10, 0))

        # Difficulty label
        difficulty_text = f"Difficulty: {self.settings.difficulty}"
        if self.settings.difficulty == "Custom":
            difficulty_text += f" ({self.size_y}x{self.size_x}, {self.total_mines} mines)"

        self.difficulty_label = Label(self.bottom_frame, text=difficulty_text, font=("Arial", 9))
        self.difficulty_label.pack(side="right")

    def toggle_auto_play(self):
        """Toggle autoplay on/off."""
        if self.game_over_flag:
            return

        if not self.ai_running:
            self.ai_running = True
            self.auto_play_btn.config(text="Stop AI")
            self.ai_status_label.config(text="AI: Running")
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
        """Load images from the images folder, scaling them to match button size"""
        self.images = {}

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
                    original_img = PhotoImage(file=filepath)
                    # Scale the image to better fit 30x30 buttons
                    # If original is smaller, zoom it; if larger, subsample it
                    orig_width = original_img.width()
                    orig_height = original_img.height()

                    if orig_width < 30 or orig_height < 30:
                        # Zoom up if too small
                        zoom_factor = max(30 // orig_width, 30 // orig_height, 1)
                        self.images[key] = original_img.zoom(zoom_factor, zoom_factor)
                    elif orig_width > 30 or orig_height > 30:
                        # Subsample down if too large
                        subsample_factor = max(orig_width // 30, orig_height // 30, 1)
                        self.images[key] = original_img.subsample(subsample_factor, subsample_factor)
                    else:
                        # Perfect size
                        self.images[key] = original_img
                else:
                    # Fallback to colored rectangles
                    self.images[key] = self.create_fallback_image(key)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                self.images[key] = self.create_fallback_image(key)

        # Load number tiles (1-8)
        self.images["numbers"] = []
        for i in range(1, 9):
            try:
                filepath = os.path.join(image_path, f"tile_{i}.gif")
                if os.path.exists(filepath):
                    original_img = PhotoImage(file=filepath)
                    # Scale the image to better fit 30x30 buttons
                    orig_width = original_img.width()
                    orig_height = original_img.height()
                    scaled_img = None
                    if orig_width < 30 or orig_height < 30:
                        # Zoom up if too small
                        zoom_factor = max(30 // orig_width, 30 // orig_height, 1)
                        scaled_img = original_img.zoom(zoom_factor, zoom_factor)
                    elif orig_width > 30 or orig_height > 30:
                        # Subsample down if too large
                        subsample_factor = max(orig_width // 30, orig_height // 30, 1)
                        scaled_img = original_img.subsample(subsample_factor, subsample_factor)
                    else:
                        # Perfect size
                        scaled_img = original_img

                    self.images["numbers"].append(scaled_img)
                else:
                    self.images["numbers"].append(self.create_fallback_number_image(i))
            except Exception as e:
                print(f"Error loading tile_{i}.gif: {e}")
                self.images["numbers"].append(self.create_fallback_number_image(i))

    def create_fallback_image(self, image_type):
        """Create fallback colored rectangle images"""
        theme = THEMES[self.settings.theme]
        img = PhotoImage(width=30, height=30)  # Changed from 20x20 to 30x30

        if image_type == "plain":
            img.put(theme["button_bg"], to=(0, 0, 30, 30))
        elif image_type == "clicked":
            img.put("#ffffff", to=(0, 0, 30, 30))
        elif image_type == "mine":
            img.put(theme["mine_color"], (0, 0, 30, 30))
        elif image_type == "flag":
            img.put(theme["flag_color"], (0, 0, 30, 30))
        elif image_type == "wrong":
            img.put("#888888", (0, 0, 30, 30))

        return img

    def create_fallback_number_image(self, number):
        """Create fallback number images"""
        img = PhotoImage(width=20, height=20)
        # img.put("#ffffff", (0, 0, 20, 20))
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
                                     width=30, height=30, bd=1, relief="raised")
                }

                tile["button"].bind(BTN_CLICK, self.on_click_wrapper(x, y))
                tile["button"].bind(BTN_FLAG, self.on_right_click_wrapper(x, y))
                tile["button"].bind("<Double-Button-1>",
                                    self.on_double_click_wrapper(x, y))  # Double-click for auto-open
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

    def show_game_over_dialog(self, won):
        """Show a custom game over dialog with larger text"""
        # Create custom dialog window
        dialog = Toplevel(self.tk)
        dialog.title("You Win!" if won else "Mine Exploded!")
        dialog.resizable(False, False)
        dialog.grab_set()  # Make it modal

        # Configure dialog styling
        dialog.configure(bg="#f0f0f0")

        # Prepare message text
        if won:
            msg = (
                f"🎉 You cleared the minefield!\n\n"
                f"Time: {self.time_taken}\n"
                f"Flags used: {self.flag_count}\n"
                f"Difficulty: {self.settings.difficulty}\n\n"
                "Would you like to play again?"
            )
        else:
            msg = (
                "💥 Boom! You hit a mine.\n\n"
                f"Flags placed: {self.flag_count}\n"
                f"Difficulty: {self.settings.difficulty}\n\n"
                "Try again?"
            )

        # Create and pack the message label with large font
        message_label = Label(
            dialog,
            text=msg,
            font=("Arial", 14, "normal"),  # Much larger font
            bg="#f0f0f0",
            fg="#333333",
            justify="center",
            padx=30,
            pady=20
        )
        message_label.pack(pady=(20, 10))

        # Frame for buttons
        button_frame = Frame(dialog, bg="#f0f0f0")
        button_frame.pack(pady=(10, 20))

        # Result variable to store user choice
        result = [False]  # Use list so it's mutable in nested function

        def on_yes():
            result[0] = True
            dialog.destroy()

        def on_no():
            result[0] = False
            dialog.destroy()

        # Create buttons with larger font
        yes_btn = Button(
            button_frame,
            text="Yes",
            command=on_yes,
            font=("Arial", 12, "bold"),
            padx=20,
            pady=8,
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            width=8
        )
        yes_btn.pack(side="left", padx=(0, 10))

        no_btn = Button(
            button_frame,
            text="No",
            command=on_no,
            font=("Arial", 12, "bold"),
            padx=20,
            pady=8,
            bg="#f44336",
            fg="white",
            activebackground="#da190b",
            width=8
        )
        no_btn.pack(side="left")

        # Center the dialog
        dialog.update_idletasks()
        width = dialog.winfo_reqwidth()
        height = dialog.winfo_reqheight()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"+{x}+{y}")

        # Set focus and bind Enter/Escape keys
        yes_btn.focus_set()
        dialog.bind("<Return>", lambda e: on_yes())
        dialog.bind("<Escape>", lambda e: on_no())

        # Wait for dialog to close
        dialog.wait_window()

        return result[0]

    def game_over(self, won, clicked_mine_x=None, clicked_mine_y=None):
        self.game_over_flag = True

        # Stop AI if running
        if self.ai_running:
            self.ai_running = False
            self.auto_play_btn.config(  text="Auto Play  ")
            self.ai_status_label.config(text="AI: Stopped")

        # Record statistics
        if self.settings.save_stats and self.start_time:
            self.time_taken = (datetime.now() - self.start_time).total_seconds()
            self.stats.add_game(self.settings.difficulty, won, self.time_taken if won else None)
            self.stats.save_stats()

        if not won:
            # Show all mines and wrong flags
            for x in range(self.size_x):
                for y in range(self.size_y):
                    tile = self.tiles[x][y]
                    if tile["is_mine"] and tile["state"] != STATE_FLAGGED:
                        # # Check if this is the mine that was clicked to trigger game over
                        # if clicked_mine_x is not None and x == clicked_mine_x and y == clicked_mine_y:
                        #     # Highlight the clicked mine with red background
                        #     tile["button"].config(image=self.images["mine"], bg="red", activebackground="red")
                        # else:
                        #     # Show other mines normally
                        #     tile["button"].config(image=self.images["mine"])
                        tile["button"].config(image=self.images["mine"])
                    elif not tile["is_mine"] and tile["state"] == STATE_FLAGGED:
                        # Show wrong flags with red background
                        tile["button"].config(image=self.images["wrong"], bg="red", activebackground="red")
        else:
            # For winning, show all unflagged mines
            for x in range(self.size_x):
                for y in range(self.size_y):
                    tile = self.tiles[x][y]
                    if tile["is_mine"] and tile["state"] != STATE_FLAGGED:
                        tile["button"].config(image=self.images["mine"])

        self.tk.update()

        if self.show_game_over_dialog(won):
            self.tk.after(100, self.restart)

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

        self.tk.after(25, self.update_timer)

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
            # Pass the coordinates of the clicked mine to highlight it
            self.game_over(False, tile["coords"]["x"], tile["coords"]["y"])
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
    window.title("Minesweeper")
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
