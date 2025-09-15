import json
import os
import platform
import random
import time
import sys
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from collections import deque
from datetime import datetime
from tkinter import *
from tkinter import ttk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

HAS_TENSORFLOW = False
tf = None
keras = None
layers = None

def load_tensorflow():
    """Lazy load TensorFlow only when needed with better error handling"""
    global HAS_TENSORFLOW, tf, keras, layers
    if not HAS_TENSORFLOW and tf is None:
        try:
            print("Loading TensorFlow... (this may take a moment)")
            
            # Try different import methods
            try:
                import tensorflow as tf_module
            except ImportError as e1:
                print(f"Standard TensorFlow import failed: {e1}")
                try:
                    # Try alternative import
                    import tensorflow.compat.v2 as tf_module
                    tf_module.enable_v2_behavior()
                except ImportError as e2:
                    print(f"TensorFlow v2 import also failed: {e2}")
                    raise ImportError("TensorFlow not available")
            
            from tensorflow import keras as keras_module  
            from tensorflow.keras import layers as layers_module
            
            # Test if TensorFlow is working
            test_tensor = tf_module.constant([1, 2, 3])
            
            # Configure TensorFlow for better compatibility
            tf_module.config.set_soft_device_placement(True)
            
            # Set CPU only to avoid GPU issues
            try:
                tf_module.config.set_visible_devices([], 'GPU')
                print("GPU disabled, using CPU only")
            except:
                print("Could not disable GPU, continuing anyway")
            
            tf = tf_module
            keras = keras_module
            layers = layers_module
            HAS_TENSORFLOW = True
            print("TensorFlow loaded successfully!")
            
        except Exception as e:
            print(f"TensorFlow loading failed: {e}")
            print("CNN will use pattern-based fallback.")
            HAS_TENSORFLOW = False
            return False
    
    return HAS_TENSORFLOW



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
        # Phase 1: Try deterministic moves first (these are fast and reliable)
        deterministic_move = self.find_deterministic_moves(game)
        if deterministic_move:
            return deterministic_move

        # Phase 2: Try CSP solving for more complex situations
        csp_move = self.solve_csp(game)
        if csp_move:
            return csp_move

        # Phase 3: Probabilistic fallback
        return self.probability_fallback(game)

    def find_deterministic_moves(self, game):
        """Find obvious safe/mine moves using basic minesweeper logic"""
        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]
                if tile["state"] == STATE_CLICKED and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    unknown_neighbors = [n for n in neighbors if n["state"] == STATE_DEFAULT]
                    flagged_neighbors = [n for n in neighbors if n["state"] == STATE_FLAGGED]

                    # Safe click: all required mines are already flagged
                    if len(flagged_neighbors) == tile["mines"] and unknown_neighbors:
                        n = unknown_neighbors[0]
                        return ('click', n["coords"]["x"], n["coords"]["y"])

                    # Sure flag: remaining unknown tiles must all be mines
                    remaining_mines = tile["mines"] - len(flagged_neighbors)
                    if remaining_mines > 0 and len(unknown_neighbors) == remaining_mines:
                        n = unknown_neighbors[0]
                        return ('flag', n["coords"]["x"], n["coords"]["y"])

        return None

    def build_constraints(self, game):
        """Build CSP constraints from the current board state"""
        constraints = []
        variables = set()

        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]
                if tile["state"] == STATE_CLICKED and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    unknown_neighbors = []
                    flagged_count = 0

                    for n in neighbors:
                        if n["state"] == STATE_DEFAULT:
                            coord = (n["coords"]["x"], n["coords"]["y"])
                            unknown_neighbors.append(coord)
                            variables.add(coord)
                        elif n["state"] == STATE_FLAGGED:
                            flagged_count += 1

                    # Only add constraint if there are unknown neighbors
                    if unknown_neighbors:
                        required_mines = tile["mines"] - flagged_count
                        # Skip invalid constraints
                        if 0 <= required_mines <= len(unknown_neighbors):
                            constraints.append((unknown_neighbors, required_mines))

        return list(variables), constraints

    def solve_csp(self, game):
        """Solve using constraint satisfaction with improved backtracking"""
        variables, constraints = self.build_constraints(game)

        if not variables or not constraints:
            return None

        # Limit problem size to avoid timeouts
        if len(variables) > 30:
            # Focus on the most constrained area
            variables = self.select_most_constrained_variables(variables, constraints, 25)
            # Filter constraints to only include selected variables
            constraints = [(vars_list, mines) for vars_list, mines in constraints
                           if any(v in variables for v in vars_list)]

        # Try to find solutions using backtracking
        solutions = []
        self.backtrack_solutions(variables, constraints, {}, 0, solutions, max_solutions=50)

        if solutions:
            # Analyze solutions to find certain moves
            certain_safe, certain_mines = self.analyze_solutions(variables, solutions)

            if certain_safe:
                x, y = certain_safe[0]
                return ('click', x, y)
            elif certain_mines:
                x, y = certain_mines[0]
                return ('flag', x, y)

        return None

    def select_most_constrained_variables(self, variables, constraints, limit):
        """Select the most constrained variables to focus CSP solving"""
        variable_scores = {}

        for var in variables:
            # Count how many constraints involve this variable
            constraint_count = sum(1 for vars_list, _ in constraints if var in vars_list)
            variable_scores[var] = constraint_count

        # Sort by constraint count (most constrained first)
        sorted_vars = sorted(variable_scores.items(), key=lambda x: x[1], reverse=True)
        return [var for var, _ in sorted_vars[:limit]]

    def backtrack_solutions(self, variables, constraints, assignment, var_index, solutions, max_solutions=100):
        """Find solutions using backtracking with pruning"""
        if len(solutions) >= max_solutions:
            return False

        if var_index == len(variables):
            if self.is_complete_solution_valid(assignment, constraints):
                solutions.append(assignment.copy())
            return True

        var = variables[var_index]

        # Try both values: 0 (safe) and 1 (mine)
        for value in [0, 1]:
            assignment[var] = value

            if self.is_partial_assignment_valid(assignment, constraints):
                if not self.backtrack_solutions(variables, constraints, assignment,
                                                var_index + 1, solutions, max_solutions):
                    break

            del assignment[var]

        return True

    def is_partial_assignment_valid(self, assignment, constraints):
        """Check if partial assignment could lead to a valid solution"""
        for vars_list, required_mines in constraints:
            assigned_vars = [v for v in vars_list if v in assignment]
            unassigned_vars = [v for v in vars_list if v not in assignment]

            if assigned_vars:
                current_mines = sum(assignment[v] for v in assigned_vars)
                remaining_mines = required_mines - current_mines

                # Check if it's possible to satisfy the constraint
                if remaining_mines < 0 or remaining_mines > len(unassigned_vars):
                    return False

        return True

    def is_complete_solution_valid(self, assignment, constraints):
        """Check if complete assignment satisfies all constraints"""
        for vars_list, required_mines in constraints:
            actual_mines = sum(assignment.get(v, 0) for v in vars_list)
            if actual_mines != required_mines:
                return False
        return True

    def analyze_solutions(self, variables, solutions):
        """Analyze solutions to find variables that are always safe or always mines"""
        certain_safe = []
        certain_mines = []

        for var in variables:
            values = [solution[var] for solution in solutions]
            if all(v == 0 for v in values):  # Always safe
                certain_safe.append(var)
            elif all(v == 1 for v in values):  # Always mine
                certain_mines.append(var)

        return certain_safe, certain_mines

    def probability_fallback(self, game):
        """Improved probabilistic fallback with better global reasoning"""
        unknown_cells = []
        constrained_cells = set()

        # Collect all unknown cells and identify which ones are constrained
        variables, constraints = self.build_constraints(game)
        constrained_cells.update(variables)

        for x in range(game.size_x):
            for y in range(game.size_y):
                if game.tiles[x][y]["state"] == STATE_DEFAULT:
                    unknown_cells.append((x, y))

        if not unknown_cells:
            return None

        # Calculate global mine probability
        total_mines = game.total_mines if hasattr(game, 'total_mines') else game.mine_count
        flagged_count = sum(1 for x in range(game.size_x) for y in range(game.size_y)
                            if game.tiles[x][y]["state"] == STATE_FLAGGED)
        remaining_mines = total_mines - flagged_count

        if remaining_mines <= 0:
            # All mines found, click any unknown cell
            x, y = unknown_cells[0]
            return ('click', x, y)

        # Prefer unconstrained cells when possible (they're often safer)
        unconstrained_cells = [(x, y) for x, y in unknown_cells if (x, y) not in constrained_cells]

        if unconstrained_cells and len(constrained_cells) > 0:
            # Calculate probabilities
            global_prob = remaining_mines / len(unknown_cells)

            # If unconstrained cells have lower probability than average, prefer them
            if len(unconstrained_cells) > remaining_mines:
                x, y = unconstrained_cells[0]
                return ('click', x, y)

        # Otherwise, click the first available unknown cell
        x, y = unknown_cells[0]
        return ('click', x, y)


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


class CNNTrainer:
    """Trains a CNN Minesweeper solver from minefields.json data."""

    def __init__(self, minefields_file: str = "minefields.json"):
        self.minefields_file = minefields_file
        self.model = None
        
        # Create model directory if it doesn't exist
        os.makedirs("model", exist_ok=True)

    # ---------------------------
    # Data generation
    # ---------------------------

    def generate_training_data(self, difficulty="Beginner", max_games=300000):
        """Generate training examples by simulating perfect games."""
        print(f"Generating training data from {self.minefields_file}...")

        try:
            with open(self.minefields_file, 'r') as f:
                all_minefields = json.load(f)
        except FileNotFoundError:
            print(f"Error: {self.minefields_file} not found. Run benchmark first.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.minefields_file}")
            return []

        if difficulty not in all_minefields:
            print(f"No data for {difficulty} difficulty")
            return []

        minefields = all_minefields[difficulty][:max_games]
        training_examples = []

        for i, minefield in enumerate(minefields):
            if i % 1000 == 0:
                print(f"Processing minefield {i + 1}/{len(minefields)}")

            try:
                examples = self._simulate_perfect_game(minefield)
                training_examples.extend(examples)
            except Exception as e:
                print(f"Error processing minefield {i}: {e}")

        print(f"Generated {len(training_examples)} training examples")
        return training_examples

    def _simulate_perfect_game(self, minefield):
        """Play out a game with an expert solver to collect training states."""
        try:
            game = HeadlessMinesweeper(minefield)
            solver = CSPStrategy()
            examples = []

            # Start with center click
            cx, cy = game.size_x // 2, game.size_y // 2
            if not game.process_move("click", cx, cy):  # unlucky first click
                return []

            step, max_steps = 0, game.size_x * game.size_y * 2
            while not game.game_over_flag and step < max_steps:
                state_matrix = self._encode_game_state(game)
                move = solver.next_move(game)
                if not move:
                    break

                move_type, x, y = move
                target = self._create_target_labels(game)

                examples.append({
                    "state": state_matrix.copy(),
                    "target": target,
                    "action": (move_type, x, y)
                })

                # Apply move
                if move_type == "click":
                    if not game.process_move("click", x, y):
                        break  # mine hit
                else:
                    game.process_move("flag", x, y)

                step += 1

            return examples
        except Exception as e:
            print(f"Error simulating game: {e}")
            return []

    def _encode_game_state(self, game):
        """Encode Minesweeper game state as multi-channel tensor for CNN."""
        channels = 4
        state = np.zeros((game.size_x, game.size_y, channels), dtype=np.float32)

        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]

                # Channel 0: Unknown cells
                if tile["state"] == 0:  # default
                    state[x, y, 0] = 1.0

                # Channel 1: Revealed numbers (normalized 0–1)
                elif tile["state"] == 1:  # clicked
                    state[x, y, 1] = tile["mines"] / 8.0

                # Channel 2: Flags
                elif tile["state"] == 2:
                    state[x, y, 2] = 1.0

                # Channel 3: constraint ratio
                if tile["state"] == 1 and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    flagged = sum(1 for n in neighbors if n["state"] == 2)
                    unknown = sum(1 for n in neighbors if n["state"] == 0)
                    remaining = tile["mines"] - flagged
                    if unknown > 0:
                        state[x, y, 3] = remaining / unknown

        return state

   
    def _create_target_labels(self, game):
        """Dense target labels: full minefield mask (1=mine, 0=safe)."""
        target = np.zeros((game.size_x, game.size_y), dtype=np.float32)
        for x in range(game.size_x):
            for y in range(game.size_y):
                # Fixed: use mine_positions instead of mines
                target[x, y] = 1.0 if (x, y) in game.mine_positions else 0.0
        return target
    # ---------------------------
    # Model
    # ---------------------------

    def build_model(self, input_shape):
        """Construct the CNN architecture."""
        model = keras.Sequential([
            layers.Input(shape=input_shape),

            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),

            layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same"),
            layers.Reshape(input_shape[:2])
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall")
            ]
        )
        return model

    # ---------------------------
    # Training
    # ---------------------------

    def train(self, difficulty="Beginner", max_games=5000, epochs=25):
        """Train the CNN with lazy TensorFlow loading and save to model folder"""
        # Load TensorFlow only when training is actually needed
        if not load_tensorflow():
            print("TensorFlow required for training.")
            return None

        training_examples = self.generate_training_data(difficulty, max_games)
        if not training_examples:
            print("No training data generated.")
            return None

        X, y = [], []
        for example in training_examples:
            X.append(example["state"])
            y.append(example["target"])
        
        import numpy as np
        X, y = np.array(X), np.array(y)

        print(f"Training data shape: X={X.shape}, y={y.shape}")

        input_shape = X.shape[1:]
        self.model = self.build_model(input_shape)

        print("Model architecture:")
        self.model.summary()

        history = self.model.fit(
            X, y,
            batch_size=(64 if difficulty == "Beginner" else (32 if difficulty == "Intermediate" else 16)),
            epochs=epochs,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
            ]
        )

        # Save model to model folder with organized naming
        #model_filename = f"minesweeper_cnn_{difficulty.lower()}.keras"
        model_filename = f"minesweeper_cnn_{difficulty.lower()}_{max_games}.keras"
        model_path = os.path.join("model", model_filename)
        
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Also save a simplified name for easy loading
        simple_filename = f"minesweeper_cnn_{difficulty.lower()}.keras"
        simple_path = os.path.join("model", simple_filename)
        self.model.save(simple_path)
        print(f"Model also saved to {simple_path}")
        
        return self.model

class TrainedCNNStrategy(AIStrategy):
    """CNN strategy using pre-trained model with lazy loading and proper difficulty matching"""

    def __init__(self, model_path=None, difficulty="Beginner"):
        self.fallback_strategy = AutoOpenStrategy()
        self.model = None
        self.difficulty = difficulty.lower()  # Store as lowercase for consistency
        self.model_path = model_path
        self.tf_loaded = False
        self.model_input_shape = None  # Track expected input shape
        self._shared_models = {}   # class-level cache
        self.predict_fn = None
        # Create model directory if it doesn't exist
        os.makedirs("model", exist_ok=True)

    def load_model(self, model_path):
        """Load trained model with lazy TensorFlow loading and shape validation"""
        if not self.tf_loaded:
            if not load_tensorflow():
                print("TensorFlow not available, using fallback strategy")
                return False
            self.tf_loaded = True

        try:
            print(f"Attempting to load model: {model_path}")
            self.model = keras.models.load_model(model_path)
            
            # Store the expected input shape for validation
            self.model_input_shape = self.model.input_shape[1:]  # Remove batch dimension
            print(f"Model loaded successfully. Expected input shape: {self.model_input_shape}")
            return True
            
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            self.model = None
            self.model_input_shape = None
            return False

    def _validate_game_compatibility(self, game):
        """Check if the game dimensions match the model's expected input shape"""
        if self.model is None or self.model_input_shape is None:
            return False
            
        expected_x, expected_y = self.model_input_shape[:2]
        actual_x, actual_y = game.size_x, game.size_y
        
        if expected_x != actual_x or expected_y != actual_y:
            print(f"Game size mismatch: Model expects ({expected_x}x{expected_y}), "
                  f"but game is ({actual_x}x{actual_y})")
            return False
            
        return True

    def next_move(self, game):
        """Get next move using CNN or fallback with proper compatibility checking"""
        # Always try deterministic moves first (fast)
        move = self._deterministic_moves(game)
        if move:
            return move

        # Load model on first use if not already loaded
        if self.model is None and not self.tf_loaded:
            model_loaded = self._try_load_appropriate_model(game)
            
            if not model_loaded:
                print(f"No compatible trained model found for {self.difficulty}.")
                print("Using fallback strategy.")

        # Use CNN if available and compatible
        if self.model is not None:
            if self._validate_game_compatibility(game):
                try:
                    move = self._cnn_prediction(game)
                    if move:
                        return move
                except Exception as e:
                    print(f"CNN prediction error: {e}")
            else:
                print("Model incompatible with current game size, using fallback")

        # Fall back to pattern-based strategy
        return self.fallback_strategy.next_move(game)
    def _try_load_appropriate_model(self, game):
        """Try to load a model that matches the current game configuration"""
        # Define model paths to try (in order of preference)
        model_paths = []
        
        # If specific path provided, try it first
        if self.model_path:
            model_paths.append(self.model_path)
        
        # Try to match difficulty and game size
        game_size_key = f"{game.size_x}x{game.size_y}"
        
        # Map common game sizes to difficulties
        size_to_difficulty = {
            "9x9": "beginner",
            "16x16": "intermediate", 
            "16x30": "expert",
            "30x16": "expert"
        }
        
        # Try the difficulty that matches the game size first
        if game_size_key in size_to_difficulty:
            matched_difficulty = size_to_difficulty[game_size_key]
            model_paths.extend([
                f"model/minesweeper_cnn_{matched_difficulty}_5000.keras",
                f"model/minesweeper_cnn_{matched_difficulty}_50000.keras",
                f"model/minesweeper_cnn_{matched_difficulty}_100000.keras",
                f"model/minesweeper_cnn_{matched_difficulty}_200000.keras",
                f"model/minesweeper_cnn_{matched_difficulty}.keras",
            ])
        
        # Validate and set default difficulty if needed
        if self.difficulty not in ["beginner", "intermediate", "expert"]:
            self.difficulty = "beginner"  # Default fallback
        
        # Then try the requested difficulty (if different from matched)
        if game_size_key not in size_to_difficulty or size_to_difficulty[game_size_key] != self.difficulty:
            model_paths.extend([
                f"model/minesweeper_cnn_{self.difficulty}_5000.keras",
                f"model/minesweeper_cnn_{self.difficulty}_50000.keras",
                f"model/minesweeper_cnn_{self.difficulty}_100000.keras",
                f"model/minesweeper_cnn_{self.difficulty}_200000.keras",
                f"model/minesweeper_cnn_{self.difficulty}.keras",
            ])
        
        # Try all other difficulties as fallback
        for diff in ["beginner", "intermediate", "expert"]:
            # Skip if we already tried this difficulty
            skip_diff = False
            if game_size_key in size_to_difficulty and size_to_difficulty[game_size_key] == diff:
                skip_diff = True
            if diff == self.difficulty:
                skip_diff = True
                
            if not skip_diff:
                model_paths.extend([
                    f"model/minesweeper_cnn_{diff}_5000.keras",
                    f"model/minesweeper_cnn_{diff}_50000.keras",
                    f"model/minesweeper_cnn_{diff}_100000.keras",
                    f"model/minesweeper_cnn_{diff}_200000.keras",
                    f"model/minesweeper_cnn_{diff}.keras",
                ])
        
        # Try to load any available model
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                print(f"Found model at: {path}")
                if self.load_model(path):
                    # Check if this model is compatible with current game
                    if self._validate_game_compatibility(game):
                        model_loaded = True
                        print(f"Model {path} is compatible with current game")
                        break
                    else:
                        print(f"Model {path} is not compatible with current game size")
                        self.model = None  # Reset model
        
        return model_loaded

    def _deterministic_moves(self, game):
        """Find certain moves (no TensorFlow needed)"""
        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]
                if tile["state"] != 1 or tile["mines"] == 0:
                    continue

                neighbors = game.get_neighbors(x, y)
                flagged = sum(1 for n in neighbors if n["state"] == 2)
                unknown = [n for n in neighbors if n["state"] == 0]
                remaining = tile["mines"] - flagged

                if remaining == 0 and unknown:
                    n = unknown[0]
                    return ("click", n["coords"]["x"], n["coords"]["y"])
                elif remaining == len(unknown) and remaining > 0:
                    n = unknown[0]
                    return ("flag", n["coords"]["x"], n["coords"]["y"])
        return None

    def _cnn_prediction(self, game):
        """Use CNN to predict mine probabilities"""
        if self.model is None:
            return None
            
        state = self._encode_state(game)
        
        # Validate state shape matches model expectations
        if state.shape != self.model_input_shape:
            print(f"State shape {state.shape} doesn't match model input {self.model_input_shape}")
            return None
            
        state_batch = tf.expand_dims(state, axis=0)

        try:
            # Create prediction function if not exists
            if self.predict_fn is None:
                self.predict_fn = tf.function(
                    lambda x: self.model(x, training=False),
                    input_signature=[tf.TensorSpec(shape=[None] + list(self.model_input_shape), dtype=tf.float32)]
                )
            
            predictions = self.predict_fn(state_batch)[0].numpy()
        except Exception as e:
            print(f"Model prediction failed: {e}")
            return None

        # Find best action
        unknown_cells = [(x, y) for x in range(game.size_x)
                        for y in range(game.size_y)
                        if game.tiles[x][y]["state"] == 0]

        if not unknown_cells:
            return None

        # Find safest cell
        safest_prob = float('inf')
        safest_cell = None

        for x, y in unknown_cells:
            if x < predictions.shape[0] and y < predictions.shape[1]:
                mine_prob = predictions[x, y]
                if mine_prob < safest_prob:
                    safest_prob = mine_prob
                    safest_cell = (x, y)

        if safest_cell:
            x, y = safest_cell
            # Flag if very confident it's a mine, click if confident it's safe
            if safest_prob > 0.8:
                return ("flag", x, y)
            elif safest_prob < 0.3:
                return ("click", x, y)

            return None
        
    def _encode_state(self, game):
        """Encode game state for CNN"""
        import numpy as np
        
        channels = 4
        state = np.zeros((game.size_x, game.size_y, channels), dtype=np.float32)

        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]

                if tile["state"] == 0:
                    state[x, y, 0] = 1.0
                elif tile["state"] == 1:
                    state[x, y, 1] = tile["mines"] / 8.0
                elif tile["state"] == 2:
                    state[x, y, 2] = 1.0

                if tile["state"] == 1 and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    flagged = sum(1 for n in neighbors if n["state"] == 2)
                    unknown = sum(1 for n in neighbors if n["state"] == 0)
                    remaining = tile["mines"] - flagged

                    if unknown > 0:
                        state[x, y, 3] = remaining / unknown

        return state


STRATEGIES = {
    "Basic": AutoOpenStrategy,
    "Probabilistics": ProbabilisticStrategy,
    "CSP": CSPStrategy,
    "Hybrid": HybridStrategy,
    "CNN": TrainedCNNStrategy,
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


class MinefieldGenerator:
    def __init__(self):
        self.saved_maps = {}

    def generate_maps(self, difficulty, count=100, seed=None):
        """Generate multiple minefield maps for a specific difficulty"""
        if seed is not None:
            random.seed(seed)

        maps = []
        for i in range(count):
            config = DIFFICULTIES[difficulty]
            size_x, size_y, mines = config["size_x"], config["size_y"], config["mines"]

            # Generate mine positions
            mine_positions = set()
            while len(mine_positions) < mines:
                x = random.randint(0, size_x - 1)
                y = random.randint(0, size_y - 1)
                mine_positions.add((x, y))

            # Create map with mine counts
            mine_map = {
                "difficulty": difficulty,
                "size_x": size_x,
                "size_y": size_y,
                "total_mines": mines,
                "mines": list(mine_positions),
                "id": f"{difficulty}_{i}"
            }
            maps.append(mine_map)

        self.saved_maps[difficulty] = maps
        return maps

    def save_maps(self, filename="minefields.json"):
        """Save all generated maps to a JSON file"""
        with open(filename, "w") as f:
            json.dump(self.saved_maps, f, indent=2)

    def load_maps(self, filename="minefields.json"):
        """Load maps from a JSON file"""
        try:
            with open(filename, "r") as f:
                self.saved_maps = json.load(f)
            return True
        except FileNotFoundError:
            print(f"File {filename} not found")
            return False
        except json.JSONDecodeError:
            print(f"Error reading JSON from {filename}")
            return False


class HeadlessMinesweeper:
    """A version of Minesweeper without UI for strategy testing"""

    def __init__(self, mine_map):
        self.size_x = mine_map["size_x"]
        self.size_y = mine_map["size_y"]
        self.total_mines = mine_map["total_mines"]
        self.mine_positions = set(tuple(pos) for pos in mine_map["mines"])

        # Initialize game state
        self.game_over_flag = False
        self.won = False
        self.moves = []
        self.first_click = True
        self.clicked_count = 0
        self.flag_count = 0
        self.correct_flag_count = 0

        # Create tiles
        self.tiles = {}
        for x in range(self.size_x):
            self.tiles[x] = {}
            for y in range(self.size_y):
                self.tiles[x][y] = {
                    "coords": {"x": x, "y": y},
                    "is_mine": (x, y) in self.mine_positions,
                    "state": STATE_DEFAULT,
                    "mines": 0  # Will be calculated
                }

        # Calculate numbers for tiles
        for x in range(self.size_x):
            for y in range(self.size_y):
                if not self.tiles[x][y]["is_mine"]:
                    mine_count = sum(1 for nx, ny in self.get_neighbors_coords(x, y)
                                     if (nx, ny) in self.mine_positions)
                    self.tiles[x][y]["mines"] = mine_count

    def get_neighbors_coords(self, x, y):
        """Get coordinates of neighboring tiles"""
        coords = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size_x and 0 <= ny < self.size_y:
                    coords.append((nx, ny))
        return coords

    def get_neighbors(self, x, y):
        """Get neighboring tile objects"""
        return [self.tiles[nx][ny] for nx, ny in self.get_neighbors_coords(x, y)]

    def process_move(self, move_type, x, y):
        """Process a move (click or flag)"""
        if self.game_over_flag:
            return False

        self.moves.append((move_type, x, y))

        if move_type == "click":
            return self.click(x, y)
        elif move_type == "flag":
            return self.flag(x, y)
        return False

    def click(self, x, y):
        """Process a click at coordinates x,y"""
        if not (0 <= x < self.size_x and 0 <= y < self.size_y):
            return False

        tile = self.tiles[x][y]

        if self.game_over_flag or tile["state"] != STATE_DEFAULT:
            return False

        # Check if mine
        if tile["is_mine"]:
            self.game_over_flag = True
            return False

        # Reveal tile
        if tile["mines"] == 0:
            self.clear_surrounding_tiles((x, y))
        else:
            tile["state"] = STATE_CLICKED
            self.clicked_count += 1

        # Check win condition
        if self.clicked_count == (self.size_x * self.size_y) - self.total_mines:
            self.game_over_flag = True
            self.won = True
            return True

        return True

    def flag(self, x, y):
        """Process a flag at coordinates x,y"""
        if not (0 <= x < self.size_x and 0 <= y < self.size_y):
            return False

        tile = self.tiles[x][y]

        if self.game_over_flag or tile["state"] == STATE_CLICKED:
            return False

        if tile["state"] == STATE_DEFAULT:
            # Flag tile
            tile["state"] = STATE_FLAGGED
            self.flag_count += 1
            if tile["is_mine"]:
                self.correct_flag_count += 1
        else:  # STATE_FLAGGED
            # Unflag tile
            tile["state"] = STATE_DEFAULT
            self.flag_count -= 1
            if tile["is_mine"]:
                self.correct_flag_count -= 1

        return True

    def clear_surrounding_tiles(self, start_coords):
        """Clear surrounding empty tiles (BFS)"""
        queue = deque([start_coords])
        visited = set()

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue

            visited.add((x, y))
            tile = self.tiles[x][y]

            if tile["state"] != STATE_DEFAULT:
                continue

            tile["state"] = STATE_CLICKED
            self.clicked_count += 1

            # If empty tile, add neighbors to queue
            if tile["mines"] == 0:
                for nx, ny in self.get_neighbors_coords(x, y):
                    queue.append((nx, ny))

class StrategyBenchmark:
    """Benchmark different AI strategies on the same minefield maps"""

    def __init__(self):
        self.generator = MinefieldGenerator()
        self.results = {}

    def prepare_maps(self, difficulty="Beginner", count=100, seed=None, load_file=None):
        """Prepare maps for benchmarking"""
        if load_file:
            success = self.generator.load_maps(load_file)
            if not success:
                print(f"Failed to load {load_file}, generating new maps instead")
                self.generator.generate_maps(difficulty, count, seed)
        else:
            self.generator.generate_maps(difficulty, count, seed)

        return self.generator.saved_maps

    def run_benchmark(self, difficulty="Beginner", strategies=None):
        """Run benchmark on all maps for given strategies with strategy reuse"""
        if strategies is None:
            strategies = STRATEGIES

        # Filter out None strategies
        valid_strategies = {name: cls for name, cls in strategies.items() if cls is not None}
        if not valid_strategies:
            print("No valid strategies found!")
            return {}

        maps = self.generator.saved_maps.get(difficulty, [])
        if not maps:
            print(f"No maps found for difficulty {difficulty}")
            return {}

        results = {}
        total_maps = len(maps)

        # Create strategy instances ONCE for all games
        strategy_instances = {}
        print("Initializing strategies...")
        for strategy_name, strategy_factory in valid_strategies.items():
            try:
                if callable(strategy_factory) and not isinstance(strategy_factory, type):
                    # It's a lambda or function, call it to get the instance
                    strategy_instances[strategy_name] = strategy_factory()
                else:
                    # It's a class, instantiate it
                    strategy_instances[strategy_name] = strategy_factory()
                print(f"✓ {strategy_name} initialized")
            except Exception as e:
                print(f"✗ Failed to initialize {strategy_name}: {e}")

        for strategy_name, strategy_instance in strategy_instances.items():
            print(f"\nRunning {strategy_name} strategy on {total_maps} {difficulty} maps...")
            strategy_results = []
            wins = 0
            total_moves = 0
            total_time = 0
            total_flags = 0
            total_correct_flags = 0

            # Use the SAME strategy instance for all games
            for i, mine_map in enumerate(maps):
                progress = f"[{i + 1}/{total_maps}]"
                if i > 0:
                    avg_moves_so_far = total_moves / i
                    print(f"\r{progress} Testing {strategy_name}: {wins}/{i} wins, avg moves: {avg_moves_so_far:.1f}", end="", flush=True)

                # Reset strategy state if needed (but keep the loaded model)
                if hasattr(strategy_instance, 'reset'):
                    strategy_instance.reset()

                # Run strategy on map with the SAME instance
                result = self._test_strategy_on_map(strategy_instance, mine_map, progress)
                strategy_results.append(result)

                # Update running statistics
                if result["won"]:
                    wins += 1
                total_moves += len(result["moves"])
                total_time += result["time"]
                total_flags += result["flagged"]
                total_correct_flags += result["correct_flag_count"]

            # Finalize results for this strategy
            avg_moves = total_moves / len(strategy_results) if strategy_results else 0
            avg_time = total_time / len(strategy_results) if strategy_results else 0
            avg_flags = total_flags / len(strategy_results) if strategy_results else 0
            flag_accuracy = total_correct_flags / total_flags if total_flags > 0 else 0

            results[strategy_name] = {
                "wins": wins,
                "games": len(strategy_results),
                "avg_moves": avg_moves,
                "avg_time": avg_time,
                "avg_flags": avg_flags,
                "flag_accuracy": flag_accuracy,
                "win_rate": wins / len(strategy_results) if strategy_results else 0,
                "details": strategy_results
            }

            # Print final result with newline
            print(f"\r{strategy_name} complete: {wins}/{total_maps} wins ({wins / total_maps * 100:.1f}%), "
                  f"avg moves: {avg_moves:.1f}, avg flags: {avg_flags:.1f} (flag acc: {flag_accuracy:.1%})  ")

        self.results[difficulty] = results
        return results

    def _test_strategy_on_map(self, strategy_instance, mine_map, progress=""):
        """Test a single strategy INSTANCE on a single map"""
        try:
            game = HeadlessMinesweeper(mine_map)

            start_time = time.time()
            move_count = 0

            # First move is always in the center for consistency
            center_x, center_y = game.size_x // 2, game.size_y // 2
            success = game.process_move("click", center_x, center_y)
            move_count += 1

            # If first move hit a mine, return early
            if not success:
                elapsed = time.time() - start_time
                return {
                    "won": False,
                    "moves": game.moves,
                    "time": elapsed,
                    "flagged": game.flag_count,
                    "correct_flag_count": game.correct_flag_count
                }

            # Run strategy until game over or move limit reached
            max_moves = game.size_x * game.size_y * 2
            while not game.game_over_flag and move_count < max_moves:
                try:
                    move = strategy_instance.next_move(game)  # Use instance, not class
                    if move is None:
                        break

                    # Validate move format
                    if not isinstance(move, (list, tuple)) or len(move) != 3:
                        print(f"Invalid move format from {strategy_instance.__class__.__name__}: {move}")
                        break

                    move_type, x, y = move
                    if move_type not in ["click", "flag"]:
                        print(f"Invalid move type: {move_type}")
                        break

                    success = game.process_move(move_type, x, y)
                    move_count += 1

                    # If move failed, strategy might be stuck
                    if not success and move_type == "click":
                        break

                except Exception as e:
                    print(f"Error in strategy {strategy_instance.__class__.__name__}: {e}")
                    break

            elapsed = time.time() - start_time

            return {
                "won": game.won,
                "moves": game.moves,
                "time": elapsed,
                "flagged": game.flag_count,
                "correct_flag_count": game.correct_flag_count
            }

        except Exception as e:
            print(f"Error testing strategy on map: {e}")
            return {
                "won": False,
                "moves": [],
                "time": 0.0,
                "flagged": 0,
                "correct_flag_count": 0
            }

    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to file"""
        try:
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def print_summary(self):
        """Print summary of benchmark results"""
        for difficulty, results in self.results.items():
            print(f"\n=== {difficulty} ===")
            sorted_results = sorted(
                results.items(),
                key=lambda x: (x[1]["win_rate"], -x[1]["avg_moves"]),
                reverse=True
            )

            for strategy, data in sorted_results:
                print(f"{strategy}:")
                print(f"  Win rate: {data['win_rate']:.2%} ({data['wins']}/{data['games']})")
                print(f"  Average moves: {data['avg_moves']:.2f}")
                print(f"  Avg time: {data['avg_time']:.3f}s")
                if data['avg_flags'] > 0:
                    print(f"  Avg flags: {data['avg_flags']:.2f} (flag_accuracy: {data['flag_accuracy']:.1%})")

class TrainedCNNStrategy(AIStrategy):
    """CNN strategy using pre-trained model with lazy loading and proper difficulty matching"""

    def __init__(self, model_path=None, difficulty="Beginner"):
        self.fallback_strategy = AutoOpenStrategy()
        self.model = None
        self.difficulty = difficulty.lower()
        self.model_path = model_path
        self.tf_loaded = False
        self.model_input_shape = None
        self.model_loaded_once = False  # Track if we've attempted loading
        
        os.makedirs("model", exist_ok=True)

    def reset(self):
        """Reset strategy state for new game (but keep loaded model)"""
        # Don't reset model or tf_loaded - keep them for reuse
        # Only reset per-game state if you have any
        pass

    def load_model(self, model_path):
        """Load trained model with lazy TensorFlow loading and shape validation"""
        if not self.tf_loaded:
            if not load_tensorflow():
                print("TensorFlow not available, using fallback strategy")
                return False
            self.tf_loaded = True

        try:
            print(f"Loading model: {model_path}")
            self.model = keras.models.load_model(model_path)
            self.model_input_shape = self.model.input_shape[1:]
            print(f"Model loaded. Input shape: {self.model_input_shape}")
            return True
            
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            self.model = None
            self.model_input_shape = None
            return False

    def _validate_game_compatibility(self, game):
        """Check if the game dimensions match the model's expected input shape"""
        if self.model is None or self.model_input_shape is None:
            return False
            
        expected_x, expected_y = self.model_input_shape[:2]
        actual_x, actual_y = game.size_x, game.size_y
        
        if expected_x != actual_x or expected_y != actual_y:
            if not hasattr(self, '_warned_about_size'):
                print(f"Game size mismatch: Model expects ({expected_x}x{expected_y}), "
                      f"but game is ({actual_x}x{actual_y})")
                self._warned_about_size = True
            return False
            
        return True

    def next_move(self, game):
        """Get next move using CNN or fallback with proper compatibility checking"""
        # Always try deterministic moves first (fast)
        move = self._deterministic_moves(game)
        if move:
            return move

        # Load model only ONCE per strategy instance
        if self.model is None and not self.model_loaded_once:
            self.model_loaded_once = True  # Prevent repeated attempts
            model_loaded = self._try_load_appropriate_model(game)
            
            if not model_loaded:
                print(f"No compatible trained model found for {self.difficulty}.")
                print("Using fallback strategy for all games.")

        # Use CNN if available and compatible
        if self.model is not None:
            if self._validate_game_compatibility(game):
                try:
                    move = self._cnn_prediction(game)
                    if move:
                        return move
                except Exception as e:
                    print(f"CNN prediction error: {e}")

        # Fall back to pattern-based strategy
        return self.fallback_strategy.next_move(game)

        

    def _try_load_appropriate_model(self, game):
        """Try to load a model that matches the current game configuration"""
        # Define model paths to try (in order of preference)
        model_paths = []
        
        # If specific path provided, try it first
        if self.model_path:
            model_paths.append(self.model_path)
        
        # Try to match difficulty and game size
        game_size_key = f"{game.size_x}x{game.size_y}"
        
        # Map common game sizes to difficulties
        size_to_difficulty = {
            "9x9": "beginner",
            "16x16": "intermediate", 
            "16x30": "expert",
            "30x16": "expert"
        }
        
        # Try the difficulty that matches the game size first
        if game_size_key in size_to_difficulty:
            matched_difficulty = size_to_difficulty[game_size_key]
            model_paths.extend([
                 model_paths.extend([
                    f"model/minesweeper_cnn_{diff}_50000.keras",
                    f"model/minesweeper_cnn_{diff}_100000.keras",
                    f"model/minesweeper_cnn_{diff}_200000.keras",
                    f"model/minesweeper_cnn_{diff}.keras",
                ])
            ])
        
        # Then try the requested difficulty
        if self.difficulty not in ["beginner", "intermediate", "expert"]:
            self.difficulty = "beginner"  # Default fallback
            
        model_paths.extend([
            model_paths.extend([
                    f"model/minesweeper_cnn_{diff}_50000.keras",
                    f"model/minesweeper_cnn_{diff}_100000.keras",
                    f"model/minesweeper_cnn_{diff}_200000.keras",
                    f"model/minesweeper_cnn_{diff}.keras",
                ])
        ])
        
        # Try all difficulties as fallback
        for diff in ["beginner", "intermediate", "expert"]:
            if diff != self.difficulty:
                model_paths.extend([
                    f"model/minesweeper_cnn_{diff}_50000.keras",
                    f"model/minesweeper_cnn_{diff}_100000.keras",
                    f"model/minesweeper_cnn_{diff}_200000.keras",
                    f"model/minesweeper_cnn_{diff}.keras",
                ])
        
        # Try to load any available model
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                print(f"Found model at: {path}")
                if self.load_model(path):
                    # Check if this model is compatible with current game
                    if self._validate_game_compatibility(game):
                        model_loaded = True
                        print(f"Model {path} is compatible with current game")
                        break
                    else:
                        print(f"Model {path} is not compatible with current game size")
                        self.model = None  # Reset model
        
        return model_loaded

    def _deterministic_moves(self, game):
        """Find certain moves (no TensorFlow needed)"""
        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]
                if tile["state"] != 1 or tile["mines"] == 0:
                    continue

                neighbors = game.get_neighbors(x, y)
                flagged = sum(1 for n in neighbors if n["state"] == 2)
                unknown = [n for n in neighbors if n["state"] == 0]
                remaining = tile["mines"] - flagged

                if remaining == 0 and unknown:
                    n = unknown[0]
                    return ("click", n["coords"]["x"], n["coords"]["y"])
                elif remaining == len(unknown) and remaining > 0:
                    n = unknown[0]
                    return ("flag", n["coords"]["x"], n["coords"]["y"])
        return None

    def _cnn_prediction(self, game):
        """Use CNN to predict mine probabilities"""
        if self.model is None:
            return None
            
        state = self._encode_state(game)
        
        # Validate state shape matches model expectations
        if state.shape != self.model_input_shape:
            print(f"State shape {state.shape} doesn't match model input {self.model_input_shape}")
            return None
            
        state_batch = tf.expand_dims(state, axis=0)

        try:
            predictions = self.model.predict(state_batch, verbose=0)[0]
        except Exception as e:
            print(f"Model prediction failed: {e}")
            return None

        # Find best action
        unknown_cells = [(x, y) for x in range(game.size_x)
                         for y in range(game.size_y)
                         if game.tiles[x][y]["state"] == 0]

        if not unknown_cells:
            return None

        # Find safest cell
        safest_prob = float('inf')
        safest_cell = None

        for x, y in unknown_cells:
            if x < predictions.shape[0] and y < predictions.shape[1]:
                mine_prob = predictions[x, y]
                if mine_prob < safest_prob:
                    safest_prob = mine_prob
                    safest_cell = (x, y)

        if safest_cell:
            x, y = safest_cell
            # Flag if very confident it's a mine, click if confident it's safe
            if safest_prob > 0.8:
                return ("flag", x, y)
            elif safest_prob < 0.3:
                return ("click", x, y)

        return None

    def _encode_state(self, game):
        """Encode game state for CNN"""
        import numpy as np
        
        channels = 4
        state = np.zeros((game.size_x, game.size_y, channels), dtype=np.float32)

        for x in range(game.size_x):
            for y in range(game.size_y):
                tile = game.tiles[x][y]

                if tile["state"] == 0:
                    state[x, y, 0] = 1.0
                elif tile["state"] == 1:
                    state[x, y, 1] = tile["mines"] / 8.0
                elif tile["state"] == 2:
                    state[x, y, 2] = 1.0

                if tile["state"] == 1 and tile["mines"] > 0:
                    neighbors = game.get_neighbors(x, y)
                    flagged = sum(1 for n in neighbors if n["state"] == 2)
                    unknown = sum(1 for n in neighbors if n["state"] == 0)
                    remaining = tile["mines"] - flagged

                    if unknown > 0:
                        state[x, y, 3] = remaining / unknown

        return state


# Enhanced TensorFlow loading function
def load_tensorflow():
    """Lazy load TensorFlow only when needed with better error handling"""
    global HAS_TENSORFLOW, tf, keras, layers
    if not HAS_TENSORFLOW and tf is None:
        try:
            print("Loading TensorFlow... (this may take a moment)")
            
            # Try different import methods
            try:
                import tensorflow as tf_module
            except ImportError as e1:
                print(f"Standard TensorFlow import failed: {e1}")
                try:
                    # Try alternative import
                    import tensorflow.compat.v2 as tf_module
                    tf_module.enable_v2_behavior()
                except ImportError as e2:
                    print(f"TensorFlow v2 import also failed: {e2}")
                    raise ImportError("TensorFlow not available")
            
            from tensorflow import keras as keras_module  
            from tensorflow.keras import layers as layers_module
            
            # Test if TensorFlow is working
            test_tensor = tf_module.constant([1, 2, 3])
            
            # Configure TensorFlow for better compatibility
            tf_module.config.set_soft_device_placement(True)
            
            # Set CPU only to avoid GPU issues
            try:
                tf_module.config.set_visible_devices([], 'GPU')
                print("GPU disabled, using CPU only")
            except:
                print("Could not disable GPU, continuing anyway")
            
            tf = tf_module
            keras = keras_module
            layers = layers_module
            HAS_TENSORFLOW = True
            print("TensorFlow loaded successfully!")
            
        except Exception as e:
            print(f"TensorFlow loading failed: {e}")
            print("CNN will use pattern-based fallback.")
            HAS_TENSORFLOW = False
            return False
    
    return HAS_TENSORFLOW


# Enhanced benchmark function with better strategy selection
def run_benchmark_cli():
    """Command-line interface for running benchmarks with proper difficulty mapping"""
    benchmark = StrategyBenchmark()
    count = 0

    print("===== Minesweeper AI Strategy Benchmark =====")
    print("1. Generate new maps")
    print("2. Load existing maps")
    choice = input("Choose option (1-2): ")

    if choice not in ["1", "2"]:
        print("Invalid choice. Please enter 1 or 2.")
        return

    if choice == "1":
        print("\nDifficulty levels:")
        print("1. Beginner (9x9, 10 mines)")
        print("2. Intermediate (16x16, 40 mines)")
        print("3. Expert (16x30, 99 mines)")

        difficulty_input = input("Select difficulty (1-3) [1]: ") or "1"

        difficulty_map = {
            "1": "Beginner",
            "2": "Intermediate", 
            "3": "Expert"
        }

        if difficulty_input not in difficulty_map:
            print(f"Invalid difficulty choice '{difficulty_input}'. Please enter 1, 2, or 3.")
            return

        difficulty = difficulty_map[difficulty_input]
        print(f"Selected: {difficulty}")

        count = int(input("Number of maps [100]: ") or "100")
        seed = input("Random seed (leave empty for random): ")
        seed = int(seed) if seed else None

        print(f"Generating {count} maps for {difficulty}...")
        benchmark.prepare_maps(difficulty, count, seed)

        save_maps = input("Save generated maps? (y/n) [y]: ").lower() != "n"
        if save_maps:
            filename = input("Filename [minefields.json]: ") or "minefields.json"
            benchmark.generator.save_maps(filename)
            print(f"Maps saved to {filename}")
    else:
        filename = input("Maps filename [minefields.json]: ") or "minefields.json"
        if benchmark.generator.load_maps(filename):
            print(f"Maps loaded from {filename}")
            difficulties = list(benchmark.generator.saved_maps.keys())
            print(f"Available difficulties: {', '.join(difficulties)}")


            if difficulties:
                print("\nSelect difficulty:")
                for i, diff in enumerate(difficulties, 1):
                    print(f"{i}. {diff}")

                diff_input = input(f"Choose difficulty (1-{len(difficulties)}) [1]: ") or "1"

                try:
                    diff_index = int(diff_input) - 1
                    if 0 <= diff_index < len(difficulties):
                        difficulty = difficulties[diff_index]
                        print(f"Selected: {difficulty}")
                    else:
                        print(f"Invalid choice '{diff_input}'. Please enter a number between 1 and {len(difficulties)}.")
                        return
                except ValueError:
                    print(f"Invalid input '{diff_input}'. Please enter a number.")
                    return
            else:
                difficulty = "Beginner"
        else:
            print(f"Could not load {filename}. Generating new Beginner maps...")
            difficulty = "Beginner"
            benchmark.prepare_maps(difficulty, 100)

    # Enhanced strategy selection with proper CNN difficulty matching
    available_strategies = [name for name, cls in STRATEGIES.items() if cls is not None]
    if not available_strategies:
        print("No strategies available! Please add strategy classes to the STRATEGIES dictionary.")
        return

    print(f"\nAvailable strategies:")
    for i, strategy in enumerate(available_strategies, 1):
        print(f"{i}. {strategy}")
    print("Enter strategy numbers (e.g., '1,3' or '1-3' for range) or press Enter for all:")

    strategy_input = input("Strategy selection: ").strip()

    if not strategy_input:
        # Select all strategies, but configure CNN with correct difficulty
        selected_strategies = {}
        for name, cls in STRATEGIES.items():
            if cls is not None:
                if name == "CNN" or "CNN" in name:
                    # Create CNN strategy with matching difficulty
                    selected_strategies[name] = lambda: TrainedCNNStrategy(difficulty=difficulty)
                else:
                    selected_strategies[name] = cls
        print("Selected: All strategies")
    else:
        selected_strategies = {}

        try:
            selections = []
            for part in strategy_input.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selections.extend(range(start, end + 1))
                else:
                    selections.append(int(part))

            for num in selections:
                if 1 <= num <= len(available_strategies):
                    strategy_name = available_strategies[num - 1]
                    strategy_cls = STRATEGIES[strategy_name]
                    
                    # Special handling for CNN strategy
                    if strategy_name == "CNN" or "CNN" in strategy_name:
                        selected_strategies[strategy_name] = lambda: TrainedCNNStrategy(difficulty=difficulty)
                    else:
                        selected_strategies[strategy_name] = strategy_cls
                else:
                    print(f"Invalid strategy number '{num}'. Please enter numbers between 1 and {len(available_strategies)}.")
                    return

            if selected_strategies:
                print(f"Selected: {', '.join(selected_strategies.keys())}")
            else:
                print("No valid strategies selected!")
                return

        except ValueError:
            print("Invalid input format. Use numbers separated by commas (e.g., '1,2,3') or ranges (e.g., '1-3').")
            return

    # Run benchmark with proper strategy instantiation
    print(f"\nRunning benchmark for {len(selected_strategies)} strategies on {difficulty}...")
    
    # Convert lambda functions to actual strategy classes for the benchmark
    strategy_instances = {}
    for name, strategy_factory in selected_strategies.items():
        if callable(strategy_factory) and not isinstance(strategy_factory, type):
            # It's a lambda or function, call it to get the instance
            strategy_instances[name] = strategy_factory
        else:
            # It's a class, use it directly
            strategy_instances[name] = strategy_factory
    
    benchmark.run_benchmark(difficulty, strategy_instances)

    # Show results
    benchmark.print_summary()

    # Save results
    save_results = input("\nSave benchmark results? (y/n) [y]: ").lower() != "n"
    if save_results:
        filename = input("Results filename [benchmark_results.json]: ") or "benchmark_results.json"
        benchmark.save_results(filename)

    generate_graphs = input("Generate graphs from results? (y/n) [y]: ").lower() != "n"
    if generate_graphs:
        try:
            base_dir = "benchmark_result"
            os.makedirs(base_dir, exist_ok=True)
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.patches import Rectangle
            import seaborn as sns

            # Set style for better looking plots
            plt.style.use('default')
            sns.set_palette("husl")

            for difficulty, results in benchmark.results.items():
                strategies = list(results.keys())
                n_strategies = len(strategies)

                # Extract all metrics
                win_rates = [results[s]["win_rate"] * 100 for s in strategies]
                avg_moves = [results[s]["avg_moves"] for s in strategies]
                avg_time = [results[s]["avg_time"] * 1000 for s in strategies]  # Convert to ms
                avg_flags = [results[s]["avg_flags"] for s in strategies]
                flag_accuracy = [results[s]["flag_accuracy"] * 100 for s in strategies]
                games_played = [results[s]["games"] for s in strategies]
                wins = [results[s]["wins"] for s in strategies]

                # Create a comprehensive dashboard
                fig = plt.figure(figsize=(20, 16))
                fig.suptitle(f'Minesweeper AI Strategy Benchmark - {difficulty}', fontsize=24, fontweight='bold')

                # Color palette
                colors = plt.cm.Set3(np.linspace(0, 1, n_strategies))

                # 1. Win Rate Comparison (Top Left)
                ax1 = plt.subplot(3, 3, 1)
                bars1 = ax1.bar(range(n_strategies), win_rates, color=colors, alpha=0.8, edgecolor='black',
                                linewidth=0.5)
                ax1.set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Win Rate (%)', fontsize=12)
                ax1.set_xticks(range(n_strategies))
                ax1.set_xticklabels(strategies, rotation=45, ha='right')
                ax1.set_ylim(0, 105)
                ax1.grid(True, alpha=0.3)

                # Add value labels
                for i, (bar, rate) in enumerate(zip(bars1, win_rates)):
                    ax1.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

                # 2. Average Moves vs Win Rate Scatter (Top Middle)
                ax2 = plt.subplot(3, 3, 2)
                scatter = ax2.scatter(avg_moves, win_rates, c=colors, s=200, alpha=0.7, edgecolors='black')
                ax2.set_title('Efficiency vs Success', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Average Moves')
                ax2.set_ylabel('Win Rate (%)')
                ax2.grid(True, alpha=0.3)

                # Add strategy labels
                for i, strategy in enumerate(strategies):
                    ax2.annotate(strategy, (avg_moves[i], win_rates[i]),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)

                # 3. Performance Time Distribution (Top Right)
                ax3 = plt.subplot(3, 3, 3)
                bars3 = ax3.bar(range(n_strategies), avg_time, color=colors, alpha=0.8, edgecolor='black',
                                linewidth=0.5)
                ax3.set_title('Average Response Time', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Time (ms)')
                ax3.set_xticks(range(n_strategies))
                ax3.set_xticklabels(strategies, rotation=45, ha='right')
                ax3.grid(True, alpha=0.3)

                for i, (bar, time) in enumerate(zip(bars3, avg_time)):
                    ax3.text(i, time + max(avg_time) * 0.01, f'{time:.1f}ms', ha='center', va='bottom', fontsize=8)

                # 4. Flagging Behavior (Middle Left)
                ax4 = plt.subplot(3, 3, 4)
                x = np.arange(n_strategies)
                width = 0.35

                bars4a = ax4.bar(x - width / 2, avg_flags, width, label='Avg Flags Used', color='lightcoral', alpha=0.8)
                bars4b = ax4.bar(x + width / 2, [f * a / 100 for f, a in zip(avg_flags, flag_accuracy)],
                                 width, label='Correct Flags', color='darkred', alpha=0.8)

                ax4.set_title('Flagging Strategy Analysis', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Number of Flags')
                ax4.set_xticks(x)
                ax4.set_xticklabels(strategies, rotation=45, ha='right')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

                # 5. Win/Loss Ratio Pie Charts (Middle Center)
                ax5 = plt.subplot(3, 3, 5)
                if len(strategies) == 1:
                    # Single strategy pie chart
                    strategy = strategies[0]
                    wins_val = wins[0]
                    losses = games_played[0] - wins_val
                    ax5.pie([wins_val, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%',
                            colors=['lightgreen', 'lightcoral'])
                    ax5.set_title(f'{strategy}\nWin/Loss Distribution', fontsize=12, fontweight='bold')
                else:
                    # Multiple strategies comparison
                    ax5.bar(range(n_strategies), wins, color='lightgreen', alpha=0.8, label='Wins')
                    ax5.bar(range(n_strategies), [g - w for g, w in zip(games_played, wins)],
                            bottom=wins, color='lightcoral', alpha=0.8, label='Losses')
                    ax5.set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')
                    ax5.set_ylabel('Number of Games')
                    ax5.set_xticks(range(n_strategies))
                    ax5.set_xticklabels(strategies, rotation=45, ha='right')
                    ax5.legend()

                # 6. Performance Radar Chart (Middle Right)
                ax6 = plt.subplot(3, 3, 6, projection='polar')

                # Normalize metrics for radar chart (0-100 scale)
                metrics = []
                labels = ['Win Rate', 'Speed\n(inv time)', 'Efficiency\n(inv moves)', 'Flag Accuracy']

                for i in range(n_strategies):
                    strategy_metrics = [
                        win_rates[i],  # Already 0-100
                        100 - min(100, (avg_time[i] / max(avg_time)) * 100) if max(avg_time) > 0 else 100,
                        # Invert time
                        100 - min(100, (avg_moves[i] / max(avg_moves)) * 100) if max(avg_moves) > 0 else 100,
                        # Invert moves
                        flag_accuracy[i] if avg_flags[i] > 0 else 50  # Flag accuracy or neutral
                    ]
                    metrics.append(strategy_metrics)

                angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                angles += angles[:1]  # Complete the circle

                for i, (strategy, strategy_metrics) in enumerate(zip(strategies, metrics)):
                    strategy_metrics += strategy_metrics[:1]  # Complete the circle
                    ax6.plot(angles, strategy_metrics, 'o-', linewidth=2, label=strategy, color=colors[i])
                    ax6.fill(angles, strategy_metrics, alpha=0.1, color=colors[i])

                ax6.set_xticks(angles[:-1])
                ax6.set_xticklabels(labels)
                ax6.set_ylim(0, 100)
                ax6.set_title('Performance Radar', fontsize=14, fontweight='bold', pad=20)
                ax6.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
                ax6.grid(True)

                # 7. Move Distribution Box Plot (Bottom Left)
                ax7 = plt.subplot(3, 3, 7)

                # Extract move counts for each strategy
                move_data = []
                for strategy in strategies:
                    strategy_moves = [len(game_result["moves"]) for game_result in results[strategy]["details"]]
                    move_data.append(strategy_moves)

                bp = ax7.boxplot(move_data, tick_labels=strategies, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax7.set_title('Move Count Distribution', fontsize=14, fontweight='bold')
                ax7.set_ylabel('Number of Moves')
                ax7.set_xticklabels(strategies, rotation=45, ha='right')
                ax7.grid(True, alpha=0.3)

                # 8. Performance Heatmap (Bottom Center)
                ax8 = plt.subplot(3, 3, 8)

                # Create performance matrix
                performance_metrics = np.array([
                    win_rates,
                    [100 - min(100, (t / max(avg_time)) * 100) if max(avg_time) > 0 else 100 for t in avg_time],
                    [100 - min(100, (m / max(avg_moves)) * 100) if max(avg_moves) > 0 else 100 for m in avg_moves],
                    flag_accuracy
                ])

                im = ax8.imshow(performance_metrics, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
                ax8.set_xticks(range(n_strategies))
                ax8.set_xticklabels(strategies, rotation=45, ha='right')
                ax8.set_yticks(range(len(labels)))
                ax8.set_yticklabels(labels)
                ax8.set_title('Performance Heatmap', fontsize=14, fontweight='bold')

                # Add text annotations
                for i in range(len(labels)):
                    for j in range(n_strategies):
                        text = ax8.text(j, i, f'{performance_metrics[i, j]:.1f}',
                                        ha="center", va="center", color="black", fontweight='bold')

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax8, shrink=0.8)
                cbar.set_label('Performance Score', rotation=270, labelpad=20)

                # 9. Strategy Rankings (Bottom Right)
                ax9 = plt.subplot(3, 3, 9)

                # Calculate overall score (weighted combination)
                overall_scores = []
                weights = {'win_rate': 0.4, 'moves': 0.2, 'time': 0.2, 'flags': 0.2}

                for i in range(n_strategies):
                    score = (
                            win_rates[i] * weights['win_rate'] +
                            (100 - min(100, (avg_time[i] / max(avg_time)) * 100) if max(avg_time) > 0 else 100) *
                            weights['time'] +
                            (100 - min(100, (avg_moves[i] / max(avg_moves)) * 100) if max(avg_moves) > 0 else 100) *
                            weights['moves'] +
                            (flag_accuracy[i] if avg_flags[i] > 0 else 50) * weights['flags']
                    )
                    overall_scores.append(score)

                # Sort by overall score
                sorted_indices = sorted(range(n_strategies), key=lambda x: overall_scores[x], reverse=True)
                sorted_strategies = [strategies[i] for i in sorted_indices]
                sorted_scores = [overall_scores[i] for i in sorted_indices]

                bars9 = ax9.barh(range(n_strategies), sorted_scores,
                                 color=[colors[strategies.index(s)] for s in sorted_strategies],
                                 alpha=0.8, edgecolor='black', linewidth=0.5)
                ax9.set_yticks(range(n_strategies))
                ax9.set_yticklabels(sorted_strategies)
                ax9.set_xlabel('Overall Performance Score')
                ax9.set_title('Strategy Rankings', fontsize=14, fontweight='bold')
                ax9.grid(True, alpha=0.3)

                # Add score labels
                for i, (bar, score) in enumerate(zip(bars9, sorted_scores)):
                    ax9.text(score + 1, i, f'{score:.1f}', ha='left', va='center', fontweight='bold')

                plt.tight_layout()
                plt.savefig(
                    os.path.join(base_dir, f"comprehensive_analysis_{difficulty}.png"),
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none'
                )
                plt.close()

                # Additional detailed plots for each strategy
                for strategy_name, strategy_data in results.items():
                    if len(strategy_data["details"]) > 1:  # Only if we have multiple games
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                        fig.suptitle(f'{strategy_name} - Detailed Analysis ({difficulty})',
                                     fontsize=16, fontweight='bold')

                        # Game-by-game performance
                        game_moves = [len(game["moves"]) for game in strategy_data["details"]]
                        game_results = [1 if game["won"] else 0 for game in strategy_data["details"]]
                        game_times = [game["time"] * 1000 for game in strategy_data["details"]]
                        game_flags = [game["flagged"] for game in strategy_data["details"]]

                        # 1. Moves per game
                        ax1.plot(game_moves, 'b-', alpha=0.7, linewidth=1)
                        ax1.scatter(range(len(game_moves)), game_moves,
                                    c=['green' if w else 'red' for w in game_results],
                                    alpha=0.6, s=30)
                        ax1.set_title('Moves per Game')
                        ax1.set_xlabel('Game Number')
                        ax1.set_ylabel('Number of Moves')
                        ax1.grid(True, alpha=0.3)
                        ax1.axhline(y=np.mean(game_moves), color='orange', linestyle='--',
                                    label=f'Average: {np.mean(game_moves):.1f}')
                        ax1.legend()

                        # 2. Time performance
                        ax2.hist(game_times, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
                        ax2.set_title('Response Time Distribution')
                        ax2.set_xlabel('Time per Game (ms)')
                        ax2.set_ylabel('Frequency')
                        ax2.axvline(x=np.mean(game_times), color='red', linestyle='--',
                                    label=f'Mean: {np.mean(game_times):.1f}ms')
                        ax2.grid(True, alpha=0.3)
                        ax2.legend()

                        # 3. Win streak analysis
                        win_streaks = []
                        current_streak = 0
                        for result in game_results:
                            if result:
                                current_streak += 1
                            else:
                                if current_streak > 0:
                                    win_streaks.append(current_streak)
                                current_streak = 0
                        if current_streak > 0:
                            win_streaks.append(current_streak)

                        if win_streaks:
                            ax3.hist(win_streaks, bins=max(1, len(set(win_streaks))),
                                     alpha=0.7, color='lightgreen', edgecolor='black')
                            ax3.set_title('Win Streak Distribution')
                            ax3.set_xlabel('Consecutive Wins')
                            ax3.set_ylabel('Frequency')
                        else:
                            ax3.text(0.5, 0.5, 'No Win Streaks', ha='center', va='center',
                                     transform=ax3.transAxes, fontsize=14)
                            ax3.set_title('Win Streak Distribution')
                        ax3.grid(True, alpha=0.3)

                        # 4. Flagging behavior over time
                        ax4.scatter(range(len(game_flags)), game_flags,
                                    c=['green' if w else 'red' for w in game_results], alpha=0.6)
                        ax4.set_title('Flagging Behavior')
                        ax4.set_xlabel('Game Number')
                        ax4.set_ylabel('Flags Used')
                        ax4.grid(True, alpha=0.3)
                        if game_flags:
                            ax4.axhline(y=np.mean(game_flags), color='orange', linestyle='--',
                                        label=f'Average: {np.mean(game_flags):.1f}')
                            ax4.legend()

                        plt.tight_layout()
                        strategy_dir = os.path.join(base_dir, strategy_name)
                        os.makedirs(strategy_dir, exist_ok=True)

                        plt.savefig(
                            os.path.join(strategy_dir, f"{strategy_name}_{difficulty}.png"),
                            dpi=300, bbox_inches='tight'
                        )
                        plt.close()
            print("Comprehensive graphs generated and saved as PNG files.")

        except ImportError:
            print("matplotlib and/or seaborn not installed. Skipping graph generation.")
            print("Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"Error generating graphs: {e}")
            import traceback
            traceback.print_exc()



def train_cnn_from_minefields():
    """Enhanced training function with interactive map generation and training"""
    
    trainer = CNNTrainer()
    
    print("\nDifficulty levels:")
    print("1. Beginner (9x9, 10 mines)")
    print("2. Intermediate (16x16, 40 mines)")
    print("3. Expert (16x30, 99 mines)")

    difficulty_input = input("Select difficulty (1-3) [1]: ") or "1"

    difficulty_map = {
        "1": "Beginner",
        "2": "Intermediate", 
        "3": "Expert"
    }

    if difficulty_input not in difficulty_map:
        print(f"Invalid difficulty choice '{difficulty_input}'. Please enter 1, 2, or 3.")
        return

    difficulty = difficulty_map[difficulty_input]
    print(f"Selected: {difficulty}")

    count = int(input("Number of maps [100]: ") or "100")
    seed = input("Random seed (leave empty for random): ")
    seed = int(seed) if seed else None

    print(f"Generating {count} maps for {difficulty}...")
    
    benchmark = StrategyBenchmark()
    
    benchmark.prepare_maps(difficulty, count, seed)

    save_maps = input("Save generated maps? (y/n) [y]: ").lower() != "n"
    filename = f"minefields_{difficulty.lower()}.json"
    
    if save_maps:
        custom_filename = input(f"Filename [{filename}]: ") or filename
        benchmark.generator.save_maps(custom_filename)
        print(f"Maps saved to {custom_filename}")
        filename = custom_filename

    # Now train the CNN for this specific difficulty
    print(f"\n=== Training CNN for {difficulty} ===")
    
    try:
        # Set training parameters based on difficulty
        if difficulty == "Beginner":
            max_games = 5000
            epochs = 40
        elif difficulty == "Intermediate":
            max_games = 10000
            epochs = 60
        else:  # Expert
            max_games = 20000
            epochs = 80
        
        # Train the model
        model = trainer.train(difficulty, max_games=max_games, epochs=epochs)
        
        if model:
            model_filename = f"minesweeper_cnn_{difficulty.lower()}_{max_games}.keras"
            print(f"✓ Training completed for {difficulty}")
            print(f"✓ Model saved as: {model_filename}")
        else:
            print(f"✗ Training failed for {difficulty}")
            
    except Exception as e:
        print(f"✗ Error training {difficulty}: {e}")

    print("\n" + "=" * 50)
    print(f"Training Summary for {difficulty}:")
    print(f"- Maps file: {filename}")
    print(f"- Model file: minesweeper_cnn_{difficulty.lower()}_{max_games}.keras")
    print("- Use TrainedCNNStrategy to play with trained model")
    print("=" * 50)


def train_all_difficulties():
    """Train CNN models for all difficulties with separate map files"""
    
    trainer = CNNTrainer()
    benchmark = StrategyBenchmark()
    difficulties = ["Intermediate", "Expert"]
    training_params = {
        "Beginner": {"max_games": 5000, "epochs": 40, "map_count": 5000},
        "Intermediate": {"max_games": 1000, "epochs": 60, "map_count": 1000},
        "Expert": {"max_games": 2000, "epochs": 80, "map_count": 2000}
    }
    
    print("=== Auto-generating maps and training CNNs for all difficulties ===")
    
    for difficulty in difficulties:
        print(f"\n{'='*20} {difficulty} {'='*20}")
        
        params = training_params[difficulty]
        filename = f"minefields.json"
        
        try:
            # Generate maps for this difficulty
            print(f"Generating {params['map_count']} maps for {difficulty}...")
            
           
            benchmark.prepare_maps(difficulty, params['map_count'], seed=None)
            benchmark.generator.save_maps(filename)
            print(f"✓ Maps saved to {filename}")
            
            # Train the model
            print(f"Training CNN for {difficulty}...")
            print(f"Parameters: {params['max_games']} games, {params['epochs']} epochs")
            
            model = trainer.train(difficulty, max_games=params['max_games'], epochs=params['epochs'])
            
            if model:
                model_filename = f"minesweeper_cnn_{difficulty.lower()}_{params['max_games']}.keras"
                print(f"✓ Training completed for {difficulty}")
                print(f"✓ Model saved as: {model_filename}")
            else:
                print(f"✗ Training failed for {difficulty}")
                
        except Exception as e:
            print(f"✗ Error with {difficulty}: {e}")
            continue

    print("\n" + "=" * 60)
    print("Final Summary:")
    print("Generated files:")
    for difficulty in difficulties:
        params = training_params[difficulty]
        print(f"  - minefields_{difficulty.lower()}.json")
        print(f"  - minesweeper_cnn_{difficulty.lower()}_{params['max_games']}.keras")
    print("\nUse TrainedCNNStrategy to play with any of the trained models")
    print("=" * 60)



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
        first_strategy = next(iter(STRATEGIES))  # gets the first key of the dict
        self.strategy_name = StringVar(value=first_strategy)

        # Initialize game variables
        self.size_x = DIFFICULTIES[self.settings.difficulty]["size_x"]
        self.size_y = DIFFICULTIES[self.settings.difficulty]["size_y"]
        self.total_mines = DIFFICULTIES[self.settings.difficulty]["mines"]

        # Initialize AI with proper strategy instance
        first_strategy_class = next(iter(STRATEGIES.values()))  # get the first class
        self.ai = MinesweeperAI(self, first_strategy_class())

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
        self.strategy_btn = Button(self.bottom_frame, text=f"Strategy: {self.strategy_name.get()} (F5)",
                                   command=self.cycle_strategy, font=("Arial", 10), padx=10, width=25)
        self.strategy_btn.pack(side="left", padx=(10, 0))

        self.auto_play_btn = Button(self.bottom_frame, text="Auto Play (F6)", command=self.toggle_auto_play,
                                    font=("Arial", 10), padx=10, width=20)  # Add fixed width
        self.auto_play_btn.pack(side="left", padx=(10, 0))

        # AI status
        self.ai_running = False
        self.ai_status_label = Label(self.bottom_frame, text="AI: Stopped", font=("Arial", 10),
                                     width=15)  # Add fixed width
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
            img.put(theme["button_bg"], to=(0, 0), width=30, height=30)
        elif image_type == "clicked":
            img.put("#ffffff", to=(0, 0), width=30, height=30)
        elif image_type == "mine":
            img.put(theme["mine_color"], to=(0, 0), width=30, height=30)
        elif image_type == "flag":
            img.put(theme["flag_color"], to=(0, 0), width=30, height=30)
        elif image_type == "wrong":
            img.put("#888888", to=(0, 0), width=30, height=30)

        return img

    def create_fallback_number_image(self, number):
        """Create fallback number images"""
        img = PhotoImage(width=20, height=20)
        img.put("#ffffff", to=(0, 0), width=30, height=30)
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
            self.auto_play_btn.config(text="Auto Play  ")
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
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        run_benchmark_cli()
        return
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
    #train_cnn_from_minefields()
    #train_all_difficulties()    