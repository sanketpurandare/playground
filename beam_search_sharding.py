#!/usr/bin/env python3
"""
Beam Search for Sharding Optimization

This implements beam search to solve the sharding optimization problem
in polynomial time while maintaining good solution quality.
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import torch
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard


@dataclass
class BeamState:
    """Represents a partial sharding solution in the beam"""

    placements: Dict[str, Placement]  # node_id -> placement
    total_cost: float
    nodes_assigned: Set[str]

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def copy(self):
        return BeamState(
            placements=self.placements.copy(),
            total_cost=self.total_cost,
            nodes_assigned=self.nodes_assigned.copy(),
        )


class BeamSearchShardingOptimizer:
    """
    Beam search optimizer for sharding problems
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        placement_options: Dict[str, List[Placement]],
        beam_width: int = 10,
        cost_function=None,
    ):
        self.graph = graph
        self.placement_options = placement_options
        self.beam_width = beam_width
        self.cost_function = cost_function or self._default_cost_function

        # Precompute topological order
        self.topo_order = list(nx.topological_sort(graph))

        # Precompute communication costs
        self.comm_cost_cache = {}
        self._precompute_communication_costs()

    def _precompute_communication_costs(self):
        """Precompute communication costs between all placement pairs"""
        all_placements = set()
        for placements in self.placement_options.values():
            all_placements.update(placements)

        for p1 in all_placements:
            for p2 in all_placements:
                self.comm_cost_cache[(p1, p2)] = self._compute_comm_cost(p1, p2)

    def _compute_comm_cost(
        self, src_placement: Placement, dst_placement: Placement
    ) -> float:
        """Compute communication cost between two placements"""
        if src_placement == dst_placement:
            return 0.0

        # Simplified cost model - replace with actual redistribute_cost
        if isinstance(src_placement, Shard) and isinstance(dst_placement, Replicate):
            return 100.0  # AllGather cost
        elif isinstance(src_placement, Replicate) and isinstance(dst_placement, Shard):
            return 50.0  # Scatter cost
        elif isinstance(src_placement, Shard) and isinstance(dst_placement, Shard):
            if src_placement.dim != dst_placement.dim:
                return 200.0  # AllToAll cost
            return 0.0
        else:
            return 10.0  # Default cost

    def _default_cost_function(self, node: str, placement: Placement) -> float:
        """Default computation cost function"""
        # Simplified - replace with actual computation cost estimation
        if isinstance(placement, Shard):
            return 1.0  # Lower cost for sharded computation
        else:
            return 2.0  # Higher cost for replicated computation

    def optimize(self) -> Tuple[Dict[str, Placement], float]:
        """
        Main beam search optimization

        Returns:
            (optimal_placements, total_cost)
        """
        # Initialize beam with empty state
        beam = [BeamState(placements={}, total_cost=0.0, nodes_assigned=set())]

        # Process nodes in topological order
        for node in self.topo_order:
            beam = self._expand_beam(beam, node)
            beam = self._prune_beam(beam)

        # Return best complete solution
        best_state = min(beam, key=lambda s: s.total_cost)
        return best_state.placements, best_state.total_cost

    def _expand_beam(self, beam: List[BeamState], node: str) -> List[BeamState]:
        """Expand beam by assigning placements to the current node"""
        new_beam = []

        for state in beam:
            for placement in self.placement_options[node]:
                new_state = self._extend_state(state, node, placement)
                if new_state is not None:
                    new_beam.append(new_state)

        return new_beam

    def _extend_state(
        self, state: BeamState, node: str, placement: Placement
    ) -> Optional[BeamState]:
        """Extend a beam state by assigning a placement to a node"""
        new_state = state.copy()

        # Compute cost for this assignment
        node_cost = self.cost_function(node, placement)

        # Compute communication costs with predecessors
        comm_cost = 0.0
        for pred in self.graph.predecessors(node):
            if pred in new_state.placements:
                pred_placement = new_state.placements[pred]
                comm_cost += self.comm_cost_cache[(pred_placement, placement)]

        # Update state
        new_state.placements[node] = placement
        new_state.total_cost += node_cost + comm_cost
        new_state.nodes_assigned.add(node)

        return new_state

    def _prune_beam(self, beam: List[BeamState]) -> List[BeamState]:
        """Prune beam to keep only top-k states"""
        # Sort by cost and keep top beam_width states
        beam.sort(key=lambda s: s.total_cost)
        return beam[: self.beam_width]


class AdaptiveBeamSearchOptimizer(BeamSearchShardingOptimizer):
    """
    Adaptive beam search that adjusts beam width based on problem complexity
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        placement_options: Dict[str, List[Placement]],
        initial_beam_width: int = 10,
        max_beam_width: int = 50,
    ):
        super().__init__(graph, placement_options, initial_beam_width)
        self.initial_beam_width = initial_beam_width
        self.max_beam_width = max_beam_width
        self.current_beam_width = initial_beam_width

    def _adapt_beam_width(self, node: str, beam: List[BeamState]) -> int:
        """Adapt beam width based on current node complexity"""
        # Increase beam width for nodes with many placement options
        num_options = len(self.placement_options[node])

        # Increase beam width for nodes with high fan-out
        fan_out = len(list(self.graph.successors(node)))

        # Complexity score
        complexity = num_options * (1 + fan_out)

        if complexity > 20:
            return min(self.max_beam_width, self.current_beam_width * 2)
        elif complexity < 5:
            return max(self.initial_beam_width // 2, self.current_beam_width // 2)
        else:
            return self.current_beam_width

    def _expand_beam(self, beam: List[BeamState], node: str) -> List[BeamState]:
        """Expand beam with adaptive width"""
        # Adapt beam width for this node
        self.current_beam_width = self._adapt_beam_width(node, beam)

        return super()._expand_beam(beam, node)


class HierarchicalBeamSearchOptimizer:
    """
    Hierarchical beam search that decomposes the problem into subproblems
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        placement_options: Dict[str, List[Placement]],
        beam_width: int = 10,
        subgraph_size: int = 10,
    ):
        self.graph = graph
        self.placement_options = placement_options
        self.beam_width = beam_width
        self.subgraph_size = subgraph_size

    def optimize(self) -> Tuple[Dict[str, Placement], float]:
        """
        Hierarchical optimization using beam search on subgraphs
        """
        # Decompose graph into subgraphs
        subgraphs = self._decompose_graph()

        # Optimize each subgraph
        global_placements = {}
        total_cost = 0.0

        for subgraph in subgraphs:
            # Extract placement options for this subgraph
            sub_placement_options = {
                node: self.placement_options[node] for node in subgraph.nodes()
            }

            # Optimize subgraph
            optimizer = BeamSearchShardingOptimizer(
                subgraph, sub_placement_options, self.beam_width
            )

            sub_placements, sub_cost = optimizer.optimize()

            # Merge results
            global_placements.update(sub_placements)
            total_cost += sub_cost

        # Add inter-subgraph communication costs
        inter_cost = self._compute_inter_subgraph_cost(subgraphs, global_placements)
        total_cost += inter_cost

        return global_placements, total_cost

    def _decompose_graph(self) -> List[nx.DiGraph]:
        """Decompose graph into smaller subgraphs"""
        # Simple decomposition: group consecutive nodes in topological order
        topo_order = list(nx.topological_sort(self.graph))
        subgraphs = []

        for i in range(0, len(topo_order), self.subgraph_size):
            subgraph_nodes = topo_order[i : i + self.subgraph_size]
            subgraph = self.graph.subgraph(subgraph_nodes).copy()
            subgraphs.append(subgraph)

        return subgraphs

    def _compute_inter_subgraph_cost(
        self, subgraphs: List[nx.DiGraph], placements: Dict[str, Placement]
    ) -> float:
        """Compute communication cost between subgraphs"""
        cost = 0.0

        for i in range(len(subgraphs) - 1):
            current_subgraph = subgraphs[i]
            next_subgraph = subgraphs[i + 1]

            # Find edges between subgraphs
            for node1 in current_subgraph.nodes():
                for node2 in next_subgraph.nodes():
                    if self.graph.has_edge(node1, node2):
                        p1, p2 = placements[node1], placements[node2]
                        cost += self._compute_comm_cost(p1, p2)

        return cost

    def _compute_comm_cost(self, p1: Placement, p2: Placement) -> float:
        """Simplified communication cost"""
        if p1 == p2:
            return 0.0
        return 10.0  # Simplified cost


class ConstrainedBeamSearchOptimizer(BeamSearchShardingOptimizer):
    """
    Beam search with additional constraints (memory, user preferences, etc.)
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        placement_options: Dict[str, List[Placement]],
        beam_width: int = 10,
        memory_limit: float = None,
        user_constraints: Dict[str, Placement] = None,
    ):
        super().__init__(graph, placement_options, beam_width)
        self.memory_limit = memory_limit
        self.user_constraints = user_constraints or {}

    def _extend_state(
        self, state: BeamState, node: str, placement: Placement
    ) -> Optional[BeamState]:
        """Extend state with constraint checking"""
        # Check user constraints
        if node in self.user_constraints:
            if placement != self.user_constraints[node]:
                return None  # Violates user constraint

        new_state = super()._extend_state(state, node, placement)

        if new_state is None:
            return None

        # Check memory constraint
        if self.memory_limit is not None:
            memory_usage = self._compute_memory_usage(new_state)
            if memory_usage > self.memory_limit:
                return None  # Violates memory constraint

        return new_state

    def _compute_memory_usage(self, state: BeamState) -> float:
        """Compute memory usage for current state"""
        # Simplified memory calculation
        total_memory = 0.0
        for node, placement in state.placements.items():
            if isinstance(placement, Replicate):
                total_memory += 1.0  # Full tensor
            else:
                total_memory += 0.5  # Sharded tensor
        return total_memory


# Example usage and testing
def create_example_transformer_graph():
    """Create a simplified transformer block graph for testing"""
    graph = nx.DiGraph()

    # Add nodes
    nodes = [
        "input",
        "ln1",
        "qkv_proj",
        "attention",
        "out_proj",
        "add1",
        "ln2",
        "ffn1",
        "ffn2",
        "add2",
        "output",
    ]
    graph.add_nodes_from(nodes)

    # Add edges (simplified transformer flow)
    edges = [
        ("input", "ln1"),
        ("ln1", "qkv_proj"),
        ("qkv_proj", "attention"),
        ("attention", "out_proj"),
        ("out_proj", "add1"),
        ("input", "add1"),
        ("add1", "ln2"),
        ("ln2", "ffn1"),
        ("ffn1", "ffn2"),
        ("ffn2", "add2"),
        ("add1", "add2"),
        ("add2", "output"),
    ]
    graph.add_edges_from(edges)

    return graph


def test_beam_search_sharding():
    """Test the beam search sharding optimizer"""
    # Create example graph
    graph = create_example_transformer_graph()

    # Define placement options
    placement_options = {}
    for node in graph.nodes():
        placement_options[node] = [
            Replicate(),
            Shard(0),  # Batch dimension
            Shard(1),  # Feature dimension
        ]

    # Test basic beam search
    print("Testing Basic Beam Search:")
    optimizer = BeamSearchShardingOptimizer(graph, placement_options, beam_width=5)
    placements, cost = optimizer.optimize()

    print(f"Total cost: {cost:.2f}")
    for node, placement in placements.items():
        print(f"  {node}: {placement}")

    # Test adaptive beam search
    print("\nTesting Adaptive Beam Search:")
    adaptive_optimizer = AdaptiveBeamSearchOptimizer(
        graph, placement_options, initial_beam_width=3, max_beam_width=10
    )
    adaptive_placements, adaptive_cost = adaptive_optimizer.optimize()

    print(f"Total cost: {adaptive_cost:.2f}")
    for node, placement in adaptive_placements.items():
        print(f"  {node}: {placement}")

    # Test hierarchical beam search
    print("\nTesting Hierarchical Beam Search:")
    hierarchical_optimizer = HierarchicalBeamSearchOptimizer(
        graph, placement_options, beam_width=3, subgraph_size=4
    )
    hierarchical_placements, hierarchical_cost = hierarchical_optimizer.optimize()

    print(f"Total cost: {hierarchical_cost:.2f}")
    for node, placement in hierarchical_placements.items():
        print(f"  {node}: {placement}")

    # Test constrained beam search
    print("\nTesting Constrained Beam Search:")
    user_constraints = {"input": Shard(0), "output": Shard(0)}
    constrained_optimizer = ConstrainedBeamSearchOptimizer(
        graph,
        placement_options,
        beam_width=5,
        memory_limit=8.0,
        user_constraints=user_constraints,
    )
    constrained_placements, constrained_cost = constrained_optimizer.optimize()

    print(f"Total cost: {constrained_cost:.2f}")
    for node, placement in constrained_placements.items():
        print(f"  {node}: {placement}")


if __name__ == "__main__":
    test_beam_search_sharding()
