"""
HIA Topology Module: Hexagonal Grid and Flower of Life Geometry
Phase 0 - Fundamental Topology and Definitions
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon


class SparseHexagonalLattice:
    """
    Hexagonal grid topology based on Flower of Life sacred geometry.
    Each node has 6 direct neighbors, forming natural resonance channels.
    """
    
    def __init__(self, radius: int = 5, phi: float = 1.618033988749895):
        """
        Initialize Flower of Life hexagonal grid.
        
        Args:
            radius: Number of layers around central hexagon (depth of flower)
            phi: Golden ratio (φ) for fractal scaling
        """
        self.radius = radius
        self.phi = phi
        self.nodes = []
        self.adjacency_matrix = None
        self.distance_matrix = None
        self._initialize_hexagonal_grid()
    
    def _initialize_hexagonal_grid(self):
        """Generate hexagonal grid coordinates using axial coordinate system."""
        self.nodes = []
        
        # Generate hexagonal coordinates (axial system)
        for q in range(-self.radius, self.radius + 1):
            for r in range(-self.radius, self.radius + 1):
                if abs(q + r) <= self.radius:
                    # Convert to Cartesian for visualization
                    x = 3/2 * q
                    y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                    self.nodes.append({
                        'axial': (q, r),
                        'cartesian': (x, y),
                        'index': len(self.nodes)
                    })
        
        self.num_nodes = len(self.nodes)
        self._compute_adjacency()
        self._compute_distances()
    
    def _compute_adjacency(self):
        """Build adjacency matrix - 6 neighbors per hexagon."""
        self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
        
        # Hexagonal neighbors in axial coordinates
        neighbors_delta = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        
        for i, node in enumerate(self.nodes):
            q, r = node['axial']
            for dq, dr in neighbors_delta:
                neighbor_axial = (q + dq, r + dr)
                # Find this neighbor in our nodes
                for j, other_node in enumerate(self.nodes):
                    if other_node['axial'] == neighbor_axial:
                        self.adjacency_matrix[i, j] = 1.0
                        break
    
    def _compute_distances(self):
        """Compute pairwise Euclidean distances."""
        positions = np.array([node['cartesian'] for node in self.nodes])
        self.distance_matrix = np.sqrt(np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2))
    
    def compute_resonance_weights(self, sigma: float = 1.0) -> np.ndarray:
        """
        Compute resonance weights via Gaussian decay on spatial distance.
        W_ij = exp(-D_ij^2 / σ^2)
        
        Args:
            sigma: Spatial decay parameter (controls resonance range)
            
        Returns:
            Resonance weight matrix (num_nodes x num_nodes)
        """
        return np.exp(-self.distance_matrix ** 2 / (sigma ** 2))
    
    def get_node_neighbors(self, node_index: int, k: int = 1) -> List[int]:
        """
        Get k-hop neighbors of a node.
        
        Args:
            node_index: Index of query node
            k: Number of hops (1 = immediate neighbors)
            
        Returns:
            List of neighbor indices
        """
        neighbors = set()
        current_layer = {node_index}
        
        for _ in range(k):
            next_layer = set()
            for node_idx in current_layer:
                adjacent = np.where(self.adjacency_matrix[node_idx] > 0)[0]
                next_layer.update(adjacent)
            neighbors.update(next_layer)
        
        neighbors.discard(node_index)
        return sorted(list(neighbors))
    
    def visualize(self, weights: np.ndarray = None, title: str = "Flower of Life Topology"):
        """
        Visualize the hexagonal topology.
        
        Args:
            weights: Optional node coloring weights (num_nodes,)
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot nodes
        positions = np.array([node['cartesian'] for node in self.nodes])
        
        if weights is not None:
            scatter = ax.scatter(positions[:, 0], positions[:, 1], c=weights, 
                               cmap='viridis', s=200, alpha=0.8, edgecolors='black')
            plt.colorbar(scatter, ax=ax, label='Activation')
        else:
            ax.scatter(positions[:, 0], positions[:, 1], s=200, alpha=0.7, 
                      c='lightblue', edgecolors='black')
        
        # Plot edges (only immediate neighbors for clarity)
        for i in range(len(self.nodes)):
            for j in np.where(self.adjacency_matrix[i] > 0)[0]:
                if i < j:  # Avoid duplicate lines
                    ax.plot([positions[i, 0], positions[j, 0]],
                           [positions[i, 1], positions[j, 1]],
                           'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax


class CoherenceField:
    """
    Measures coherence (internal stability) across the hexagonal topology.
    C_t = measure of phase synchronization and energy distribution.
    """
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.coherence_history = []
    
    def compute_coherence(self, state: np.ndarray, phases: np.ndarray) -> float:
        """
        Compute coherence as mean resultant vector length of phases.
        C_t = ||mean(exp(i*φ_t))||
        
        Args:
            state: System state vector (num_nodes,)
            phases: Phase angles (num_nodes,)
            
        Returns:
            Coherence value in [0, 1]
        """
        complex_phases = np.exp(1j * phases)
        coherence = np.abs(np.mean(complex_phases))
        self.coherence_history.append(coherence)
        return coherence


class GoldenRatioScaler:
    """
    Implements fractal time scaling via golden ratio φ.
    Enables multi-scale processing from microseconds to hours.
    """
    
    def __init__(self, phi: float = 1.618033988749895, num_scales: int = 8):
        """
        Initialize golden ratio scaling.
        
        Args:
            phi: Golden ratio
            num_scales: Number of temporal scales
        """
        self.phi = phi
        self.num_scales = num_scales
        self.time_scales = [phi ** i for i in range(num_scales)]
    
    def project_to_scales(self, value: float) -> np.ndarray:
        """
        Project a value across φ^k scales.
        
        Returns:
            Array of projections (num_scales,)
        """
        return np.array([value / scale for scale in self.time_scales])
    
    def compress_from_scales(self, projections: np.ndarray) -> float:
        """
        Reconstruct value from multi-scale projections.
        """
        weights = np.array([1.0 / (scale ** 2) for scale in self.time_scales])
        weights /= weights.sum()
        return np.dot(projections, weights)


if __name__ == "__main__":
    # Test topology
    topology = SparseHexagonalLattice(radius=3)
    print(f"Total nodes: {topology.num_nodes}")
    print(f"Adjacency matrix shape: {topology.adjacency_matrix.shape}")
    
    # Test resonance weights
    weights = topology.compute_resonance_weights(sigma=1.5)
    print(f"Resonance weights (sample mean): {weights.mean():.4f}")
    
    # Test neighbors
    neighbors = topology.get_node_neighbors(0, k=1)
    print(f"First node has {len(neighbors)} immediate neighbors: {neighbors}")
    
    # Visualize
    topology.visualize()
    plt.show()
