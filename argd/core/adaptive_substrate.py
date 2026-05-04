import torch
import torch.nn as nn
import math
from typing import Tuple


class AdaptiveGraphSubstrate(nn.Module):
    """
    Manages the dynamic growth and pruning of the system's topological graph
    without destroying the computational graph.

    Two operating modes:
      Hard mode  — binary active_mask updated by Top-K heuristic (fast, non-differentiable).
      Soft mode  — differentiable gate_logits -> sigmoid soft_mask trained by a separate
                   gate_optimizer, enabling topology selection to participate in gradient flow.

    The two modes are kept in sync: after each gate optimizer step, call sync_hard_mask()
    to project the soft mask back to the binary active_mask used for logging and Top-K.
    """

    def __init__(self, max_nodes: int, initial_active: int, phi: float = 1.618033988749895):
        super().__init__()
        self.max_nodes = max_nodes
        self.phi = phi
        self.initial_active = initial_active

        # Fundamental harmonics for expansion calculation
        self.register_buffer('harmonics', torch.tensor([1.0, 3.0, 6.0, 9.0, 11.0]))
        self.pi_phi = nn.Parameter(torch.tensor(math.pi * self.phi), requires_grad=False)
        self.Lambda_21 = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.Psi_42 = nn.Parameter(torch.tensor(0.01), requires_grad=False)

        # --- Hard binary mask (for logging, Top-K expansion, and pruning) ---
        self.register_buffer('active_mask', torch.zeros(max_nodes))
        self.active_mask[:initial_active] = 1.0

        # --- Soft differentiable gate (learned via gate_optimizer) ---
        # Core nodes initialized to logit +2.0 (sigmoid ≈ 0.88 = active)
        # Reserve nodes initialized to logit -2.0 (sigmoid ≈ 0.12 = dormant)
        gate_init = torch.full((max_nodes,), -2.0)
        gate_init[:initial_active] = 2.0
        self.gate_logits = nn.Parameter(gate_init)

        # Track node activity for pruning
        self.register_buffer('node_activity_ema', torch.zeros(max_nodes))

        # Temporal centroid for distance measurements
        self.register_buffer('temporal_centroid', torch.zeros(1))

        # Error tracking for expansion pressure (G_t)
        self.register_buffer('error_ema', torch.tensor(0.0))

    def update_error(self, loss_val: float, decay: float = 0.9):
        """Called externally by orchestrator to track prediction error."""
        self.error_ema.copy_(decay * self.error_ema + (1.0 - decay) * loss_val)

    # ------------------------------------------------------------------
    # Soft differentiable interface
    # ------------------------------------------------------------------

    def get_soft_mask(self) -> torch.Tensor:
        """
        Returns the differentiable soft mask m_i = sigmoid(gate_logits_i).
        Shape: (max_nodes,). Gradient flows through gate_logits.
        """
        return torch.sigmoid(self.gate_logits)

    def differentiable_gate_loss(self, G_t: float, adjacency: torch.Tensor,
                                  sparsity_weight: float = 0.01) -> torch.Tensor:
        """
        Differentiable loss that trains gate_logits via a separate gate_optimizer.

        Two competing pressures create a learned equilibrium:

          (a) Sparsity pressure  — L1 penalty on soft_mask.sum() pushes all nodes
              toward dormancy. This is equivalent to a learned sparsity prior.

          (b) Expansion pressure — when G_t is high (high error + low coherence),
              border-adjacent nodes are rewarded for being active. Gradient flows
              through soft_mask = sigmoid(gate_logits) for border nodes only.

        The equilibrium is: a node remains active if and only if its activation
        reduces prediction error enough to overcome the sparsity penalty — i.e.,
        the model learns its own capacity budget from data.

        Relation to known methods:
          - Adaptive Computation Time (Graves 2016): per-step halting probability
            trained by a differentiable penalty. gate_logits is the spatial analogue.
          - Neural Architecture Search (DARTS, Liu et al. 2018): architecture
            parameters trained jointly with weights via gradient descent.
          - Continual learning (PackNet, Mallya & Lazebnik 2018): dynamic capacity
            allocation without forgetting. Here G_t replaces a manual schedule.

        Args:
            G_t: scalar float — topological expansion pressure in [0, 1].
            adjacency: (max_nodes, max_nodes) float adjacency matrix.
            sparsity_weight: coefficient for sparsity penalty (default 0.01).

        Returns:
            Scalar tensor, differentiable w.r.t. gate_logits only.
        """
        soft_mask = self.get_soft_mask()  # (max_nodes,), differentiable

        # (a) Sparsity: push all nodes toward inactive
        sparsity_loss = sparsity_weight * soft_mask.sum()

        # (b) Expansion: when G_t is high, reward border nodes being active.
        # Border = adjacent to current cluster but not yet clearly active (< 0.5).
        # Detach the neighbor computation so gradient only flows through the border
        # nodes' own gate_logits, not through the whole cluster.
        with torch.no_grad():
            neighbor_signal = torch.matmul(adjacency.float(), soft_mask.detach())
            border_nodes = (neighbor_signal > 0).float() * (1.0 - (soft_mask.detach() > 0.5).float())

        # negative sign: we minimize, so reward = -G_t * (border nodes that are active)
        expansion_reward = -float(G_t) * (border_nodes * soft_mask).sum()

        return sparsity_loss + expansion_reward

    def sync_hard_mask(self, threshold: float = 0.5):
        """
        Projects soft gate back to binary active_mask (for logging, Top-K, pruning).
        Call after each gate optimizer step.
        Protects the initial core cluster: nodes 0..initial_active-1 are never deactivated.
        """
        with torch.no_grad():
            new_mask = (self.get_soft_mask() > threshold).float()
            new_mask[:self.initial_active] = 1.0  # protect core
            self.active_mask.copy_(new_mask)

    # ------------------------------------------------------------------
    # Hard-mode interface (Top-K heuristic, kept for backward compatibility)
    # ------------------------------------------------------------------

    def compute_theta_full(self, current_state: torch.Tensor, t: float, phase_sync_energy: float) -> torch.Tensor:
        """
        Computes the Topological Expansion Potential (Theta).
        """
        mean_state = current_state.mean(dim=(0, 1)) if current_state.dim() == 3 else current_state.mean(dim=0)

        if self.temporal_centroid.shape != mean_state.shape:
            self.temporal_centroid = mean_state.clone().detach()
        else:
            self.temporal_centroid = 0.95 * self.temporal_centroid + 0.05 * mean_state.detach()

        # Distance to subconscious core
        dist = torch.norm(current_state - self.temporal_centroid, p=2, dim=-1) + 0.01
        t_tensor = torch.tensor(t, device=current_state.device, dtype=torch.float32)

        theta_sum = torch.zeros_like(dist)

        for n in self.harmonics:
            wave = torch.sin(t_tensor * n * self.phi * self.pi_phi)
            term = (wave * phase_sync_energy * self.Lambda_21) / dist
            theta_sum += term

        return theta_sum + self.Psi_42

    def attempt_topology_expansion(self, adjacency_matrix: torch.Tensor, expansion_potential: torch.Tensor, k: int = 2) -> int:
        """
        Top-K Probabilistic Expansion.
        Finds the border of the active network and activates the 'k' highest-potential reserve nodes.
        """
        # Find neighbors touching the currently active nodes
        neighbors_matrix = torch.matmul(adjacency_matrix.float(), self.active_mask)
        # Isolate nodes that are neighbors but are currently sleeping
        potential_new_nodes = (neighbors_matrix > 0).float() * (1.0 - self.active_mask)

        if potential_new_nodes.sum() == 0:
            return 0  # No room to grow

        # Growth signal modulated by expansion potential (mean across batch/time)
        theta_mean = expansion_potential.mean(dim=0)

        # Restrict topk to candidate nodes only (avoids picking active nodes
        # when growth_signal is negative at the border)
        candidate_indices = potential_new_nodes.nonzero(as_tuple=True)[0]
        candidate_scores = theta_mean[candidate_indices]

        # Select Top-K highest potential nodes
        num_to_activate = min(k, len(candidate_indices))
        if num_to_activate > 0:
            _, topk_local = torch.topk(candidate_scores, num_to_activate)
            topk_indices = candidate_indices[topk_local]
            self.active_mask[topk_indices] = 1.0
            # Warm-start the soft gate so the gate_optimizer starts from a reasonable point
            with torch.no_grad():
                self.gate_logits[topk_indices] = self.gate_logits[topk_indices].clamp(min=0.5)

        return num_to_activate

    def entropy_gated_pruning(self, rigidity: float, min_active: int = 7):
        """
        Puts nodes back to sleep if they are inactive or overly rigid,
        preserving energy and preventing massive monolithic structures.
        """
        if self.active_mask.sum() <= min_active:
            return 0  # Don't prune below core size

        # Nodes with low activity and high systemic rigidity are candidates
        prune_candidates = (self.node_activity_ema < 0.1) & (rigidity > 0.8) & (self.active_mask > 0)

        # Don't prune the absolute core (nodes 0 to min_active-1)
        prune_candidates[:min_active] = False

        num_pruned = prune_candidates.sum().item()
        if num_pruned > 0:
            self.active_mask[prune_candidates] = 0.0
            self.node_activity_ema[prune_candidates] = 0.0
            # Push gate_logits of pruned nodes below threshold so soft gate stays consistent
            with torch.no_grad():
                self.gate_logits[prune_candidates] = self.gate_logits[prune_candidates].clamp(max=-0.5)

        return num_pruned

    def get_mask(self) -> torch.Tensor:
        return self.active_mask
