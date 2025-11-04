#!/usr/bin/env python3
"""
Gradient Similarity Module for Component-wise Gradient Analysis

This module computes and stores similarity matrices between gradients from different reward components.
Memory-efficient implementation that only retains similarity values, not full gradients.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class GradientSimilarityAnalyzer:
    """
    Computes and stores gradient similarity matrices between reward components.

    Memory-efficient: Only stores similarity values, not full gradients.
    """

    def __init__(self, component_names: List[str] = None):
        """
        Initialize the gradient similarity analyzer.

        Args:
            component_names: List of component names (default: ['correctness', 'format', 'length'])
        """
        if component_names is None:
            component_names = ['correctness', 'format', 'length']

        self.component_names = component_names

        # Store similarity matrices from each update step
        self.similarity_history = []  # List of similarity matrices over time
        self.current_similarities = {}  # Most recent similarity matrix

        # Temporary storage for gradient computation (cleared after each similarity calculation)
        self._temp_gradients = defaultdict(dict)

    def collect_gradients(self, model: nn.Module, component_name: str) -> None:
        """
        Temporarily collect gradients from a model for a specific component.

        Args:
            model: PyTorch model with computed gradients
            component_name: Name of the component (e.g., 'correctness')
        """
        if component_name not in self.component_names:
            raise ValueError(f"Unknown component: {component_name}. "
                           f"Expected one of {self.component_names}")

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Store gradient temporarily (always clone for safety)
                self._temp_gradients[component_name][name] = param.grad.clone().detach()

    def compute_and_store_similarity_matrix(self) -> Dict[str, float]:
        """
        Compute similarity matrix from collected gradients and store it.

        This should be called after collecting gradients for all components.
        Automatically clears temporary gradient storage after computation.

        Returns:
            Dictionary representing the similarity matrix
        """
        similarity_matrix = {}

        # Get parameter names that exist in all components
        all_param_sets = [set(self._temp_gradients[comp].keys())
                         for comp in self.component_names if comp in self._temp_gradients]

        if not all_param_sets:
            print("Warning: No gradients collected for similarity computation")
            return similarity_matrix

        common_params = list(set.intersection(*all_param_sets))

        if not common_params:
            print("Warning: No common parameters found across all components")
            return similarity_matrix

        # Compute pairwise similarities
        for i, comp1 in enumerate(self.component_names):
            for j, comp2 in enumerate(self.component_names):
                if i >= j:  # Skip diagonal and duplicate pairs
                    continue

                similarities = []
                weights = []

                for param_name in common_params:
                    grad1 = self._temp_gradients[comp1].get(param_name)
                    grad2 = self._temp_gradients[comp2].get(param_name)

                    if grad1 is not None and grad2 is not None:
                        # Compute cosine similarity
                        sim = self._compute_cosine_similarity(grad1, grad2)
                        similarities.append(sim)
                        # Weight by parameter size
                        weights.append(grad1.numel())

                # Compute weighted average similarity
                if similarities:
                    final_sim = np.average(similarities, weights=weights)
                    key = f"{comp1}_vs_{comp2}"
                    similarity_matrix[key] = final_sim

        # Store the similarity matrix
        self.current_similarities = similarity_matrix
        self.similarity_history.append(similarity_matrix.copy())

        # Clear temporary gradients to free memory
        self._temp_gradients.clear()

        return similarity_matrix

    @staticmethod
    def _compute_cosine_similarity(grad1: torch.Tensor, grad2: torch.Tensor,
                                  epsilon: float = 1e-8) -> float:
        """
        Compute cosine similarity between two gradient tensors.

        Args:
            grad1: First gradient tensor
            grad2: Second gradient tensor
            epsilon: Small value for numerical stability

        Returns:
            Cosine similarity value between -1 and 1
        """
        # Flatten gradients
        g1_flat = grad1.view(-1)
        g2_flat = grad2.view(-1)

        # Compute dot product and norms
        dot_product = torch.dot(g1_flat, g2_flat)
        norm1 = torch.norm(g1_flat)
        norm2 = torch.norm(g2_flat)

        # Compute cosine similarity with numerical stability
        cosine_sim = dot_product / (norm1 * norm2 + epsilon)

        return cosine_sim.item()

    def get_current_similarity_matrix(self) -> Dict[str, float]:
        """Get the most recent similarity matrix."""
        return self.current_similarities.copy()

    def get_similarity_history(self) -> List[Dict[str, float]]:
        """Get the full history of similarity matrices."""
        return self.similarity_history.copy()

    def clear(self) -> None:
        """Clear temporary storage and reset history if needed."""
        self._temp_gradients.clear()
        # Optionally clear history to save memory
        # self.similarity_history.clear()
        # self.current_similarities.clear()


