"""
Callback for calculating mutual information between DistilBERT HIDDEN STATES using CLUB estimators.
This version calculates MI between full hidden states to detect redundancy.
"""

import torch
import torch.nn as nn
from transformers import TrainerCallback
from CLUB.mi_estimators import CLUB, CLUBSample
import os


class MutualInformationCallback(TrainerCallback):
    """
    Custom callback to calculate and log mutual information between layer hidden states.
    Uses CLUB (Contrastive Log-ratio Upper Bound) estimators.

    This calculates MI between FULL HIDDEN STATES (not just residuals) to detect redundancy.
    High MI between adjacent layers = redundant representations (can be penalized).
    """

    def __init__(
        self,
        model_config,
        hidden_dim=768,
        num_layers=6,
        hidden_size=512,
        log_interval=100,
        use_sample_club=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        trainer=None
    ):
        """
        Args:
            model_config: Model configuration to get hidden dimensions
            hidden_dim: Hidden dimension of the model (default 768 for DistilBERT)
            num_layers: Number of transformer layers (default 6 for DistilBERT)
            hidden_size: Hidden size for CLUB estimator networks
            log_interval: How often to calculate MI (in steps)
            use_sample_club: Whether to use CLUBSample (faster) or CLUB (more accurate)
            device: Device to run MI estimation on
            trainer: Reference to the trainer (will be set automatically)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.log_interval = log_interval
        self.device = device
        self.trainer = trainer

        # Create CLUB estimators for adjacent layer hidden states
        # High MI = redundant information = can be penalized
        self.club_estimators = nn.ModuleDict()

        EstimatorClass = CLUBSample if use_sample_club else CLUB

        # Estimators for hidden states between adjacent layers
        for i in range(num_layers - 1):
            self.club_estimators[f'hidden_{i}_{i+1}'] = EstimatorClass(
                hidden_dim, hidden_dim, hidden_size
            ).to(device)

        # Optimizers for training CLUB estimators
        self.club_optimizers = {}
        for name, estimator in self.club_estimators.items():
            self.club_optimizers[name] = torch.optim.Adam(estimator.parameters(), lr=1e-4)

        self.mi_history = {name: [] for name in self.club_estimators.keys()}

    def _flatten_residuals(self, residuals, max_samples=512):
        """
        Flatten residuals from [batch_size, seq_length, hidden_dim] to [batch_size * seq_length, hidden_dim]
        and subsample to avoid memory issues.

        Args:
            residuals: Tensor of shape [batch_size, seq_length, hidden_dim]
            max_samples: Maximum number of samples to keep (default: 512)
        """
        batch_size, seq_length, hidden_dim = residuals.shape
        flattened = residuals.reshape(batch_size * seq_length, hidden_dim)

        # Subsample if too many samples
        num_samples = flattened.shape[0]
        if num_samples > max_samples:
            # Randomly sample max_samples indices
            indices = torch.randperm(num_samples, device=flattened.device)[:max_samples]
            flattened = flattened[indices]

        return flattened

    def train_club_estimators(self, hidden_states, num_iterations=5):
        """
        Train CLUB estimators to approximate q(Y|X) using hidden states.

        Args:
            hidden_states: List of hidden states, one per layer
            num_iterations: Number of training iterations for CLUB
        """
        for _ in range(num_iterations):
            # Train estimators for adjacent hidden states
            for i in range(self.num_layers - 1):
                x = self._flatten_residuals(hidden_states[i]).detach()
                y = self._flatten_residuals(hidden_states[i + 1]).detach()

                self.club_optimizers[f'hidden_{i}_{i+1}'].zero_grad()
                loss = self.club_estimators[f'hidden_{i}_{i+1}'].learning_loss(x, y)
                loss.backward()
                self.club_optimizers[f'hidden_{i}_{i+1}'].step()

    def calculate_mi(self, hidden_states):
        """
        Calculate mutual information estimates using trained CLUB estimators.

        Returns:
            Dictionary of MI estimates for each pair of adjacent layers
        """
        mi_estimates = {}

        with torch.no_grad():
            # MI between adjacent layer hidden states
            for i in range(self.num_layers - 1):
                x = self._flatten_residuals(hidden_states[i])
                y = self._flatten_residuals(hidden_states[i + 1])
                mi = self.club_estimators[f'hidden_{i}_{i+1}'](x, y)
                mi_estimates[f'MI_hidden_L{i}_L{i+1}'] = mi.item()

        return mi_estimates

    def on_step_end(self, args, state, control, model, **kwargs):
        """
        Called at the end of each training step.
        Calculates MI at regular intervals.
        """
        if state.global_step % self.log_interval == 0 and state.global_step > 0:
            model.eval()

            # Get a batch of data to calculate MI
            # We'll use the training batch that was just processed
            if hasattr(model, 'module'):
                # Handle DataParallel/DistributedDataParallel
                actual_model = model.module
            else:
                actual_model = model

            # Get inputs from the trainer
            try:
                # Access inputs from trainer's stored batch
                inputs = None
                if self.trainer is not None and hasattr(self.trainer, 'current_inputs'):
                    inputs = self.trainer.current_inputs

                if inputs is not None:
                    # Move inputs to device
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = actual_model(**inputs, output_residuals=True)

                    if hasattr(outputs, 'hidden_after_layer'):
                        # Use full hidden states for redundancy detection
                        hidden_states = [r.to(self.device) for r in outputs.hidden_after_layer]

                        # Train CLUB estimators
                        self.train_club_estimators(hidden_states)

                        # Calculate MI
                        mi_estimates = self.calculate_mi(hidden_states)

                        # Log to TensorBoard via trainer
                        if self.trainer is not None and hasattr(self.trainer, 'log'):
                            self.trainer.log(mi_estimates)

                        # Alternative: Log via callback integration
                        if hasattr(self.trainer, 'callback_handler') and hasattr(self.trainer.callback_handler, 'callbacks'):
                            for callback in self.trainer.callback_handler.callbacks:
                                if hasattr(callback, '_SummaryWriter__class__'):  # TensorBoard callback
                                    # Get the TensorBoard writer
                                    if hasattr(callback, 'tb_writer') and callback.tb_writer is not None:
                                        for name, value in mi_estimates.items():
                                            callback.tb_writer.add_scalar(name, value, state.global_step)

                        # Store in history
                        for name, value in mi_estimates.items():
                            key = name.replace('MI_', '')
                            if key in self.mi_history:
                                self.mi_history[key].append((state.global_step, value))

                        print(f"\n[Step {state.global_step}] Mutual Information (Hidden States):")
                        for name, value in sorted(mi_estimates.items()):
                            print(f"  {name}: {value:.4f}")
                        print(f"  → High MI = Redundant representations between layers")
                    else:
                        print(f"Warning: No hidden states available at step {state.global_step}")

                else:
                    print(f"Warning: No inputs available at step {state.global_step}")

            except Exception as e:
                print(f"Warning: Could not calculate MI at step {state.global_step}: {e}")
                import traceback
                traceback.print_exc()

            model.train()

        return control

    def save_mi_history(self, output_dir):
        """Save MI history to file."""
        import json
        os.makedirs(output_dir, exist_ok=True)

        history_file = os.path.join(output_dir, "mi_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.mi_history, f, indent=2)

        print(f"MI history saved to {history_file}")
