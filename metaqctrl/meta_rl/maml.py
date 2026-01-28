"""
First Order Model-Agnostic Meta-Learning (MAML) Implementation

Meta-learns an initialization π₀ that adapts quickly to new tasks.
© 2025 The MITRE Corporation, All Rights Reserved   
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from copy import deepcopy

try:
    import higher  # For differentiable optimization
    HIGHER_AVAILABLE = True
except ImportError:
    HIGHER_AVAILABLE = False
    higher = None
    
class MAML:
    """ 
    FOMAML algorithm for meta-learning quantum control policies.
    In both cases (higher = True and False) --> first order is used. 
    
    Algorithm:
    1. Sample batch of tasks θ ~ P, sample tasks as noise 
    2. For each task:
       a. Clone meta-parameters: φ = π₀, take meta parameters
       b. Take K gradient steps: φ → φ - α∇_φ L(φ; θ) , take K gradient steps for that task 
       c. Evaluate on validation data: L_val(φ; θ) , evaluate validation data 
    3. Meta-update: π₀ → π₀ - β∇_π₀ Σ_θ L_val(AdaptK(π₀; θ); θ), update , update policy duraing adaptaion phase. 
    """
    
    def __init__(
        self,
        policy: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        meta_lr: float = 0.001,
        first_order: bool = False,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            policy: Policy network (will be the meta-initialization) 
            inner_lr: Learning rate for inner loop adaptation
            inner_steps: Number K of inner gradient steps
            meta_lr: Learning rate for outer meta-update
            first_order: If True, use first-order MAML (FOMAML) - faster but less accurate
            device: torch device
        """
        ## Policy 
        self.policy = policy.to(device) 
        ##Inner LR 
        self.inner_lr = inner_lr
        ##Inner Steps 
        self.inner_steps = inner_steps
        # Meta learning rate 
        self.meta_lr = meta_lr
        #Determines first order 
        self.first_order = first_order
        #Device 
        self.device = device
        
        # Meta-optimizer (updates π₀) --> makes the optimizer module 
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=meta_lr)
        
        # Logging
        self.meta_train_losses = []
        self.meta_pre_adapt_losses = []  # Track pre-adaptation losses
        self.meta_val_losses = []

        self._warned_no_higher = False
    
    def inner_loop(
        self,
        task_data: Dict,
        loss_fn: Callable,
        num_steps: Optional[int] = None
    ) -> Tuple[nn.Module, List[float]]:
        """
        Perform K-step inner loop adaptation on a single task.

        Args:
            task_data: Dictionary with 'support' and 'query' data
            loss_fn: Loss function L(policy, data) → scalar
            num_steps: Number of gradient steps (defaults to self.inner_steps)

        Returns:
            adapted_policy: Policy after K adaptation steps
            losses: List of losses at each step
        """
        ##Number of adaptation steps (K)
        num_steps = num_steps or self.inner_steps

        # Clone policy for this task
        ## Make policy
        adapted_policy = deepcopy(self.policy)
         
        #Train the policy
        adapted_policy.train()

        # Inner optimizer
        inner_optimizer = optim.SGD(adapted_policy.parameters(), lr=self.inner_lr)
        
        losses = []
        support_data = task_data['support']
        for step in range(num_steps):
            inner_optimizer.zero_grad() 
            loss = loss_fn(adapted_policy, support_data)
            losses.append(loss.item())
            # Gradient step
            loss.backward()
            inner_optimizer.step() 
        return adapted_policy, losses
    
    def inner_loop_higher(
        self,
        task_data: Dict,
        loss_fn: Callable,
        num_steps: Optional[int] = None
    ) -> Tuple:
        """  
        Inner loop using `higher` library for differentiable optimization.
        This enables second-order MAML (backprop through inner loop).

        Returns:
            fmodel: Functional model after adaptation
            losses: Inner loop losses
        """
        if not HIGHER_AVAILABLE:
            raise ImportError("The 'higher' library is required for second-order MAML. "
                            "Install with: pip install higher")

        num_steps = num_steps or self.inner_steps
        support_data = task_data['support']
        losses = []
            
        inner_opt = optim.SGD(self.policy.parameters(), lr=self.inner_lr)

        with higher.innerloop_ctx(
            self.policy,
            inner_opt,
            copy_initial_weights=True,
            track_higher_grads=(not self.first_order)
        ) as (fmodel, diffopt):

            for step in range(num_steps): 
                loss = loss_fn(fmodel, support_data)
                losses.append(loss.item()) 
                diffopt.step(loss) 
            return fmodel, losses
    
    def meta_train_step(
        self,
        task_batch: List[Dict],
        loss_fn: Callable,
        use_higher: bool = True
    ) -> Dict[str, float]:
        """ 
        Single meta-training step on a batch of tasks.  

        Args:
            task_batch: List of task dictionaries, each with 'support' and 'query'
            loss_fn: Loss function
            use_higher: If True, use higher library for second-order gradients

        Returns:
            metrics: Dictionary of training metrics
        """
        self.meta_optimizer.zero_grad()

        meta_loss_tensor = None  
        task_losses = []
        task_pre_adapt_losses = []  
        use_manual_grads = False   

        for task_data in task_batch:
            with torch.no_grad():
                pre_adapt_loss = loss_fn(self.policy, task_data['query'])
                task_pre_adapt_losses.append(pre_adapt_loss.item())
            if use_higher and not self.first_order and HIGHER_AVAILABLE:
                ## Modified ....instead just using FO MAML for both --> both if and else are the same 
                support_data = task_data['support']
                query_data = task_data['query']

                inner_losses = []
                inner_opt = optim.SGD(self.policy.parameters(), lr=self.inner_lr)

                with higher.innerloop_ctx(
                    self.policy,
                    inner_opt,
                    copy_initial_weights=True,
                    track_higher_grads=True   
                ) as (fmodel, diffopt):
                    # Inner loop adaptation on support set
                    for step in range(self.inner_steps):
                        loss = loss_fn(fmodel, support_data)
                        inner_losses.append(loss.item())
                        diffopt.step(loss)

                    # Compute query loss INSIDE context for proper gradient flow
                    query_loss = loss_fn(fmodel, query_data)
                    fmodel_params = list(fmodel.parameters())
                    meta_params = list(self.policy.parameters())

                    # Compute gradients w.r.t. adapted parameters
                    # create_graph=True for second-order (gradient of gradient)
                    adapted_grads = autograd.grad(
                        query_loss,
                        fmodel_params,
                        create_graph=True,  # Second-order: need gradients of gradients
                        allow_unused=True
                    )

                    # Manually accumulate gradients to meta-parameters
                    for meta_param, adapted_grad in zip(meta_params, adapted_grads):
                        if adapted_grad is not None:
                            if meta_param.grad is None:
                                meta_param.grad = adapted_grad.clone()
                            else:
                                meta_param.grad = meta_param.grad + adapted_grad.clone()

            else:
                # First-order MAML - use manual gradient computation
                use_manual_grads = True

                if HIGHER_AVAILABLE:
                    # Use higher library for inner loop
                    fmodel, inner_losses = self.inner_loop_higher(task_data, loss_fn)

                    # Compute query loss
                    query_loss = loss_fn(fmodel, task_data['query'])
                    meta_params = list(self.policy.parameters())
                    adapted_params = list(fmodel.parameters())
                    adapted_grads = autograd.grad(
                        query_loss,
                        adapted_params,
                        create_graph=False,   
                        allow_unused=True
                    )

                    for meta_param, adapted_grad in zip(meta_params, adapted_grads):
                        if adapted_grad is not None:
                            if meta_param.grad is None:
                                meta_param.grad = adapted_grad.clone()
                            else:
                                meta_param.grad += adapted_grad.clone()

                else:
                    adapted_policy, inner_losses = self.inner_loop(task_data, loss_fn)
                    query_loss = loss_fn(adapted_policy, task_data['query'])

                    meta_params = list(self.policy.parameters())
                    adapted_params = list(adapted_policy.parameters())

                    adapted_grads = autograd.grad(
                        query_loss,
                        adapted_params,
                        create_graph=False,
                        allow_unused=True
                    )

                    for meta_param, adapted_grad in zip(meta_params, adapted_grads):
                        if adapted_grad is not None:
                            if meta_param.grad is None:
                                meta_param.grad = adapted_grad.clone()
                            else:
                                meta_param.grad += adapted_grad.clone()

                    if not self._warned_no_higher:
                        print("WARNING: First-order MAML without 'higher' library may not train correctly!")
                        print("Install with: pip install higher")
                        self._warned_no_higher = True

            # FIXED: Check for NaN/Inf and skip task entirely to avoid gradient issues
            if torch.isnan(query_loss) or torch.isinf(query_loss):
                print(f"WARNING: Invalid loss detected (NaN or Inf): {query_loss.item()}")
                print(f"  Inner losses: {inner_losses}")
                print(f"  Skipping this task to preserve gradient flow")

                continue

            if meta_loss_tensor is None:
                meta_loss_tensor = query_loss
            else:
                meta_loss_tensor = meta_loss_tensor + query_loss

            task_losses.append(query_loss.item())

        n_valid_tasks = len(task_losses)

        if n_valid_tasks == 0:
            print("ERROR: No valid tasks in batch (all were NaN/Inf)")
            return {
                'meta_loss': float('nan'),
                'mean_task_loss': float('nan'),
                'std_task_loss': float('nan'),
                'min_task_loss': float('nan'),
                'max_task_loss': float('nan'),
                'error': 'no_valid_tasks'
            }

        meta_loss_tensor = meta_loss_tensor / n_valid_tasks
        meta_loss = meta_loss_tensor.item()

        if np.isnan(meta_loss) or np.isinf(meta_loss):
            print(f"ERROR: Invalid meta_loss detected: {meta_loss}")
            print("  Skipping this meta-update to prevent corruption")
            return {
                'meta_loss': float('nan'),
                'mean_task_loss': np.mean(task_losses),
                'std_task_loss': np.std(task_losses),
                'min_task_loss': np.min(task_losses),
                'max_task_loss': np.max(task_losses),
                'error': 'invalid_loss'
            }

        for param in self.policy.parameters():
            if param.grad is not None:
                param.grad = param.grad / n_valid_tasks


        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"WARNING: Invalid gradient norm detected: {grad_norm.item()}")
            print("  Skipping optimizer step")
        else:
            self.meta_optimizer.step()

        # Logging
        metrics = {
            'meta_loss': meta_loss,   
            'mean_task_loss': np.mean(task_losses),
            'std_task_loss': np.std(task_losses),
            'min_task_loss': np.min(task_losses),
            'max_task_loss': np.max(task_losses),
            'grad_norm': grad_norm.item(),
            'mean_pre_adapt_loss': np.mean(task_pre_adapt_losses) if task_pre_adapt_losses else float('nan'),
            'std_pre_adapt_loss': np.std(task_pre_adapt_losses) if task_pre_adapt_losses else float('nan'),
            'adaptation_gain': np.mean(task_pre_adapt_losses) - meta_loss if task_pre_adapt_losses else float('nan')
        }

        self.meta_train_losses.append(meta_loss)  
        pre_adapt_loss = np.mean(task_pre_adapt_losses) if task_pre_adapt_losses else float('nan')
        self.meta_pre_adapt_losses.append(pre_adapt_loss)
        return metrics
    
    def meta_validate(
        self,
        val_tasks: List[Dict],
        loss_fn: Callable
    ) -> Dict[str, float]:
        """
        Evaluate meta-learned initialization on validation tasks.

        Args:
            val_tasks: Validation task batch
            loss_fn: Loss function

        Returns:
            metrics: Validation metrics
        """
        self.policy.eval()

      
        try:
            val_losses = []
            adapted_losses = []
            for task_data in val_tasks:
                with torch.no_grad():
                    pre_loss = loss_fn(self.policy, task_data['query'])
                    val_losses.append(pre_loss.item())

                adapted_policy, _ = self.inner_loop(task_data, loss_fn)

                # Loss after adaptation (no grad needed)
                with torch.no_grad():
                    post_loss = loss_fn(adapted_policy, task_data['query'])
                    adapted_losses.append(post_loss.item())
        finally:
            self.policy.train()

        metrics = {
            'val_loss_pre_adapt': np.mean(val_losses),
            'val_loss_post_adapt': np.mean(adapted_losses),
            'adaptation_gain': np.mean(val_losses) - np.mean(adapted_losses),
            'std_post_adapt': np.std(adapted_losses)
        }

        self.meta_val_losses.append(metrics['val_loss_post_adapt'])

        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save meta-learned initialization and training state."""
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'inner_steps': self.inner_steps,
            'meta_train_losses': self.meta_train_losses,
            'meta_pre_adapt_losses': self.meta_pre_adapt_losses,
            'meta_val_losses': self.meta_val_losses,
            **kwargs
        }
        torch.save(checkpoint, path) 
        policy_only_path = path.replace('.pt', '_policy.pt')
        torch.save(self.policy.state_dict(), policy_only_path)

        print(f"Checkpoint saved to {path}")
        print(f"Policy weights saved to {policy_only_path}")
    
    def load_checkpoint(self, path: str) -> int:
 
        """Load meta-learned initialization and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.inner_lr = checkpoint['inner_lr']
        self.inner_steps = checkpoint['inner_steps']
        self.meta_train_losses = checkpoint.get('meta_train_losses', [])
        self.meta_pre_adapt_losses = checkpoint.get('meta_pre_adapt_losses', [])
        self.meta_val_losses = checkpoint.get('meta_val_losses', [])
        
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {path} (epoch {epoch})")
        
        return epoch


class MAMLTrainer:
    """
    High-level trainer for FOMAML experiments.
    Handles task sampling, data generation, and training loop.
    """
    
    def __init__(
        self,
        maml: MAML,
        task_sampler: Callable,
        data_generator: Callable,
        loss_fn: Callable,
        n_support: int = 10,
        n_query: int = 10,
        log_interval: int = 10,
        val_interval: int = 50
    ):
        """
        Args:
            maml: MAML instance
            task_sampler: Function that samples tasks from P
            data_generator: Function that generates support/query data for a task
            loss_fn: Loss function
            n_support: Number of support trajectories per task
            n_query: Number of query trajectories per task
            log_interval: Log every N iterations
            val_interval: Validate every N iterations
        """
        self.maml = maml
        self.task_sampler = task_sampler
        self.data_generator = data_generator
        self.loss_fn = loss_fn
        self.n_support = n_support
        self.n_query = n_query
        self.log_interval = log_interval
        self.val_interval = val_interval
        
        self.iteration = 0
        self.best_val_loss = float('inf')

        self.training_history = {
            'iterations': [],          # Iteration numbers
            'meta_loss': [],           # Meta-loss (query loss)
            'val_fidelity': [],        # Validation fidelity
            'val_error': [],           # Validation error
            'val_iteration': [],       # Iterations where validation occurred
            'val_fidelity_std': [],    # Validation fidelity std
            'grad_norms': [],          # Gradient norms
            'nan_count': []  ,          # NaN/Inf incidents
            'val_post_adapt': [], 
            'val_pre_adapt': [] 
        }
    
    def generate_task_batch(self, n_tasks: int, split: str = 'train') -> List[Dict]:
        """
        Generate a batch of tasks with support/query data.
        
        Args:
            n_tasks: Number of tasks to sample
            split: 'train', 'val', or 'test'
            
        Returns:
            task_batch: List of task dictionaries
        """
        tasks = self.task_sampler(n_tasks, split=split)
        
        task_batch = []
        for task_params in tasks:
            support_data = self.data_generator(
                task_params,
                n_trajectories=self.n_support,
                split='support'
            )
            query_data = self.data_generator(
                task_params,
                n_trajectories=self.n_query,
                split='query'
            )
            
            task_batch.append({
                'task_params': task_params,
                'support': support_data,
                'query': query_data
            })
        
        return task_batch
    
    def train(
        self,
        n_iterations: int,
        tasks_per_batch: int = 4,
        val_tasks: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Main training loop.
        
        Args:
            n_iterations: Number of meta-training iterations
            tasks_per_batch: Number of tasks per meta-batch
            val_tasks: Number of tasks for validation
            save_path: Path to save checkpoints
        """
        print(f"Starting MAML training for {n_iterations} iterations...")
        print(f"Tasks per batch: {tasks_per_batch}")
        print(f"Inner steps: {self.maml.inner_steps}, Inner LR: {self.maml.inner_lr}")
        print(f"Meta LR: {self.maml.meta_lr}\n")
        
        for iteration in range(n_iterations):
            self.iteration = iteration
            
            # Sample task batch
            task_batch = self.generate_task_batch(tasks_per_batch, split='train')
            
            # Meta-training step
            train_metrics = self.maml.meta_train_step(task_batch, self.loss_fn)

            # Track training metrics for figure generation
            self.training_history['iterations'].append(iteration)
            self.training_history['meta_loss'].append(train_metrics['meta_loss'])
            self.training_history['grad_norms'].append(train_metrics.get('grad_norm', 0.0))

            pre_adapt_loss = train_metrics.get('mean_pre_adapt_loss', float('nan'))

            has_nan = train_metrics.get('error') == 'invalid_loss' or train_metrics.get('error') == 'no_valid_tasks'
            self.training_history['nan_count'].append(1 if has_nan else 0)

            # Logging
            if iteration % self.log_interval == 0:
                grad_norm = train_metrics.get('grad_norm', 0.0)
                pre_adapt_loss = train_metrics.get('mean_pre_adapt_loss', float('nan'))
                post_adapt_loss = train_metrics['meta_loss']
                adapt_gain = train_metrics.get('adaptation_gain', float('nan'))

                # Convert losses to fidelities (assuming loss = 1 - fidelity)
                pre_adapt_fidelity = 1.0 - pre_adapt_loss
                post_adapt_fidelity = 1.0 - post_adapt_loss

                print(f"Iter {iteration}/{n_iterations}")
                print(f"  Pre-adapt:  Loss={pre_adapt_loss:.4f}, Fidelity={pre_adapt_fidelity:.4f}")
                print(f"  Post-adapt: Loss={post_adapt_loss:.4f}, Fidelity={post_adapt_fidelity:.4f}")
                print(f"  Adaptation Gain: {adapt_gain:.4f} | Grad Norm: {grad_norm:.4f}")

                if iteration % (self.log_interval * 5) == 0:  # Every 5th log interval
                    zero_grad_count = 0
                    total_params = 0
                    for name, param in self.maml.policy.named_parameters():
                        total_params += 1
                        if param.grad is None or param.grad.abs().max() < 1e-10:
                            zero_grad_count += 1
                    if zero_grad_count > 0:
                        print(f"  [DIAGNOSTIC] {zero_grad_count}/{total_params} parameters have zero/no gradients")
            

            if iteration % self.val_interval == 0 and iteration > 0:
                val_task_batch = self.generate_task_batch(val_tasks, split='val')
                val_metrics = self.maml.meta_validate(val_task_batch, self.loss_fn)

                val_fidelity = 1.0 - val_metrics['val_loss_post_adapt']
                val_error = val_metrics['val_loss_post_adapt']
                val_fidelity_std = val_metrics['std_post_adapt']

                self.training_history['val_fidelity'].append(val_fidelity)
                self.training_history['val_error'].append(val_error)
                self.training_history['val_iteration'].append(iteration)
                self.training_history['val_fidelity_std'].append(val_fidelity_std)
                self.training_history['val_pre_adapt'].append(val_metrics['val_loss_pre_adapt'])
                self.training_history['val_post_adapt'].append(val_metrics['val_loss_post_adapt'])
                

                print(f"\n[Validation] Iter {iteration}")
                print(f"  Pre-adapt loss:  {val_metrics['val_loss_pre_adapt']:.4f}")
                print(f"  Post-adapt loss: {val_metrics['val_loss_post_adapt']:.4f}")
                print(f"  Val Fidelity: {val_fidelity:.4f} ± {val_fidelity_std:.4f}")
                print(f"  Val Error: {val_error:.4f}")
                print(f"  Adaptation gain: {val_metrics['adaptation_gain']:.4f}\n")

                # Save best model
                if save_path and val_metrics['val_loss_post_adapt'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss_post_adapt']
                    best_path = save_path.replace('.pt', '_best.pt')
                    self.maml.save_checkpoint(best_path, iteration, **val_metrics)
        
        if save_path:
            self.maml.save_checkpoint(save_path, n_iterations)

            self.save_training_history(save_path)

        print("\nTraining complete!")

    def save_training_history(self, checkpoint_path: str):
        """Save training history to JSON file."""
        import json
        from pathlib import Path

        history_path = Path(checkpoint_path).parent / "training_history.json"

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"Training history saved to: {history_path}")


# Example usage
if __name__ == "__main__":
    from metaqctrl.meta_rl.policy import PulsePolicy
    
    # Create policy
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2
    )
    
    # Initialize MAML
    maml = MAML(
        policy=policy,
        inner_lr=0.01,
        inner_steps=5,
        meta_lr=0.001,
        first_order=False
    )
    
    print(f"MAML initialized with policy: {policy.count_parameters():,} parameters")
    print(f"Inner loop: {maml.inner_steps} steps @ lr={maml.inner_lr}")
    print(f"Meta-learning rate: {maml.meta_lr}")
    
    # Dummy loss function for testing
    def dummy_loss_fn(policy, data):
        task_features = data['task_features']
        controls = policy(task_features)
        # Dummy loss: minimize control magnitude
        return torch.mean(controls ** 2)
    
    # Dummy task data
    dummy_task = {
        'support': {
            'task_features': torch.randn(10, 3)
        },
        'query': {
            'task_features': torch.randn(10, 3)
        }
    }
    
    # Test inner loop
    print("\nTesting inner loop...")
    adapted_policy, losses = maml.inner_loop(dummy_task, dummy_loss_fn)
    print(f"Inner loop losses: {[f'{l:.4f}' for l in losses]}")
    
    # Test meta-step
    print("\nTesting meta-training step...")
    task_batch = [dummy_task for _ in range(4)]
    metrics = maml.meta_train_step(task_batch, dummy_loss_fn, use_higher=False)
    print(f"Meta-training metrics: {metrics}")
