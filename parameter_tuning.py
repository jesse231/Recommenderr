import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from scipy.stats import qmc
import os
import gc

# ==========================================
# 1. Environment & Network Definitions
# ==========================================

class Environment:
    def __init__(self, kg_adj, entity_embeddings, relation_embeddings, item_ids, max_steps=3):
        self.kg_adj = kg_adj
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.item_ids = item_ids
        
        self.item_ids_list = list(item_ids)
        self.item_embeddings_matrix = self.entity_embeddings[self.item_ids_list]
        self.max_steps = max_steps

        self.user = None
        self.entity = None
        self.last_entity = None
        self.last_relation = None
        self.history = set()
        self.tstep = 0
        self.max_user_score = 1.0 
        
    def get_state(self):
        return (self.user, self.entity, self.last_entity, self.last_relation)
        
    def get_valid_actions(self):
        valid_actions = []
        for relation, tail in self.kg_adj[self.entity]:
            if tail not in self.history and tail != self.user:
                valid_actions.append((relation, tail))
        return valid_actions

    def reset(self, user):
        self.user = user
        self.entity = user
        self.history = set()
        self.tstep = 0
        self.last_entity = user
        self.last_relation = 0 
        
        u_emb = self.entity_embeddings[user]
        all_item_scores = np.dot(self.item_embeddings_matrix, u_emb) 
        self.max_user_score = max(np.max(all_item_scores), 1e-9)
        
        return self.get_state()
        
    def step(self, action):
        relation, next_entity = action
        
        self.history.add(self.entity)
        self.last_entity = self.entity
        self.last_relation = relation
        self.entity = next_entity
        self.tstep += 1
        
        done = self.tstep >= self.max_steps
        reward = 0.0
        
        if done:
            if self.entity in self.item_ids:
                u_emb = self.entity_embeddings[self.user]
                i_emb = self.entity_embeddings[self.entity]
                raw_score = np.dot(u_emb, i_emb)
                normalized_score = raw_score / self.max_user_score
                reward = max(0.0, float(normalized_score)) 
                
        return self.get_state(), reward, done, {}

class ActionPruner:
    def __init__(self, entity_embeddings, relation_embeddings, max_actions=400):
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.max_actions = max_actions

    def prune(self, user_id, valid_actions):
        if not valid_actions: return []
        u_emb = self.entity_embeddings[user_id]
        action_scores = []
        
        for relation_id, tail_id in valid_actions:
            r_emb = self.relation_embeddings[relation_id]
            e_emb = self.entity_embeddings[tail_id]
            score = np.dot(u_emb + r_emb, e_emb)
            action_scores.append((score, (relation_id, tail_id)))

        action_scores.sort(key=lambda x: x[0], reverse=True)
        return [act for score, act in action_scores[:self.max_actions]]

class PolicyValueNetwork(nn.Module):
    def __init__(self, state_dim=400, action_space_size=400, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.5):
        super(PolicyValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.policy_out = nn.Linear(hidden_dim2, action_space_size)
        self.value_out = nn.Linear(hidden_dim2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU() 

    def forward(self, state, action_mask):
        x = self.elu(self.fc1(state))
        x = self.dropout(x)
        x = self.elu(self.fc2(x))
        x = self.dropout(x)

        logits = self.policy_out(x)
        masked_logits = logits.masked_fill(action_mask == 0, -1e9)
        action_probs = F.softmax(masked_logits, dim=-1)
        state_value = self.value_out(x)

        return action_probs, state_value

# ==========================================
# 2. Data Loading
# ==========================================

def load_data():
    print("Loading Knowledge Graph and Embeddings...")
    ratings = pd.read_csv('./ml-1m/ratings.csv', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    interactions = ratings[['UserID', 'MovieID']].copy()
    interactions['Head'] = 'user_' + interactions['UserID'].astype(str)
    interactions['Relation'] = 'user.interact.movie'
    interactions['Tail'] = 'movie_' + interactions['MovieID'].astype(str)

    kg = pd.read_csv('kg.txt', sep='\t', names=['Head', 'Relation', 'Tail'])
    kg['Head'] = kg['Head'].astype(str)
    kg['Tail'] = 'entity_' + kg['Tail'].astype(str)

    graph = pd.concat([interactions[['Head', 'Relation', 'Tail']], kg], ignore_index=True)
    graph = pd.concat([graph, graph.rename(columns={'Head': 'Tail', 'Tail': 'Head'})], ignore_index=True)
    graph = pd.concat([graph, pd.DataFrame({'Head': graph['Head'], 'Relation': 'self_loop', 'Tail': graph['Head']})], ignore_index=True)

    entities = pd.unique(graph[['Head', 'Tail']].values.ravel('K'))
    entity_to_id = {entity: i for i, entity in enumerate(sorted(entities))}
    relations = pd.unique(graph['Relation'])
    relation_to_id = {relation: i for i, relation in enumerate(sorted(relations))}

    graph['Head_ID'] = graph['Head'].map(entity_to_id)
    graph['Tail_ID'] = graph['Tail'].map(entity_to_id)
    graph['Relation_ID'] = graph['Relation'].map(relation_to_id)

    numerical_graph = graph[['Head_ID', 'Relation_ID', 'Tail_ID']].values

    entity_embeddings = np.load('entity_embeddings.npy')
    relation_embeddings = np.load('relation_embeddings.npy')

    item_ids = {entity_id for name, entity_id in entity_to_id.items() if name.startswith('movie_')}
    valid_user_ids = [entity_id for name, entity_id in entity_to_id.items() if name.startswith('user_')]

    kg_adj = defaultdict(list)
    for head, relation, tail in numerical_graph:
        kg_adj[head].append((relation, tail))
        
    return kg_adj, entity_embeddings, relation_embeddings, item_ids, valid_user_ids

# ==========================================
# 3. Training Loop (GPU Enabled)
# ==========================================

def train_pgpr(env, policy_net, pruner, entity_embeddings, relation_embeddings, valid_user_ids, 
               config, device, num_episodes=50000): # Lowered episodes for tuning speed; adjust as needed
    
    optimizer = optim.Adam(policy_net.parameters(), lr=config['lr'])
    batch_size = config['batch_size']
    gamma = config['gamma']
    
    policy_net.train()
    optimizer.zero_grad()
    
    batch_loss = 0.0
    batch_reward = 0.0
    history_rewards = []

    for episode in range(1, num_episodes + 1):
        user_id = np.random.choice(valid_user_ids) 
        state_tuple = env.reset(user=user_id)
        
        trajectory = []
        done = False
        
        # Phase 1: Collect Trajectory
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions: break 
                
            pruned_actions = pruner.prune(user_id, valid_actions) 
            
            u, e_t, e_last, r_t = state_tuple
            state_vec = np.concatenate([
                entity_embeddings[u], entity_embeddings[e_t],
                entity_embeddings[e_last], relation_embeddings[r_t]
            ])
            # Send to GPU
            state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
            action_mask = torch.zeros(1, pruner.max_actions).to(device)
            action_mask[0, :len(pruned_actions)] = 1
            
            action_probs, state_value = policy_net(state_tensor, action_mask)
            
            m = torch.distributions.Categorical(probs=action_probs)
            sampled_action_idx = m.sample()
            sampled_action = pruned_actions[sampled_action_idx.item()]
            
            log_prob = m.log_prob(sampled_action_idx)
            next_state, reward, done, _ = env.step(sampled_action)
            
            trajectory.append({'log_prob': log_prob, 'value': state_value, 'reward': reward})
            state_tuple = next_state

        # Phase 2: Compute Loss
        if trajectory:
            returns = []
            G = 0
            for step_data in reversed(trajectory):
                G = step_data['reward'] + gamma * G
                returns.insert(0, G)
                
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                
            policy_loss, value_loss = [], []
            
            for step_data, G_val in zip(trajectory, returns):
                advantage = G_val - step_data['value'].item()
                policy_loss.append(-step_data['log_prob'] * advantage)
                G_tensor = torch.tensor([G_val], dtype=torch.float32).unsqueeze(0).to(device)
                value_loss.append(F.mse_loss(step_data['value'], G_tensor))
                
            trajectory_loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
            loss = trajectory_loss / batch_size
            loss.backward()
            
            batch_loss += trajectory_loss.item()
            batch_reward += sum([t['reward'] for t in trajectory])

        # Phase 3: Optimize
        if episode % batch_size == 0 or episode == num_episodes:
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            history_rewards.append(batch_reward / batch_size)
            batch_loss = 0.0
            batch_reward = 0.0

    # Return the average reward of the last 10% of batches as the performance metric
    final_metric_slice = max(1, len(history_rewards) // 10)
    return np.mean(history_rewards[-final_metric_slice:])

# ==========================================
# 4. Hyperparameter Search (Latin Hypercube)
# ==========================================

def run_hyperparameter_optimization(num_samples=10, episodes_per_trial=25000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")
    
    kg_adj, entity_embeddings, relation_embeddings, item_ids, valid_user_ids = load_data()
    
    # Define LHS Sampler (4 dimensions: lr, gamma, hd1, hd2)
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=num_samples)
    
    # Scale samples to actual hyperparameter ranges
    lrs = 10 ** (sample[:, 0] * -3 - 2)           # Log scale: 1e-5 to 1e-2
    gammas = 0.9 + sample[:, 1] * 0.099           # Linear: 0.9 to 0.999
    hidden_dims_1 = np.floor(256 + sample[:, 2] * (1024 - 256)).astype(int) # 256 to 1024
    hidden_dims_2 = np.floor(128 + sample[:, 3] * (512 - 128)).astype(int)  # 128 to 512
    
    best_reward = -float('inf')
    best_config = None
    
    print(f"\n--- Starting LHS Optimization ({num_samples} configurations) ---")
    
    for i in range(num_samples):
        config = {
            'lr': float(lrs[i]),
            'gamma': float(gammas[i]),
            'hidden_dim1': int(hidden_dims_1[i]),
            'hidden_dim2': int(hidden_dims_2[i]),
            'batch_size': 64 # Kept static, but can be added to LHS if desired
        }
        
        print(f"\n[Trial {i+1}/{num_samples}] Testing config: {config}")
        
        # Initialize isolated instances for this trial
        env = Environment(kg_adj, entity_embeddings, relation_embeddings, item_ids, max_steps=3)
        pruner = ActionPruner(entity_embeddings, relation_embeddings, max_actions=400)
        policy_net = PolicyValueNetwork(
            state_dim=400, 
            action_space_size=400,
            hidden_dim1=config['hidden_dim1'],
            hidden_dim2=config['hidden_dim2']
        ).to(device) # Move model to GPU
        
        # Train and get evaluation metric
        avg_terminal_reward = train_pgpr(
            env, policy_net, pruner, entity_embeddings, relation_embeddings, 
            valid_user_ids, config, device, num_episodes=episodes_per_trial
        )
        
        print(f"[Trial {i+1} Result] Avg Terminal Reward: {avg_terminal_reward:.4f}")
        
        if avg_terminal_reward > best_reward:
            best_reward = avg_terminal_reward
            best_config = config
            print(">>> New Best Configuration Found! <<<")
            # Save the best model
            torch.save(policy_net.state_dict(), "best_pgpr_model.pth")
            
        # Clean up memory on GPU
        del policy_net
        del env
        del pruner
        torch.cuda.empty_cache()
        gc.collect()

    print("\n=========================================")
    print("Optimization Complete.")
    print(f"Best Reward: {best_reward:.4f}")
    print(f"Best Configuration: {best_config}")
    print("Best model weights saved to 'best_pgpr_model.pth'")
    print("=========================================")

if __name__ == "__main__":
    # Ensure your kg.txt, ratings.csv, and .npy embeddings are in the directory
    # Adjust `num_samples` (how many hyperparam combos to try) and `episodes_per_trial` (how long to train each)
    run_hyperparameter_optimization(num_samples=10, episodes_per_trial=30000)