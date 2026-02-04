"""
Graph Convolution Layer for Knowledge-Aware Group Recommendation

Phase 2: Information Propagation through GCN
Implements neighbor aggregation and representation 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Graph Convolution Network Layer for Knowledge Graph propagation
    
    This layer performs:
    1. Neighbor Aggregation (Eq. 1 and Eq. 3 from paper)
    2. Representation Update using GCN Aggregator (Eq. 5 from paper)
    
    The layer operates on a knowledge graph where nodes are users and entities (items),
    and edges represent interactions or relations.
    """
    
    def __init__(self, in_dim, out_dim, aggregation_type='bi-interaction'):
        """
        Initialize GCN Layer
        
        Args:
            in_dim: Input embedding dimension
            out_dim: Output embedding dimension
            aggregation_type: Type of aggregation (bi-interaction, gcn, graphsage)
        """
        super(GCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregation_type = aggregation_type
        
        # Transformation weights for different aggregation types
        if aggregation_type == 'bi-interaction':
            # Bi-Interaction aggregation uses two separate transforms
            # W1 for self-connection, W2 for neighbor aggregation
            self.W1 = nn.Linear(in_dim, out_dim, bias=False)
            self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        elif aggregation_type == 'gcn':
            # Standard GCN uses single weight matrix
            self.W = nn.Linear(in_dim, out_dim, bias=False)
        elif aggregation_type == 'graphsage':
            # GraphSAGE concatenates self and neighbor representations
            self.W = nn.Linear(2 * in_dim, out_dim, bias=False)
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, ego_embeddings, adjacency_matrix, relation_embeddings=None):
        """
        Forward pass of GCN layer
        
        Args:
            ego_embeddings: Current node embeddings, shape (num_nodes, in_dim)
                           Represents e_u^(l) for users or e_i^(l) for items at layer l
            adjacency_matrix: Sparse adjacency matrix, shape (num_nodes, num_nodes)
                             Normalized Laplacian matrix for message passing
            relation_embeddings: Optional relation embeddings for knowledge-aware propagation
        
        Returns:
            Updated embeddings after aggregation, shape (num_nodes, out_dim)
            
        Implements:
            Eq. 1 (Neighbor Aggregation):
                e_N(u)^(l) = sum_{i in N(u)} (1/sqrt(|N(u)||N(i)|)) * e_i^(l)
            
            Eq. 3 (Message Construction):
                m_{u,i}^(l) = f(e_u^(l), e_i^(l), r_{u,i})
                where f is the aggregation function
            
            Eq. 5 (Representation Update via GCN Aggregator):
                e_u^(l+1) = LeakyReLU(W1 * e_u^(l) + W2 * e_N(u)^(l))
                This combines self-connection with aggregated neighbor information
        """
        
        # Step 1: Neighbor Aggregation (Eq. 1)
        # Aggregate information from neighbors using normalized adjacency
        # neighbor_embeddings = A * ego_embeddings
        # where A is the normalized adjacency matrix (Laplacian normalization)
        neighbor_embeddings = torch.sparse.mm(adjacency_matrix, ego_embeddings)
        
        # Step 2: Representation Update (Eq. 5)
        # Different aggregation strategies for combining self and neighbor info
        
        if self.aggregation_type == 'bi-interaction':
            # Bi-Interaction: Separate transforms for self and neighbors, then element-wise product
            # This captures the interaction between user and item representations
            # e^(l+1) = LeakyReLU(W1 * e^(l) + W2 * e_N^(l) + (W1 * e^(l)) * (W2 * e_N^(l)))
            self_part = self.W1(ego_embeddings)
            neighbor_part = self.W2(neighbor_embeddings)
            
            # Bi-linear interaction term (element-wise multiplication)
            interaction = self_part * neighbor_part
            
            # Combine all three components
            output = self.activation(self_part + neighbor_part + interaction)
            
        elif self.aggregation_type == 'gcn':
            # Standard GCN aggregation
            # e^(l+1) = LeakyReLU(W * (e^(l) + e_N^(l)))
            combined = ego_embeddings + neighbor_embeddings
            output = self.activation(self.W(combined))
            
        elif self.aggregation_type == 'graphsage':
            # GraphSAGE-style aggregation with concatenation
            # e^(l+1) = LeakyReLU(W * [e^(l) || e_N^(l)])
            combined = torch.cat([ego_embeddings, neighbor_embeddings], dim=1)
            output = self.activation(self.W(combined))
        
        return output
    
    def forward_relation_aware(self, ego_embeddings, edge_index, edge_type, 
                               relation_embeddings, num_nodes):
        """
        Relation-aware forward pass for knowledge graph with typed edges
        
        This variant incorporates relation embeddings into message passing,
        useful when different edge types carry different semantic meanings
        
        Args:
            ego_embeddings: Node embeddings, shape (num_nodes, in_dim)
            edge_index: Edge indices, shape (2, num_edges)
                       edge_index[0] = source nodes, edge_index[1] = target nodes
            edge_type: Relation type for each edge, shape (num_edges,)
            relation_embeddings: Embedding for each relation type, shape (num_relations, rel_dim)
            num_nodes: Total number of nodes in the graph
        
        Returns:
            Updated embeddings incorporating relation information
            
        Implements relation-aware message passing:
            m_{u,i,r}^(l) = W_r * (e_u^(l) + e_i^(l))
            where W_r depends on the relation type r
        """
        
        # Initialize aggregated messages
        aggregated = torch.zeros_like(ego_embeddings)
        
        # Get source and target node indices
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Retrieve embeddings for source nodes (neighbors)
        source_embeddings = ego_embeddings[source_nodes]
        
        # Retrieve relation embeddings for each edge
        edge_relations = relation_embeddings[edge_type]
        
        # Apply relation-specific transformation
        # For simplicity, we use element-wise multiplication with relation embeddings
        # More complex: messages = W_r * (source + relation) for each relation type
        messages = source_embeddings * edge_relations
        
        # Aggregate messages to target nodes
        # Use scatter_add to sum messages going to the same target node
        aggregated.index_add_(0, target_nodes, messages)
        
        # Normalize by degree (number of incoming edges per node)
        # Count incoming edges per node
        degree = torch.bincount(target_nodes, minlength=num_nodes).float()
        degree = degree.clamp(min=1.0)  # Avoid division by zero
        
        # Normalize aggregated messages
        aggregated = aggregated / degree.unsqueeze(1)
        
        # Final transformation
        if self.aggregation_type == 'bi-interaction':
            self_part = self.W1(ego_embeddings)
            neighbor_part = self.W2(aggregated)
            interaction = self_part * neighbor_part
            output = self.activation(self_part + neighbor_part + interaction)
        else:
            combined = ego_embeddings + aggregated
            output = self.activation(self.W(combined))
        
        return output


"""End of GCN layer definitions."""
