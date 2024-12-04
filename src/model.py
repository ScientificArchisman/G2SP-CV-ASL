from torch import nn
import torch

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """Initialize the GCN layer.

        Args:
            input_dim (int): Input dimension of the layer.
            output_dim (int): Output dimension of the layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialize weight matrix W with Xavier initialization
        self.W = nn.Parameter(torch.empty(input_dim, output_dim))
        torch.nn.init.xavier_uniform_(self.W)

        # Residual connection transformation (if input and output dimensions differ)
        self.residual_transform = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

        # Apply Xavier initialization to residual transform weights
        if self.residual_transform is not None:
            torch.nn.init.xavier_uniform_(self.residual_transform.weight)
            if self.residual_transform.bias is not None:
                torch.nn.init.zeros_(self.residual_transform.bias)

    def calculate_degree_matrix(self, A: torch.Tensor):
        """Calculate the normalized adjacency matrix and D^{-1/2}.

        Args:
            A (torch.Tensor): Adjacency matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Normalized adjacency matrix and D^{-1/2}.
        """
        A_hat = A + torch.eye(A.size(1), device=A.device)  # Add self-connections
        degrees = A_hat.sum(dim=1)
        D_neg_sqrt = torch.diag_embed(degrees.pow(-0.5))
        return A_hat, D_neg_sqrt

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        """Forward pass of the GCN layer.

        Args:
            X (torch.Tensor): Input feature matrix of shape (N, F).
            A (torch.Tensor): Adjacency matrix of shape (N, N).

        Returns:
            torch.Tensor: Output feature matrix of shape (N, output_dim).
        """
        A_hat, D_neg_sqrt = self.calculate_degree_matrix(A)

        # Graph convolution operation
        support = torch.matmul(D_neg_sqrt, torch.matmul(A_hat, D_neg_sqrt))
        output = torch.matmul(support, torch.matmul(X, self.W))

        # Residual connection
        if self.residual_transform is not None:
            X_transformed = self.residual_transform(X)
        else:
            X_transformed = X

        # Add the residual connection to the output
        output += X_transformed

        # Activation function
        return self.activation(output)


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_landmarks: int, num_classes: int, dropout: float = 0.5):
        """Initialize the GCN model.

        Args:
            input_dim (int): Input dimension of the model.
            hidden_dims (list): List of hidden dimensions.
            num_landmarks (int): Number of landmarks per graph.
            num_classes (int): Number of classes in the dataset.
            dropout (float, optional): Dropout rate. Defaults to 0.5.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # Extra layers
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dims[1])

        # Initialize the GCN layers
        self.layers = nn.ModuleList()
        current_input_dim = input_dim
        for idx, hidden_dim in enumerate(hidden_dims):
            self.layers.append(GCNLayer(current_input_dim, hidden_dim))
            self.layers.append(self.dropout)
            current_input_dim = hidden_dim

        # Insert batch normalization after the second layer
        self.layers.insert(3, self.batch_norm)

        # Output layer
        self.output_layer = nn.Linear(current_input_dim * num_landmarks, num_classes)

        # Apply Xavier initialization to the output layer
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GCN model.

        Args:
            X (torch.Tensor): Input feature matrix of shape (batch_size, N, F).
            A (torch.Tensor): Adjacency matrix of shape (N, N).

        Returns:
            torch.Tensor: Output feature matrix of shape (batch_size, num_classes).
        """
        for layer in self.layers:
            # If layer is dropout or batch normalization, don't pass adjacency matrix
            if isinstance(layer, nn.Dropout):
                X = layer(X)
            elif isinstance(layer, nn.BatchNorm1d):
                original_shape = X.shape
                X = layer(X.view(-1, X.size(-1)))  # BatchNorm1d expects (batch_size * N, num_features)
                X = X.view(original_shape)         # Reshape back to original shape
            else:
                X = layer(X, A)

        # Flatten the output and pass it through the output layer
        X = X.view(X.size(0), -1)  # Flatten features
        output = self.output_layer(X)

        return output


if __name__ == "__main__":
    model = GCN(input_dim= 4, hidden_dims=[2, 2], num_classes=100)


    print("Model architecture: ", model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")    