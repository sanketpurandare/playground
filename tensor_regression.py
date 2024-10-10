import torch

if __name__ == "__main__":
    # N - number of simulations
    # D - number of days
    # Z - number of features/coefficients

    N, D, Z = 128, 1024, 3
    dev = torch.cuda.current_device()
    # Coefficients vector is of size (Z)
    coeff = torch.randn(Z, device=dev)
    # Single simulation is of size (D, Z)
    sim = torch.randn(D, Z, device=dev)
    # Epsilon is of size (D, 1)
    epsilon = torch.randn(D, device=dev)
    res = sim @ coeff + epsilon
    # the res shape will be (D, 1)
    print(res.shape)

    # Now when we have N simulations, the size is (N, D, Z)
    sims = torch.randn(N, D, Z, device=dev)

    # To obtain the result, we need to reshape it to (N * D, Z)
    sims = sims.reshape(N * D, Z)
    # The result will be of shape (N * D, 1), hence we reshape it to (N, D)
    # For adding the epsilon we need to broadcast the same epsilon of shape (D) to N simulations
    # Hence we use the unsqueeze operation to do the broadcasting
    res = (sims @ coeff).reshape(N, D) + epsilon.unsqueeze(0)
    # The final result shape is (N, D)
    print(res.shape)
