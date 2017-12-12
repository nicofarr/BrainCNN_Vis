# -*- coding: utf-8 -*-

## author : Amine Echraibi, Nicolas Farrugia
## Based on functions from https://github.com/AmineEch/BrainCNN/blob/master/BrainCNN.ipynb


# generating injury signitures S_1 ans S_2

def get_symmetric_noise(m, n):
    """Return a random noise image of size m x n with values between 0 and 1."""

    # Generate random noise image.
    noise_img = np.random.rand(m, n)

    # Make the noise image symmetric.
    noise_img = noise_img + noise_img.T

    # Normalize between 0 and 1.
    noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())

    assert noise_img.max() == 1  # Make sure is between 0 and 1.
    assert noise_img.min() == 0
    assert (noise_img.T == noise_img).all()  # Make sure symmetric.

    return noise_img

def simulate_injury(X, weight_A, sig_A):
    denom = (np.ones(X.shape) + (weight_A * sig_A))
    X_sig_AB = np.divide(X, denom)
    return X_sig_AB

def apply_injury_and_noise(X, Sig_A, weight_A,noise_weight):
    """Returns a symmetric, signed, noisy, adjacency matrix with simulated injury from two sources."""
    X_sig_AB = simulate_injury(X, weight_A, Sig_A)
    # Get the noise image.
    noise_img = get_symmetric_noise(X.shape[0], X.shape[1])

    # Weight the noise image.
    weighted_noise_img = noise_img * noise_weight

    # Add the noise to the original image.
    X_sig_AB_noise = X_sig_AB + weighted_noise_img
    

    assert (X_sig_AB_noise[0,:,:].T == X_sig_AB_noise[0,:,:]).all()  # Make sure still is symmetric.

    return X_sig_AB_noise


def generate_injury_signatures(X_mn, r_state,sig_indexes):
        """Generates the signatures that represent the underlying signal in our synthetic experiments.

        d : (integer) the size of the input matrix (assumes is size dxd)
        """

        # Get the strongest regions, which we will apply simulated injuries
        sig_indexes = sig_indexes
        d = X_mn.shape[0]

        S = []

        # Create a signature for
        for idx, sig_idx in enumerate(sig_indexes):
            # Okay, let's make some signature noise vectors.
            A_vec = r_state.rand((d))
            # B_vec = np.random.random((n))

            # Create the signature matrix.
            A = np.zeros((d, d))
            A[:, sig_idx] = A_vec
            A[sig_idx, :] = A_vec
            S.append(A)
            
            assert (A.T == A).all()  # Check if matrix is symmetric.

        return np.asarray(S)
def sample_injury_strengths(n_samples, X_mn, A, noise_weight):
        """Returns n_samples connectomes with simulated injury from two sources."""
        mult_factor = 10

        n_classes = 1

        # Range of values to predict.
        n_start = 0.5
        n_end = 1.4
        # amt_increase = 0.1

        # These will be our Y.
        A_weights = np.random.uniform(n_start, n_end, [n_samples])

        X_h5 = np.zeros((n_samples, 1, X_mn.shape[0], X_mn.shape[1]), dtype=np.float32)
        Y_h5 = np.zeros((n_samples, n_classes), dtype=np.float32)

        for idx in range(n_samples):
            w_A = A_weights[idx]

            # Get the matrix.
            X_sig = apply_injury_and_noise(X_mn, A, w_A * mult_factor, noise_weight)

            # Normalize.
            X_sig = (X_sig - X_sig.min()) / (X_sig.max() - X_sig.min())

            # Put in h5 format.
            X_h5[idx, 0, :, :] = X_sig
            Y_h5[idx, :] = [w_A]

        return X_h5, Y_h5