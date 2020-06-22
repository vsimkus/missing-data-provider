from itertools import product

import pytest
import scipy.stats as ss
import torch

from missing_data_provider import MissingDataProvider, _generate_patterns


@pytest.mark.skip(reason=('Since the fraction of incomplete samples is '
                          'determined in advance, and then patterns are '
                          'sampled at random, we cannot guarantee the '
                          'exact missingness that we will get.'))
def test_generate_correct_total_missingness():
    N, D = 1000, 20
    max_patterns = 50
    total_miss = [0.9, 0.5, 0.1]
    miss_types = ('MCAR', 'MNAR', 'MAR')
    data = torch.randn(N, D)

    for miss_type, miss in product(miss_types, total_miss):
        data_provider = MissingDataProvider(data,
                                            total_miss=miss,
                                            miss_type=miss_type,
                                            max_patterns=max_patterns,
                                            should_fit_to_data=True)

        X, M = data_provider[:]
        assert X.shape == data.shape and M.shape == data.shape,\
            'Output data shape does not match input!'

        total_values = M.numel()
        total_missing = (total_values - M.sum()).item()

        # Allowing 2% error due to sampling
        assert abs(total_missing/total_values - miss) < 0.02,\
            'Total missingness is not correct!'


def test_little_mcar():
    N, D = 1000, 20
    total_miss = 0.6
    A = torch.randn(D, D)
    mean = torch.randn(D)
    data = torch.randn(N, D) @ A + mean
    cov = A.T @ A

    dataset = torch.utils.data.TensorDataset(data)
    data_provider = MissingDataProvider(dataset,
                                        total_miss=total_miss,
                                        miss_type='MCAR')

    data, mask = data_provider[:]
    d2, df = _little_MCAR_statistics(data, mask,
                                     mean=mean,
                                     cov=cov)

    p_value = 1 - ss.chi2.cdf(d2, df).item()
    assert not p_value < 0.05


def test_little_mnar():
    N, D = 10000, 20
    max_patterns = 50
    total_miss = 0.6
    A = torch.randn(D, D)
    mean = torch.randn(D)
    data = torch.randn(N, D) @ A + mean
    cov = A.T @ A

    dataset = torch.utils.data.TensorDataset(data)
    data_provider = MissingDataProvider(dataset,
                                        total_miss=total_miss,
                                        max_patterns=max_patterns,
                                        miss_type='MNAR',
                                        should_fit_to_data=True)

    data, mask = data_provider[:]
    d2, df = _little_MCAR_statistics(data, mask,
                                     mean=mean,
                                     cov=cov)

    p_value = 1 - ss.chi2.cdf(d2, df).item()
    assert p_value < 0.05


def test_little_mcar_with_custom_patterns():
    N, D = 10000, 6
    max_patterns = 100
    total_miss = 0.6
    A = torch.randn(D, D)
    mean = torch.randn(D)
    data = torch.randn(N, D) @ A + mean
    cov = A.T @ A

    # Create patterns
    patterns, rel_freqs = _generate_patterns(max_patterns, D, total_miss=total_miss+0.1)

    # Setting all weights to zero makes the mechanism strictly MCAR
    weights = torch.zeros_like(patterns, dtype=torch.float)

    dataset = torch.utils.data.TensorDataset(data)
    data_provider = MissingDataProvider(dataset,
                                        total_miss=total_miss,
                                        miss_type='patterns',
                                        patterns=patterns,
                                        weights=weights,
                                        rel_freqs=rel_freqs,
                                        should_fit_to_data=True)

    data, mask = data_provider[:]
    d2, df = _little_MCAR_statistics(data, mask,
                                     mean=mean,
                                     cov=cov)

    p_value = 1 - ss.chi2.cdf(d2, df).item()
    assert not p_value < 0.05


def test_little_mnar_with_custom_patterns():
    N, D = 10000, 6
    max_patterns = 100
    total_miss = 0.6
    A = torch.randn(D, D)
    mean = torch.randn(D)
    data = torch.randn(N, D) @ A + mean
    cov = A.T @ A

    # Create patterns
    patterns, rel_freqs = _generate_patterns(max_patterns, D, total_miss=total_miss+0.1)

    # Creat strict MNAR weights
    weights = torch.ones_like(patterns, dtype=torch.float)
    weights[patterns] = 0.

    dataset = torch.utils.data.TensorDataset(data)
    data_provider = MissingDataProvider(dataset,
                                        total_miss=total_miss,
                                        miss_type='patterns',
                                        patterns=patterns,
                                        weights=weights,
                                        rel_freqs=rel_freqs,
                                        should_fit_to_data=True)

    data, mask = data_provider[:]
    d2, df = _little_MCAR_statistics(data, mask,
                                     mean=mean,
                                     cov=cov)

    p_value = 1 - ss.chi2.cdf(d2, df).item()
    print(p_value)
    # asd
    assert p_value < 0.05


def _little_MCAR_statistics(data, mask, mean=None, cov=None):
    # TODO: make a utility
    if mean is None and cov is None:
        # Fit Gaussian parameters using EM
        mean, cov = _fit_multivariate_Gaussian_from_incomplete_data(data, mask)
        # Unbiased estimator
        cov = data.shape[0] / (data.shape[0] - 1) * cov

    # Remove fully-observed
    incomp_mask = mask.sum(dim=1) != data.shape[-1]
    data = data[incomp_mask, :]
    mask = mask[incomp_mask, :]

    # Count patterns and group them
    unique_masks, inverse_idx, counts = torch.unique(mask,
                                                     dim=0,
                                                     return_inverse=True,
                                                     return_counts=True)
    counts = counts.to(torch.float)

    d2 = 0
    df = -data.shape[-1]
    # TODO: vectorise
    for j in range(len(unique_masks)):
        X_j = data[inverse_idx == j, :]
        mask_j = unique_masks[j, :]

        cov_j = cov[mask_j, :]
        cov_j = cov_j[:, mask_j]
        X_j = X_j[:, mask_j]
        emean_j = X_j.mean(dim=0)
        mean_j = mean[mask_j]
        dif = emean_j - mean_j

        # Workaround for fully unobserved datapoints
        if dif.shape[0] > 0:
            Z = (dif[None, :] @ torch.inverse(cov_j) @ dif[:, None]).flatten().item()
        else:
            #NOTE ????
            Z = 0
        d2 += counts[j] * Z
        df += (~mask_j).sum()
        print(d2)

    return d2, df


def _fit_multivariate_Gaussian_from_incomplete_data(data, mask, max_iter=100, eta=1e-4):
    # TODO: factor out into a utility
    mask = mask
    mask_not = ~mask

    assert data.shape == mask.shape
    data = data.clone()

    # Initialise parameters - mean and covariance
    mean = torch.zeros((data.shape[-1], ), dtype=torch.float)
    # Generate random pos-definite covariance matrix
    cov = torch.randn((data.shape[-1], data.shape[-1]), dtype=torch.float)
    cov = cov @ cov.T

    for j in range(max_iter):
        # E-step
        mean_mis_all = []
        cov_mis_all = []
        # Vectorising this code is a little difficult, so let's leave it
        # TODO: vectorise
        for i in range(data.shape[0]):
            # Select relevant submatrices
            cov_mm = cov[mask_not[i, :], :]
            cov_mm = cov_mm[:, mask_not[i, :]]
            cov_mo = cov[mask_not[i, :], :]
            cov_mo = cov_mo[:, mask[i, :]]
            cov_oo = cov[mask[i, :], :]
            cov_oo = cov_oo[:, mask[i, :]]

            # Compute the posterior cov and set it in the matrix
            Z = cov_mo @ torch.inverse(cov_oo)
            c = cov_mm - Z @ cov_mo.T
            cov_mis = torch.zeros_like(cov)
            idx = torch.nonzero(mask_not[i, :])
            cov_mis[idx, idx.T] = c
            cov_mis_all.append(cov_mis)

            # Compute the posterior mean
            mean_mis = torch.zeros_like(mean)
            mean_mis[mask_not[i, :]] = \
                (mean[mask_not[i, :]] + Z @ (data[i, mask[i, :]] - mean[mask[i, :]]))
            mean_mis_all.append(mean_mis)

        # Concatenate tensors
        cov_mis = torch.stack(cov_mis_all)
        mean_mis = torch.stack(mean_mis_all)

        # M-step
        data[mask_not] = mean_mis[mask_not]
        old_mean = mean
        mean = data.mean(dim=0)

        old_cov = cov
        dif = data - mean
        cov = (dif.unsqueeze(-1) @ dif.unsqueeze(-2) + cov_mis).mean(dim=0)

        if (torch.norm(old_mean - mean) < eta
                and torch.norm(old_cov - cov) < eta):
            break

    print(f'Finished EM in {j}/{max_iter} iterations.')

    return mean, cov
