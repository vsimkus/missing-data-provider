from functools import reduce
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.distributions.beta import Beta


__all__ = ['MissingDataProvider',
           '_generate_patterns',
           '_random_weights_for_patterns']


class MissingDataProvider(Dataset):
    """
    Generates missing data in the given dataset.
    The missingness is denoted by adding additional tensor M,
    whose size is the same as X.
    Args:
        dataset (torch.utils.data): Fully-observed PyTorch dataset
        target_idx (int): If the dataset returns tuples, then this should be the index
            of the target data in the tuple for which the missing mask is added.
        total_miss (float): Total fraction of values to be made missing
        miss_type (str): The type of missingness to be generated
            - if `patterns`, then the provided patterns are used
            - if `MCAR`, then generates a uniformly distributed mask on the whole data
            - if `MAR`, `MNAR` (or `NMAR`), then generates #max_patterns random patterns
                and generates the missing masks using the patterns
        patterns (torch.Tensor or np.ndarray): Patterns to be used if miss_type == `patterns`
        rel_freqs (torch.Tensor or np.ndarray): Relative frequencies of the given patterns
        weights (torch.Tensor or np.ndarray): Mechanism to be used with the given patterns
        balances (torch.Tensor or np.ndarray): If the weights were previously fitted on some
            data then the fitted balances can be set too.
        max_patterns (int): Maximum number of patterns to generate for the chosen miss_type
            if the miss_type is `MAR`, `MNAR`, or `NMAR` (but not for MCAR).
        should_fit_to_data (bool): Whether the weights should be fitted to the data and new balance
            terms should be computed. Most often you want this to be set to True, unless you're
            reusing weights and balances fitted onto another say, e.g. fitted on training data,
            and reusing on test data. In this case weights and balances should be provided.
        rand_generator (torch.Generator): (optional) PyTorch random number generator.
    """
    def __init__(self,
                 dataset: torch.Tensor,
                 target_idx: int = 0,
                 total_miss: float = 0.00,
                 miss_type: str = 'MCAR',
                 patterns: Union[np.ndarray, torch.Tensor] = None,
                 rel_freqs: Union[np.ndarray, torch.Tensor] = None,
                 weights: Union[np.ndarray, torch.Tensor] = None,
                 balances: Union[np.ndarray, torch.Tensor] = None,
                 max_patterns: int = None,
                 should_fit_to_data: bool = True,
                 rand_generator: torch.Generator = None):
        super().__init__()
        self.dataset = dataset
        self.target_idx = target_idx
        self.total_miss = total_miss
        self.miss_type = miss_type
        self.should_fit_to_data = should_fit_to_data
        self.max_patterns = max_patterns

        self.patterns = patterns
        self.rel_freqs = rel_freqs
        self.weights = weights
        self.balances = balances

        self._validate_args()

        # Initialise pseudo-random number generator
        self.rand_generator = (rand_generator
                               if rand_generator is not None
                               else torch.Generator())

        # Any preparations that need to be done before sampling masks
        self.prepare_prerequisites()

        # Sample the missingness mask
        self.init_miss_mask()

    def prepare_prerequisites(self):
        if self.miss_type == 'MCAR':
            # No preparation needed for MCAR
            return

        # Get target data
        data = self._get_target_data()

        # If the type is one of the below then we generate random missingness patterns
        if self.miss_type in ('MNAR', 'NMAR', 'MAR'):
            # When generating patterns we generally want a little higher missing value
            # fraction in the patterns, so that we can have some completely-observed
            # cases too.
            pattern_miss = min(Beta(2, 5).sample().item(), 1)
            pattern_miss = (1 - self.total_miss) * pattern_miss + self.total_miss
            self.patterns, self.rel_freqs = _generate_patterns(
                                        self.max_patterns,
                                        D=data.shape[-1],
                                        total_miss=pattern_miss,
                                        rand_generator=self.rand_generator)

            self.weights = _random_weights_for_patterns(
                                        self.patterns,
                                        miss_mech=self.miss_type,
                                        rand_generator=self.rand_generator)

        # Convert total % of missing values to % of incomplete samples
        self.incomp_frac = self._incomplete_sample_fraction(
                                        self.patterns,
                                        self.rel_freqs,
                                        self.total_miss,
                                        data.shape)

        # Choose samples to be made incomplete
        self.incomp_idxs, _ = \
            self._split_comp_and_incomp_idxs(data, self.incomp_frac)

        if self.should_fit_to_data:
            self.fit_to_data(data[self.incomp_idxs, :])

    def init_miss_mask(self):
        # Get all target data
        data = self._get_target_data()

        if self.miss_type == 'MCAR':
            self.miss_mask = self._uniform_mask(data)
        elif self.miss_type in ('patterns', 'MNAR', 'NMAR', 'MAR'):
            self.miss_mask = self._sample_miss_mask(data)
        else:
            raise Exception('No such missingness mechanism type allowed:'
                            f' {self.miss_type}!')

    def __getitem__(self, idx):
        data = self.dataset[idx]
        miss_mask = self.miss_mask[idx]
        if isinstance(data, tuple):
            # Insert missingness mask after the target_idx tensor to which it corresponds
            data = (data[:self.target_idx+1]
                    + (miss_mask,)
                    + data[self.target_idx+1:])
        else:
            data = (data, miss_mask)

        return data

    def __len__(self):
        return len(self.dataset)

    def fit_to_data(self, incomp_data):
        """ Fits weights and balance terms
        """
        # Compute the scores for each data point and its pattern's weight
        # other scores are set to zero
        wx = incomp_data @ self.weights.T

        # Where score is always 0, the pattern is MCAR
        mcar_dims = torch.all(wx == 0, dim=0)

        # Compute the standardised z-score for each pattern
        score_std, score_mean = torch.std_mean(wx, unbiased=True, dim=0)
        # Prevent division by zero for MCAR patterns
        score_std[mcar_dims] = 1.
        wx_standardised = (wx - score_mean) / score_std

        # Compute the balance term for each pattern
        b = torch.log(self.rel_freqs) - torch.mean(wx_standardised, axis=0)

        self.weights = self.weights / score_std[:, None]
        self.balances = b - (score_mean / score_std)

    def _sample_miss_mask(self, data):
        incomp_data = data[self.incomp_idxs, :]

        # Sample missingness masks
        incomp_mask_idxs = self._sample_miss_mask_idx(incomp_data,
                                                      self.weights,
                                                      self.balances)
        incomp_masks = self.patterns[incomp_mask_idxs]

        # Create a mask of all 1s for the fully-observed data-points
        all_masks = torch.ones_like(data, dtype=torch.bool)
        all_masks[self.incomp_idxs] = incomp_masks

        return all_masks

    def _get_target_data(self, idx=slice(None)):
        # NOTE: this won't work with large datasets that do not fit into memory
        data = self.dataset[idx]
        if isinstance(data, tuple):
            # Get the data for which the missing masks are generated
            data = data[self.target_idx]
        return data

    def _validate_args(self):
        assert self.miss_type in ('patterns', 'MCAR', 'MAR', 'MNAR', 'NMAR'),\
            f'Invalid missingness mechanism type: {self.miss_type}!'

        assert 0 < self.total_miss <= 1,\
            f'Invalid total missingness: {self.total_miss:.2f}!'

        if self.miss_type == 'patterns':
            assert None not in (self.patterns, self.rel_freqs, self.weights),\
                'For miss_type==patterns, patterns, rel_freqs, and weights must be provided!'

        if self.patterns is not None:
            assert (self.patterns.shape[0] == self.rel_freqs.shape[0]
                    and self.patterns.shape[0] == self.weights.shape[0]
                    and (self.balances is None
                         or self.patterns.shape[0] == self.balances.shape[0])),\
                'Shapes of patterns, rel_freqs, or weights (and balances) do not match!'

        if self.miss_type == 'patterns' and not self.should_fit_to_data:
            assert self.weights is not None and self.balances is not None,\
                ('For miss_type==`patterns` and should_fit_to_data==False, the fitted weights '
                 'and balances should be provided!')

        if self.miss_type in ('MCAR', 'MAR', 'MNAR', 'NMAR'):
            assert self.should_fit_to_data, \
                'If generating missingness patterns, then should_fit_to_data should be set to True!'

        if self.miss_type in ('MAR', 'MNAR', 'NMAR'):
            assert self.max_patterns is not None, \
                'If generating missingness patterns, then max_patterns should be set given!'

    def _uniform_mask(self, data):
        # Works for PyTorch and Numpy
        total_values = reduce(lambda x, y: x*y, data.shape, 1)

        # Generate appropriate number of missing values
        miss_mask = torch.ones(total_values, dtype=torch.bool)
        miss_mask[:int(self.total_miss*total_values)] = 0

        # Randomise mask
        rand_idx = torch.randperm(total_values, generator=self.rand_generator)
        miss_mask = miss_mask[rand_idx]
        return miss_mask.reshape_as(data)

    def _incomplete_sample_fraction(self, patterns, rel_freqs,
                                    total_miss, data_shape):
        total_values = reduce(lambda x, y: x*y, data_shape, 1)
        miss_values = total_miss * total_values

        # The number of incomplete samples for each pattern
        C = miss_values * rel_freqs / torch.sum(~patterns, dim=1)

        # Total number of incomplete samples for all patterns
        C = torch.sum(C)

        assert C < data_shape[0],\
            ('The calculated incomplete sample fraction is greater than the '
             'number of samples. This means that the patterns and relative '
             'frequencies are not compatible with the requested total '
             'missingness fraction.')

        # The incomplete fraction of all data points
        return C / data_shape[0]

    def _split_comp_and_incomp_idxs(self, data, frac_incomp_samples):
        # The total number of incomplete data points
        total_incomp = int(torch.floor(frac_incomp_samples * data.shape[0]))

        # Randomly split the data into incomplete and complete subsets
        all_idx = torch.randperm(data.shape[0], generator=self.rand_generator)
        return all_idx[:total_incomp], all_idx[total_incomp:]

    def _sample_miss_mask_idx(self, incomp_data, weights, balances):
        scores = incomp_data @ weights.T + balances

        probs = torch.nn.functional.softmax(scores, dim=-1)
        return torch.multinomial(probs, 1,
                                 replacement=False,
                                 generator=self.rand_generator).squeeze()


def _generate_patterns(max_patterns, D, total_miss, rand_generator=None):
    """
    Generate missingness patterns as binary masks. 1 is observed and 0 is missing.
    Args:
        max_patterns (int): maximum number of binary missingness patterns
        D (int): dimensionality of each patterns
        total_miss (float): The total fraction of missing values in the
            patterns, between 0 and 1.
        rand_generator (torch.Generator): (optional) PyTorch random number generator.
    """
    rand_generator = (rand_generator if rand_generator is not None
                      else torch.Generator())

    # Create an array with the desired fraction of missing values
    total_vals = max_patterns*D
    miss_vals = int(total_vals*total_miss)
    patterns = torch.ones((total_vals, ), dtype=torch.bool)
    patterns[:miss_vals] = 0.

    # Shuffle the array and reshape to the desired pattern shape
    rand_idx = torch.randperm(total_vals, generator=rand_generator)
    patterns = patterns[rand_idx]
    patterns = patterns.reshape(max_patterns, D)

    # Only return unique patterns and get their relative frequency
    # TODO: maybe not unique?
    patterns, rel_freqs = torch.unique(patterns, dim=0, return_counts=True)
    rel_freqs = rel_freqs.float()
    rel_freqs = rel_freqs / rel_freqs.sum()

    # If there is a full-observed mask, we want to remove it
    fully_observed = torch.all(patterns, dim=-1)
    if torch.any(fully_observed):
        full_rel_freqs = rel_freqs[fully_observed]

        not_fully_observed = ~fully_observed
        rel_freqs = rel_freqs[not_fully_observed]
        patterns = patterns[not_fully_observed]
        num_pattern_ones = patterns.sum(dim=1)

        # Redistribute the ones by changing the relative freqs of the
        # incomplete patterns
        delta = full_rel_freqs.sum()*D/patterns.shape[0]
        rel_freqs += delta
        rel_freqs /= num_pattern_ones

        # Renormalise the freqs
        rel_freqs = rel_freqs / rel_freqs.sum()

    return patterns, rel_freqs


def _random_weights_for_patterns(patterns, miss_mech, rand_generator=None):
    """
    Generate a random weight matrix for each pattern. Weight values in [-1, 1)
    Args:
        patterns (torch.Tensor): Patterns for which to generate random weights
        miss_mech (str): MNAR (zeros where patterns==1) or MAR (zeros where patterns==0)
        rand_generator (torch.Generator): (optional) PyTorch random number generator.
    """
    rand_generator = (rand_generator if rand_generator is not None
                      else torch.Generator())

    # Generate random weights in [-1, 1)
    weights = torch.rand(*(patterns.shape))*2-1

    if miss_mech in ('MNAR', 'NMAR'):
        # Make sure the patterns depend on missing variables only
        weights *= ~patterns
    elif miss_mech == 'MAR':
        # Make sure the patterns depend on observed variables only
        weights *= patterns
    else:
        raise Exception('No such missingness mechanism type allowed:'
                        f' {miss_mech}!')

    return weights
