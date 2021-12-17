#!/usr/bin/env python
# -*- coding: utf-8 -*-


from math import e, lgamma, log, pi

from fpylll import BKZ as fplll_bkz, GSO, IntegerMatrix, LLL
from fpylll.tools.bkz_simulator import simulate
from fpylll.util import gaussian_heuristic

from g6k.algorithms.bkz import default_dim4free_fun
from g6k.utils.util import load_lwe_challenge


def delta_0f(k):
    """
    Auxiliary function giving root Hermite factors. Small values
    experimentally determined, otherwise from [Chen13]

    :param k: BKZ blocksize for which the root Hermite factor is required

    """
    small = (( 2, 1.02190),  # noqa
             ( 5, 1.01862),  # noqa
             (10, 1.01616),
             (15, 1.01485),
             (20, 1.01420),
             (25, 1.01342),
             (28, 1.01331),
             (40, 1.01295))

    k = float(k)
    if k <= 2:
        return (1.0219)
    elif k < 40:
        for i in range(1, len(small)):
            if small[i][0] > k:
                return (small[i-1][1])
    elif k == 40:
        return (small[-1][1])
    else:
        return (k/(2*pi*e) * (pi*k)**(1./k))**(1/(2*(k-1.)))


def log_gh_svp_q(d, delta, eta, n, q):
    """
    First calculates where the expected lengths (via root Hermite factor) of
    Gram--Schmidt norms are longer than q, after this point uses delta^-2 as
    the decrease factor in their length.

    Uses the volumes implied by these estimated Gram--Schmidt lengths to give
    the Gaussian heuristic in context [d - eta: d], for eta the dimension of
    some SVP call.

    :param d: the dimension of the embedding lattice = m + 1
    :param delta: the root Hermite factor of the reduction used
    :param eta: the dimension of the block [d - eta: d] for the final SVP call
    :param n: the dimension of the LWE secret
    :param q: the modulus of the LWE problem
    """

    def log_rhf_estimate(delta, dim, log_vol):
        """
        Returns the log of the estimated length of the first basis vector
        calculated using the root Hermite factor

        :param delta: the root Hermite factor of the reduction used
        :param dim: the dimension of the lattice being reduced
        :param log_vol: the log of the volume of the lattice being reduced
        """
        return (dim - 1) * log(delta) + (1./dim) * log_vol

    for i in range(d):
        dim = d - i
        # can explicitly calculate volume when presuming q vectors prior
        log_vol = (d - n - 1 - i) *  log(q)
        log_est = log_rhf_estimate(delta, dim, log_vol)
        if log_est < log(q):
            # index and height for which we no longer find q vectors
            slope_begin = i
            log_slope_height = log_est
            break

    print(slope_begin)

    # q_index represents the indices up to which we expect q vectors
    # b_0, ..., b_q_index
    q_index = slope_begin - 1 if slope_begin > 0 else None

    if q_index is None:
        # can calculate as standard using rhf estimate for b_0 and GSA
        return log_gh_svp(d, delta, eta, n, q)
    else:
        if eta <= d - slope_begin:
            # have q vectors but none feature directly in volume calculation
            log_vol = eta * log_slope_height + eta * (eta + 2*slope_begin - 2*d + 1) * log(delta)
        else:
            # number of q vectors hit is slope_begin - d + eta
            q_part = (slope_begin - d + eta) * log(q)
            # cover the entire slope
            slope_part = (d - slope_begin) * log_slope_height - (d - i) * (d - i - 1) * log(delta)
            log_vol = q_part + slope_part

    normalised_log_vol = (1./eta) * log_vol
    ball_part = ((1./eta)*lgamma((eta/2.)+1))-(.5*log(pi))

    return ball_part + normalised_log_vol


def log_gh_svp(d, delta, eta, n, q):
    """
    Calculates the log of the Gaussian heuristic of the context in which
    SVP will try and discover the projected embedded error, when no q vectors
    are assumed to be present.

    The volume component of the Gaussian heuristic (in particular the lengths
    of the appropriate Gram--Schmidt vectors) is estimated using the GSA
    [Schnorr03] with the multiplicative factor = delta^-2.

    :param d: the dimension of the embedding lattice = m + 1
    :param delta: the root Hermite factor given by the BKZ reduction
    :param eta: the dimension of the SVP call in context [d-eta:d]
    :param n: the dimension of the LWE secret
    :param q: the modulus of the LWE instance

    """
    ball_part = ((1./eta)*lgamma((eta/2.)+1))-(.5*log(pi))
    vol_part = ((1./d)*(d-n-1)*log(q))+((eta-d)*log(delta))
    return ball_part + vol_part


def gsa_params(n, alpha, q=None, samples=None, d=None, decouple=False):
    """
    Finds winning parameters (a BKZ reduction dimension and a final SVP call
    dimension) for a given Darmstadt LWE instance (n, alpha).

    :param n: the dimension of the LWE secret
    :param alpha: the noise rate of the LWE instance
    :param q: the modulus of the LWE instance. ``None`` means determine by
        reloading the challenge
    :param samples: maximum number of LWE samples to use for the embedding
        lattice. ``None`` means ``5*n``
    :param d: find best parameters for a dimension ``d`` embedding lattice
    :param decouple: if True the BKZ dimension and SVP dimension may differ

    """
    if q is None or samples is None:
        A, _, q = load_lwe_challenge(n, alpha)
        samples = A.nrows

    stddev = alpha*q

    params = decoupler(decouple, n, samples, q, stddev, d)
    min_cost_param = find_min_complexity(params)
    if min_cost_param is not None:
        return min_cost_param


def decoupler(decouple, n, samples, q, stddev, d):
    """
    Creates valid (bkz_dim, svp_dim, d) triples, as determined by
    ``primal_parameters`` and determines which succeed in the recovery of the
    embedded error.

    :param decouple: if True the BKZ dimension and SVP dimension may differ
    :param n: the dimension of the LWE secret
    :param samples: maximum number of LWE samples to use for the embedding
        lattice. ``None`` means ``5*n``
    :param q: the modulus of the LWE instance
    :param stddev: the standard deviation of the distribution from which the
        error vector components were uniformly and indepedently drawn
    :param d: find best parameters for dimension ``d`` embedding lattice

    """
    params = []

    if d is not None:
        ms = [d - 1]
    else:
        ms = range(n, min(5*n+1, samples+1))

    for m in ms:
        beta_bound = min(m+1, 110+default_dim4free_fun(110))
        svp_bound = min(m+1, 156)
        for bkz_block_size in range(40, beta_bound):
            delta_0 = delta_0f(bkz_block_size)
            if decouple:
                svp_dims = range(40, svp_bound)
            else:
                svp_dims = [min(bkz_block_size, svp_bound)]

            for svp_dim in svp_dims:
                d = float(m + 1)
                rhs = log_gh_svp(d, delta_0, svp_dim, n, q)
                if rhs - log(stddev) - log(svp_dim)/2. >= 0:
                    params.append([bkz_block_size, svp_dim, m+1])

    return params


def find_min_complexity(params):
    """
    For each valid and solving triple (bkz_dim, svp_dim, d) determines an
    approximate (!) cost and minimises.

    :param params: a list of all solving (bkz_dim, svp_dim, d) triples

    """
    min_cost = None
    min_cost_param = None
    expo = .349

    for param in params:

        bkz_block_size = param[0] - default_dim4free_fun(param[0])
        svp_dim = param[1] - default_dim4free_fun(param[1])
        d = param[2]

        bkz_cost = 2 * d * (2 ** (expo * bkz_block_size))
        finisher_svp_cost = 2 ** ((expo * svp_dim))
        new_cost = bkz_cost + finisher_svp_cost

        if min_cost is None or new_cost < min_cost:
            min_cost = new_cost
            min_cost_param = param

    return min_cost_param


def sim_params(n, alpha):
    A, c, q = load_lwe_challenge(n, alpha)
    stddev = alpha*q
    winning_params = []
    for m in range(60, min(2*n+1, A.nrows+1)):
        B = primal_lattice_basis(A, c, q, m=m)
        M = GSO.Mat(B)
        M.update_gso()
        beta_bound = min(m+1, 110+default_dim4free_fun(110)+1)
        svp_bound = min(m+1, 151)
        rs = [M.get_r(i, i) for i in range(M.B.nrows)]
        for beta in range(40, beta_bound):
            rs, _ = simulate(rs, fplll_bkz.EasyParam(beta, max_loops=1))
            for svp_dim in range(40, svp_bound):
                gh = gaussian_heuristic(rs[M.B.nrows-svp_dim:])
                if svp_dim*(stddev**2) < gh:
                    winning_params.append([beta, svp_dim, m+1])
                    break
    min_param = find_min_complexity(winning_params)
    return min_param


def primal_lattice_basis(A, c, q, m=None):
    """
    Construct primal lattice basis for LWE challenge
    ``(A,c)`` defined modulo ``q``.

    :param A: LWE matrix
    :param c: LWE vector
    :param q: integer modulus
    :param m: number of samples to use (``None`` means all)

    """
    if m is None:
        m = A.nrows
    elif m > A.nrows:
        raise ValueError("Only m=%d samples available." % A.nrows)
    n = A.ncols

    B = IntegerMatrix(m+n+1, m+1)
    for i in range(m):
        B[i, i] = q
        for j in range(n):
            B[j + m, i] = A[i, j]
        B[-1, i] = c[i]
    B[-1, -1] = 1

    B = LLL.reduction(B)
    assert(B[:n] == IntegerMatrix(n, m+1))
    B = B[n:]

    return B
