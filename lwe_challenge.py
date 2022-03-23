#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LWE Challenge Solving Command Line Client
"""

import copy
import re
import sys
import time

from collections import OrderedDict # noqa
from math import log

from fpylll import BKZ as fplll_bkz
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic, set_threads

from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.algorithms.pump import pump
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer, dummy_tracer
from g6k.utils.util import load_lwe_challenge, load_matrix_file

from g6k.utils.lwe_estimation import gsa_params, primal_lattice_basis


def lwe_kernel(arg0, params=None, seed=None):
    """
    Run the primal attack against Darmstadt LWE instance (n, alpha).

    :param n: the dimension of the LWE-challenge secret
    :param params: parameters for LWE:

        - lwe/alpha: the noise rate of the LWE-challenge

        - lwe/m: the number of samples to use for the primal attack

        - lwe/goal_margin: accept anything that is
          goal_margin * estimate(length of embedded vector)
          as an lwe solution

        - lwe/svp_bkz_time_factor: if > 0, run a larger pump when
          svp_bkz_time_factor * time(BKZ tours so far) is expected
          to be enough time to find a solution

        - bkz/blocksizes: given as low:high:inc perform BKZ reduction
          with blocksizes in range(low, high, inc) (after some light)
          prereduction

        - bkz/tours: the number of tours to do for each blocksize

        - bkz/jump: the number of blocks to jump in a BKZ tour after
          each pump

        - bkz/extra_dim4free: lift to indices extra_dim4free earlier in
          the lattice than the currently sieved block

        - bkz/fpylll_crossover: use enumeration based BKZ from fpylll
          below this blocksize

        - bkz/dim4free_fun: in blocksize x, try f(x) dimensions for free,
          give as 'lambda x: f(x)', e.g. 'lambda x: 11.5 + 0.075*x'

        - pump/down_sieve: sieve after each insert in the pump-down
          phase of the pump

        - dummy_tracer: use a dummy tracer which captures less information

        - verbose: print information throughout the lwe challenge attempt

    """

    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    params = copy.copy(params)

    # params for underlying BKZ
    extra_dim4free = params.pop("bkz/extra_dim4free")
    jump = params.pop("bkz/jump")
    dim4free_fun = params.pop("bkz/dim4free_fun")
    pump_params = pop_prefixed_params("pump", params)
    fpylll_crossover = params.pop("bkz/fpylll_crossover")
    blocksizes = params.pop("bkz/blocksizes")
    tours = params.pop("bkz/tours")

    # flow of the lwe solver
    svp_bkz_time_factor = params.pop("lwe/svp_bkz_time_factor")
    goal_margin = params.pop("lwe/goal_margin")

    # generation of lwe instance and Kannan's embedding
    alpha = params.pop("lwe/alpha")
    m = params.pop("lwe/m")
    decouple = svp_bkz_time_factor > 0

    # misc
    dont_trace = params.pop("dummy_tracer")
    verbose = params.pop("verbose")

    threads = params.get("threads", None)
    if threads is not None:
        set_threads(threads)

    A, c, q = load_lwe_challenge(n=n, alpha=alpha)
    print("-------------------------")
    print("Primal attack, LWE challenge n=%d, alpha=%.4f" % (n, alpha))

    if m is None:
        try:
            min_cost_param = gsa_params(n=A.ncols, alpha=alpha, q=q,
                                        decouple=decouple)
            (b, s, m) = min_cost_param
        except TypeError:
            raise TypeError("No winning parameters.")
    else:
        try:
            min_cost_param = gsa_params(n=A.ncols, alpha=alpha, q=q,
                                        decouple=decouple)
            (b, s, _) = min_cost_param
        except TypeError:
            raise TypeError("No winning parameters.")
    print("Chose %d samples. Predict solution at bkz-%d + svp-%d" % (m, b, s))
    print("")

    # no use in having a very small b
    b = max(b, s-65)

    if blocksizes is not None:
        blocksizes = list(range(10, 40)) + eval("range(%s)" % re.sub(":", ",", blocksizes)) # noqa
    else:
        blocksizes = list(range(10, 50)) + list(reversed(range(b-14, 60, -10))) + list(range(b - 12, b + 25, 2)) # noqa

    loadblocksize = 0
#    loadblocksize=113
#    loadtour=0
#    loadtime=96696.556
    if loadblocksize == 0:
        B = primal_lattice_basis(A, c, q, m=m)
    else:
        print("Loading matrix file: lwechallenge/B_n%d_a%.4f_block%d_tour%d.mat" % (n, alpha, loadblocksize, loadtour))
        sys.stdout.flush()
        B, _ = load_matrix_file("lwechallenge/B_n%d_a%.4f_block%d_tour%d.mat" % (n, alpha, loadblocksize, loadtour), doLLL=False, high_prec=True)

    g6k = Siever(B, params)
    print("GSO precision: ", g6k.M.float_type)

    if dont_trace:
        tracer = dummy_tracer
    else:
        tracer = SieveTreeTracer(g6k, root_label=("lwe"), start_clocks=True)

    d = g6k.full_n
    g6k.lll(0, g6k.full_n)

    target_norm = (alpha*q)**2 * m + 1
    full_gh = gaussian_heuristic([g6k.M.get_r(i, i) for i in range(d)])
    target_norm_slack = min(goal_margin * target_norm, 0.98 * full_gh)

    slope = basis_quality(g6k.M)["/"]
    print("Intial Slope = %.5f\n" % slope)

    T0 = time.time()
    T0_BKZ = time.time()
    for blocksize in blocksizes:
        for tt in range(tours):
            if blocksize < loadblocksize:
                continue
            if blocksize == loadblocksize:
                T0_BKZ = time.time() - loadtime
                T0 = T0_BKZ
            else:
                # BKZ tours

                if blocksize < fpylll_crossover:
                    if verbose:
                        print("Starting a fpylll BKZ-%d tour. " % (blocksize), end='')
                        sys.stdout.flush()
                    bkz = BKZReduction(g6k.M)
                    par = fplll_bkz.Param(blocksize,
                                          strategies=fplll_bkz.DEFAULT_STRATEGY,
                                          max_loops=1)
                    bkz(par)

                else:
                    if verbose:
                        print("Starting a pnjBKZ-%d tour. " % (blocksize))

                    pump_n_jump_bkz_tour(g6k, tracer, blocksize, jump=jump,
                                         verbose=verbose,
                                         extra_dim4free=extra_dim4free,
                                         dim4free_fun=dim4free_fun,
                                         goal_r0=target_norm_slack,
                                         pump_params=pump_params)

            T_BKZ = time.time() - T0_BKZ

            if verbose:
                slope = basis_quality(g6k.M)["/"]
                fmt = "slope: %.5f, walltime: %.3f sec"
                print(fmt % (slope, time.time() - T0))

            g6k.lll(0, g6k.full_n)

            if blocksize >= fpylll_crossover and blocksize > loadblocksize:
                fn = open("lwechallenge/B_n%d_a%.4f_block%d_tour%d.mat" % (n, alpha, blocksize, tt), "w")
                fn.write(str(g6k.M.B))
                fn.close()

            if g6k.M.get_r(0, 0) <= target_norm_slack:
                break

            # overdoing n_max would allocate too much memory, so we are careful
            svp_Tmax = svp_bkz_time_factor * T_BKZ
            expo = 0.292
            # maximal d s.t. 2^{expo * d} / param.threads <= svp_Tmax
            # cannot figure out the additive 58, will leave for now
            n_max = int(58+2+(1./expo) * log(svp_Tmax*params.threads)/log(2.))
            rr = [g6k.M.get_r(i, i) for i in range(d)]
            for n_expected in range(2, d-2):
                x = target_norm * n_expected/(1.*d)
                if 4./3. * gaussian_heuristic(rr[d-n_expected:]) > x:
                    break
            # ensure a BigSVP attempt after the last bkz tour
            if blocksize >= blocksizes[-1] and tt >= tours-1:
                n_max = max(n_max, n_expected+2)
            print( "Without otf, would expect solution at pump-%d. n_max=%d in the given time." % (n_expected, n_max) )# noqa
            sys.stdout.flush()
            if n_expected >= n_max - 1:
                continue
            n_max += 1
            # Larger SVP
            llb = d - n_max
            while llb >= 1 and gaussian_heuristic([g6k.M.get_r(i, i) for i in range(llb, d)]) < target_norm * (d - llb)/(1.*d): # noqa
                print("gh([%d,%d])=%f < %f" % (llb, d, (gaussian_heuristic([g6k.M.get_r(i, i) for i in range(llb, d)])), target_norm * (d - llb)/(1.*d)))
                sys.stdout.flush()
                llb -= 1
            if llb < 0:
                print(" llb < 0 ")
                sys.stdout.flush()
                raise ValueError("llb < 0")

            # llb is kappa for pump, so the point to which solutions are lifted
            # lift_slack attempts to ensure we do not miss solution purely for
            # not lifting enough. Note that while this increases beta = d-llb
            # it increases dim4free f by the same amount, so the only extra
            # cost incurred is from more lifting

            lift_slack = 3
            llb = max(0, llb-lift_slack)

            f = max(d-llb-n_expected, 0)
            if verbose:
                print( "Starting svp pump_{%d, %d, %d}, n_max = %d, Tmax= %.2f sec" % (llb, d-llb, f, n_max, svp_Tmax) ) # noqa
                sys.stdout.flush()

            proj_target_norm = target_norm * (d - llb)/(1.*d)
            proj_gh = gaussian_heuristic([g6k.M.get_r(i, i) for i in range(llb, d)]) # noqa
            proj_target_norm_slack = min(goal_margin * proj_target_norm,
                                         0.98 * proj_gh)

            pump(g6k, tracer, llb, d-llb, f, verbose=verbose,
                 goal_r0=proj_target_norm_slack)

            if verbose:
                slope = basis_quality(g6k.M)["/"]
                fmt = "\n slope: %.5f, walltime: %.3f sec"
                print(fmt % (slope, time.time() - T0))
                print

            g6k.lll(0, g6k.full_n)
            T0_BKZ = time.time()
            if g6k.M.get_r(0, 0) <= target_norm:
                break

        if g6k.M.get_r(0, 0) <= target_norm:
            print("Finished! TT=%.2f sec" % (time.time() - T0))
            print(g6k.M.B[0])
            alpha_ = int(alpha*1000)
            filename = 'lwechallenge/%03d-%03d-solution.txt' % (n, alpha_)
            fn = open(filename, "w")
            fn.write(str(g6k.M.B[0]))
            fn.close()
            return

    raise ValueError("No solution found.")


def lwe():
    """
    Attempt to solve an lwe challenge.

    """
    description = lwe.__doc__

    args, all_params = parse_args(description,
                                  lwe__alpha=0.005,
                                  lwe__m=None,
                                  lwe__goal_margin=1.5,
                                  lwe__svp_bkz_time_factor=1,
                                  bkz__blocksizes=None,
                                  bkz__tours=1,
                                  bkz__jump=1,
                                  bkz__extra_dim4free=12,
                                  bkz__fpylll_crossover=51,
                                  bkz__dim4free_fun="default_dim4free_fun",
                                  pump__down_sieve=True,
                                  dummy_tracer=True,  # set to control memory
                                  verbose=True
                                  )

    stats = run_all(lwe_kernel, all_params.values(), # noqa
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed)


if __name__ == '__main__':
    lwe()
