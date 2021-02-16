#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full Sieve Command Line Client
"""

import logging
import pickle as pickler
from collections import OrderedDict


from g6k.algorithms.workout import workout
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer
from g6k.utils.util import load_svpchallenge_and_randomize, load_matrix_file, db_stats
from fpylll import BKZ as BKZ_FPYLLL
from fpylll.tools.bkz_stats import dummy_tracer


def full_sieve_kernel(arg0, params=None, seed=None):
    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    pump_params = pop_prefixed_params("pump", params)
    workout_params = pop_prefixed_params("workout", params)
    verbose = params.pop("verbose")
    load_matrix = params.pop("load_matrix")
    pre_bkz = params.pop("pre_bkz")
    trace = params.pop("trace")

    reserved_n = n
    params = params.new(reserved_n=reserved_n, otf_lift=False)

    if load_matrix is None:
        A, bkz = load_svpchallenge_and_randomize(n, s=0, seed=seed)
        if verbose:
            print("Loaded challenge dim %d" % n)
        if pre_bkz is not None:
            par = BKZ_FPYLLL.Param(pre_bkz, strategies=BKZ_FPYLLL.DEFAULT_STRATEGY, max_loops=1)
            bkz(par)

    else:
        A, _ = load_matrix_file(load_matrix, doLLL=False, high_prec=False)
        if verbose:
            print("Loaded file '%s'" % load_matrix)


    g6k = Siever(A, params, seed=seed)
    if trace:
        tracer = SieveTreeTracer(g6k, root_label=("full-sieve", n), start_clocks=True)
    else:
        tracer = dummy_tracer

    # Actually runs a workout with very large decrements, so that the basis is kind-of reduced
    # for the final full-sieve
    workout(g6k, tracer, 0, n, dim4free_min=0, dim4free_dec=15, pump_params=pump_params, verbose=verbose, **workout_params)

    g6k.output_bench()

    tracer.exit()

    if hasattr(tracer, "trace"):
        return tracer.trace
    else:
        return None

def full_sieve():
    """
    Run a a full sieve (with some partial sieve as precomputation).
    """
    description = full_sieve.__doc__

    args, all_params = parse_args(description, trace=True)

    stats = run_all(full_sieve_kernel, all_params.values(),
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed)

    inverse_all_params = OrderedDict([(v, k) for (k, v) in all_params.iteritems()])

    for (n, params) in stats:
        stat = stats[(n, params)]
        if stat[0] is None:
            logging.info("Trace disabled")
            continue
        cputime = sum([float(node["cputime"]) for node in stat])/len(stat)
        walltime = sum([float(node["walltime"]) for node in stat])/len(stat)
        avr_db, max_db = db_stats(stat)
        fmt = "%48s :: m: %1d, n: %2d, cputime :%7.4fs, walltime :%7.4fs, avr_max |db|: 2^%2.2f, max_max db |db|: 2^%2.2f"  # noqa
        logging.info(fmt %(inverse_all_params[params], params.threads, n, cputime, walltime, avr_db, max_db))

    if args.pickle:
        pickler.dump(stats, open("full-sieve-%d-%d-%d-%d.sobj" %
                                 (args.lower_bound, args.upper_bound, args.step_size, args.trials), "wb"))


if __name__ == '__main__':
    full_sieve()
