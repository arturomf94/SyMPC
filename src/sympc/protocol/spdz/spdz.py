import operator
import sympc
from concurrent.futures import ThreadPoolExecutor, wait

from sympc.protocol import beaver
from sympc.tensor.utils import modulo
from sympc.tensor import ShareTensor


EXPECTED_OPS = {"mul", "matmul"}


""" Functions that are executed at the orchestrator """
def mul_master(x, y, op_str):

    """
    [c] = [a * b]
    [eps] = [x] - [a]
    [delta] = [y] - [b]

    Open eps and delta
    [result] = [c] + eps * [b] + delta * [a] + eps * delta
    """

    if op_str not in EXPECTED_OPS:
        raise ValueError(f"{op_str} should be in {EXPECTED_OPS}")

    a_sh, b_sh, c_sh = beaver.build_triples(x, y, op_str)
    eps = x - a_sh
    delta = y - b_sh
    session = x.session
    nr_parties = len(session.session_ptr)

    eps_plaintext = eps.reconstruct(decode=False)
    delta_plaintext = delta.reconstruct(decode=False)

    with ThreadPoolExecutor(max_workers=nr_parties, thread_name_prefix="spdz_mul_master") as executor:
        args = list(zip(session.session_ptr, a_sh.shares_ptr, b_sh.shares, c_sh.shares))
        futures = [
            executor.submit(
                session.parties[i].sympc.protocol.spdz.mul_parties,
                *args[i],
                eps_plaintext,
                delta_plaintext,
                op_str
            )
            for i in range(nr_parties)
        ]

    shares = [f.result() for f in futures]
    return shares


""" Functions that are executed at each party """
def mul_parties(session, a_share, b_share, c_share, eps, delta, op_str):
    op = getattr(operator, op_str)

    eps_b = op(eps, b_share)
    delta_a = op(delta, a_share)

    share = c_share + eps_b + delta_a
    if session.rank == 0:
        delta_eps = op(delta, eps)
        share = share + delta_eps

    return share
