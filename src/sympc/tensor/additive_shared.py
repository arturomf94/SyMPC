import torch

from .fixed_precision import FixedPrecisionTensor

from copy import deepcopy
import operator


class AdditiveSharingTensor:

    def __init__(self, secret=None, shape=None, shares=None, session=None):
        if not session:
            raise ValueError("Session should not be None")

        if len(session.session_ptr) == 0:
            raise ValueError("setup_mpc was not called on the session")

        self.session = session
        self.shape = None

        if secret is not None:
            secret, shape, is_remote_secret = AdditiveSharingTensor.sanity_checks(secret, shape, session)
            parties = session.parties
            self.shape = shape

            if is_remote_secret:
                # If the secret is remote we use PRZS (Pseudo-Random-Zero Shares) and the
                # party that holds the secret will add it to it's share
                self.shares = AdditiveSharingTensor.generate_przs(self.shape, self.session)
                for i, share in enumerate(self.shares):
                    if share.client == secret.client:
                        self.shares[i] = self.shares[i] + secret
            else:
                self.shares = []
                shares = AdditiveSharingTensor.generate_shares(secret, self.session)
                for share, party in zip(shares, self.session.parties):
                    self.shares.append(share.send(party))

        elif shares is not None:
            self.shares = shares


    @staticmethod
    def sanity_checks(secret, shape, session):
        is_remote_secret = False

        if "Pointer" in type(secret).__name__:
            is_remote_secret = True
            if shape is None:
                raise ValueError("Shape must be specified if secret is at another worker")
        else:
            if isinstance(secret, (int, float)):
                secret = torch.Tensor([secret])

            if isinstance(secret, torch.Tensor):
                secret = FixedPrecisionTensor(secret, session.config)

            shape = secret.shape

        return secret, shape, is_remote_secret


    @staticmethod
    def generate_przs(shape, session):
        shape = tuple(shape)

        shares = []
        for session_ptr, generators_ptr in zip(session.session_ptr, session.przs_generators):
            share = session_ptr.przs_generate_random_elem(shape, generators_ptr)
            shares.append(share)

        return shares

    @staticmethod
    def generate_shares(secret, session):
        parties = session.parties
        nr_parties = len(parties)

        shape = secret.shape
        min_value = session.config.min_value
        max_value = session.config.max_value

        random_shares = []
        for _ in range(nr_parties - 1):
            rand_long = torch.randint(min_value, max_value, shape).long()
            fpt_rand = FixedPrecisionTensor(data=rand_long,  config=session.config)
            random_shares.append(fpt_rand)

        shares = []
        for i in range(len(parties)):
            if i == 0:
                share = random_shares[i]
            elif i < nr_parties - 1:
                share = random_shares[i] - random_shares[i-1]
            else:
                share = secret - random_shares[i-1]

            shares.append(share)
        return shares

    def reconstruct(self, decode=True):
        plaintext = FixedPrecisionTensor(data=0, config=self.session.config)

        for share in self.shares:
            print(f"Request share from {share.client}")
            share.request(block=True)
            plaintext += share.get()

        if decode:
            plaintext = plaintext.decode()

        return plaintext

    def add(self, y):
        return self.__apply_op(y, "add")

    def sub(self, y):
        return self.__apply_op(y, "sub")

    def mul(self, y):
        return self.__apply_op(y, "mul")

    def div(self, y):
        return self.__apply_op(y, "div")

    def __apply_private_op(self, y, op_str):
        if y.session.uuid != self.session.uuid:
            raise ValueError(f"Need same session {self.session.uuid} and {y.session.uuid}")

        if op_str in {"mul"}:
            from ..protocol import spdz

            shares = spdz.mul_master(self, y, op_str)
            result = AdditiveSharingTensor(shares=shares, session=self.session)
        elif op_str in {"sub", "add"}:
            op = getattr(operator, op_str)
            result = AdditiveSharingTensor(session=self.session)
            result.shares = [
                    op(*share_tuple)
                    for share_tuple in zip(self.shares, y.shares)
            ]

        return result

    def __apply_public_op(self, y, op_str):
        op = getattr(operator, op_str)
        # Here are two sens: one for modulo and one for op
        # TODO: Make only one operation

        if op_str in {"mul"}:
            shares = [op(share, y) for share in self.shares]
        else:
            operands_shares = [y] + [0 for _ in range(len(self.shares)-1)]
            shares = [
                op(*tuple_shares)
                for tuple_shares in zip(self.shares, operands_shares)
            ]

        result = AdditiveSharingTensor(shares=shares, session=self.session)
        return result

    def __apply_op(self, y, op):
        is_private = isinstance(y, AdditiveSharingTensor)

        if is_private:
            result = self.__apply_private_op(y, op)
        else:
            result = self.__apply_public_op(y, op)

        return result


    def __str__(self):
        type_name = type(self).__name__
        out = f"[{type_name}]"

        for share in self.shares:
            out = f"{out}\n\t| {share.client} -> {share.__name__}"
        return out

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __mul__ = mul
    __rmul__ = mul
