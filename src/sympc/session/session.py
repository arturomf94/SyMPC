from uuid import uuid1

from ..config import Config
from copy import deepcopy
import random


# TODO: Remove this from here
import torch

class Session:
    def __init__(self, config=None, parties=None, ttp=None, uuid=None):
        self.uuid = uuid1() if uuid is None else uuid

        # Each worker will have the rank as the index in the list
        self.parties = parties

        # Some protocols require a trusted third party
        # Ex: SPDZ
        self.trusted_third_party = ttp
        self.crypto_store = {}
        self.protocol = None
        self.config = config if config else Config()

        self.przs_generators = None

        # Those will be populated in the setup_mpc
        self.rank = None
        self.session_ptr = []

    def get_copy(self):
        session_copy = Session()
        session_copy.uuid = deepcopy(self.uuid)
        session_copy.parties = [party for party in self.parties]
        session_copy.trusted_third_party = self.trusted_third_party
        session_copy.crypto_store = {}
        session_copy.protocol = self.protocol
        session_copy.config = deepcopy(self.config)
        session_copy.przs_generators = self.przs_generators
        session_copy.rank = self.rank
        session_copy.session_ptr = [s_ptr for s_ptr in self.session_ptr]

        return session_copy

    def przs_generate_random_elem(self, shape, generators):
        from ..tensor import FixedPrecisionTensor

        gen0, gen1 = generators
        min_value = self.config.min_value
        max_value = self.config.max_value

        current_share = torch.randint(min_value, max_value, size=shape, generator=gen0)
        next_share = torch.randint(min_value, max_value, size=shape, generator=gen1)

        share = FixedPrecisionTensor(data=current_share - next_share, config=self.config)

        return share

    @staticmethod
    def setup_mpc(session):
        for rank, party in enumerate(session.parties):
            # Assign a new rank before sending it to another party
            session.rank = rank
            session.session_ptr.append(session.send(party))

        Session.setup_przs(session)


    @staticmethod
    def setup_przs(session):
        nr_parties = len(session.parties)

        # Create the remote lists where we add the generators
        session.przs_generators = [party.python.List([None, None]) for party in session.parties]

        for rank in range(nr_parties):

            # TODO: Need to change this with something secure
            seed = random.randint(-2**32, 2**32)
            next_rank = (rank + 1) % nr_parties

            gen_current = session.parties[rank].torch.Generator()
            gen_current.manual_seed(seed)

            gen_next = session.parties[next_rank].torch.Generator()
            gen_next.manual_seed(seed)

            session.przs_generators[rank][1] = gen_current
            session.przs_generators[next_rank][0] = gen_next
