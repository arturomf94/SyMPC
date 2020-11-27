from sympc.encoder import FixedPointEncoder
from sympc.tensor.utils import modulo
from sympc.session import Session
import operator


class ShareTensor:
    """
        This class represents only 1 share  (from n) that a party
        can generate when secretly sharing that a party holds
    """

    __slots__ = {"_tensor", "session", "fp_encoder"}

    def __init__(
            self,
            data=None,
            session=None,
            encoder_base=10,
            encoder_precision=4,
            ring_size=2**62,
        ):

        if session is None:
            self.session = Session(
                ring_size=ring_size,
                encoder_base=encoder_base,
                encoder_precision=encoder_precision,
            )
        else:
            self.session = session

        # TODO: It looks like the same logic as above
        self.fp_encoder = FixedPointEncoder(
            base=self.config.encoder_base,
            precision=self.config.encoder_precision,
        )

        self._tensor = None
        if data is not None:
            self._tensor = self.fp_encoder.encode(data)

    @staticmethod
    def sanity_checks(x, y):
        if not isinstance(y, ShareTensor):
            y = ShareTensor(data=y, config=x.config)

        x_prec = x.config.encoder_precision
        y_prec = y.config.encoder_precision

        x_base = x.config.encoder_base
        y_base = y.config.encoder_base

        if x_prec != y_prec and x_prec * y_prec != 0:
            # Check both value has a specified precision and they differ
            raise ValueError(f"The precisions do not match {x_prec} with {y_prec}")

        if x_base != y_base:
            raise ValueError(f"The bases do not match {x_base} with {y_base}")

        return y

    def apply_function(self, y, op_str):
        op = getattr(operator, op_str)
        value = op(self._tensor, y._tensor)
        res = ShareTensor(config=self.config)
        res._tensor = value
        return res

    def add(self, y):
        y = ShareTensor.sanity_checks(self, y)
        res = self.apply_function(y, "add")
        res._tensor = modulo(res._tensor, res.config)
        return res

    def sub(self, y):
        y = ShareTensor.sanity_checks(self, y)
        res = self.apply_function(y, "sub")
        res._tensor = modulo(res._tensor, res.config)
        return res

    def mul(self, y):
        y = ShareTensor.sanity_checks(self, y)
        res = self.apply_function(y, "mul")
        res._tensor = res._tensor // self.fp_encoder.scale
        res._tensor = modulo(res._tensor, res.config)
        return res

    def div(self, y):
        # TODO
        pass

    def __getattr__(self, attr_name):
        # Default to some tensor specific attributes like
        # size, shape, etc.
        import pdb; pdb.set_trace()
        tensor = self._tensor
        return getattr(tensor, attr_name)

    def __gt__(self, y):
        y = ShareTensor.sanity_checks(self, y)
        res = self._tensor < y._tensor
        return res

    def __lt__(self, y):
        y = ShareTensor.sanity_checks(self, y)
        res = self._tensor < y._tensor
        return res

    def __str__(self):
        type_name = type(self).__name__
        out = f"[{type_name}]"
        out = f"{out}\n\t| {self.fp_encoder}"
        out = f"{out}\n\t| Data: {self._tensor}"

        return out

    def __eq__(self, other):
        if not (self._tensor == other._tensor).all():
            return False

        if not (self.config == other.config):
            return False

        return True

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = sub
    __mul__ = mul
    __rmul__ = mul
    __div__ = div
