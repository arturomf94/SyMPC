from ..encoder import FixedPointEncoder
from .utils import modulo
from ..config import Config
import operator


class FixedPrecisionTensor:

    def __init__(
            self,
            data=None,
            config=None,
            encoder_base=10,
            encoder_precision=4,
            ring_size=2**62,
        ):

        if config is None:
            self.config = Config(
                encoder_base=encoder_base,
                encoder_precision=encoder_precision,
                ring_size=ring_size
            )
        else:
            self.config = config

        # TODO: It looks like the same logic as above
        self.fp_encoder = FixedPointEncoder(
            base=self.config.encoder_base,
            precision=self.config.encoder_precision,
            ring_size=self.config.ring_size
        )

        self._tensor = None
        if data is not None:
            self._tensor = self.fp_encoder.encode(data)

    def decode(self):
        return self.fp_encoder.decode(self._tensor)

    @staticmethod
    def sanity_checks(x, y):
        if not isinstance(y, FixedPrecisionTensor):
            y = FixedPrecisionTensor(data=y, config=x.config)

        x_prec = x.config.encoder_precision
        y_prec = y.config.encoder_precision

        x_base = x.config.encoder_base
        y_base = y.config.encoder_base

        if x_prec != y_prec and x_prec * y_prec != 0:
            raise ValueError(f"The precisions do not match {x_prec} with {y_prec}")

        if x_base != y_base:
            raise ValueError(f"The bases do not match {x_base} with {y_base}")

        return y

    def apply_function(self, y, op_str):
        op = getattr(operator, op_str)
        value = op(self._tensor, y._tensor)
        res = FixedPrecisionTensor(config=self.config)
        res._tensor = value
        return res

    def add(self, y):
        y = FixedPrecisionTensor.sanity_checks(self, y)
        res = self.apply_function(y, "add")
        res._tensor = modulo(res._tensor, res.config)
        return res

    def sub(self, y):
        y = FixedPrecisionTensor.sanity_checks(self, y)
        res = self.apply_function(y, "sub")
        res._tensor = modulo(res._tensor, res.config)
        return res

    def mul(self, y):
        y = FixedPrecisionTensor.sanity_checks(self, y)
        res = self.apply_function(y, "mul")

        if self.fp_encoder.precision and y.fp_encoder.precision:
            res._tensor = res._tensor // self.fp_encoder.scale
            print("here", self.fp_encoder.scale, self, y)

        fp_encoder = FixedPointEncoder(
            base=res.fp_encoder.base,
            precision=max(self.fp_encoder.precision, y.fp_encoder.precision)
        )

        res._tensor = modulo(res._tensor, res.config)
        return res

    def div(self, y):
        # TODO
        pass

    def __getattr__(self, attr_name):
        # Default to some tensor specific attributes like
        # size, shape, etc.
        tensor = self._tensor
        return getattr(tensor, attr_name)

    def __gt__(self, y):
        y = FixedPrecisionTensor.sanity_checks(self, y)
        res = self._tensor < y._tensor
        return res

    def __lt__(self, y):
        y = FixedPrecisionTensor.sanity_checks(self, y)
        res = self._tensor < y._tensor
        return res

    def __str__(self):
        type_name = type(self).__name__
        out = f"[{type_name}]"
        out = f"{out}\n\t{self.fp_encoder}"
        out = f"{out}\n\tData: {self._tensor}"

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
