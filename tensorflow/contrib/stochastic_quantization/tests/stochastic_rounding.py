import numpy as np


class StochasticRoundingFormat(object):
    def __init__(self, bits_int, bits_frac):
        self._bits_int = bits_int
        self._bits_int_mask = 2**(self._bits_int)
        self._bits_frac = bits_frac
        self._bits_frac_mask = 2**(self._bits_frac)
        self._total_bits_mask = 2**(self._bits_int +
                                    self._bits_frac)
        self._frac_mask = 2**(self._bits_frac)
        self._max_frac = (self._frac_mask - 1) / self._frac_mask
        self._min_frac = 1.0 / self._frac_mask

        self._max_value = (2**self._bits_int-1) + self._max_frac
        self._min_value = -2**self._bits_int - self._max_frac

    def round(self, data_raw):
        data = np.array(data_raw, dtype=np.float)

        overflow_data = (data > self._max_value).astype(np.float)
        neg_overflow_data = (data < self._min_value).astype(np.float)

        non_clamped_data = ((data <= self._max_value).astype(np.float) *
                            (data >= self._min_value).astype(np.float))

        clamped_data = (data * non_clamped_data +
                        overflow_data * float(self._max_value) +
                        neg_overflow_data * self._min_value)
        scaled_repr = (clamped_data.astype(np.float) *
                       float(self._bits_frac_mask))
        scaled_floor = np.floor(scaled_repr)

        data_round = (np.random.rand(*data.shape) <
                      (scaled_repr - scaled_floor)).astype(np.float)

        return (scaled_floor + data_round).astype(np.int32)


class StochasticRounding(object):
    def __init__(self, number_format, shape, data=None):
        self._format = number_format
        if data is not None:
            self._data = number_format.round(data)
        else:
            self._data = np.zeros(shape, dtype=np.int32)

    @classmethod
    def from_numpy(cls, number_format, data):
        return cls(number_format, data.shape, data)

    @classmethod
    def new_format(cls, bits_int, bits_frac, shape, data=None):
        number_format = StochasticRoundingFormat(bits_int, bits_frac)
        return cls(number_format=number_format,
                   shape=shape,
                   data=data)

    def copy_format(self, shape=None, data=None):
        if shape is None:
            shape = self._data.shape

        return StochasticRounding(self._format,
                                  shape=shape, data=data)

    def to_float(self):
        return self._data.astype(np.float)/self._format._bits_frac_mask

    def op(self, operation, other):

        result = operation(self.to_float(), other.to_float())
        return self.copy_format(shape=result.shape, data=result)


def test_basic_rounding():
    print(">>>> Test 1...")
    number_format = StochasticRoundingFormat(bits_int=2, bits_frac=5)

    a_np = np.array([-8, -2, -1, 0, 1, 2, 3, 8], dtype=float)
    a = StochasticRounding.from_numpy(number_format=number_format,
                                      data=a_np)

    expected = np.array([-4-number_format._max_frac, -2, -1, 0,
                         1, 2, 3, 3+number_format._max_frac],
                        dtype=np.float)
    np.testing.assert_allclose(expected,
                               a.to_float(),
                               atol=number_format._min_frac,
                               rtol=0.0)


def test_stochastic_rounding():
    print(">>>> Test 2...")
    number_format = StochasticRoundingFormat(bits_int=5, bits_frac=5)

    a_np = np.array([-64, -2, -1, 0, 1, 2, 3, 64], dtype=np.float)
    a = StochasticRounding.from_numpy(number_format,
                                      data=a_np)
    b_np = np.array([16, 16, 16, 16, 16, 16, 16, 16], dtype=np.float)
    b = a.copy_format(data=b_np)

    op = np.divide

    a_np[0] = -32 - (2**5-1)/2**5
    a_np[7] = 31 + (2**5-1)/2**5
    expected_result = np.array(a_np/b_np, dtype=np.float)

    observed_result = a.op(op, b)

    np.testing.assert_allclose(expected_result,
                               observed_result.to_float(),
                               atol=number_format._min_frac,
                               rtol=0.0)


def test_rounding_ops(iteration):
    print(">>>> Test 3(%d)..." % iteration)
    
    a_np = np.random.rand(10000)
    b_np = np.random.rand(10000)
    number_format = StochasticRoundingFormat(bits_int=2, bits_frac=5)
    a = StochasticRounding.from_numpy(number_format, a_np)
    b = StochasticRounding.from_numpy(number_format, b_np)

    observed_result = a.op(np.multiply, b)

    max_error = 3 * number_format._min_frac + number_format._min_frac**2
    np.testing.assert_allclose(a_np,
                               a.to_float(), rtol=0.0,
                               atol=number_format._min_frac,
                               err_msg="test3: rounding a failed")
    np.testing.assert_allclose(b_np,
                               b.to_float(), rtol=0.0,
                               atol=number_format._min_frac,
                               err_msg="test3: rounding b failed")
    np.testing.assert_allclose(a_np*b_np,
                               observed_result.to_float(), rtol=0.0,
                               atol=max_error,
                               err_msg="test3: multiply test failed")

if __name__ == "__main__":
    test_basic_rounding()
    test_stochastic_rounding()
    for i in range(10000):
        test_rounding_ops(i)
