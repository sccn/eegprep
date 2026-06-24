import math
import numpy as np
import warnings

def round_mat(x, decimals=0):
    """MATLAB-style rounding function."""
    if isinstance(x, (float, int)):
        if math.isnan(x) or math.isinf(x):
            return x
        xp = math
    else:
        xp = np
        x = np.asarray(x)

    if decimals == 0:
        return xp.copysign(xp.floor(abs(x) + 0.5), x)

    if decimals > 0:
        factor = 10.0**decimals
        y = xp.copysign(xp.floor(abs(x) * factor + 0.5), x)
        return y / factor

    factor = 10.0 ** (-decimals)
    y = xp.copysign(xp.floor(abs(x) / factor + 0.5), x)
    return y * factor

def parity_matmul(left, right):
    """Matrix multiplication wrapper that enforces parity-safe rounding."""
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        res = left @ right
    return round_mat(res, 15)

def parity_weight_update(weights, update):
    """Weight update wrapper that enforces parity-safe rounding."""
    res = weights + update
    return round_mat(res, 15)

class MatlabRNG:
    """Ziggurat algorithm for normal distribution generation identical to MATLAB's randn."""
    def __init__(self, seed=5489):
        self.bg = np.random.MT19937()
        self.bg.state = np.random.RandomState(seed).get_state()
        
        self.ZIGGURAT_NOR_R = 3.6541528853610088
        self.ZIGGURAT_NOR_INV_R = 0.27366123732975828
        self.NOR_SECTION_AREA = 0.00492867323399
        self.NMANTISSA = 9007199254740992.0 # 2^53
        
        self.wi = np.zeros(256, dtype=np.float64)
        self.fi = np.zeros(256, dtype=np.float64)
        self.ki = np.zeros(256, dtype=np.int64)
        self._create_ziggurat_tables()

    def _create_ziggurat_tables(self):
        x1 = self.ZIGGURAT_NOR_R
        self.wi[255] = x1 / self.NMANTISSA
        self.fi[255] = math.exp(-0.5 * x1 * x1)
        
        self.ki[0] = int(x1 * self.fi[255] / self.NOR_SECTION_AREA * self.NMANTISSA)
        self.wi[0] = self.NOR_SECTION_AREA / self.fi[255] / self.NMANTISSA
        self.fi[0] = 1.0
        
        for i in range(254, 0, -1):
            x = math.sqrt(-2.0 * math.log(self.NOR_SECTION_AREA / x1 + self.fi[i+1]))
            self.ki[i+1] = int(x / x1 * self.NMANTISSA)
            self.wi[i] = x / self.NMANTISSA
            self.fi[i] = math.exp(-0.5 * x * x)
            x1 = x
            
        self.ki[1] = 0

    def randu53(self):
        while True:
            raw = self.bg.random_raw(2)
            a = int(raw[0]) >> 5
            b = int(raw[1]) >> 6
            if a != 0 or b != 0:
                break
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

    def rand_normal(self):
        while True:
            raw = self.bg.random_raw(2)
            lo = int(raw[0])
            hi = int(raw[1]) & 0x3FFFFF
            r = (hi << 32) | lo
            
            rabs = r >> 1
            idx = rabs & 0xFF
            sign = -1 if (r & 1) else 1
            x = sign * rabs * self.wi[idx]
            
            if rabs < self.ki[idx]:
                return x
            elif idx == 0:
                while True:
                    u1 = self.randu53()
                    u2 = self.randu53()
                    xx = -self.ZIGGURAT_NOR_INV_R * math.log(u1)
                    yy = -math.log(u2)
                    if yy + yy > xx * xx:
                        break
                return -self.ZIGGURAT_NOR_R - xx if (rabs & 0x100) else self.ZIGGURAT_NOR_R + xx
            elif (self.fi[idx-1] - self.fi[idx]) * self.randu53() + self.fi[idx] < math.exp(-0.5 * x * x):
                return x

    def randn(self, *shape):
        n = math.prod(shape) if shape else 1
        res = np.zeros(n, dtype=np.float64)
        for i in range(n):
            res[i] = self.rand_normal()
        
        if shape:
            return res.reshape(shape, order='F')
        return res[0]

    def rand(self, *shape):
        n = math.prod(shape) if shape else 1
        res = np.zeros(n, dtype=np.float64)
        for i in range(n):
            res[i] = self.randu53()
        if shape:
            return res.reshape(shape, order='F')
        return res[0]

    def randint(self, low, high, size=None):
        n = size if isinstance(size, int) else (math.prod(size) if size else 1)
        res = np.zeros(n, dtype=np.int64)
        diff = high - low
        for i in range(n):
            res[i] = low + int(math.floor(self.randu53() * diff))
        if size and not isinstance(size, int):
            return res.reshape(size, order='F')
        elif size:
            return res
        return res[0]

