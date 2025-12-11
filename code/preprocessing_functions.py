def standard_scale(arr: np.ndarray) -> np.ndarray:
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    stds = np.where(stds == 0, 1, stds)  # Prevent division by zero for constant columns
    scaled = (arr - means) / stds
    return scaled

def random_padding(x,max_time_steps= 100):
    ##assuming seq shape is time_steps x 7 (imu only)
    x= np.array(x)
    if x.shape[0] < max_time_steps:
        r = torch.randint(0,max_time_steps - x.shape[0], size = (1,))
        final_x = np.vstack((torch.zeros(r,x.shape[1]),x,torch.zeros(max_time_steps-r-x.shape[0],x.shape[1])))
        assert final_x.shape[0] == max_time_steps, "Error: Shape issue in padding!!"
        # final_x = pd.DataFrame(final_x)
        # final_x.columns = x.columns
        return final_x
    else:
        return  x[:max_time_steps]

class SignalTransform:
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p
    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y

class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)
      
class GaussianNoise(SignalTransform):
    def __init__(
        self, always_apply: bool = False, 
        p: float = 0.5, max_noise_amplitude: float = 0.20, **kwargs
    ):
        super().__init__(always_apply, p)
        self.noise_amplitude = (0.0, max_noise_amplitude)

    def apply(self, x: np.ndarray, **params):
        noise_amplitude = np.random.uniform(*self.noise_amplitude)
        noise = np.random.randn(*x.shape)  # shape (L, N)
        augmented = (x + noise * noise_amplitude).astype(x.dtype)
        return augmented
      
class PinkNoiseSNR(SignalTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt((y ** 2).max(axis=0))  # shape: (N,)
        a_noise = a_signal / (10 ** (snr / 20))   # shape: (N,)
        pink_noise = np.stack([cn.powerlaw_psd_gaussian(1, len(y)) for _ in range(y.shape[1])], axis=1)
        a_pink = np.sqrt((pink_noise ** 2).max(axis=0))  # shape: (N,)
        pink_noise_normalized = pink_noise * (a_noise / a_pink)
        augmented = (y + pink_noise_normalized).astype(y.dtype)
        return augmented
      
class TimeStretch(SignalTransform):
    def __init__(self, max_rate=1.5, min_rate=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.always_apply = always_apply
        self.p = p

    def apply(self, x: np.ndarray):
        """
        Stretch a 1D or 2D array in time using linear interpolation.
        - x: np.ndarray of shape (L,) or (L, N)
        - rate: float, e.g., 1.2 for 20% longer, 0.8 for 20% shorter
        """
        rate = np.random.uniform(self.min_rate, self.max_rate)
        L = x.shape[0]
        L_new = int(L / rate)
        orig_idx = np.linspace(0, L - 1, num=L)
        new_idx = np.linspace(0, L - 1, num=L_new)

        if x.ndim == 1:
            return np.interp(new_idx, orig_idx, x)
        elif x.ndim == 2:
            return np.stack([
                np.interp(new_idx, orig_idx, x[:, i]) for i in range(x.shape[1])
            ], axis=1)
        else:
            raise ValueError("Only 1D or 2D arrays are supported.")
          
class TimeShift(SignalTransform):
    def __init__(self, always_apply=False, p=0.5, max_shift_pct=0.25, padding_mode="replace"):
        super().__init__(always_apply, p)
        
        assert 0 <= max_shift_pct <= 1.0, "`max_shift_pct` must be between 0 and 1"
        assert padding_mode in ["replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'"
        
        self.max_shift_pct = max_shift_pct
        self.padding_mode = padding_mode

    def apply(self, x: np.ndarray, **params):
        assert x.ndim == 2, "`x` must be a 2D array with shape (L, N)"
        
        L = x.shape[0]
        max_shift = int(L * self.max_shift_pct)
        shift = np.random.randint(-max_shift, max_shift + 1)

        # Roll along time axis (axis=0)
        augmented = np.roll(x, shift, axis=0)

        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift, :] = 0
            elif shift < 0:
                augmented[shift:, :] = 0
        return augmented
      
class ButterFilter(SignalTransform):
    def __init__(self, always_apply=False, p=0.5, cutoff_freq=20, sampling_rate=200, order=4):
        super().__init__(always_apply, p)
        
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order

    def apply(self, x: np.ndarray, **params):
        assert x.ndim == 2, "`x` must be a 2D array with shape (L, N)"
        return self.butter_lowpass_filter(x)

    def butter_lowpass_filter(self, data: np.ndarray):
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        filtered_data = lfilter(b, a, data, axis=0)  # filter each channel independently
        return filtered_data
