# Open-Source TinyCES
* As technology emerges, every businesses want to implement AI into their product.<br />
* As AI is an attractive technology, it is too bulky to implement on their product (Hardware Devices).<br />
* In order to solve their problem, I'm sharing the smallest AI as an open-source so they can enhance their product.<br />
* This model is specifically for a signal enhancement purposes in medical field (ECG sensor).<br />
* This model is based on the 3 research papers (I'll put a reference down below in case anyone is curious about it).<br /><br />
  
## Software Use-case
* Use *Google Colab* when training AI model and generate API code for simple integration (Compatible for majority of the embedding engineers out there).<br />
* Once you run the Google Colab in your computer, the trained model is going to be automatically downloaded in your computer.<br /><br />

## Filter
* *Bandpass Filter*: Removes DC baseline wander (low-frequency noise) and high-frequency artifacts from ECG signal.<br />
* *Anti-aliasing Resampling*: Prevents aliasing during downsampling from 260 Hz to 250 Hz by implicit low-pass filtering in polyphase interpolation.<br />
* *Additive Gaussian Noise*: Augments data by simulating sensor noise, improving model robustness without altering signal structure.<br />
* *Dropout (Regularization)*: Acts as a stochastic "filter" during training to prevent overfitting by randomly dropping units (implicit noise injection).<br /><br />

## Algorithm & Mathematical Equations<br />
| Algorithm | Mathematical Equation | Purpose/Context in Script |
| :--- | :--- | :--- |
| **Polyphase Resampling** | L' = L × (f's / f_s)<br>Reconstruction via sinc interpolation:<br>x(t) = ∑ x[n] · sinc(t - nT) | Downsamples ECG signals from 360 Hz to 250 Hz while preserving annotation timings;<br>Used in MIT-BIH loading (Cell 2). |
| **Min-Max Normalization** | $x' = \frac{x-\min(x)}{\max(x)-\min(x)+\epsilon}$<br>where $\epsilon = 10^{-8}$ | Scales filtered ECG segments to [0,1] range per window;<br>Prepares data for int8 quantization and stable training (Cell 3). |
| **Binary Focal Loss** | $FL(p_t) = -\alpha_t(1 - p_t)^\gamma \log(p_t)$<br>where $p_t = y \cdot p + (1 - y)(1 - p)$<br>$\alpha_t = \alpha \text{ (if } y=1) \text{ or } 1 - \alpha$;<br>Defaults: $\alpha=0.25, \gamma=2.0$ | Handles class imbalance in arrhythmia detection (fewer abnormal beats);<br>Used as training loss for the CNN (Cell 4). |
| **1D Convolution (in ReTinyCES CNN)** | $(x * w)[n] = \sum_{m} x[n - m] \cdot w[m] + b$<br>Followed by ReLU:<br>$f(z) = \max(0, z)$ | Extracts temporal features from ECG windows;<br>Layers: 16 filters (kernel=8), 32 filters (kernel=5) (Cell 4). |
| **Sigmoid Activation (Output)** | $\sigma(z) = \frac{1}{1+e^{-z}}$ | Produces binary probability (normal vs. abnormal) at the final dense layer (Cell 4). |
| **Butterworth IIR Filter Design** | Transfer Function (low-pass prototype):<br>$H(s) = \frac{\omega_c^2}{s^2+\sqrt{2}\omega_c s+\omega_c^2}$<br>Cascaded for bandpass; Digital:<br>$y[n] = \sum_k b_k x[n - k] - \sum_k a_k y[n - k]$ | Designs coefficients for bandpass filtering (0.5–40 Hz);<br>Normalized frequencies: $f_{norm} = f / (f_s / 2)$ (Cell 3). |
