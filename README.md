# Edge-AI-for-ECG-Sensor
* Edge AI use-case for sensor Synthetic Noise addition for &amp; high accuracy purposes.<br />
* As Sensors are being used in various industries, it's being used in various situations.<br />
* In various situations, sensors are not capable to give you an accurate signal as it's disrupted.<br />
* In order to solve this problem, they needs to embedded with small size edge-AI to give you the reasonable signal.<br />
* In this repository, we're going to focus on solving problems at medical field (ECG Sensor).<br />
* ECG sensor is being used for detecting and recording the electrical activity of the heart, used in majority of the ER (Emergency) cases.<br />
* As ECG sensor is being used in 79% of ER cases (122 million patients per year), it's being used in various situations and perform badly in 3% ~ 5% (3.7 million to 6.1 million patients per year) for respiration state (The process by which organisms produce energy from food), physical movement, and electrode contact issues (wrong placement which often happens).<br />
* IBM Granite models brings a solution here. They are great when placing edge deployment with Synthetic noise application and embedding with small hardware products.<br />
* For ECG Sensor deployment, we decided to use IBM Granite TSPulse (granite-timeseries-tspulse-r1).<br />
* TSPulse is one of the best AI model for this use-case because it is pretrained with *dual-space masked recontruction* (time + frequency domains), making it naturally strong at imputation and Synthetic noise addition especially on physiological signals.<br />
* Use the *hybrid-dualhead-512-p8-r1* variant for imputation/denoising. It excels at zero-shot reconstruction and works perfectly with fine-tuning.<br /><br />
  
## Software Use-case
* Use *Google Colab* when training AI model and generate API code for simple integration(Compatibility for majority of the embedding engineers out there).<br />
* Real-time Arduino integration: 512-1024 samples and send data via Serial/WiFi.<br /><br />

## Synthetic Noise Application
* *Gaussian white noise* (for powerline/thermal interference).<br />
* *Baseline Wander* (low-frequency drift via sine square waves).<br />
* *EMG/muscle artifacts* (high-frequency modulated Gaussian) to train the TSPulse model on noisy-clean ECG pairs.<br /><br /><br />

## Algorithm & Mathematical Equation Use-case<br />
| Noise Type                          | Real-World Cause                              | Standard Mathematical Model                                                                                                           |
|-------------------------------------|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| **Additive White Gaussian Noise (AWGN)** | Thermal noise, quantization error, amplifier noise | $n_{\text{AWGN}}(t) = \sigma_g \cdot w(t),\quad w(t)\sim\mathcal{N}(0,1)$ $\sigma_g = \sqrt{\frac{P_s}{10^{\text{SNR[dB]}/10}}}$ |
| **Baseline Wander (BW)**            | Respiration, body movement, electrode drift  | $n_{\text{BW}}(t) = \sum_{k=1}^{3} b_k \sin(2\pi k f_{\text{resp}} t + \phi_k)$ $f_{\text{resp}} \in [0.15, 0.4]\ \text{Hz}$ |
| **Powerline Interference (PLI)**    | 50/60 Hz electromagnetic coupling             | $n_{\text{PL}}(t) = A_{\text{pl}} \sum_{k=1}^{H} c_k \sin(2\pi k f_{\text{pl}} t + \phi_k)$ $(H = 1-5\ \text{harmonics})$ |
| **Muscle Artifact (MA/EMG)**        | Skeletal muscle contraction, tremor, shivering | Gold standard: MIT-BIH NSTDB "ma" record (real recorded EMG)<br><br>**Synthetic model (most realistic & widely used):**<br>$n_{\text{MA}}(t) = \sigma_{\text{ma}}(t) \cdot g(t) $<br>$ g(t):\ \text{band-pass filtered Gaussian (20-500 Hz)}$<br>$ \sigma_{\text{ma}}(t):\ \text{slow envelope (0.1-10 Hz)}$ |
| **Electrode Motion Artifact (EM)**      | Skin stretching, loose contact, cable movement       | Gold standard: MIT-BIH NSTDB "em" record (real recording)<br><br>**Synthetic model (biphasic exponential pulse train):**<br>$ n_{\text{EM}}(t) = \sum_i A_i \left( e^{-(t-t_i)/\tau_1} - e^{-(t-t_i)/\tau_2} \right) $<br>$ \tau_1 \gg \tau_2\ \text{(sharp rise + slow decay)}$ |
