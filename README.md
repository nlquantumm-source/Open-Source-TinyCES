# Edge-AI-for-ECG-Sensor
* Edge AI use-case for sensor Synthetic Noise addition for &amp; high accuracy purposes.<br />
* As Sensors are being used in various industries, it's being used in various situations.<br />
* In various situations, sensors are not capable to give you an accurate signal as it's disrupted.<br />
* In order to solve this problem, they needs to embedded with small size edge-AI to give you the reasonable signal.<br />
* In this repository, it's being used for medical field (ECG Sensor).<br />
* ECG sensor is being used for detects and records the electrical activity of the heart, used in majority of the ER (Emergency) cases.<br />
* As ECG sensor is being used in 79% of ER cases (122 million patients per year), it's being used in various situations and perform badly in 3% ~ 5% (3.7 million to 6.1 million patients per year).<br />
* In order to help out 3.7 to 6.1 million patients per year from suffering through poor signal performance of ECG sensor, I brought a open-source soluation that can be easily embedded with majority of the ECG sensor.<br />
* ECG sensor performs at low accuracy especially when patients are at respiration state (The process by which organisms produce energy from food), physical movement, and electrode contact issues (wrong placement which often happens).<br />
* IBM Granite models brings a solution here. They are great when placing edge deployment with Synthetic noise application and embedding with small hardware products. For ECG Sensor deployment, we decided to use IBM Granite TSPulse (granite-timeseries-tspulse-r1).<br />
* TSPulse is one of the best AI model for this use-case because it is pretrained with *dual-space masked recontruction* (time + frequency domains), making it naturally strong at imputation and Synthetic noise addition especially on physiological signals.<br />
* Use the *hybrid-dualhead-512-p8-r1* variant for imputation/denoising. It excels at zero-shot reconstruction and works perfectly with fine-tuning.<br /><br />
  
# Software Use-case
* Use *Google Colab* when training AI model (Compatibility for majority of the embedding engineers out there).<br />
* Real-time Arduino integration: 512-1024 samples and send data via Serial/WiFi.<br /><br />

# Synthetic Noise Application
* *Gaussian white noise* (for powerline/thermal interference).<br />
* *Baseline Wander* (low-frequency drift via sine square waves).<br />
* *EMG/muscle artifacts* (high-frequency modulated Gaussian) to train the TSPulse model on noisy-clean ECG pairs.<br /><br /><br />

# Algorithm & Mathematical Equation Use-case
* <table>
  <thead>
    <tr>
      <th>Noise Type</th>
      <th>Real-World Cause</th>
      <th>Standard Mathematical Model</th>
      <th>Typical Parameters<br><small>(fs = 250–500 Hz, ECG std ≈ 1)</small></th>
      <th>Use-Case in 2020–2025 SOTA Denoising</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Additive White Gaussian Noise (AWGN)</strong></td>
      <td>Thermal noise, quantization, amplifier noise</td>
      <td>$$n_{\text{AWGN}}(t) = \sigma_g \cdot w(t),\quad w(t)\sim\mathcal{N}(0,1)$$</td>
      <td>SNR ∈ [−6, 24] dB<br>σ_g ≈ 0.01–1.0 × std(ECG)</td>
      <td>Universal benchmark. Near-perfect removal at SNR ≥ 6 dB with any modern method</td>
    </tr>
    <tr>
      <td><strong>Baseline Wander (BW)</strong></td>
      <td>Respiration, movement, electrode drift</td>
      <td>$$n_{\text{BW}}(t) = A_{\text{bw}} \sin(2\pi \cdot 0.25 t)$$</td>
      <td>A_bw = 0.1–0.5 × std(ECG)<br>f ≈ 0.15–0.4 Hz</td>
      <td>Critical for R-peak detection. Adaptive/EMD/deep models excel</td>
    </tr>
    <tr>
      <td><strong>Powerline Interference (PLI)</strong></td>
      <td>50/60 Hz electromagnetic coupling</td>
      <td>$$n_{\text{PL}}(t) = A_{\text{pl}} \sum_{k=1^H c_k \sin(2\pi k f_{\text{pl}} t + \phi_k)$$</td>
      <td>A_pl = 1–20 % of ECG amplitude<br>f_pl = 50 or 60 Hz</td>
      <td>Easiest (notch filter). Modern nets remove it implicitly</td>
    </tr>
    <tr>
      <td><strong>Muscle Artifact (MA/EMG)</strong></td>
      <td>Muscle contraction, tremor</td>
      <td>NSTDB “ma” or bursty high-pass filtered Gaussian with envelope</td>
      <td>σ_ma = 0.05–0.25 × std(ECG)<br>NSTDB ma scaled up to ×3</td>
      <td>Hardest broadband noise. 2023–2025: diffusion + masked models (TSPulse) win</td>
    </tr>
    <tr>
      <td><strong>Electrode Motion (EM)</strong></td>
      <td>Skin stretch, loose electrodes</td>
      <td>NSTDB “em” record (only realistic source)</td>
      <td>Scaled 0.1–2.0<br>Transients up to 5× ECG amplitude</td>
      <td>Clinically most dangerous (mimics ischemia/PVC). Only latest foundation models (TSPulse 2024–2025) preserve >95 % morphology</td>
    </tr>
  </tbody>
</table>
