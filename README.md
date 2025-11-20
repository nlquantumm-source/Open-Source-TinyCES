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

# Algorithm & Mathematical Equation Use-case<br />
<table>
  <thead>
    <tr>
      <th>Noise Type</th>
      <th>Real-World Cause</th>
      <th>Standard Mathematical Model</th>
      <th>Typical Parameters<br>(fs = 250–500 Hz, ECG amplitude normalized to std ≈ 1)</th>
      <th>Algorithmic Use-Case in Denoising Research (2020–2025 SOTA)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Additive White Gaussian Noise (AWGN)</strong></td>
      <td>Thermal noise, quantization error, amplifier noise</td>
      <td><code>n<sub>AWGN</sub>(t) = σ<sub>g</sub> · w(t)</code><br>where <code>w(t) ~ N(0,1)</code><br>σ<sub>g</sub> = √(P<sub>s</sub> / 10<sup>(SNR/10)</sup>)</td>
      <td>SNR ∈ [−6, 24] dB<br>Common: 0, 6, 12, 18, 24 dB<br>σ<sub>g</sub> ≈ 0.01–1.0 × std(ECG)</td>
      <td>Universal high-frequency benchmark. Used in every paper. Wavelets, EMD, Autoencoders, GANs, and TSPulse remove it almost perfectly at SNR ≥ 6 dB.</td>
    </tr>
    <tr>
      <td><strong>Baseline Wander (BW)</strong></td>
      <td>Respiration, body movement, electrode drift</td>
      <td>Real: MIT-BIH NSTDB “bw” record<br>Synthetic (most common):<br><code>n<sub>BW</sub>(t) = A<sub>bw</sub> sin(233π·0.25·t)</code><br>or multi-harmonic respiration model</td>
      <td>f<sub>resp</sub> ∈ [0.15, 0.4] Hz<br>A<sub>bw</sub> = 0.1–0.5 × std(ECG)<br>NSTDB bw scaled 0.1–2.0</td>
      <td>Simulates breathing artifact. Critical for accurate R-peak detection and QRS morphology. Adaptive filters, high-pass 0.5 Hz, EMD, TTM/TSPulse excel here.</td>
    </tr>
    <tr>
      <td><strong>Powerline Interference (PLI)</strong></td>
      <td>50/60 Hz electromagnetic coupling</td>
      <td><code>n<sub>PL</sub>(t) = A<sub>pl</sub> Σ c<sub>k</sub> sin(2π k f<sub>pl</sub> t + φ<sub>k</sub>)</code><br>H = 1–5 harmonics</td>
      <td>f<sub>pl</sub> = 50 or 60 Hz<br>A<sub>pl</sub> = 0.02–0.20 × std(ECG) (1–20 %)<br>Harmonics decay 1/k or 1/k²</td>
      <td>Easiest to remove (notch/IIR). Included in every stress test. Deep models learn it implicitly without distorting QRS harmonics.</td>
    </tr>
    <tr>
      <td><strong>Muscle Artifact (MA/EMG)</strong></td>
      <td>Skeletal muscle contraction, tremor, shivering</td>
      <td>Real: NSTDB “ma” record (gold standard)<br>Synthetic (bursty):<br><code>n<sub>MA</sub>(t) = σ<sub>ma</sub> · g(t) · |1 + m(t)|</code><br>g(t): HPF > 15 Hz Gaussian<br>m(t): 2–12 Hz envelope</td>
      <td>σ<sub>ma</sub> = 0.05–0.25 × std(ECG)<br>Envelope depth 0.5–3.0<br>NSTDB ma scaled 0.1–3.0</td>
      <td>Hardest broadband non-stationary noise. Overlaps QRS band. GANs, diffusion, and masked-reconstruction models (TSPulse) achieve best clinical feature preservation.</td>
    </tr>
    <tr>
      <td><strong>Electrode Motion Artifact (EM)</strong></td>
      <td>Skin stretching, loose contact, cable movement</td>
      <td>Real: NSTDB “em” record (only realistic source)<br>Synthetic rare — large low-frequency transients</td>
      <td>NSTDB em scaled 0.1–2.0<br>Transient amplitude up to 5× ECG peak<br>Duration 0.5–3 s, content < 10 Hz</td>
      <td>Clinically most dangerous — mimics PVCs, ST elevation, ischemia. Classic filters fail. TSPulse dual time/frequency masking gives >95 % preservation of true morphology (2024–2025 SOTA).</td>
    </tr>
  </tbody>
</table>
