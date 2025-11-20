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
* Use *Google Colab* when training AI model (Compatibility for majority of the embedding engineers out there).<br />
* Real-time Arduino integration: 512-1024 samples and send data via Serial/WiFi.<br /><br />

## Synthetic Noise Application
* *Gaussian white noise* (for powerline/thermal interference).<br />
* *Baseline Wander* (low-frequency drift via sine square waves).<br />
* *EMG/muscle artifacts* (high-frequency modulated Gaussian) to train the TSPulse model on noisy-clean ECG pairs.<br /><br /><br />

## Algorithm & Mathematical Equation Use-case<br />
<table>
  <thead>
    <tr>
      <th>Noise Type</th>
      <th>Real-World Cause</th>
      <th>Standard Mathematical Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Additive White Gaussian Noise (AWGN)</strong></td>
      <td>Thermal noise, quantization error, amplifier noise</td>
      <td>$$ n_{\text{AWGN}}(t) = \sigma_g \cdot w(t),\quad w(t)\sim\mathcal{N}(0,1) $$<br>$$ \sigma_g \text{ set via target SNR: } \sigma_g = \sqrt{\frac{P_s}{10^{\text{SNR}/10}}} $$</td>
    </tr>
    <tr>
      <td><strong>Baseline Wander (BW)</strong></td>
      <td>Respiration, body movement, electrode drift</td>
      <td>$$ n_{\text{BW}}(t) = A_{\text{bw}} \sin(2\pi f_{\text{resp}} t) $$<br>$$ \text{or multi-harmonic: } \sum_{k=1}^{3} b_k \sin(2\pi k f_{\text{resp}} t + \phi_k) $$<br>$$ f_{\text{resp}} \in [0.15, 0.4]\text{ Hz} $$</td>
    </tr>
    <tr>
      <td><strong>Powerline Interference (PLI)</strong></td>
      <td>50/60 Hz electromagnetic coupling</td>
      <td>$$ n_{\text{PL}}(t) = A_{\text{pl}} \sum_{k=1}^{H} c_k \sin(2\pi k f_{\text{pl}} t + \phi_k) $$<br>$$ H = 1-5 \text{ harmonics} $$</td>
    </tr>
    <tr>
      <td><strong>Muscle Artifact (MA/EMG)</strong></td>
      <td>Skeletal muscle contraction, tremor, shivering</td>
      <td>$$ \text{Real (gold standard): MIT-BIH NSTDB ``ma'' record} $$<br>$$ \text{Synthetic bursty model (most realistic approximation):} $$<br>$$ n_{\text{MA}}(t) = \sigma_{\text{ma}} \cdot g(t) \cdot |1 + m(t)| $$<br>$$ g(t):\ \text{HPF}>15\,\text{Hz}\ \mathcal{N}(0,1),\ m(t):\ 2-12\,\text{Hz envelope} $$</td>
    </tr>
    <tr>
      <td><strong>Electrode Motion Artifact (EM)</strong></td>
      <td>Skin stretching, loose contact, cable movement</td>
      <td>$$ \text{Real (gold standard): MIT-BIH NSTDB ``em'' record (most reliable)} $$<br>$$ \text{Synthetic approximation (biphasic exponential transients, used in some studies):} $$<br>$$ n_{\text{EM}}(t) = \sum_{i} A_i \left( e^{-(t-t_i)/\tau_1} - e^{-(t-t_i)/\tau_2} \right) $$<br>$$ \tau_1 \ll \tau_2\ (sharp rise + slow decay),\ \text{random } A_i, t_i, \tau $$</td>
    </tr>
  </tbody>
</table>
