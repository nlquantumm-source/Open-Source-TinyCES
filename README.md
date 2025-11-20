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
      <td><img style="vertical-align: middle;" src="https://latex.codecogs.com/svg.latex?\color{White}\Large%20n_{\text{AWGN}}(t)%20=%20\sigma_g%20\cdot%20w(t),\quad%20w(t)\sim%20\mathcal{N}(0,1)" alt="AWGN model"></td>
    </tr>
    <tr>
      <td><strong>Baseline Wander (BW)</strong></td>
      <td>Respiration, body movement, electrode drift</td>
      <td><img style="vertical-align: middle;" src="https://latex.codecogs.com/svg.latex?\color{White}\Large%20n_{\text{BW}}(t)%20=%20A_{\text{bw}}\sin(2\pi%20f_{\text{resp}}t)%20\text{%20or%20}%20\sum_{k=1}^{3}b_k\sin(2\pi%20k%20f_{\text{resp}}t%20+%20\phi_k)" alt="Baseline Wander model"></td>
    </tr>
    <tr>
      <td><strong>Powerline Interference (PLI)</strong></td>
      <td>50/60 Hz electromagnetic coupling</td>
      <td><img style="vertical-align: middle;" src="https://latex.codecogs.com/svg.latex?\color{White}\Large%20n_{PL}(t)=A_{pl}\sum_{k=1}^{H}c_k\sin(2\pi%20k%20f_{pl}%20t%20+%20\phi_k)" alt="Powerline Interference model"></td>
    </tr>
    <tr>
      <td><strong>Muscle Artifact (MA/EMG)</strong></td>
      <td>Skeletal muscle contraction, tremor, shivering</td>
      <td><img style="vertical-align: middle;" src="https://latex.codecogs.com/svg.latex?\color{White}\Large%20n_{\text{MA}}(t)%20=%20\sigma_{\text{ma}}%20\cdot%20g(t)\cdot|1%20+%20m(t)|,%20%20\text{g(t): HPF%20>%2015Hz,%20m(t):%202-12%20Hz%20envelope}" alt="Muscle Artifact model"></td>
    </tr>
    <tr>
      <td><strong>Electrode Motion Artifact (EM)</strong></td>
      <td>Skin stretching, loose contact, cable movement</td>
      <td><img style="vertical-align: middle;" src="https://latex.codecogs.com/svg.latex?\color{White}\Large%20\text{MIT-BIH%20NSTDB%20``em''%20record%20(real%20recording,%20no%20reliable%20synthetic%20model)}" alt="Electrode Motion Artifact"></td>
    </tr>
  </tbody>
</table>
