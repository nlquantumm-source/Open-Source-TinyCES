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
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Noise Types Reference Table</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f9f9f9;
            color: #333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            font-size: 14px;
            line-height: 1.5;
        }
        th {
            background-color: #2c3e50;
            color: white;
            padding: 14px 10px;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 12px 10px;
            border-bottom: 1px solid #ddd;
            vertical-align: top;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #eef5ff;
        }
        .noise-type {
            font-weight: bold;
            font-size: 15px;
            white-space: nowrap;
        }
        .mathjax-block {
            display: block;
            margin: 8px 0;
            text-align: center;
        }
    </style>
</head>
<body>

<h1 style="text-align:center; color:#2c3e50;">ECG Noise Types and Standard Models (2020–2025 Research)</h1>

<table>
    <thead>
        <tr>
            <th>Noise Type</th>
            <th>Real-World Cause</th>
            <th>Standard Mathematical Model (with LaTeX)</th>
            <th>Typical Parameters<br><small>(fs = 250–500 Hz, ECG amplitude normalized to std = 1 or peak ≈ 1–3 mV)</small></th>
            <th>Algorithmic Use-Case in Denoising Research<br><small>(2020–2025 SOTA)</small></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="noise-type">Additive White Gaussian Noise (AWGN)</td>
            <td>Thermal noise, quantization error, amplifier noise</td>
            <td>\[ n_{\text{AWGN}}(t) = \sigma_g \cdot w(t) \]<br>
                where \( w(t) \sim \mathcal{N}(0,1) \)<br><br>
                \[ \sigma_g = \sqrt{\frac{P_s}{10^{\text{SNR}/10}}} \]<br>
                \( P_s \) = signal power</td>
            <td>σ_g set for SNR ∈ [−6, 24] dB<br>
                Most common: 0, 6, 12, 18, 24 dB<br>
                σ_g ≈ 0.01–1.0 × std(ECG)</td>
            <td>Universal high-frequency noise benchmark. Used in every ECG denoising paper to test general robustness. Wavelets, EMD, deep models (Autoencoders, GANs, TSPulse) all show near-perfect removal at SNR ≥ 6 dB.</td>
        </tr>
        <tr>
            <td class="noise-type">Baseline Wander (BW)</td>
            <td>Respiration, body movement, electrode drift</td>
            <td>Real: NSTDB “bw” record<br>
                Synthetic (most widely adopted):<br>
                \[ n_{\text{BW}}(t) = A_{\text{bw}} \sum_{k=1}^{3} b_k \sin(2\pi k f_{\text{resp}} t + \phi_k) \]<br>
                or single sinusoid \( A_{\text{bw}} \sin(2\pi \cdot 0.25 t) \)</td>
            <td>f_resp ∈ [0.15, 0.4] Hz (breathing 9–24 breaths/min)<br>
                A_bw = 0.1–0.5 × std(ECG)<br>
                NSTDB bw scaled by 0.1–2.0<br>
                Typical single-component amplitude 0.2–0.3 mV peak-to-peak</td>
            <td>Simulates breathing artifact. Critical for accurate R-peak detection and QRS morphology. Adaptive filters, high-pass (0.5 Hz), EMD, and modern foundation models (TTM/TSPulse) excel. Failure here causes false arrhythmia detection.</td>
        </tr>
        <tr>
            <td class="noise-type">Powerline Interference (PLI)</td>
            <td>50/60 Hz electromagnetic coupling</td>
            <td>\[ n_{\text{PL}}(t) = A_{\text{pl}} \sum_{k=1}^{H} c_k \sin(2\pi k f_{\text{pl}} t + \phi_k) \]<br>
                H = 1–5 harmonics</td>
            <td>f_pl = 50 or 60 Hz<br>
                A_pl = 0.02–0.20 × std(ECG) (1–20 % of ECG amplitude, often 5–10 %)<br>
                Harmonics amplitude decay 1/k or 1/k²<br>
                NSTDB does not include PLI → always synthetic</td>
            <td>Easiest to remove (notch/IIR). Included in almost every stress test for completeness. Deep models learn it implicitly without explicit notch, preserving harmonic content of QRS complex.</td>
        </tr>
        <tr>
            <td class="noise-type">Muscle Artifact (MA/EMG)</td>
            <td>Skeletal muscle contraction, tremor, shivering</td>
            <td>Real: NSTDB “ma” record (gold standard)<br>
                Synthetic (bursty, recommended):<br>
                \[ n_{\text{MA}}(t) = \sigma_{\text{ma}} \cdot g(t) \cdot \|1 + m(t)\| \]<br>
                where g(t) ~ N(0,1) filtered HPF > 15 Hz,<br>
                m(t) = low-frequency envelope (bandpass 2–12 Hz)</td>
            <td>σ_ma = 0.05–0.25 × std(ECG)<br>
                Envelope modulation depth 0.5–3.0<br>
                Burst duration 0.2–1 s<br>
                NSTDB ma scaled 0.1–3.0 (very challenging at >1.0)</td>
            <td>Most difficult broadband non-stationary noise. Overlaps QRS band → linear filters distort morphology. GANs, diffusion models, and masked-reconstruction models (TSPulse, TimesFM) achieve best preservation of clinical features (2023–2025 papers).</td>
        </tr>
        <tr>
            <td class="noise-type">Electrode Motion Artifact (EM)</td>
            <td>Skin stretching, loose contact, cable movement</td>
            <td>Real: NSTDB “em” record (only realistic option)<br>
                Synthetic (rarely used): sporadic transients modeled as<br>
                \[ n_{\text{EM}}(t) = A_{\text{em}} \cdot \text{rect}(t-t_0) * h(t) \]<br>
                or large low-frequency swings + saturation</td>
            <td>NSTDB em scaled 0.1–2.0 (at scale = 1 already mimics PVCs/ST changes)<br>
                Transient amplitude up to 5× ECG peak, duration 0.5–3 s<br>
                Frequency content < 10 Hz with abrupt onsets</td>
            <td>Hardest artifact clinically — mimics pathological ST elevation, PVCs, ischemia. Classic filters fail completely. TSPulse’s dual time/frequency masking gives best distinction from true morphology (2024–2025 papers show >95 % clinical feature preservation).</td>
        </tr>
    </tbody>
</table>

</body>
</html>
