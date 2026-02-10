# recon_image-for-ultrasound
# 📷 Photoacoustic Reconstruction App (Kivy + Python)

This application performs **2D k-space linear FFT reconstruction** of photoacoustic (PA) RF data and displays:

- ✅ US ENV (averaged)
- ✅ PA RF (averaged)
- ✅ PA Reconstruction (k-space method)

The app is built using:
- Python
- Kivy (GUI)
- NumPy
- Custom reconstruction core (`recon_core.py`)

---

## 📂 Project Structure

project_folder/
│
├── main.py # Kivy GUI application
├── recon_core.py # Core reconstruction logic
├── requirements.txt # Python dependencies
└── README.md


---

## ⚙️ Features

- File picker for:
  - `_env.raw`
  - `_rf.raw`
- Adjustable frame averaging (`Navg`)
- Separate image dialog for:
  - US ENV
  - PA RF averaged
  - PA Reconstruction
- Per-image:
  - Depth control
  - Contrast control (percentile-based windowing)
- Works on Windows (desktop)
- Can be packaged into Android APK (via Buildozer)

---

## 🧠 Reconstruction Algorithm

Uses a **k-space linear reconstruction** method based on:

- FFT in time and lateral dimension
- Dispersion relation mapping
- Interpolation in k-space
- Inverse FFT back to spatial domain
- Optional positivity enforcement

Original method based on:
> Bradley Treeby & Ben Cox (k-Wave Toolbox)

---

## 🖥️ Running on Windows (Desktop)

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Application
python main.py
📱 Build Android APK (Recommended via WSL2)
Install Buildozer in Ubuntu (WSL2)
sudo apt update
sudo apt install -y python3 python3-pip git zip openjdk-17-jdk
pip install buildozer cython
Initialize
buildozer init
Edit buildozer.spec:

requirements = python3,kivy,numpy
android.api = 33
android.minapi = 24
Build APK
buildozer -v android debug
APK will appear in:

bin/
📦 requirements.txt
kivy
numpy
📁 Input File Format
The app expects:

Clarius .raw files

_env.raw for ultrasound envelope data

_rf.raw for PA RF data

Both must be selected before reconstruction.

🔧 Troubleshooting
❌ NumPy ptp error
If using NumPy 2.0+, use:

np.ptp(img)
instead of:

img.ptp()
❌ RuntimeWarning: invalid value in divide
Safe to ignore — handled by:

np.maximum(..., 0)
📜 License
Educational / Research Use

Reconstruction method based on:
k-Wave Toolbox (LGPL)

👨‍💻 Author
Ritik Tomar
