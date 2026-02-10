# recon_core.py
import numpy as np
from pathlib import Path

# =========================
# rdataread (unchanged)
# =========================
def rdataread(filename, frames):
    with open(filename, "rb") as f:
        hinfo = np.fromfile(f, dtype=np.int32, count=5)
        if hinfo.size < 5:
            raise ValueError("Invalid RAW header")

        header = {
            "id": int(hinfo[0]),
            "frames": int(hinfo[1]),
            "lines": int(hinfo[2]),
            "samples": int(hinfo[3]),
            "sampleSize": int(hinfo[4]),
        }

        frames = min(frames, header["frames"])
        hid = header["id"]
        lines = header["lines"]
        samples = header["samples"]
        ts = np.zeros(frames, dtype=np.int64)

        if hid == 1:  # ENV
            data = np.zeros((frames, samples, lines), dtype=np.uint8)
            for fidx in range(frames):
                ts[fidx] = np.fromfile(f, dtype=np.int64, count=1)[0]

                buf = np.fromfile(f, dtype=np.uint8, count= samples * lines)
                data[fidx] = buf.reshape(samples, lines, order="F")

        elif hid == 2:  # RF
            data = np.zeros((frames, samples, lines), dtype=np.int16)
            for fidx in range(frames):
                ts[fidx] = np.fromfile(f, dtype=np.int64, count=1)[0]

                buf = np.fromfile(f, dtype=np.int16, count=samples * lines)
                data[fidx] = buf.reshape(samples, lines, order="F")

        else:
            raise ValueError(f"Unsupported RAW type id={hid}")

    return data, header, ts

# =========================
# kspaceLineRecon (unchanged)
# =========================
def _make_grid(Nx, dx, Ny, dy):
    kx = 2*np.pi*np.fft.fftfreq(Nx, dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, dy)
    kx, ky = np.fft.fftshift(kx), np.fft.fftshift(ky)
    KY, KX = np.meshgrid(ky, kx, indexing="xy")
    return KX, KY, np.sqrt(KX**2 + KY**2)

def _interp_nearest(w, col, wq):
    idx = np.searchsorted(w, wq, side="left")
    idx = np.clip(idx, 1, len(w)-1)
    left, right = w[idx-1], w[idx]
    idx -= (wq-left < right-wq)
    out = col[idx]
    out[(wq < w[0]) | (wq > w[-1])] = 0
    return out

def kspaceLineRecon(p, dy, dt, c):
    p = np.vstack([np.flip(p, 0), p[1:]])
    Nt, Ny = p.shape

    dx = dt * c
    KX, KY, K = _make_grid(Nt, dx, Ny, dy)
    w = c * KX
    w_new = c * K

    sf = (c**2 * np.sqrt(np.maximum((w/c)**2 - KY**2, 0))) / (2*w)
    sf[(w == 0) & (KY == 0)] = c/2

    P = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(p))) * sf
    P[np.abs(w) < np.abs(c*KY)] = 0

    P2 = np.zeros_like(P)
    for j in range(Ny):
        P2[:, j] = _interp_nearest(w[:, 0], P[:, j], w_new[:, j])

    img = np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(P2))))
    img = img[(Nt+1)//2 - 1:] * (4/c)
    img[img < 0] = 0
    return img

# =========================
# Public API
# =========================
def run_from_raw_files(rf_raw, env_raw=None, navg=10, frames=1000):
    dy = 260.416666e-6
    dt = 1/30e6
    c = 1500.0

    data_rf, _, _ = rdataread(rf_raw, frames)
    data_env = None
    if env_raw:
        data_env, _, _ = rdataread(env_raw, frames)

    us_img = np.mean(data_env, 0).squeeze() if data_env is not None else None

    navg = min(navg, data_rf.shape[0])
    y = np.mean(data_rf[:navg, :data_rf.shape[1]//2, :], 0).squeeze()

    recon = kspaceLineRecon(y, dy, dt, c)
    return us_img, y[:, :192], np.abs(recon[:, :192])
