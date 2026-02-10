import os
import re
import tarfile
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")  

import matplotlib.pyplot as plt
plt.ion()  

# GUI
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


# ============================================================
# EXACT PORT: rdataread.m
# ============================================================
def rdataread(filename, frames):
    with open(filename, "rb") as f:
        hinfo = np.fromfile(f, dtype=np.int32, count=5)
        if hinfo.size < 5:
            raise ValueError("File too small / missing header")

        header = {
            "id": int(hinfo[0]),
            "frames": int(hinfo[1]),
            "lines": int(hinfo[2]),
            "samples": int(hinfo[3]),
            "sampleSize": int(hinfo[4]),
        }

        if frames > header["frames"]:
            frames = header["frames"]

        hid = header["id"]
        lines = header["lines"]
        samples = header["samples"]

        ts = np.zeros(frames, dtype=np.int64)

        if hid in (0, 3):  # IQ / PW IQ
            data = np.zeros((frames, samples * 2, lines), dtype=np.int16)
            for fi in range(frames):
                ts_val = np.fromfile(f, dtype=np.int64, count=1)
                if ts_val.size != 1:
                    raise ValueError("Unexpected EOF while reading timestamp")
                ts[fi] = ts_val[0]
                oneline = np.fromfile(f, dtype=np.int16, count=(samples * 2 * lines))
                if oneline.size != samples * 2 * lines:
                    raise ValueError("Unexpected EOF while reading IQ frame")
                oneline = oneline.reshape((samples * 2, lines), order="F")  # MATLAB column-major
                data[fi, :, :] = oneline

        elif hid == 1:  # ENV uint8
            data = np.zeros((frames, samples, lines), dtype=np.uint8)
            for fi in range(frames):
                ts_val = np.fromfile(f, dtype=np.int64, count=1)
                if ts_val.size != 1:
                    raise ValueError("Unexpected EOF while reading timestamp")
                ts[fi] = ts_val[0]
                oneline = np.fromfile(f, dtype=np.uint8, count=(samples * lines))
                if oneline.size != samples * lines:
                    raise ValueError("Unexpected EOF while reading ENV frame")
                oneline = oneline.reshape((samples, lines), order="F")
                data[fi, :, :] = oneline

        elif hid == 2:  # RF int16
            data = np.zeros((frames, samples, lines), dtype=np.int16)
            for fi in range(frames):
                ts_val = np.fromfile(f, dtype=np.int64, count=1)
                if ts_val.size != 1:
                    raise ValueError("Unexpected EOF while reading timestamp")
                ts[fi] = ts_val[0]
                frame_data = np.fromfile(f, dtype=np.int16, count=(samples * lines))
                if frame_data.size != samples * lines:
                    raise ValueError("Unexpected EOF while reading RF frame")
                frame_data = frame_data.reshape((samples, lines), order="F")
                data[fi, :, :] = frame_data

        else:
            raise ValueError(f"Unsupported header.id={hid}")

    return data, header, ts


# ============================================================
# EXACT PORT: kspaceLineRecon.m (default Interp='*nearest')
# ============================================================
def _make_grid(Nx, dx, Ny, dy):
    kx_vec = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky_vec = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx_vec = np.fft.fftshift(kx_vec)
    ky_vec = np.fft.fftshift(ky_vec)
    KY, KX = np.meshgrid(ky_vec, kx_vec, indexing="xy")
    K = np.sqrt(KX**2 + KY**2)
    return KX, KY, K

def _interp_column_nearest(w_vec, col, wq):
    idx = np.searchsorted(w_vec, wq, side="left")
    idx = np.clip(idx, 1, len(w_vec) - 1)
    left = w_vec[idx - 1]
    right = w_vec[idx]
    choose_right = (wq - left) >= (right - wq)
    idx = idx - 1 + choose_right.astype(np.int64)
    out = col[idx]
    out[(wq < w_vec[0]) | (wq > w_vec[-1])] = np.nan
    return out

def kspaceLineRecon(p, dy, dt, c, DataOrder="ty", Interp="*nearest", Plot=False, PosCond=False):
    p = np.asarray(p)

    if DataOrder == "yt":
        p = p.T
    elif DataOrder != "ty":
        raise ValueError("DataOrder must be 'ty' or 'yt'")

    # mirror: [flipdim(p,1); p(2:end,:)]
    p = np.vstack([np.flip(p, axis=0), p[1:, :]])
    Nt, Ny = p.shape

    dx = dt * c
    kx, ky, k = _make_grid(Nt, dx, Ny, dy)

    w = c * kx
    w_new = c * k

    arg = (w / c) ** 2 - ky**2
    sqrt_term = np.sqrt(np.maximum(arg, 0.0))

    sf = (c**2) * sqrt_term / (2.0 * w)
    sf[(w == 0) & (ky == 0)] = c / 2.0

    P = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(p)))
    P = sf * P

    # exclude inhomogeneous part
    P[np.abs(w) < np.abs(c * ky)] = 0

    mode = Interp.replace("*", "").lower()
    if mode != "nearest":
        raise ValueError("This script matches MATLAB default '*nearest' only.")

    w_vec = w[:, 0]
    P2 = np.empty_like(P)
    for j in range(Ny):
        P2[:, j] = _interp_column_nearest(w_vec, P[:, j], w_new[:, j])

    P = np.nan_to_num(P2, nan=0.0)

    p_rec = np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(P))))

    # keep positive half
    Nt0 = (Nt + 1) // 2
    p_rec = p_rec[Nt0 - 1 :, :]

    # scaling: 2*2*p/c
    p_rec = (4.0 / c) * p_rec

    if PosCond:
        p_rec[p_rec < 0] = 0

    if Plot:
        plt.figure()
        plt.imshow(p_rec, aspect="equal")
        plt.title("kspaceLineRecon output (internal)")
        plt.xlabel("y (sensor index)")
        plt.ylabel("x (depth index)")
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.5)

    return p_rec


# ============================================================
# GUI helpers
# ============================================================
def pick_tar_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Clarius TAR File",
        filetypes=[
            ("Clarius Archive", "*.tar *.tar.gz *.tgz"),
            ("TAR", "*.tar"),
            ("TAR.GZ", "*.tar.gz"),
            ("TGZ", "*.tgz"),
            ("All files", "*.*"),
        ],
    )

def ask_navg(max_frames):
    root = tk.Tk()
    root.withdraw()
    return simpledialog.askinteger(
        "PA Frame Averaging",
        f"Enter number of PA frames to average (1–{max_frames}):",
        initialvalue=max_frames,
        minvalue=1,
        maxvalue=max_frames,
    )


# ============================================================
# Archive helpers
# ============================================================
def extract_tar(selected_tar: Path, temp_extract_folder: Path):
    temp_extract_folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(selected_tar, "r:*") as tar:
        tar.extractall(path=temp_extract_folder)

def decompress_lzo(temp_extract_folder: Path, lzop_exe: Path):
    lzo_files = list(temp_extract_folder.rglob("*.lzo"))
    for f in lzo_files:
        subprocess.run([str(lzop_exe), "-d", str(f)], check=True)

def find_raws(temp_extract_folder: Path):
    raw_files = list(temp_extract_folder.rglob("*.raw"))
    env_raw = None
    rf_raw = None
    for rp in raw_files:
        n = rp.name.lower()
        if n.endswith("_env.raw"):
            env_raw = rp
        elif n.endswith("_rf.raw"):
            rf_raw = rp
    return env_raw, rf_raw, raw_files


# ============================================================
# Plot + Save
# ============================================================
def save_plot_png(out_path: Path):
    plt.savefig(out_path, dpi=250, bbox_inches="tight")

def show_now():
    plt.show(block=False)
    plt.pause(0.5)


# ============================================================
# Main
# ============================================================
def main():
    # Your lzop path
    lzop_folder = Path(r"C:\Users\tomar\Desktop\New folder (3)")
    lzop_exe = lzop_folder / "lzop.exe"
    if not lzop_exe.exists():
        messagebox.showerror("lzop.exe not found", f"Could not find:\n{lzop_exe}")
        return

    selected = pick_tar_file()
    if not selected:
        print("No file selected. Exiting.")
        return

    selected_tar = Path(selected)
    lzo_path = selected_tar.parent

    base_name = selected_tar.stem
    if base_name.endswith(".tar"):
        base_name = base_name[:-4]

    temp_extract_folder = lzo_path / f"{base_name}_temp"

    print(f"\nSelected file: {selected_tar}")
    print(f"Temporary extraction folder: {temp_extract_folder}\n")

    # Extract + Decompress
    extract_tar(selected_tar, temp_extract_folder)
    decompress_lzo(temp_extract_folder, lzop_exe)

    # Find raw files
    env_raw, rf_raw, _raw_files = find_raws(temp_extract_folder)
    if rf_raw is None:
        messagebox.showerror("Missing RF", "Could not find *_rf.raw after extraction/decompression.")
        shutil.rmtree(temp_extract_folder, ignore_errors=True)
        return

    # Load raw data
    data_us = None
    if env_raw is not None:
        data_us, hdr_us, ts_us = rdataread(env_raw, frames=1000)
        print(f"US ENV detected: {env_raw.name} | size={data_us.shape} | header={hdr_us}")
    else:
        print("Warning: *_env.raw not found. US ENV plot will be skipped.")

    data_pa, hdr_pa, ts_pa = rdataread(rf_raw, frames=1000)
    print(f"PA RF detected: {rf_raw.name} | size={data_pa.shape} | header={hdr_pa}")

    # Ask Navg
    nframes_total = data_pa.shape[0]
    navg = ask_navg(nframes_total)
    if navg is None:
        print("Averaging selection cancelled.")
        shutil.rmtree(temp_extract_folder, ignore_errors=True)
        return
    print(f"Using {navg} PA frames for averaging")

    # Output folder
    output_folder = lzo_path / f"{base_name}_extracted__avg{navg}"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Copy extracted contents to output folder
    for item in temp_extract_folder.iterdir():
        dest = output_folder / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    excel_file = output_folder / "Processed_Plots.xlsx"
    print(f"All results will be saved in: {output_folder}")

    # MATLAB constants
    dy = 260.416666e-6
    dt = 1 / 30e6
    c = 1500.0
    cols_to_show = 192

    # Excel writer mode
    writer_mode = "w"

    try:
        # ==========================
        # FIGURE 1: US ENV averaged
        # ==========================
        if data_us is not None:
            us_img = np.mean(data_us, axis=0).squeeze()
            Ny_us, Nx_us = us_img.shape
            y_mm_us = np.arange(Ny_us) * c * dt * 1e3
            x_mm_us = np.arange(Nx_us) * (dy / 2) * 1e3

            plt.figure()
            plt.imshow(
                us_img,
                extent=[x_mm_us[0], x_mm_us[-1], y_mm_us[-1], y_mm_us[0]],
                aspect="equal",
                cmap="gray",
            )
            plt.gca().invert_yaxis()
            plt.title("US ENV (averaged)")
            plt.xlabel("Lateral (mm)")
            plt.ylabel("Depth (mm)")
            plt.colorbar()
            save_plot_png(output_folder / "FIG1_US_ENV_averaged.png")
            show_now()

            with pd.ExcelWriter(excel_file, engine="openpyxl", mode=writer_mode) as w:
                pd.DataFrame(us_img).to_excel(w, sheet_name="US_ENV_Image", index=False, header=False)
                pd.DataFrame(x_mm_us).to_excel(w, sheet_name="US_ENV_X_mm", index=False, header=False)
                pd.DataFrame(y_mm_us).to_excel(w, sheet_name="US_ENV_Y_mm", index=False, header=False)
            writer_mode = "a"

        # ==========================
        # PA RF averaging
        # ==========================
        print("STEP A: computing PA average y...")
        half_samples = data_pa.shape[1] // 2
        y = np.mean(data_pa[:navg, :half_samples, :], axis=0).squeeze()
        print("STEP A done. y shape =", y.shape)

        if y.ndim != 2:
            raise RuntimeError(f"Expected y to be 2D (Ny,Nx). Got shape {y.shape}")

        Ny, Nx = y.shape
        y_mm = np.arange(Ny) * c * dt * 1e3
        x_mm = np.arange(Nx) * dy * 1e3
        cols = min(cols_to_show, Nx)

        # ==========================
        # FIGURE 2: PA RF averaged
        # ==========================
        print("STEP B: plotting PA RF...")
        plt.figure()
        plt.imshow(
            y[:, :cols],
            extent=[x_mm[0], x_mm[cols - 1], y_mm[-1], y_mm[0]],
            aspect="equal",
            cmap="gray",
        )
        plt.gca().invert_yaxis()
        plt.title(f"PA RF Averaged ({navg} frames)")
        plt.xlabel("Lateral (mm)")
        plt.ylabel("Depth (mm)")
        plt.colorbar()
        save_plot_png(output_folder / f"FIG2_PA_RF_averaged_{navg}frames.png")
        show_now()
        print("STEP B done (RF figure should be visible).")

        # ==========================
        # FIGURE 3: internal recon plot
        # ==========================
        print("STEP C: running kspaceLineRecon...")
        p_xy = kspaceLineRecon(y, dy, dt, c, Plot=True, PosCond=True)
        print("STEP C done. p_xy shape =", p_xy.shape)

        # ==========================
        # FIGURE 4: PA Reconstruction
        # ==========================
        print("STEP D: plotting reconstruction...")
        img = np.abs(p_xy[:, :cols])
        vmin, vmax = np.percentile(img, [1, 99])

        plt.figure()
        plt.imshow(
            img,
            extent=[x_mm[0], x_mm[cols - 1], y_mm[-1], y_mm[0]],
            aspect="equal",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        plt.gca().invert_yaxis()
        plt.title(f"PA Reconstruction ({navg} frames)")
        plt.xlabel("Lateral (mm)")
        plt.ylabel("Depth (mm)")
        plt.colorbar()
        save_plot_png(output_folder / f"FIG4_PA_Reconstruction_{navg}frames.png")
        show_now()
        print("STEP D done (Recon figure should be visible).")

        # ==========================
        # Excel outputs for PA
        # ==========================
        with pd.ExcelWriter(excel_file, engine="openpyxl", mode=writer_mode) as w:
            pd.DataFrame(y[:, :cols]).to_excel(w, sheet_name="PA_RF_Image", index=False, header=False)
            pd.DataFrame(x_mm[:cols]).to_excel(w, sheet_name="PA_RF_X_mm", index=False, header=False)
            pd.DataFrame(y_mm).to_excel(w, sheet_name="PA_RF_Y_mm", index=False, header=False)

            pd.DataFrame(img).to_excel(w, sheet_name=f"PA_Recon_{navg}frames", index=False, header=False)
            pd.DataFrame(x_mm[:cols]).to_excel(w, sheet_name="PA_Reconstruction_X_mm", index=False, header=False)
            pd.DataFrame(y_mm).to_excel(w, sheet_name="PA_Reconstruction_Y_mm", index=False, header=False)

    except Exception as e:
        import traceback
        print("\n ERROR occurred. Full traceback:\n")
        traceback.print_exc()
        messagebox.showerror("Error", f"Script failed:\n{e}")
        # Keep temp folder for debugging if failure happens
        return

    # Cleanup temp folder
    shutil.rmtree(temp_extract_folder, ignore_errors=True)
    print("Temp folder removed.")
    messagebox.showinfo("Done", f"Saved all results in:\n{output_folder}")

    # Keep windows open until user closes
    print("Close all plot windows to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
