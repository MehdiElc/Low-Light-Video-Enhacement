
import cv2
import numpy as np
import os
import gc
from scipy.fftpack import fft2, ifft2
from scipy.ndimage import convolve
from bm3d import bm3d_rgb

# =================== 🎛 بخش تنظیمات کلی ===================

# مسیر ویدیوها
INPUT_VIDEO = r"C:\Users\AsuS\OneDrive\Documents\VScode.V\3X.mp4"
OUTPUT_VIDEO = r"C:\Users\AsuS\OneDrive\Documents\VScode.V\output_video3X.mp4"
OUTPUT_FRAMES_FOLDER = r"C:\Users\AsuS\OneDrive\Documents\VScode.V\output_frames3X"

# --- پارامترهای LIME ---
LIME_ALPHA = 0.03         # smoothness در refinement
LIME_GAMMA = 0.5          # کنترل روشنایی (gamma correction)
LIME_DENOISE = True       # فعال/غیرفعال کردن BM3D
LIME_BM3D_SIGMA = 0.8    # شدت حذف نویز BM3D

# --- پارامترهای Illumination Refinement ---
REFINE_RHO = 1.5          # نرخ افزایش μ
REFINE_MU_INIT = 0.01     # مقدار اولیه μ
REFINE_MAX_ITER = 60      # تعداد iteration ها

# --- میانگین‌گیری زمانی ---
TEMPORAL_AVG = True
TEMPORAL_WEIGHTS = (0.25, 0.5, 0.25)

# --- تنظیم رزولوشن ---
RESIZE_ENABLED = True
RESIZE_WIDTH = 1280  # مثلا 720 یا 1280 برای کاهش مصرف RAM

# ===========================================================


def max_rgb_illumination(img):
    return np.max(img, axis=2)

def gradient(img):
    kernel_h = np.array([[1, -1]])
    kernel_v = np.array([[1], [-1]])
    grad_h = convolve(img, kernel_h, mode='wrap')
    grad_v = convolve(img, kernel_v, mode='wrap')
    return grad_h, grad_v

def divergence(grad_h, grad_v):
    kernel_h = np.array([[1, -1]])
    kernel_v = np.array([[1], [-1]])
    div_h = convolve(grad_h, -kernel_h, mode='wrap')
    div_v = convolve(grad_v, -kernel_v, mode='wrap')
    return div_h + div_v

def shrink(x, eps):
    return np.sign(x) * np.maximum(np.abs(x) - eps, 0)

def refine_illumination(T_hat):
    alpha = LIME_ALPHA
    rho = REFINE_RHO
    mu_init = REFINE_MU_INIT
    max_iter = REFINE_MAX_ITER

    T = np.zeros_like(T_hat)
    G_h = np.zeros_like(T_hat)
    G_v = np.zeros_like(T_hat)
    Z_h = np.zeros_like(T_hat)
    Z_v = np.zeros_like(T_hat)
    mu = mu_init

    eps = 1e-3
    grad_h_T_hat, grad_v_T_hat = gradient(T_hat)
    W_h = 1.0 / (np.abs(grad_h_T_hat) + eps)
    W_v = 1.0 / (np.abs(grad_v_T_hat) + eps)

    for _ in range(max_iter):
        T_hat_fft = fft2(2 * T_hat)
        GZ_h = G_h - Z_h / mu
        GZ_v = G_v - Z_v / mu
        div_GZ = divergence(GZ_h, GZ_v)
        rhs = T_hat_fft + mu * fft2(div_GZ)

        h, w = T_hat.shape
        fx = np.cos(2 * np.pi * np.arange(w) / w)
        fy = np.cos(2 * np.pi * np.arange(h) / h)
        denom = 2 + mu * ((2 - 2 * fx)[np.newaxis, :] + (2 - 2 * fy)[:, np.newaxis])
        T = np.real(ifft2(rhs / denom))

        grad_h_T, grad_v_T = gradient(T)
        G_h = shrink(grad_h_T + Z_h / mu, alpha * W_h / mu)
        G_v = shrink(grad_v_T + Z_v / mu, alpha * W_v / mu)

        Z_h += mu * (grad_h_T - G_h)
        Z_v += mu * (grad_v_T - G_v)
        mu *= rho

    return np.clip(T, 1e-3, 1)

def lime_enhance(img):
    img_f = img.astype(np.float32) / 255.0
    T_hat = max_rgb_illumination(img_f)
    T_refined = refine_illumination(T_hat)
    T_gamma = np.power(T_refined, LIME_GAMMA)

    R = img_f / (T_gamma[..., np.newaxis] + 1e-8)
    R = np.clip(R, 0, 1)

    if LIME_DENOISE:
        R_denoised = bm3d_rgb(R, sigma_psd=LIME_BM3D_SIGMA)
        R = R * T_gamma[..., np.newaxis] + R_denoised * (1 - T_gamma[..., np.newaxis])
        R = np.clip(R, 0, 1)

    return (R * 255).astype(np.uint8)

def process_video():
    if not os.path.exists(OUTPUT_FRAMES_FOLDER):
        os.makedirs(OUTPUT_FRAMES_FOLDER)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("❌ خطا در باز کردن ویدیو")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if RESIZE_ENABLED and RESIZE_WIDTH < orig_w:
        scale = RESIZE_WIDTH / orig_w
        width = int(orig_w * scale)
        height = int(orig_h * scale)
    else:
        width, height = orig_w, orig_h

    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    ret_prev, frame_prev = cap.read()
    if RESIZE_ENABLED and ret_prev:
        frame_prev = cv2.resize(frame_prev, (width, height))

    ret_curr, frame_curr = cap.read()
    if RESIZE_ENABLED and ret_curr:
        frame_curr = cv2.resize(frame_curr, (width, height))

    frame_index = 0
    while ret_curr:
        ret_next, frame_next = cap.read()
        if RESIZE_ENABLED and ret_next:
            frame_next = cv2.resize(frame_next, (width, height))

        if TEMPORAL_AVG and ret_prev and ret_next:
            prev_f = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB).astype(np.float32)
            curr_f = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2RGB).astype(np.float32)
            next_f = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB).astype(np.float32)
            blended = (TEMPORAL_WEIGHTS[0]*prev_f +
                       TEMPORAL_WEIGHTS[1]*curr_f +
                       TEMPORAL_WEIGHTS[2]*next_f).astype(np.uint8)
            enhanced = lime_enhance(blended)
        else:
            enhanced = lime_enhance(cv2.cvtColor(frame_curr, cv2.COLOR_BGR2RGB))

        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        out.write(enhanced_bgr)
        cv2.imwrite(os.path.join(OUTPUT_FRAMES_FOLDER, f"frame_{frame_index:06d}.png"), enhanced_bgr)

        frame_index += 1
        frame_prev, frame_curr = frame_curr, frame_next
        ret_curr = ret_next

        del enhanced_bgr, enhanced
        gc.collect()

    cap.release()
    out.release()
    print(f"✅ خروجی ذخیره شد: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    process_video()
