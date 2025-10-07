import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import imageio.v2 as imageio
import cv2, math

# --------- PARAMETERS (tune these) ----------
LOW_PATH  = "C L.png"     # low-pass source
HIGH_PATH = "H H.png"   # high-pass source
OUT_PATH  = "Hybrid_Result.png"

sigmaHigh  = 25    # e.g., 12~30 (details)
sigmaLow   = 10    # e.g., 8~20  (blur)
alpha      = 1.0   # mix strength for high-pass (0.5~1.5 typical)
high_boost = 1.2   # contrast on high-pass before mixing (1.0 = none)
output_gamma = 1.0 # 1.0 = off; try 0.9~1.2 if tones feel off
# --------------------------------------------

def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
    centerI = numRows//2
    centerJ = numCols//2
    yy, xx = np.ogrid[:numRows, :numCols]
    coef = np.exp(-((yy-centerI)**2 + (xx-centerJ)**2) / (2.0 * sigma**2))
    return (1.0 - coef) if highPass else coef

def filterDFT(imageMatrix, filterMatrix):
    shiftedDFT = fftshift(fft2(imageMatrix))
    filteredDFT = shiftedDFT * filterMatrix
    return ifft2(ifftshift(filteredDFT))

def lowPass(imageMatrix, sigma):
    n, m = imageMatrix.shape
    return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=False))

def highPass(imageMatrix, sigma):
    n, m = imageMatrix.shape
    return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=True))

def normalize01(x):
    x = np.real(x)
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x)
    return x

if __name__ == "__main__":
    # 1) Load grayscale
    img_low  = imageio.imread(LOW_PATH,  mode='L').astype(np.float32)
    img_high = imageio.imread(HIGH_PATH, mode='L').astype(np.float32)

    # 2) Resize to match
    h = min(img_low.shape[0],  img_high.shape[0])
    w = min(img_low.shape[1],  img_high.shape[1])
    img_low  = cv2.resize(img_low,  (w, h))
    img_high = cv2.resize(img_high, (w, h))

    # 3) Filter
    lowPassed   = lowPass(img_low,  sigmaLow)            # smooth base
    highPassed  = highPass(img_high, sigmaHigh)          # details
    highPassed  = np.real(highPassed) * high_boost       # contrast boost on details

    # 4) Blend (Photoshop-like: low + alpha*high)
    hybrid = np.real(lowPassed) + alpha * highPassed

    # 5) Optional tone (gamma)
    hybrid01 = normalize01(hybrid)
    if output_gamma != 1.0:
        hybrid01 = np.power(np.clip(hybrid01, 0, 1), 1.0/output_gamma)

    # 6) Save
    out = (np.clip(hybrid01, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(OUT_PATH, out)

    # (Optional) also save components for inspection
    imageio.imwrite("lowPassed.png",  (normalize01(lowPassed)  * 255).astype(np.uint8))
    imageio.imwrite("highPassed.png", (normalize01(highPassed) * 255).astype(np.uint8))
    print(f"Saved: {OUT_PATH}  (sigmaHigh={sigmaHigh}, sigmaLow={sigmaLow}, alpha={alpha}, high_boost={high_boost})")

    # ==============================
    # EXTRA: COLOR VERSION
    # ==============================
    print("Generating color hybrid...")

    # reload in RGB (0–1 float)
    img_low_color  = imageio.imread(LOW_PATH).astype(np.float32) / 255.0
    img_high_color = imageio.imread(HIGH_PATH).astype(np.float32) / 255.0

    # resize both
    img_low_color  = cv2.resize(img_low_color,  (w, h))
    img_high_color = cv2.resize(img_high_color, (w, h))

    channels = []
    for c in range(3):  # R,G,B
        lowPassed   = lowPass(img_low_color[:,:,c],  sigmaLow)
        highPassed  = highPass(img_high_color[:,:,c], sigmaHigh)
        highPassed  = np.real(highPassed) * high_boost
        hybrid      = np.real(lowPassed) + alpha * highPassed
        hybrid01    = normalize01(hybrid)
        if output_gamma != 1.0:
            hybrid01 = np.power(np.clip(hybrid01, 0, 1), 1.0/output_gamma)
        channels.append(hybrid01)

    # merge channels back
    hybrid_color = np.stack(channels, axis=2)
    out_color = (np.clip(hybrid_color, 0, 1) * 255).astype(np.uint8)

    imageio.imwrite("Hybrid_Result_Color.png", out_color)
    print(f"✅ Saved color hybrid: Hybrid_Result_Color.png")
