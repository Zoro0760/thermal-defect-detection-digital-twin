import numpy as np
import cv2
import os
import subprocess
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# CONFIGURATION — EDIT THESE PATHS BEFORE RUNNING
# =============================================================================

INPUT_FOLDER  = r'C:\blade_defect_project\data\raw_ir'
OUTPUT_PHASE_FOLDER      = r'C:\blade_defect_project\data\phase_images'
OUTPUT_AMPLITUDE_FOLDER  = r'C:\blade_defect_project\data\amplitude_images'
OUTPUT_COMPARISON_FOLDER = r'C:\blade_defect_project\data\comparison_images'
TESTING_IMAGE_FOLDER     = r'C:\blade_defect_project\Testing_Image'
LOG_FOLDER    = r'C:\blade_defect_project\outputs\logs'

# FFT frequency index to extract (1 = fundamental frequency, best for subsurface defects [1])
# Try 1, 2, 3 if defects are not clearly visible — lower = deeper defects
FREQ_INDEX = 3

# Minimum frames required for reliable FFT (less than this = skip)
MIN_FRAMES = 50

# Save amplitude image alongside phase image (useful for comparison)
SAVE_AMPLITUDE = True

# Save comparison plot (phase vs amplitude vs thermal frame)
SAVE_COMPARISON_PLOT = True

# =============================================================================
# STEP 1: VIDEO READING
# =============================================================================

def read_video_frames(video_path):
    """
    Read all frames from a video file and convert to grayscale float32.

    Returns numpy array of shape (T, H, W) or None if reading fails.
    T = number of frames, H = height, W = width.

    Grayscale conversion is correct here because thermal cameras
    encode temperature as intensity — colour information is just
    a false-colour overlay added by the camera software [3].
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"   ERROR: Cannot open video: {video_path}")
        return None

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to grayscale — thermal intensity is in luminance [3]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames.append(gray)
        frame_count += 1

    cap.release()

    if frame_count == 0:
        print(f"   ERROR: Zero frames read from {video_path}")
        return None

    if frame_count < MIN_FRAMES:
        print(f"   WARNING: Only {frame_count} frames — FFT may be unreliable (min={MIN_FRAMES})")

    print(f"   Read {frame_count} frames, size: {frames[0].shape[1]}x{frames[0].shape[0]} px")
    return np.stack(frames, axis=0)  # shape: (T, H, W)


# =============================================================================
# STEP 2: VIDEO REPAIR USING FFMPEG
# =============================================================================

def repair_video(input_path, output_path):
    """
    Attempt to repair a corrupted video using FFmpeg re-encoding.

    FFmpeg reads the raw stream and re-encodes it to H.264 + AAC,
    fixing corrupt headers, missing keyframes, and container errors.
    Returns True if repair succeeded and output file exists.
    """
    try:
        command = [
            "ffmpeg",
            "-y",                        # overwrite output if exists
            "-i", input_path,            # input file
            "-c:v", "libx264",           # re-encode video to H.264
            "-preset", "fast",           # encoding speed
            "-crf", "18",                # quality (lower = better, 18 is near-lossless)
            "-an",                       # remove audio (thermal videos typically have no audio track)
            output_path
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120                  # 2 minute timeout per video
        )
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000

    except subprocess.TimeoutExpired:
        print(f"   WARNING: FFmpeg repair timed out for {input_path}")
        return False
    except FileNotFoundError:
        print("   WARNING: FFmpeg not found — skipping repair.")
        print("   Install FFmpeg: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"   WARNING: Repair failed — {e}")
        return False


# =============================================================================
# STEP 3: FFT-BASED PHASE EXTRACTION
# =============================================================================

def extract_phase_and_amplitude(frames_array, freq_index=1):
    """
    Apply pixel-wise FFT along the time axis and extract phase and amplitude.

    Theory (Maldague & Marinetti 1996) [1]:
    - Surface temperature after sinusoidal heating = sum of harmonics
    - Phase angle at each pixel = delay of thermal wave response
    - Defects below the surface cause extra phase delay (air gap = insulator)
    - Phase is independent of surface emissivity and heating non-uniformity
    - This makes phase images cleaner than raw thermal frames

    Args:
        frames_array: numpy array (T, H, W) of grayscale float32 frames
        freq_index:   which FFT harmonic to extract (1=fundamental, 2=second, etc.)

    Returns:
        phase_image:     (H, W) float array of phase angles in radians
        amplitude_image: (H, W) float array of FFT magnitude
    """
    # FFT along time axis (axis=0) — pixel-wise across all frames [4]
    fft_result = np.fft.rfft(frames_array, axis=0)

    # Clamp freq_index to valid range
    max_idx = fft_result.shape[0] - 1
    if freq_index > max_idx:
        print(f"   WARNING: freq_index={freq_index} exceeds max={max_idx}, using {max_idx}")
        freq_index = max_idx

    # Phase angle = arctan(imag / real) — the delay in radians [6]
    phase_image     = np.angle(fft_result[freq_index])

    # Amplitude = magnitude of complex FFT at that frequency
    amplitude_image = np.abs(fft_result[freq_index])

    return phase_image, amplitude_image


def normalize_to_uint8(image):
    """Normalize any float image to 0-255 uint8 for saving as PNG."""
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


# =============================================================================
# STEP 4: QUALITY CHECK
# =============================================================================

def check_phase_image_quality(phase_image, amplitude_image):
    """
    Assess if the phase image is likely to contain useful defect information.

    Returns a quality score (0-100) and a dictionary of metrics.
    Score above 50 = acceptable. Below 50 = consider different freq_index.
    """
    metrics = {}

    # Standard deviation — low std means uniform image (no defects visible)
    metrics['phase_std']     = float(np.std(phase_image))
    metrics['amplitude_std'] = float(np.std(amplitude_image))

    # Dynamic range — low range = poor contrast
    metrics['phase_range']   = float(phase_image.max() - phase_image.min())

    # SNR estimate — mean / std of amplitude
    amp_mean = amplitude_image.mean()
    amp_std  = amplitude_image.std()
    metrics['amplitude_snr'] = float(amp_mean / amp_std) if amp_std > 0 else 0

    # Score (heuristic)
    score = 0
    if metrics['phase_std']     > 0.05:  score += 25
    if metrics['phase_range']   > 0.3:   score += 25
    if metrics['amplitude_snr'] > 2.0:   score += 25
    if metrics['amplitude_std'] > 10:    score += 25

    metrics['quality_score'] = score

    return score, metrics


# =============================================================================
# STEP 5: SAVE COMPARISON PLOT
# =============================================================================

def save_comparison_plot(frames_array, phase_image, amplitude_image,
                          output_path, video_name, metrics):
    """
    Save a side-by-side comparison: thermal frame | phase | amplitude | profile.
    This is the key visual you will use in your seminar and report.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor('#0f0f1a')

    text_color = '#e8e8e8'
    mid_frame  = frames_array[len(frames_array) // 4]  # quarter-period frame

    def style(ax, title):
        ax.set_title(title, color=text_color, fontsize=10, fontweight='bold', pad=6)
        ax.tick_params(colors=text_color, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333355')
        ax.set_facecolor('#0a0a14')

    # Thermal frame
    im0 = axes[0].imshow(mid_frame, cmap='hot')
    style(axes[0], 'Raw thermal frame')
    plt.colorbar(im0, ax=axes[0], fraction=0.046).ax.tick_params(colors=text_color)

    # Phase image
    im1 = axes[1].imshow(phase_image, cmap='RdBu_r')
    style(axes[1], f'Phase image (freq_idx={FREQ_INDEX})')
    plt.colorbar(im1, ax=axes[1], fraction=0.046).ax.tick_params(colors=text_color)

    # Amplitude image
    im2 = axes[2].imshow(amplitude_image, cmap='plasma')
    style(axes[2], 'Amplitude image')
    plt.colorbar(im2, ax=axes[2], fraction=0.046).ax.tick_params(colors=text_color)

    # Horizontal cross-section through image center
    h, w = phase_image.shape
    cx   = h // 2
    axes[3].plot(phase_image[cx, :],     color='#fd79a8', linewidth=2,   label='Phase')
    axes[3].plot(amplitude_image[cx, :] / amplitude_image[cx, :].max() *
                 (phase_image[cx, :].max() - phase_image[cx, :].min()) +
                 phase_image[cx, :].min(),
                 color='#fdcb6e', linewidth=1.5, linestyle='--', label='Amplitude (norm)')
    axes[3].set_xlabel('Pixel x', color=text_color, fontsize=9)
    axes[3].set_ylabel('Phase (rad)', color=text_color, fontsize=9)
    style(axes[3], 'Horizontal cross-section')
    axes[3].legend(fontsize=8, facecolor='#1a1a2e', labelcolor=text_color)
    axes[3].grid(True, color='#333355', alpha=0.5)

    quality = metrics.get('quality_score', 0)
    fig.suptitle(
        f"{video_name}  |  Quality score: {quality}/100  |  "
        f"Phase range: {metrics.get('phase_range', 0):.3f} rad  |  "
        f"SNR: {metrics.get('amplitude_snr', 0):.2f}",
        color=text_color, fontsize=10, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


# =============================================================================
# STEP 6: SAFE PROCESSING WITH AUTO-REPAIR
# =============================================================================

def video_to_phase_safe(video_path, phase_dir, amp_dir, comp_dir, test_dir, base_name, freq_index=1):
    """
    Full processing pipeline for one video:
      1. Try to read frames normally
      2. If fails → repair with FFmpeg and retry
      3. If still fails → skip and log
      4. Run FFT → extract phase + amplitude
      5. Quality check
      6. Save outputs

    Returns dict with results, or None if completely failed.
    """
    print(f"\n{'-'*60}")
    print(f"Processing: {base_name}")
    print(f"{'-'*60}")

    # ── Step 1: Read frames ──────────────────────────────
    frames = read_video_frames(video_path)

    # ── Step 2: Repair if needed ─────────────────────────
    if frames is None:
        print(f"   Attempting FFmpeg repair...")
        repaired_path = os.path.join(
            os.path.dirname(video_path),
            base_name + "_repaired.mp4"
        )
        success = repair_video(video_path, repaired_path)

        if success:
            print(f"   Repair succeeded. Re-reading repaired video...")
            frames = read_video_frames(repaired_path)
        else:
            print(f"   Repair failed. Skipping {base_name}.")
            return None

    # ── Step 3: Final check ───────────────────────────────
    if frames is None:
        print(f"   SKIPPED: Cannot read video even after repair.")
        return None

    # ── Step 4: Extract phase and amplitude ───────────────
    print(f"   Extracting phase at freq_index={freq_index}...")
    phase_image, amplitude_image = extract_phase_and_amplitude(frames, freq_index)

    # ── Step 5: Quality check ─────────────────────────────
    quality_score, metrics = check_phase_image_quality(phase_image, amplitude_image)
    print(f"   Quality score: {quality_score}/100")
    if quality_score < 50:
        print(f"   WARNING: Low quality. Try freq_index=2 or freq_index=3.")

    # ── Step 6: Save phase image ─────────────────────────
    phase_out_path = os.path.join(phase_dir, base_name + "_phase.png")
    phase_normalized = normalize_to_uint8(phase_image)
    cv2.imwrite(phase_out_path, phase_normalized)
    print(f"   Saved phase image: {base_name}_phase.png")

    # Save a copy to testing directory
    testing_out_path = os.path.join(test_dir, base_name + "_phase.png")
    cv2.imwrite(testing_out_path, phase_normalized)
    print(f"   Saved testing copy: {base_name}_phase.png")

    # ── Step 7: Save amplitude image ─────────────────────
    if SAVE_AMPLITUDE:
        amp_out_path = os.path.join(amp_dir, base_name + "_amplitude.png")
        cv2.imwrite(amp_out_path, normalize_to_uint8(amplitude_image))
        print(f"   Saved amplitude image: {base_name}_amplitude.png")

    # ── Step 8: Save comparison plot ─────────────────────
    if SAVE_COMPARISON_PLOT:
        plot_out_path = os.path.join(comp_dir, base_name + "_comparison.png")
        save_comparison_plot(frames, phase_image, amplitude_image,
                              plot_out_path, base_name, metrics)
        print(f"   Saved comparison plot: {base_name}_comparison.png")

    return {
        'name':          base_name,
        'frames':        frames.shape[0],
        'resolution':    f"{frames.shape[2]}x{frames.shape[1]}",
        'quality_score': quality_score,
        'phase_range':   round(float(phase_image.max() - phase_image.min()), 4),
        'amplitude_snr': round(metrics.get('amplitude_snr', 0), 2),
        'output_files': [
            base_name + "_phase.png",
            base_name + "_amplitude.png" if SAVE_AMPLITUDE else None,
            base_name + "_comparison.png" if SAVE_COMPARISON_PLOT else None,
        ]
    }


# =============================================================================
# MAIN — BATCH PROCESSING LOOP
# =============================================================================

def main():
    # ── Setup folders ────────────────────────────────────
    os.makedirs(INPUT_FOLDER,  exist_ok=True)
    os.makedirs(OUTPUT_PHASE_FOLDER, exist_ok=True)
    os.makedirs(TESTING_IMAGE_FOLDER, exist_ok=True)
    if SAVE_AMPLITUDE:
        os.makedirs(OUTPUT_AMPLITUDE_FOLDER, exist_ok=True)
    if SAVE_COMPARISON_PLOT:
        os.makedirs(OUTPUT_COMPARISON_FOLDER, exist_ok=True)
    os.makedirs(LOG_FOLDER,    exist_ok=True)

    print("=" * 60)
    print("PHASE IMAGE BATCH PROCESSOR")
    print("Wind Turbine Blade Defect Detection Project")
    print("=" * 60)
    print(f"Input  folder: {INPUT_FOLDER}")
    print(f"Phase Output : {OUTPUT_PHASE_FOLDER}")
    if SAVE_AMPLITUDE:
        print(f"Amp Output   : {OUTPUT_AMPLITUDE_FOLDER}")
    if SAVE_COMPARISON_PLOT:
        print(f"Comp Output  : {OUTPUT_COMPARISON_FOLDER}")
    print(f"Freq index   : {FREQ_INDEX}")
    print("=" * 60)

    # ── Find all video files ─────────────────────────────

    video_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.gif'))
    ]

    if not video_files:
        print(f"No video files found in input folder: {INPUT_FOLDER}")
        print("Please place your raw thermal videos (.mp4, .avi, .mov, .mkv, .gif) inside the above directory and run this script again.")
        return

    print(f"Found {len(video_files)} video(s). Starting processing...\n")

    # ── Process each video ───────────────────────────────
    results      = []
    failed_files = []

    for i, filename in enumerate(sorted(video_files), 1):
        print(f"[{i}/{len(video_files)}]")

        # Extract base name without extension (used for output filenames)
        base_name  = os.path.splitext(filename)[0]
        video_path = os.path.join(INPUT_FOLDER, filename)

        result = video_to_phase_safe(
            video_path   = video_path,
            phase_dir    = OUTPUT_PHASE_FOLDER,
            amp_dir      = OUTPUT_AMPLITUDE_FOLDER,
            comp_dir     = OUTPUT_COMPARISON_FOLDER,
            test_dir     = TESTING_IMAGE_FOLDER,
            base_name    = base_name,
            freq_index   = FREQ_INDEX
        )

        if result:
            results.append(result)
        else:
            failed_files.append(filename)

    # ── Save processing log ──────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path    = os.path.join(LOG_FOLDER, f"processing_log_{timestamp}.json")
    log_data    = {
        'timestamp':     timestamp,
        'total_videos':  len(video_files),
        'successful':    len(results),
        'failed':        len(failed_files),
        'failed_files':  failed_files,
        'freq_index':    FREQ_INDEX,
        'results':       results
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    # ── Final summary ────────────────────────────────────
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Total videos    : {len(video_files)}")
    print(f"  Successfully processed : {len(results)}")
    print(f"  Failed / skipped       : {len(failed_files)}")
    if failed_files:
        print(f"  Failed files    : {failed_files}")
    print(f"  Log saved to    : {log_path}")
    print(f"  Phase outputs in : {OUTPUT_PHASE_FOLDER}")
    print("=" * 60)

    if results:
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        print(f"\n  Average quality score : {avg_quality:.1f}/100")
        low_quality = [r['name'] for r in results if r['quality_score'] < 50]
        if low_quality:
            print(f"  Low quality images    : {low_quality}")
            print(f"  Tip: Re-run these with freq_index=2 or freq_index=3")

    print("\nPhase images are ready. Next step: upload to Roboflow for annotation.")


if __name__ == "__main__":
    main()
