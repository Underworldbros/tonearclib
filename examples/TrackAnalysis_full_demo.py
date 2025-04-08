import os, json
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import sys
import time
import argparse
from sklearn.preprocessing import minmax_scale
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import entropy
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description="Run full audio analysis and report.")
parser.add_argument("file", help="Path to the .wav file")
parser.add_argument("--out", help="Output path or folder for the results", default=None)
args = parser.parse_args()

file_path = args.file
output_path = args.out


########################################################
#  HELPER FUNCTIONS  #
########################################################

def update_progress(step, total, label):
    percent = int((step / total) * 100)
    bar = ('#' * (percent // 2)).ljust(50)
    sys.stdout.write(f"\r[{bar}] {percent}% - {label}")
    sys.stdout.flush()


def guess_mode(chroma_vector, root_note_index):
    """
    Quick heuristic to guess scale mode given a chroma vector and a root index.
    We'll compare how well the chroma aligns with mode intervals.
    """
    # Mode definitions (major, minor, phrygian, etc.) are approximate.
    # Each mode is a set of semitones from the root.
    # We'll do a simple correlation approach.

    # Notes in semitones for some common modes (Ionian = major, Aeolian = minor, etc.)
    mode_templates = {
        "Ionian (Major)":        [0, 2, 4, 5, 7, 9, 11],
        "Dorian":                [0, 2, 3, 5, 7, 9, 10],
        "Phrygian":              [0, 1, 3, 5, 7, 8, 10],
        "Lydian":                [0, 2, 4, 6, 7, 9, 11],
        "Mixolydian":            [0, 2, 4, 5, 7, 9, 10],
        "Aeolian (Natural Minor)": [0, 2, 3, 5, 7, 8, 10],
        "Locrian":               [0, 1, 3, 5, 6, 8, 10]
    }

    # We'll rotate the chroma vector so root_note_index is at 0
    # That way, we can compare to base mode patterns.
    rotated = np.roll(chroma_vector, -root_note_index)

    best_mode = None
    best_score = -np.inf

    for mode_name, semitones in mode_templates.items():
        # We'll build a mask that has 1 for each semitone in the mode, 0 otherwise.
        # Then measure the sum of rotated[mask] as a crude alignment score.
        mode_mask = np.zeros(12)
        for st in semitones:
            mode_mask[st % 12] = 1
        score = np.sum(rotated * mode_mask)
        if score > best_score:
            best_score = score
            best_mode = mode_name

    return best_mode


def compute_peak_moments(rms_curve, spec_cent, times_rms, times_cent):
    """
    Identify the times of maximum RMS and spectral centroid.
    Return them along with their values.
    """
    # Find index of max RMS
    idx_rms_peak = np.argmax(rms_curve)
    rms_peak_val = rms_curve[idx_rms_peak]
    rms_peak_time = times_rms[idx_rms_peak]

    # Find index of max centroid
    idx_cent_peak = np.argmax(spec_cent)
    cent_peak_val = spec_cent[idx_cent_peak]
    cent_peak_time = times_cent[idx_cent_peak]

    return (rms_peak_time, rms_peak_val, cent_peak_time, cent_peak_val)


def compute_temporal_centroid(rms_curve, times_rms):
    """
    Weighted average of times by RMS amplitude.
    If the track is front-loaded, the centroid < mid-time.
    If it's back-loaded, centroid > mid-time.
    """
    total_energy = np.sum(rms_curve)
    if total_energy < 1e-12:
        return 0.0
    tc = np.sum(rms_curve * times_rms) / total_energy
    return tc


def segment_by_features(rms_curve, spec_cent, n_segments=2):
    """
    Simple segmenter using KMeans on (RMS, centroid) per frame.
    Returns a label array for each frame.
    """
    # Combine RMS + Centroid into feature vectors.
    # We'll do log scale for RMS to reduce dynamic range.
    # We'll also do log scale for centroid.

    eps = 1e-12
    f_rms = np.log1p(rms_curve + eps)
    f_cent = np.log1p(spec_cent + eps)

    X = np.column_stack((f_rms, f_cent))
    kmeans = KMeans(n_clusters=n_segments, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X)

    return labels


########################################################
#  SCRIPT BODY  #
########################################################

# Load the track
y, sr = librosa.load(file_path, sr=None)

update_progress(1, 14, "Calculating basic stats")

# Basic Stats
duration = librosa.get_duration(y=y, sr=sr)
rms = np.sqrt(np.mean(y**2))
peak = np.max(np.abs(y))
dynamic_range = np.max(y) - np.min(y)

# Check if Stereo
is_stereo = False
if len(y.shape) == 2 and y.shape[0] == 2:
    is_stereo = True

update_progress(2, 14, "Separating percussive for tempo analysis")
harmonic, percussive = librosa.effects.hpss(y)

tempo, beats = librosa.beat.beat_track(y=percussive, sr=sr, start_bpm=120, trim=True, units='time')
from librosa.feature.rhythm import tempo as tempo_feature

tempo_confidence = tempo_feature(y=percussive, sr=sr, aggregate=None)

update_progress(3, 14, "Detecting onsets")
onsets = librosa.onset.onset_detect(y=percussive, sr=sr, backtrack=True, units='time')

if len(beats) > 1000:
    beats = beats[:1000]
if len(onsets) > 1000:
    onsets = onsets[:1000]

update_progress(4, 14, "Computing spectral features")
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spec_cent_times = librosa.times_like(spectral_centroid, sr=sr)
spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

update_progress(5, 14, "Extracting loudness curve")
frame_length = 2048
hop_length = 512
rms_curve = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times_rms = librosa.times_like(rms_curve, sr=sr, hop_length=hop_length)

update_progress(6, 14, "Calculating zero-crossing rate")
zcr = librosa.feature.zero_crossing_rate(y)[0]
zcr_mean = np.mean(zcr)

update_progress(7, 14, "Generating MFCCs + Deltas")
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc, axis=1)
mfcc_delta = librosa.feature.delta(mfcc)
mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
mfcc_std = np.std(mfcc, axis=1)

update_progress(8, 14, "Estimating key and tonnetz")
chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
chroma_mean = np.mean(chroma, axis=1)
key_index = np.argmax(chroma_mean)
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
estimated_root = note_names[key_index]
tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

# Attempt a mode guess
best_mode = guess_mode(chroma_mean, key_index)

update_progress(9, 14, "Detecting silence intervals")
intervals = librosa.effects.split(y, top_db=20)
silence_durations = [(start / sr, end / sr) for start, end in intervals]

update_progress(10, 14, "Exporting features")
tempo_val = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
tempo_conf = float(np.max(tempo_confidence))
track_name = file_path.split("/")[-1]

onset_env = librosa.onset.onset_strength(y=percussive, sr=sr)
beat_strength = np.mean(onset_env[librosa.time_to_frames(beats, sr=sr)]) if len(beats) > 0 else 0

h_energy = np.sum(harmonic**2)
p_energy = np.sum(percussive**2)
hpr_ratio = h_energy / (h_energy + p_energy) if (h_energy + p_energy) > 0 else 0

tonnetz_spread = np.var(tonnetz)

chroma_std = np.std(chroma, axis=1)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
rolloff_mean = np.mean(rolloff)

crest_factor = peak / rms if rms != 0 else 0

beats_per_sec = len(beats) / duration if duration > 0 else 0
onsets_per_sec = len(onsets) / duration if duration > 0 else 0
silence_per_sec = len(silence_durations) / duration if duration > 0 else 0

if onsets_per_sec < 0.5 or beat_strength < 0.05:
    tempo_val = 0.0
    tempo_conf = 0.0

# Spectral Entropy
spec_entropy = entropy(spectral_flatness + 1e-12)

# Symbolic Mood Tag
if rolloff_mean < 3000 and np.mean(spectral_flatness) < 0.01:
    mood_tag = "Dark / Still / Tonal"
elif rms > 0.4 and crest_factor < 4:
    mood_tag = "Compressed / Forward / Present"
else:
    mood_tag = "Neutral / Mixed"

# ========== Additional Features ==========

update_progress(11, 14, "Analyzing track arcs")

# 1) Peak Moments
rms_peak_time, rms_peak_val, cent_peak_time, cent_peak_val = compute_peak_moments(rms_curve, spectral_centroid, times_rms, spec_cent_times)

# 2) Temporal Centroid
time_centroid = compute_temporal_centroid(rms_curve, times_rms)

# 3) Basic KMeans Segmentation
labels = segment_by_features(rms_curve, spectral_centroid, n_segments=2)

# 4) If stereo, do mid/side ratio
mid_side_ratio = None
if is_stereo:
    # y shape: (2, n_samples)
    L = y[0]
    R = y[1]
    mid = (L + R) / 2.0
    side = (L - R) / 2.0
    mid_power = np.sum(mid**2)
    side_power = np.sum(side**2)
    if (mid_power + side_power) > 0:
        mid_side_ratio = mid_power / (mid_power + side_power)
    else:
        mid_side_ratio = 1.0

update_progress(12, 14, "Combining features")

features = {
    'track_name': os.path.splitext(os.path.basename(file_path))[0],
    'duration': duration,
    'rms': rms,
    'peak': peak,
    'dynamic_range': dynamic_range,
    'tempo': tempo_val,
    'tempo_confidence': tempo_conf,
    'key_root': estimated_root,
    'key_mode': best_mode,
    'zcr_mean': zcr_mean,
    'spectral_bandwidth_mean': np.mean(spec_bandwidth),
    'spectral_flatness_mean': np.mean(spectral_flatness),
    'mfcc_1': mfcc_mean[0],
    'mfcc_2': mfcc_mean[1],
    'mfcc_3': mfcc_mean[2],
    'mfcc_4': mfcc_mean[3],
    'mfcc_5': mfcc_mean[4],
    'crest_factor': crest_factor,
    'beat_strength': beat_strength,
    'harmonic_ratio': hpr_ratio,
    'tonnetz_spread': tonnetz_spread,
    'rolloff_mean': rolloff_mean,
    'chroma_std_1': chroma_std[0],
    'chroma_std_2': chroma_std[1],
    'chroma_std_3': chroma_std[2],
    'beats_per_sec': beats_per_sec,
    'onsets_per_sec': onsets_per_sec,
    'silence_per_sec': silence_per_sec,
    'spectral_entropy': spec_entropy,
    'mood_tag': mood_tag,
    'rms_peak_time': rms_peak_time,
    'rms_peak_val': rms_peak_val,
    'cent_peak_time': cent_peak_time,
    'cent_peak_val': cent_peak_val,
    'temporal_centroid': time_centroid,
    'is_stereo': is_stereo,
    'mid_side_ratio': mid_side_ratio if mid_side_ratio is not None else 'N/A'
}

update_progress(13, 14, "Exporting to CSV")

df = pd.DataFrame([features])
print("\n[INFO] Feature data exported to audio_features.csv")

update_progress(14, 14, "Creating PDF report")

def convert_numpy_types(obj):
	if isinstance(obj, (np.integer, int)):
		return int(obj)
	elif isinstance(obj, (np.floating, float, np.float64)):
		return float(obj)
	elif isinstance(obj, (np.ndarray,)):
		return obj.tolist()
	return obj

# Output JSON
if output_path:
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(
        output_path,
        f"{features['track_name']}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({k: convert_numpy_types(v) for k, v in features.items()}, f, indent=2)
    print(f"[OK] JSON report saved to {output_file}")
else:
    print(json.dumps({k: convert_numpy_types(v) for k, v in features.items()}, indent=2))

########################################################
# PDF EXPORT
########################################################
pdf_path = os.path.join(output_path if output_path else '.', f"{features['track_name']}_report.pdf")
with PdfPages(pdf_path) as pdf:
    table_data = [["Track Name", track_name]]

    numeric_items = [(k, v) for k, v in features.items()
                     if k not in ('track_name','key_root','key_mode','mood_tag','is_stereo','mid_side_ratio')
                     and isinstance(v, (int, float, np.floating, np.integer))]
    if numeric_items:
        max_item = max(numeric_items, key=lambda x: x[1])
        min_item = min(numeric_items, key=lambda x: x[1])
    else:
        max_item = min_item = (None, None)

    # Build table data
    for key, value in features.items():
        if key == 'track_name':
            continue
        if isinstance(value, (int, float, np.floating, np.integer)):
            label = "↑ Peak" if (key, value) == max_item else ("↓ Valley" if (key, value) == min_item else "")
            table_data.append([f"{key} {label}".strip(), str(round(value, 5))])
        else:
            table_data.append([key, str(value)])

    # Create the figure for the table
    table_height = len(table_data) * 0.3 + 0.4
    # Clamp the figure height so it doesn't blow up if table_data is huge
    table_height = min(table_height, 10)
    fig, ax = plt.subplots(figsize=(8.5, table_height))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=["Feature", "Value"],
        loc='center'
    )
    # Tweak font size
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Scale: (width_scale, height_scale)
    # Reducing below 1.0 for height_scale shrinks row height
    table.scale(1, 0.9)

    # Use tight_layout and bbox_inches to reduce whitespace
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ========== Plot 1: Spectral Centroid over Time ==========
    plt.figure(figsize=(14, 4))
    plt.semilogy(spec_cent_times, spectral_centroid, color='magenta')
    plt.ylabel('Spectral Centroid (Hz)')
    plt.title("Spectral Brightness (Centroid) Over Time")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # ========== Plot 2: Energy Curve (RMS) ==========
    plt.figure(figsize=(14, 4))
    plt.plot(times_rms, rms_curve, color='orange')
    plt.title("Energy Curve Over Time (RMS)")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Energy")
    # Mark the RMS peak
    plt.axvline(rms_peak_time, color='red', linestyle='--', label=f'RMS Peak @ {rms_peak_time:.2f}s')
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # ========== Plot 3: Chroma Frequencies ==========
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=sr)
    plt.colorbar()
    plt.title('Chroma Frequencies (Key & Tonal Shift Indicator)')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # ========== Plot 4: Waveform + Beats/Onsets ==========
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.vlines(beats, -1, 1, color='r', linestyle='--', label='Beats')
    plt.vlines(onsets, -1, 1, color='g', linestyle=':', label='Onsets')
    plt.title("Waveform with Beats and Onsets")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # ========== Plot 5: Detected Non-Silent Intervals ==========
    plt.figure(figsize=(14, 2))
    for start, end in silence_durations:
        plt.axvspan(start, end, color='gray', alpha=0.4)
    plt.title("Detected Non-Silent Intervals")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # ========== Plot 6: Onset Strength Envelope ==========
    plt.figure(figsize=(14, 4))
    onset_times = librosa.times_like(onset_env, sr=sr)
    plt.plot(onset_times, onset_env, color='blue')
    plt.title("Onset Strength Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Strength")
    pdf.savefig()
    plt.close()

    # ========== Plot 7: Spectral Flatness Over Time ==========
    plt.figure(figsize=(14, 4))
    flat_times = librosa.times_like(spectral_flatness, sr=sr)
    plt.plot(flat_times, spectral_flatness, color='purple')
    plt.title("Spectral Flatness Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Flatness")
    pdf.savefig()
    plt.close()

    # ========== Plot 8: Spectral Rolloff ==========
    plt.figure(figsize=(14, 4))
    roll_times = librosa.times_like(rolloff, sr=sr)
    plt.plot(roll_times, rolloff, color='green')
    plt.title("Spectral Rolloff (85%)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    pdf.savefig()
    plt.close()

    # ========== Plot 9: Chroma Std Dev ==========
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(chroma_std)), chroma_std, color='teal')
    plt.title("Chroma Standard Deviation (Key Variance)")
    plt.xlabel("Chroma Bin")
    plt.ylabel("Std Dev")
    pdf.savefig()
    plt.close()

    # ========== Plot 10: MFCC Heatmap ==========
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title("MFCC Heatmap")
    pdf.savefig()
    plt.close()

    # ========== Plot 11: Segment-based Plot (KMeans) ==========
    # We'll color the frames by their cluster label.
    labels_times = librosa.times_like(rms_curve, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(14, 4))
    plt.title("Segment-based Clustering (2 groups)")
    plt.xlabel("Time (s)")
    plt.ylabel("log(RMS)")
    X_rms = np.log1p(rms_curve+1e-12)
    plt.scatter(labels_times, X_rms, c=labels, cmap='cool')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("\n✅ [INFO] Combined PDF report saved: audio_analysis_report.pdf")

########################################################
#  PRINTING STATS
########################################################
print("\n⚡️ SUPER ANALYZER v3 RESULTS ⚡️")
print(f"Duration: {duration:.2f} seconds")
print(f"RMS Loudness: {rms:.4f}")
print(f"Peak Amplitude: {peak:.4f}")
print(f"Dynamic Range: {dynamic_range:.4f}")
print(f"Estimated Tempo: {tempo_val:.2f} BPM (Confidence: {tempo_conf:.2f})")
print(f"Estimated Key Root: {estimated_root}")
print(f"Likely Mode: {best_mode}")
print(f"Total Beats Detected: {len(beats)}")
print(f"Total Onsets Detected: {len(onsets)}")
print(f"Silence Sections: {len(silence_durations)}")
print(f"Mean ZCR: {zcr_mean:.4f}")
print(f"Spectral Bandwidth Mean: {np.mean(spec_bandwidth):.2f}")
print(f"Spectral Flatness Mean: {np.mean(spectral_flatness):.4f}")
print(f"MFCC Mean Coefficients: {mfcc_mean.round(2)}")
print(f"MFCC Std Dev: {mfcc_std.round(2)}")
print(f"Tonal Centroids Shape: {tonnetz.shape}")
print(f"Spectral Entropy: {spec_entropy:.4f}")
print(f"Mood Tag: {mood_tag}")

print("\n[ Peak Moments ]")
print(f"RMS Peak at {rms_peak_time:.2f}s = {rms_peak_val:.4f}")
print(f"Centroid Peak at {cent_peak_time:.2f}s = {cent_peak_val:.2f} Hz")

print("\n[ Temporal Distribution ]")
print(f"Temporal Centroid: {time_centroid:.2f} s (track mid ~ {duration/2:.2f} s)")

if is_stereo:
    print("\n[ Stereo Analysis ]")
    print(f"Mid/Side Ratio: {mid_side_ratio:.3f} (1.0 = all mid, 0.0 = all side)")
else:
    print("\n[ Stereo Analysis ] Mono track detected.")

# Done.
