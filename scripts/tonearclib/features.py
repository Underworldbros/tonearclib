import librosa
import numpy as np


def guess_mode(key: int, tonic: int) -> str:
	mode_diff = (key - tonic) % 12
	if mode_diff in [0, 2, 4, 5, 7, 9, 11]:
		return "Major"
	elif mode_diff in [3, 5, 7, 8, 10]:
		return "Minor"
	return "Unknown"


def compute_temporal_centroid(S: np.ndarray) -> float:
	t = np.arange(S.shape[1])
	energy = np.sum(S, axis=0)
	total_energy = np.sum(energy)
	if total_energy == 0:
		return 0.0
	return np.sum(t * energy) / total_energy


def extract_features(file_path: str) -> dict:
	y, sr = librosa.load(file_path, sr=None)
	duration = librosa.get_duration(y=y, sr=sr)

	# Feature extraction
	rms = np.mean(librosa.feature.rms(y=y))
	spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
	spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
	spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
	tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr))
	tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
	beat_count = len(beats)

	# Harmonic analysis
	chroma = librosa.feature.chroma_stft(y=y, sr=sr)
	chroma_mean = np.mean(chroma, axis=1)
	tonic = np.argmax(chroma_mean)
	key = np.argmax(np.sum(chroma, axis=1))
	mode = guess_mode(key, tonic)

	# MFCCs
	mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
	mfccs_mean = np.mean(mfccs, axis=1).tolist()

	# Temporal centroid (rhythmic density over time)
	S = librosa.feature.melspectrogram(y=y, sr=sr)
	temporal_centroid = compute_temporal_centroid(S)

	return {
		"track_name": file_path.split("/")[-1],
		"duration": duration,
		"rms": rms,
		"spectral_centroid": spectral_centroid,
		"spectral_bandwidth": spectral_bandwidth,
		"spectral_contrast": spectral_contrast,
		"tonnetz": tonnetz,
		"tempo": tempo,
		"beat_count": beat_count,
		"chroma_mean": chroma_mean.tolist(),
		"key": key,
		"mode": mode,
		"mfccs_mean": mfccs_mean,
		"temporal_centroid": temporal_centroid
	}
