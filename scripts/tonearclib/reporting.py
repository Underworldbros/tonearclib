import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
from tonearclib.models import TrackProfile
import os

def get_track_name(profile):
	if isinstance(profile, dict):
		return profile.get("track", "Unknown")
	return get_track_name(profile)

def save_json(profile: TrackProfile, out_path: str):
	with open(out_path, "w") as f:
		json.dump(profile.__dict__, f, indent=2)

def plot_summary(profile: TrackProfile, out_path: str):
	plt.figure(figsize=(10, 6))
	plt.title(f"{get_track_name(profile)} – Feature Overview")
	plt.bar(["RMS", "Tempo", "Spectral Centroid"],
			[profile.rms, profile.tempo, profile.spectral_centroid],
			color="skyblue")
	plt.ylabel("Value")
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()

def generate_pdf_report(profile: TrackProfile, out_path: str):
	with PdfPages(out_path) as pdf:
		plt.figure(figsize=(10, 5))
		plt.title(f"MFCC Mean Coefficients – {get_track_name(profile)}")
		plt.plot(profile.mfccs_mean, marker='o')
		plt.xlabel("Coefficient")
		plt.ylabel("Value")
		plt.grid(True)
		pdf.savefig()
		plt.close()

		plt.figure(figsize=(8, 4))
		plt.title(f"Chroma Profile – {get_track_name(profile)}")
		plt.bar(range(12), profile.chroma_mean, color='orchid')
		plt.xlabel("Pitch Class")
		plt.ylabel("Mean Value")
		pdf.savefig()
		plt.close()

		plt.figure(figsize=(6, 3))
		plt.title("Basic Feature Metrics")
		plt.bar(["RMS", "Tempo", "Centroid"], [profile.rms, profile.tempo, profile.spectral_centroid])
		pdf.savefig()
		plt.close()