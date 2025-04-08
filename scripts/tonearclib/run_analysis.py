import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
from tonearclib import analyze_track
from tonearclib.reporting import save_json, generate_pdf_report


def main():
	parser = argparse.ArgumentParser(description="Run ToneArcLib analysis on a WAV/MP3 file.")
	parser.add_argument("filepath", help="Path to audio file")
	parser.add_argument("--out", help="Output directory", default="./output")
	args = parser.parse_args()

	if not os.path.exists(args.filepath):
		print("[ERROR] File does not exist:", args.filepath)
		return

	os.makedirs(args.out, exist_ok=True)
	profile = analyze_track(args.filepath)

	json_path = os.path.join(args.out, profile.track_name + ".json")
	pdf_path = os.path.join(args.out, profile.track_name + ".pdf")

	save_json(profile, json_path)
	generate_pdf_report(profile, pdf_path)

	print(f"\nAnalysis complete for {profile.track_name}")
	print(f"Saved: {json_path}\n       {pdf_path}")


if __name__ == "__main__":
	main()
