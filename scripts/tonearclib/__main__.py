import argparse
import os
import json
from tonearclib import analyze_track
from tonearclib.reporting import generate_pdf_report

def main():
	parser = argparse.ArgumentParser(description="ToneArcLib: Semantic Audio Analysis")
	parser.add_argument("filepath", help="Path to input audio file (.wav)")
	parser.add_argument("--extended", action="store_true", help="Return full analysis as JSON-compatible dict")
	parser.add_argument("--out", type=str, help="Optional output path or directory for JSON or PDF report")

	args = parser.parse_args()
	filepath = args.filepath

	if not os.path.isfile(filepath):
		print(f"[ERROR] File not found: {filepath}")
		exit(1)

	try:
		mode = "extended" if args.extended else "standard"
		result = analyze_track(filepath, mode=mode)

		filename = os.path.splitext(os.path.basename(filepath))[0]
		ext = ".json" if mode == "extended" else ".pdf"

		if args.out:
			if os.path.isdir(args.out) or args.out.endswith(("/", "\\")):
				os.makedirs(args.out, exist_ok=True)
				out_path = os.path.join(args.out, f"{filename}{ext}")
			else:
				os.makedirs(os.path.dirname(args.out), exist_ok=True)
				out_path = args.out

			if mode == "extended":
				with open(out_path, "w", encoding="utf-8") as f:
					json.dump(result, f, indent=2, default=str)
				print(f"[OK] JSON saved to {out_path}")
			else:
				generate_pdf_report(result, out_path)
				print(f"[OK] PDF report saved to {out_path}")
		else:
			print(json.dumps(result, indent=2) if mode == "extended" else result)

	except Exception as e:
		print(f"[FATAL] {type(e).__name__}: {e}")
		exit(1)

if __name__ == "__main__":
	main()
