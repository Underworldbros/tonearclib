from .features import extract_features
from .models import TrackProfile

def analyze_track(file_path: str, mode: str = 'standard') -> TrackProfile | dict:
	"""
	Analyze a track and return either a TrackProfile or full extended dictionary.

	Args:
		file_path (str): Path to audio file
		mode (str): 'standard' (returns TrackProfile) or 'extended' (returns dict)

	Returns:
		TrackProfile or dict: depending on mode
	"""
	profile = extract_features(file_path)

	if mode == 'extended':
		if isinstance(profile, dict):
			return {
				"track": profile.get("track_name", "unknown"),
				"duration": profile.get("duration"),
				"sample_rate": profile.get("sample_rate"),
				"key": profile.get("key"),
				"bpm": profile.get("tempo"),
				"sections": profile.get("sections", []),
				"timbre": profile.get("spectral_centroid"),
				"tonality": profile.get("mode"),
				"rhythm": {
					"beat_count": profile.get("beat_count"),
					"temporal_centroid": profile.get("temporal_centroid")
				},
				"mood_tag": profile.get("mood_tag", "Undefined"),
				"notes": profile.get("notes", [])
			}
		else:
			return {
				"track": profile.track_name,
				"duration": profile.duration,
				"sample_rate": profile.sample_rate,
				"key": profile.key,
				"bpm": profile.bpm,
				"sections": profile.sections,
				"timbre": profile.timbre_profile,
				"tonality": profile.tonality,
				"rhythm": profile.rhythmic_features,
				"mood_tag": profile.mood_tag,
				"notes": profile.notes
			}

		return profile
