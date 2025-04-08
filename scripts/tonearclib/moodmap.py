def tag_mood(profile: dict) -> str:
	"""
	Assign a simple mood tag based on extracted audio features.
	You may replace this with a trained model or rule system in the future.
	"""
	energy = profile.get("rms", 0)
	tempo = profile.get("tempo", 0)
	mode = profile.get("mode", "Unknown")
	spectral_centroid = profile.get("spectral_centroid", 0)
	temporal_centroid = profile.get("temporal_centroid", 0)

	if mode == "Major" and tempo > 120 and energy > 0.05:
		return "Energetic"
	elif mode == "Minor" and energy < 0.04 and spectral_centroid < 2500:
		return "Somber"
	elif spectral_centroid > 3500 and temporal_centroid > 0.6:
		return "Bright"
	elif tempo < 90 and mode == "Minor":
		return "Melancholic"
	elif energy > 0.07 and tempo > 130:
		return "Intense"
	else:
		return "Neutral"