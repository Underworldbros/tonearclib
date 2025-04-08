from dataclasses import dataclass
from typing import List

@dataclass
class TrackProfile:
	track_name: str
	duration: float
	rms: float
	spectral_centroid: float
	spectral_bandwidth: float
	spectral_contrast: float
	tonnetz: float
	tempo: float
	beat_count: int

	chroma_mean: List[float]
	key: int
	mode: str

	mfccs_mean: List[float]
	temporal_centroid: float
