from __future__ import annotations
from typing import Tuple, Optional

def _fuzzy_match_class(target: str, detected_class: str, recognized_name: Optional[str] = None) -> bool:
	"""
	Fuzzy matching for object class names with improved logic.
	Supports person name matching for face recognition.
	
	Args:
		target: Target class name (e.g., "cup", "reshad", "person")
		detected_class: Detected class name (e.g., "drinking cup", "mug", "cup", "person")
		recognized_name: Optional recognized person name (for face recognition)
	
	Returns:
		True if classes match, False otherwise
	"""
	target_lower = str(target).lower().strip()
	detected_lower = str(detected_class).lower().strip()
	
	# Check recognized name for person queries
	if recognized_name:
		recognized_lower = str(recognized_name).lower().strip()
		# If target is a person name, match against recognized_name
		if target_lower == recognized_lower:
			return True
		# Also check if target matches recognized name (substring match)
		if target_lower in recognized_lower or recognized_lower in target_lower:
			return True
	
	# Exact match
	if target_lower == detected_lower:
		return True
	
	# Plural/singular matching (bottles <-> bottle, cups <-> cup)
	# Convert both to singular for comparison
	target_singular = target_lower.rstrip('s') if target_lower.endswith('s') and len(target_lower) > 3 else target_lower
	detected_singular = detected_lower.rstrip('s') if detected_lower.endswith('s') and len(detected_lower) > 3 else detected_lower
	if target_singular == detected_singular and len(target_singular) > 2:
		return True
	
	# Special case: if target is a person name and detected class is "person", 
	# check if recognized_name matches (handled above)
	if detected_lower == "person" and recognized_name:
		if target_lower == str(recognized_name).lower().strip():
			return True
	
	# Substring match (target in detected or detected in target)
	if target_lower in detected_lower or detected_lower in target_lower:
		return True
	
	# Word-level matching (e.g., "cup" matches "drinking cup")
	target_words = set(target_lower.split())
	detected_words = set(detected_lower.split())
	
	# If any word from target is in detected words (and vice versa)
	if target_words and detected_words:
		if target_words.intersection(detected_words):
			return True
	
	# Common synonyms/alternatives
	synonyms = {
		"cup": ["mug", "drinking cup", "coffee cup", "tea cup"],
		"bottle": ["water bottle", "drink bottle"],
		"phone": ["mobile phone", "cell phone", "smartphone"],
		"laptop": ["notebook", "computer"],
		"keyboard": ["computer keyboard"],
		"mouse": ["computer mouse"],
		"pen": ["ballpoint pen", "pencil"],
		"book": ["notebook", "textbook"],
		"person": ["people", "human", "man", "woman", "person"],
	}
	
	# Check if target matches any synonym of detected class
	for syn_list in synonyms.values():
		if target_lower in syn_list and detected_lower in syn_list:
			return True
		if detected_lower in syn_list and target_lower in syn_list:
			return True
	
	return False

def angles_to_direction(angle_x: float, angle_y: float) -> str:
	"""
	Convert angle_x (horizontal) and angle_y (vertical) to human-friendly directions.
	
	Args:
		angle_x: Horizontal angle in degrees (positive = right, negative = left)
		angle_y: Vertical angle in degrees (positive = up, negative = down)
	
	Returns:
		Human-friendly direction string like "front left", "top right", "exactly in front"
	"""
	# Thresholds for direction classification (in degrees)
	FRONT_THRESHOLD = 15  # Within ±15° is considered "front"
	SIDE_THRESHOLD = 45   # Beyond ±45° is considered strong left/right
	VERTICAL_THRESHOLD = 30  # Beyond ±30° is considered up/down
	
	# Determine horizontal direction
	if abs(angle_x) <= FRONT_THRESHOLD:
		horizontal = "front"
	elif angle_x > SIDE_THRESHOLD:
		horizontal = "right"
	elif angle_x < -SIDE_THRESHOLD:
		horizontal = "left"
	elif angle_x > 0:
		horizontal = "front-right"
	else:
		horizontal = "front-left"
	
	# Determine vertical direction
	if abs(angle_y) <= VERTICAL_THRESHOLD:
		vertical = ""
	elif angle_y > VERTICAL_THRESHOLD:
		vertical = "top"
	else:
		vertical = "down"
	
	# Combine directions
	if abs(angle_x) <= FRONT_THRESHOLD and abs(angle_y) <= VERTICAL_THRESHOLD:
		return "exactly in front"
	elif vertical:
		if horizontal == "front":
			return f"{vertical}"
		else:
			return f"{vertical} {horizontal}"
	else:
		return horizontal

def format_location_response(object_name: str, distance: float, angle_x: float, angle_y: float) -> str:
	"""
	Format a location response with human-friendly directions.
	
	Args:
		object_name: Name of the object
		distance: Distance in meters
		angle_x: Horizontal angle in degrees
		angle_y: Vertical angle in degrees
	
	Returns:
		Formatted location string like "Pen is about 1.2 meters away, exactly in front"
	"""
	direction = angles_to_direction(angle_x, angle_y)
	
	if distance <= 0:
		return f"There is a {object_name} {direction}."
	else:
		return f"There is a {object_name} about {distance:.1f} meters away, {direction}."
