#!/usr/bin/env python3
"""
Flight Track Extractor
====================
Extracts aircraft track data (timestamp, lat, lon, altitude, speed) from
flight tracking platform screen recordings using OCR.

Author: Built for OSINT research
Version: 1.0.0
"""

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

# Conditional imports with helpful error messages
try:
    import pytesseract
except ImportError:
    print("ERROR: pytesseract not installed. Run: pip install pytesseract")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ROIConfig:
    """Region of Interest configuration for flight tracker UI elements.
    
    All coordinates are ratios (0.0 to 1.0) of frame dimensions.
    Calibrated for flight tracker UI at 1914x914 resolution (actual recording).
    """
    # Timeline panel - full timestamp display (bottom center of screen)
    # "Dec 7, 2025 | 16:53:26 UTC"
    timestamp: Tuple[float, float, float, float] = (0.310, 0.700, 0.420, 0.740)
    
    # Left sidebar - coordinates section (very bottom of panel, above toolbar)
    # Values are BELOW the labels in flight tracker layout
    # Left panel is ~13.5% of width (0 to 0.135)
    latitude: Tuple[float, float, float, float] = (0.008, 0.730, 0.068, 0.765)
    longitude: Tuple[float, float, float, float] = (0.070, 0.730, 0.135, 0.765)
    
    # Left sidebar - altitude and speed (mid-panel)
    # BAROMETRIC ALT row
    altitude: Tuple[float, float, float, float] = (0.008, 0.430, 0.068, 0.465)
    vertical_speed: Tuple[float, float, float, float] = (0.070, 0.430, 0.135, 0.465)
    
    # GROUND SPEED row (below Speed & Altitude graph section)
    ground_speed: Tuple[float, float, float, float] = (0.008, 0.530, 0.068, 0.565)
    
    # TRACK row (same row as GPS ALTITUDE)
    track_heading: Tuple[float, float, float, float] = (0.070, 0.460, 0.135, 0.495)
    
    # Left sidebar - identification (above coordinates)
    # ICAO 24-BIT ADDRESS and SQUAWK row
    icao_address: Tuple[float, float, float, float] = (0.008, 0.695, 0.068, 0.730)
    squawk: Tuple[float, float, float, float] = (0.070, 0.695, 0.135, 0.730)


@dataclass
class ExtractedPoint:
    """Single extracted data point."""
    frame_number: int
    timestamp_utc: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_ft: Optional[int] = None
    ground_speed_kt: Optional[int] = None
    vertical_speed_fpm: Optional[int] = None
    track_deg: Optional[int] = None
    icao_hex: Optional[str] = None
    squawk: Optional[str] = None
    track_name: Optional[str] = None
    ocr_confidence: float = 0.0
    source: str = "screen_OCR"
    
    def is_valid_position(self) -> bool:
        """Check if we have valid lat/lon."""
        return (
            self.latitude is not None and 
            self.longitude is not None and
            -90 <= self.latitude <= 90 and
            -180 <= self.longitude <= 180
        )


# =============================================================================
# OCR PREPROCESSING
# =============================================================================

class OCRPreprocessor:
    """Image preprocessing for optimal OCR results."""
    
    @staticmethod
    def preprocess_for_ocr(image: np.ndarray, dark_bg: bool = True) -> np.ndarray:
        """Preprocess image region for OCR.
        
        Args:
            image: BGR image region
            dark_bg: True if text is light on dark background (flight tracker default)
        
        Returns:
            Preprocessed grayscale image optimized for OCR
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Invert if dark background (white text on dark)
        if dark_bg:
            gray = cv2.bitwise_not(gray)
        
        # Resize for better OCR (scale up small text)
        scale = 3
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        binary = cv2.medianBlur(binary, 3)
        
        return binary
    
    @staticmethod
    def preprocess_timestamp(image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for timestamp region."""
        return OCRPreprocessor.preprocess_for_ocr(image, dark_bg=True)
    
    @staticmethod
    def preprocess_coordinates(image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for coordinate values."""
        return OCRPreprocessor.preprocess_for_ocr(image, dark_bg=True)


# =============================================================================
# OCR ENGINE
# =============================================================================

class FlightOCREngine:
    """OCR engine specialized for flight tracker UI elements."""
    
    # Tesseract configurations for different field types
    CONFIG_TIMESTAMP = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:'
    CONFIG_DATE = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz, '
    CONFIG_COORDINATE = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-'
    CONFIG_INTEGER = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-'
    CONFIG_HEX = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFabcdef'
    
    def __init__(self):
        self.preprocessor = OCRPreprocessor()
    
    def extract_timestamp(self, image: np.ndarray) -> Optional[str]:
        """Extract full timestamp (date + time) from combined region.
        
        Expected format: "Dec 7, 2025 | 16:25:25 UTC" or similar
        Returns: "2025-12-07T16:25:25" or None
        """
        scale = 4
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Month mapping
        months = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        # Try multiple preprocessing methods
        preprocessed_images = []
        
        # OTSU thresholding (best for dark bg with white text)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(cv2.resize(otsu, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        
        # Inverted OTSU
        preprocessed_images.append(cv2.resize(cv2.bitwise_not(otsu), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        
        # Simple scaling
        preprocessed_images.append(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        
        for img in preprocessed_images:
            text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
            
            # Handle "|" separator and OCR errors (- instead of :)
            text = text.replace('-', ':').replace('|', ' ')
            
            # Extract date: "Dec 7, 2025" or similar
            date_match = re.search(r'([A-Za-z]{3})\s*(\d{1,2}),?\s*(\d{4})', text)
            
            # Extract time: "HH:MM:SS"
            time_match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})', text)
            
            if date_match and time_match:
                month_str = date_match.group(1).lower()
                day = int(date_match.group(2))
                year = int(date_match.group(3))
                
                h = int(time_match.group(1))
                m = int(time_match.group(2))
                s = int(time_match.group(3))
                
                # Validate
                if (month_str in months and 1 <= day <= 31 and 2020 <= year <= 2030 and
                    0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
                    date_str = f"{year}-{months[month_str]}-{day:02d}"
                    time_str = f"{h:02d}:{m:02d}:{s:02d}"
                    return f"{date_str}T{time_str}"
        
        return None
    
    def extract_coordinate(self, image: np.ndarray) -> Optional[float]:
        """Extract decimal coordinate value."""
        # Scale up significantly for small text
        scale = 5
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Use PSM 6 (block of text) which works better for these images
        # Also try PSM 11 (sparse text) as backup
        configs = [
            '--oem 3 --psm 6',   # Block of text - best for multi-line
            '--oem 3 --psm 11',  # Sparse text
            '--oem 3 --psm 4',   # Single column of text
        ]
        
        for config in configs:
            text = pytesseract.image_to_string(scaled, config=config)
            
            # Clean up common OCR errors
            text = text.replace(',', '.').replace('O', '0').replace('o', '0')
            text = text.replace('l', '1').replace('I', '1').replace('|', '1')
            text = text.replace('S', '5').replace('s', '5')
            
            # Find all decimal numbers in the text (coordinate format: X.XXXXX)
            matches = re.findall(r'-?\d+\.\d{3,6}', text)
            
            for match in matches:
                try:
                    value = float(match)
                    # Validate as coordinate
                    if -180 <= value <= 180:
                        return value
                except ValueError:
                    continue
        
        return None
    
    def extract_integer(self, image: np.ndarray, field_type: str = 'generic') -> Optional[int]:
        """Extract integer value (altitude, speed, etc.)."""
        scale = 4
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Prepare multiple preprocessing variants
        preprocessed_images = []
        
        # OTSU thresholding (best for mixed backgrounds)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(cv2.resize(otsu, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        
        # Inverted OTSU
        preprocessed_images.append(cv2.resize(cv2.bitwise_not(otsu), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        
        # Simple scaling
        preprocessed_images.append(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        
        # Try each preprocessing method
        for scaled in preprocessed_images:
            text = pytesseract.image_to_string(scaled, config='--oem 3 --psm 6')
            
            # Handle N/A
            if 'N/A' in text.upper() or 'N/ A' in text.upper():
                return None
            
            # Clean up commas
            text_clean = text.replace(',', '')
            
            # Try to extract based on field type
            result = self._parse_integer_by_type(text_clean, field_type, scaled)
            if result is not None:
                return result
        
        return None
    
    def _parse_integer_by_type(self, text_clean: str, field_type: str, scaled_img: np.ndarray) -> Optional[int]:
        """Parse integer from text based on field type."""
        
        if field_type == 'speed':
            # Look for number before "kt" or "kts"
            match = re.search(r'(\d+)\s*kt', text_clean, re.IGNORECASE)
            if match:
                speed = int(match.group(1))
                # Sanity check: if > 1000, likely OCR noise prepended
                if speed > 1000:
                    speed_str = str(speed)
                    speed = int(speed_str[1:]) if len(speed_str) > 1 else speed
                return speed if speed <= 600 else None
                
        elif field_type == 'altitude':
            # Remove common noise from icons (arrows, +, =, etc.)
            text_clean = re.sub(r'[+\-=<>^v|]', '', text_clean)
            
            # Look for number before "ft" - handle comma/period/space in thousands (22,000 ft or 22.000 ft)
            # Pattern for XX,XXX ft or XX.XXX ft or XXXXX ft
            match = re.search(r'(\d{1,3})[\s,.]?(\d{3})\s*ft', text_clean, re.IGNORECASE)
            if match:
                alt_str = match.group(1) + match.group(2)
                return int(alt_str)
            
            # Fallback: simple number before ft
            match = re.search(r'(\d+)\s*ft', text_clean, re.IGNORECASE)
            if match:
                return int(match.group(1))
                
        elif field_type == 'vertical_speed':
            # Look for number before "fpm" (can be negative or zero)
            # Also handle OCR errors like "fom" for "fpm"
            match = re.search(r'([+-]?\d+)\s*f[op]m', text_clean, re.IGNORECASE)
            if match:
                return int(match.group(1))
                
        elif field_type == 'heading':
            # Look for number before "°" 
            match = re.search(r'(\d{1,3})\s*[°]', text_clean)
            if match:
                val = int(match.group(1))
                if 0 <= val <= 360:
                    return val
            
            # Fallback: find any 2-3 digit number (likely heading)
            matches = re.findall(r'\b(\d{2,3})\b', text_clean)
            for m in matches:
                val = int(m)
                if 0 <= val <= 360:
                    return val
            
            # Last resort: any number 0-360
            matches = re.findall(r'\b(\d{1,3})\b', text_clean)
            for m in matches:
                val = int(m)
                if 0 <= val <= 360:
                    return val
        
        # Generic fallback: find largest reasonable number
        text_clean = text_clean.replace('O', '0').replace('o', '0')
        text_clean = text_clean.replace('l', '1').replace('I', '1').replace('|', '1')
        
        matches = re.findall(r'-?\d+', text_clean)
        valid_numbers = []
        for match in matches:
            try:
                value = int(match)
                if len(match) >= 2:
                    valid_numbers.append(value)
            except ValueError:
                continue
        
        if valid_numbers:
            return max(valid_numbers, key=lambda x: abs(x))
        
        return None
    
    def extract_hex(self, image: np.ndarray) -> Optional[str]:
        """Extract hex value (ICAO address)."""
        processed = self.preprocessor.preprocess_coordinates(image)
        text = pytesseract.image_to_string(processed, config=self.CONFIG_HEX).strip()
        
        # Clean and validate hex
        text = text.replace(' ', '').upper()
        match = re.search(r'[0-9A-F]{6}', text)
        if match:
            return match.group()
        return None
    
    def extract_squawk(self, image: np.ndarray) -> Optional[str]:
        """Extract squawk code (4 digits)."""
        processed = self.preprocessor.preprocess_coordinates(image)
        text = pytesseract.image_to_string(processed, config=self.CONFIG_INTEGER).strip()
        
        match = re.search(r'\d{4}', text.replace(' ', ''))
        if match:
            code = match.group()
            # Validate squawk range (0000-7777 octal)
            if all(c in '01234567' for c in code):
                return code
        return None


# =============================================================================
# FRAME EXTRACTOR
# =============================================================================

class FrameExtractor:
    """Extract and process frames from video."""
    
    def __init__(self, video_path: str, sample_interval: float = 2.0):
        """
        Args:
            video_path: Path to video file
            sample_interval: Seconds between frame samples
        """
        self.video_path = video_path
        self.sample_interval = sample_interval
        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.duration = 0
        self.width = 0
        self.height = 0
    
    def open(self) -> bool:
        """Open video file and get properties."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return True
    
    def close(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
    
    def get_frame_numbers(self) -> List[int]:
        """Get list of frame numbers to sample."""
        frame_interval = int(self.fps * self.sample_interval)
        if frame_interval < 1:
            frame_interval = 1
        return list(range(0, self.total_frames, frame_interval))
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get specific frame by number."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def extract_roi(self, frame: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract region of interest from frame.
        
        Args:
            frame: Full frame image
            roi: (x1, y1, x2, y2) as ratios 0.0-1.0
        
        Returns:
            Cropped image region
        """
        h, w = frame.shape[:2]
        x1 = int(roi[0] * w)
        y1 = int(roi[1] * h)
        x2 = int(roi[2] * w)
        y2 = int(roi[3] * h)
        
        # Ensure valid bounds
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        return frame[y1:y2, x1:x2]


# =============================================================================
# MAIN EXTRACTOR
# =============================================================================

class FlightTrackExtractor:
    """Main class for extracting track data from flight tracker recordings."""
    
    def __init__(self, video_path: str, output_dir: str = None, 
                 sample_interval: float = 2.0, roi_config: ROIConfig = None,
                 debug: bool = False, debug_frame: int = 0):
        """
        Args:
            video_path: Path to video file
            output_dir: Output directory (default: same as video)
            sample_interval: Seconds between samples
            roi_config: ROI configuration (default: standard standard 1080p)
            debug: Save debug images of ROI extractions
            debug_frame: Which frame to save debug images for
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent
        self.sample_interval = sample_interval
        self.roi_config = roi_config or ROIConfig()
        self.debug = debug
        self.debug_frame = debug_frame
        
        self.frame_extractor = FrameExtractor(str(self.video_path), sample_interval)
        self.ocr_engine = FlightOCREngine()
        
        self.extracted_points: List[ExtractedPoint] = []
        self.diagnostics: List[Dict] = []
        
        # Manual values (set from calibration config)
        self.manual_track_name: Optional[str] = None
        self.manual_icao: Optional[str] = None
        self.manual_squawk: Optional[str] = None
        
        # Create debug directory if needed
        if self.debug:
            self.debug_dir = self.output_dir / "debug_rois"
            self.debug_dir.mkdir(exist_ok=True)
    
    def extract(self, progress_callback=None) -> List[ExtractedPoint]:
        """Run extraction on video.
        
        Args:
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            List of extracted data points
        """
        if not self.frame_extractor.open():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        try:
            print(f"Video: {self.video_path.name}")
            print(f"Resolution: {self.frame_extractor.width}x{self.frame_extractor.height}")
            print(f"Duration: {self.frame_extractor.duration:.1f}s ({self.frame_extractor.total_frames} frames)")
            print(f"FPS: {self.frame_extractor.fps:.2f}")
            print(f"Sample interval: {self.sample_interval}s")
            print()
            
            frame_numbers = self.frame_extractor.get_frame_numbers()
            print(f"Processing {len(frame_numbers)} frames...")
            
            for i, frame_num in enumerate(tqdm(frame_numbers, desc="Extracting")):
                frame = self.frame_extractor.get_frame(frame_num)
                if frame is None:
                    continue
                
                point = self._extract_from_frame(frame, frame_num)
                self.extracted_points.append(point)
                
                if progress_callback:
                    progress_callback(i + 1, len(frame_numbers))
            
            # Post-process: validate and clean data
            self._postprocess()
            
            return self.extracted_points
            
        finally:
            self.frame_extractor.close()
    
    def _extract_from_frame(self, frame: np.ndarray, frame_number: int) -> ExtractedPoint:
        """Extract all data from a single frame."""
        point = ExtractedPoint(frame_number=frame_number)
        confidence_scores = []
        
        # Debug: save full frame and all ROIs
        save_debug = self.debug and (frame_number == self.debug_frame or self.debug_frame < 0)
        if save_debug:
            cv2.imwrite(str(self.debug_dir / f"frame_{frame_number}_full.png"), frame)
        
        # Extract timestamp (combined date + time)
        ts_roi = self.frame_extractor.extract_roi(frame, self.roi_config.timestamp)
        if save_debug:
            cv2.imwrite(str(self.debug_dir / f"frame_{frame_number}_timestamp.png"), ts_roi)
        point.timestamp_utc = self.ocr_engine.extract_timestamp(ts_roi)
        
        if point.timestamp_utc:
            confidence_scores.append(1.0)
        else:
            confidence_scores.append(0.0)
        
        # Extract coordinates
        lat_roi = self.frame_extractor.extract_roi(frame, self.roi_config.latitude)
        if save_debug:
            cv2.imwrite(str(self.debug_dir / f"frame_{frame_number}_latitude.png"), lat_roi)
        point.latitude = self.ocr_engine.extract_coordinate(lat_roi)
        if point.latitude and -90 <= point.latitude <= 90:
            confidence_scores.append(1.0)
        else:
            confidence_scores.append(0.0)
        
        lon_roi = self.frame_extractor.extract_roi(frame, self.roi_config.longitude)
        if save_debug:
            cv2.imwrite(str(self.debug_dir / f"frame_{frame_number}_longitude.png"), lon_roi)
        point.longitude = self.ocr_engine.extract_coordinate(lon_roi)
        if point.longitude and -180 <= point.longitude <= 180:
            confidence_scores.append(1.0)
        else:
            confidence_scores.append(0.0)
        
        # Extract altitude
        alt_roi = self.frame_extractor.extract_roi(frame, self.roi_config.altitude)
        if save_debug:
            cv2.imwrite(str(self.debug_dir / f"frame_{frame_number}_altitude.png"), alt_roi)
        point.altitude_ft = self.ocr_engine.extract_integer(alt_roi, field_type='altitude')
        
        # Extract ground speed
        spd_roi = self.frame_extractor.extract_roi(frame, self.roi_config.ground_speed)
        if save_debug:
            cv2.imwrite(str(self.debug_dir / f"frame_{frame_number}_ground_speed.png"), spd_roi)
        point.ground_speed_kt = self.ocr_engine.extract_integer(spd_roi, field_type='speed')
        
        # Extract vertical speed
        vs_roi = self.frame_extractor.extract_roi(frame, self.roi_config.vertical_speed)
        if save_debug:
            cv2.imwrite(str(self.debug_dir / f"frame_{frame_number}_vertical_speed.png"), vs_roi)
        point.vertical_speed_fpm = self.ocr_engine.extract_integer(vs_roi, field_type='vertical_speed')
        
        # Extract track heading
        trk_roi = self.frame_extractor.extract_roi(frame, self.roi_config.track_heading)
        if save_debug:
            cv2.imwrite(str(self.debug_dir / f"frame_{frame_number}_track.png"), trk_roi)
        point.track_deg = self.ocr_engine.extract_integer(trk_roi, field_type='heading')
        
        # Use manual values (constant for entire flight)
        if self.manual_track_name:
            point.track_name = self.manual_track_name
        
        if self.manual_icao:
            point.icao_hex = self.manual_icao
        
        if self.manual_squawk:
            point.squawk = self.manual_squawk
        
        # Calculate overall confidence
        point.ocr_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        if save_debug:
            print(f"\nDebug frame {frame_number} saved to {self.debug_dir}")
            print(f"  Timestamp: {time_str} | Date: {date_str}")
            print(f"  Lat: {point.latitude} | Lon: {point.longitude}")
            print(f"  Alt: {point.altitude_ft} | GS: {point.ground_speed_kt}")
            print(f"  ICAO: {point.icao_hex} | Squawk: {point.squawk}")
        
        return point
    
    def _postprocess(self):
        """Clean and validate extracted data."""
        if not self.extracted_points:
            return
        
        # Remove duplicate timestamps
        seen_timestamps = set()
        unique_points = []
        for point in self.extracted_points:
            if point.timestamp_utc and point.timestamp_utc not in seen_timestamps:
                seen_timestamps.add(point.timestamp_utc)
                unique_points.append(point)
            elif not point.timestamp_utc:
                unique_points.append(point)
        
        self.extracted_points = unique_points
        
        # Sort by timestamp
        self.extracted_points.sort(key=lambda p: p.timestamp_utc or "")
        
        # Flag suspicious jumps (speed > 600 kt between points)
        for i in range(1, len(self.extracted_points)):
            prev = self.extracted_points[i-1]
            curr = self.extracted_points[i]
            
            if prev.is_valid_position() and curr.is_valid_position():
                # Rough distance calculation
                dlat = curr.latitude - prev.latitude
                dlon = curr.longitude - prev.longitude
                dist_nm = ((dlat * 60)**2 + (dlon * 60 * 0.7)**2)**0.5  # Approximate
                
                # If distance > 20nm in 2s interval, flag as suspicious
                if dist_nm > 20:
                    self.diagnostics.append({
                        'frame': curr.frame_number,
                        'issue': 'position_jump',
                        'details': f"Large jump: {dist_nm:.1f} nm"
                    })
    
    def save_csv(self, filename: str = None) -> Path:
        """Save extracted data to CSV with issues column for data review."""
        if filename is None:
            filename = self.video_path.stem + "_track.csv"
        
        output_path = self.output_dir / filename
        
        valid_points = [p for p in self.extracted_points if p.is_valid_position()]
        
        # Calculate median altitude to detect OCR errors
        altitudes = [p.altitude_ft for p in valid_points if p.altitude_ft and p.altitude_ft > 0]
        median_alt = sorted(altitudes)[len(altitudes) // 2] if altitudes else 0
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'track_name', 'timestamp_utc', 'latitude', 'longitude', 'altitude_ft',
                'ground_speed_kt', 'vertical_speed_fpm', 'track_deg',
                'icao_hex', 'squawk', 'issues'
            ])
            
            for point in valid_points:
                issues = []
                
                # Flag clearly impossible altitudes (likely OCR errors)
                # e.g., 87 ft when median is 7750 ft
                if point.altitude_ft and median_alt > 5000 and point.altitude_ft < 500:
                    issues.append(f"ALT_CHECK({point.altitude_ft}ft)")
                
                writer.writerow([
                    point.track_name or '',
                    point.timestamp_utc or '',
                    f"{point.latitude:.6f}" if point.latitude is not None else '',
                    f"{point.longitude:.6f}" if point.longitude is not None else '',
                    point.altitude_ft if point.altitude_ft is not None else '',
                    point.ground_speed_kt if point.ground_speed_kt is not None else '',
                    point.vertical_speed_fpm if point.vertical_speed_fpm is not None else '',
                    point.track_deg if point.track_deg is not None else '',
                    point.icao_hex or '',
                    point.squawk or '',
                    '; '.join(issues) if issues else ''
                ])
        
        print(f"Saved CSV: {output_path}")
        return output_path
    
    def save_kml(self, filename: str = None) -> Path:
        """Save extracted data to KML for Google Earth."""
        if filename is None:
            filename = self.video_path.stem + "_track.kml"
        
        output_path = self.output_dir / filename
        
        valid_points = [p for p in self.extracted_points if p.is_valid_position()]
        
        if not valid_points:
            print("Warning: No valid points to save to KML")
            return output_path
        
        # Get track name
        track_name = valid_points[0].track_name if valid_points[0].track_name else self.video_path.stem
        
        # Calculate median altitude for OCR error detection
        altitudes = [p.altitude_ft for p in valid_points if p.altitude_ft and p.altitude_ft > 0]
        median_alt = sorted(altitudes)[len(altitudes) // 2] if altitudes else 0
        
        # Build coordinates - only filter clearly impossible altitudes
        coord_lines = []
        last_good_alt = median_alt
        alt_filtered = 0
        
        for point in valid_points:
            lat = point.latitude
            lon = point.longitude
            alt_ft = point.altitude_ft if point.altitude_ft else 0
            
            # Filter clearly impossible altitudes (OCR errors like 87 instead of 7875)
            use_alt = alt_ft
            if median_alt > 5000 and alt_ft > 0 and alt_ft < 500:
                use_alt = last_good_alt
                alt_filtered += 1
            elif alt_ft > 0:
                last_good_alt = alt_ft
            
            alt_m = use_alt * 0.3048
            coord_lines.append(f"          {lon:.6f},{lat:.6f},{alt_m:.0f}")
        
        # Store for report
        self._kml_alt_filtered = alt_filtered
        self._kml_median_alt = median_alt
        
        coords_str = "\n".join(coord_lines)
        
        # KML with blue line, width 3, 50% opacity (7f = 127 = 50%)
        # KML color format: aabbggrr (alpha, blue, green, red)
        # Blue with 50% opacity: 7fff0000
        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{track_name}</name>
    <Style id="trackStyle">
      <LineStyle>
        <color>7fff0000</color>
        <width>3</width>
      </LineStyle>
      <PolyStyle>
        <color>40ffffff</color>
        <fill>1</fill>
        <outline>0</outline>
      </PolyStyle>
    </Style>
    <Placemark>
      <name>{track_name}</name>
      <description>Points: {len(valid_points)}</description>
      <styleUrl>#trackStyle</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>
{coords_str}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""
        
        with open(output_path, 'w') as f:
            f.write(kml_content)
        
        return output_path


    def save_geojson(self, filename: str = None) -> Path:
        """Save extracted data to GeoJSON."""
        if filename is None:
            filename = self.video_path.stem + "_track.geojson"
        
        output_path = self.output_dir / filename
        
        valid_points = [p for p in self.extracted_points if p.is_valid_position()]
        
        # Get track name from first point or use video name
        track_name = valid_points[0].track_name if valid_points and valid_points[0].track_name else self.video_path.stem
        
        # Build GeoJSON structure
        coordinates = []
        for point in valid_points:
            alt = point.altitude_ft * 0.3048 if point.altitude_ft else 0
            coordinates.append([point.longitude, point.latitude, alt])
        
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": track_name,
                        "source": "screen_OCR",
                        "points_count": len(valid_points)
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    }
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Saved GeoJSON: {output_path}")
        return output_path
    


# =============================================================================
# GOOGLE DRIVE SUPPORT
# =============================================================================

def download_from_gdrive(file_id: str, output_path: str) -> bool:
    """Download file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        output_path: Local path to save file
    
    Returns:
        True if successful
    """
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown not installed. Run: pip install gdown")
        return False
    
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive: {file_id}")
    
    try:
        gdown.download(url, output_path, quiet=False)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def parse_gdrive_url(url: str) -> Optional[str]:
    """Extract file ID from Google Drive URL."""
    # Handle various Google Drive URL formats
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'^([a-zA-Z0-9_-]{25,})$'  # Direct file ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract flight track data from flight tracker screen recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s video.mp4                    # Auto-detects video_calibration.json
  %(prog)s video.mp4 -c config.json     # Specify calibration file
  %(prog)s video.mp4 -f all             # Output CSV, KML, and GeoJSON
  %(prog)s video.mp4 -i 5               # Sample every 5 seconds
        '''
    )
    
    parser.add_argument('input', nargs='?', help='Video file path')
    parser.add_argument('--config', '-c', 
                        help='Calibration JSON file (auto-detected if not specified)')
    parser.add_argument('--output', '-o', help='Output directory (default: same as video)')
    parser.add_argument('--interval', '-i', type=float, default=2.0,
                        help='Sample interval in seconds (default: 2.0)')
    parser.add_argument('--formats', '-f', nargs='+', default=['csv'],
                        choices=['csv', 'kml', 'geojson', 'all'],
                        help='Output formats (default: csv)')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run grid calibration before extraction')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug images of ROI extractions')
    parser.add_argument('--debug-frame', type=int, default=0,
                        help='Frame number for debug output (default: 0)')
    parser.add_argument('--version', '-v', action='version', 
                        version='Flight Track Extractor v1.0.0')
    
    args = parser.parse_args()
    
    # If no input provided, ask user interactively
    video_path = args.input
    if not video_path:
        print("=" * 60)
        print("Flight Track Extractor")
        print("=" * 60)
        print("\nPlease enter the full path to your video file.")
        print("\nTip - Copy file path:")
        print("  Windows: Shift + Right-click file → 'Copy as path'")
        print("  Mac:     Right-click file → Hold Option → 'Copy as Pathname'")
        print("")
        video_path = input("Video file path: ").strip()
        # Remove quotes if user pasted with them
        video_path = video_path.strip('"').strip("'")
    
    if not video_path:
        print("ERROR: No video file specified")
        sys.exit(1)
    
    # Validate input
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)
    
    # Setup organized output directories
    video_folder = Path(video_path).parent
    video_stem = Path(video_path).stem
    
    # Create output subfolder for extraction results
    output_dir = args.output or str(video_folder / "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calibration folder
    calib_folder = video_folder / "calibration"
    
    # Run calibration if requested
    if args.calibrate:
        try:
            from calibrate_grid import run_calibration
            calibration = run_calibration(video_path, frame_number=50)
            if not calibration:
                sys.exit(1)
            config_path = calib_folder / f"{video_stem}_calibration.json"
            args.config = str(config_path)
        except ImportError:
            print("ERROR: calibrate_grid.py not found in same directory")
            sys.exit(1)
    
    # Auto-detect calibration file if not specified
    if not args.config:
        # Try organized structure first, then legacy locations
        possible_configs = [
            calib_folder / f"{video_stem}_calibration.json",
            calib_folder / "calibration.json",
            video_folder / f"{video_stem}_calibration.json",
            video_folder / f"{video_stem}.calibration.json",
            video_folder / "calibration.json",
        ]
        
        for config_path in possible_configs:
            if config_path.exists():
                args.config = str(config_path)
                print(f"Found calibration: {config_path}")
                break
        
        if not args.config:
            print("No calibration file found.")
            print(f"   Run: python calibrate_grid.py")
            print("   Then paste your video path when prompted.")
            sys.exit(1)
    
    # Load ROI configuration
    roi_config = ROIConfig()
    manual_values = {}
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                calib_data = json.load(f)
            
            # Apply calibration ROIs to config
            rois = calib_data.get('rois', {})
            for field_name, data in rois.items():
                if hasattr(roi_config, field_name):
                    setattr(roi_config, field_name, tuple(data['roi']))
            
            # Load manual values
            manual_values = calib_data.get('manual', {})
            
            print(f"Loaded calibration: {args.config}")
            print(f"  Fields configured: {', '.join(rois.keys())}")
            if manual_values:
                print(f"  Manual values: {', '.join(f'{k}={v}' for k, v in manual_values.items())}")
        except Exception as e:
            print(f"WARNING: Could not load config {args.config}: {e}")
            print("  Using default ROI positions")
    
    # Run extraction
    extractor = FlightTrackExtractor(
        video_path=video_path,
        output_dir=output_dir,
        sample_interval=args.interval,
        roi_config=roi_config,
        debug=args.debug,
        debug_frame=args.debug_frame if args.debug else -1
    )
    
    # Set manual values
    extractor.manual_track_name = manual_values.get('track_name')
    extractor.manual_icao = manual_values.get('icao_address')
    extractor.manual_squawk = manual_values.get('squawk')
    
    try:
        points = extractor.extract()
        
        # Calculate statistics
        valid_points = [p for p in points if p.is_valid_position()]
        valid_count = len(valid_points)
        
        print(f"\n{'='*60}")
        print(f"📊 EXTRACTION REPORT")
        print(f"{'='*60}")
        
        # Video properties
        print(f"\n🎬 Video: {Path(video_path).name}")
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            print(f"   Resolution:  {width}x{height}")
            print(f"   Duration:    {duration/60:.1f} minutes")
        except:
            pass
        
        # Calibration info
        if manual_values:
            print(f"\n🎯 Calibration:")
            if manual_values.get('track_name'):
                print(f"   Track name:  {manual_values['track_name']}")
            if manual_values.get('icao_address'):
                print(f"   ICAO:        {manual_values['icao_address']}")
            if manual_values.get('squawk'):
                print(f"   Squawk:      {manual_values['squawk']}")
        
        # Extraction results
        print(f"\n📈 Extraction results:")
        print(f"   Total samples:     {len(points)}")
        print(f"   Valid positions:   {valid_count} ({100*valid_count/len(points):.1f}%)")
        
        if valid_points:
            # Field success rates
            ts_ok = sum(1 for p in valid_points if p.timestamp_utc)
            alt_ok = sum(1 for p in valid_points if p.altitude_ft and p.altitude_ft > 0)
            spd_ok = sum(1 for p in valid_points if p.ground_speed_kt is not None)
            trk_ok = sum(1 for p in valid_points if p.track_deg is not None)
            vs_ok = sum(1 for p in valid_points if p.vertical_speed_fpm is not None)
            
            print(f"\n📋 Field success rates:")
            print(f"   Timestamp:       {ts_ok}/{valid_count} ({100*ts_ok/valid_count:.1f}%)")
            print(f"   Altitude:        {alt_ok}/{valid_count} ({100*alt_ok/valid_count:.1f}%)")
            print(f"   Ground speed:    {spd_ok}/{valid_count} ({100*spd_ok/valid_count:.1f}%)")
            print(f"   Track heading:   {trk_ok}/{valid_count} ({100*trk_ok/valid_count:.1f}%)")
            print(f"   Vertical speed:  {vs_ok}/{valid_count} ({100*vs_ok/valid_count:.1f}%)")
            
            # Time range
            timestamps = [p.timestamp_utc for p in valid_points if p.timestamp_utc]
            if timestamps:
                print(f"\n⏱️  Time range: {timestamps[0]} to {timestamps[-1]}")
        
        # Save outputs
        formats = args.formats
        if 'all' in formats:
            formats = ['csv', 'kml', 'geojson']
        
        print(f"\n📁 Output files (in {output_dir}):")
        if 'csv' in formats:
            path = extractor.save_csv()
            print(f"   {Path(path).name}")
            print(f"      └─ Check 'issues' column for flagged data")
        if 'kml' in formats:
            path = extractor.save_kml()
            print(f"   {Path(path).name}")
            alt_f = getattr(extractor, '_kml_alt_filtered', 0)
            if alt_f > 0:
                median = getattr(extractor, '_kml_median_alt', 0)
                print(f"      └─ {alt_f} impossible altitudes corrected (median={median} ft)")
        if 'geojson' in formats:
            path = extractor.save_geojson()
            print(f"   {Path(path).name}")
        
        print(f"\n{'='*60}")
        print(f"✅ Done!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
