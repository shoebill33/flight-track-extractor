#!/usr/bin/env python3
"""
Flight Tracker Grid Calibration Tool
==========================
Creates a gridded reference image from a video frame and prompts
user to identify field locations using spreadsheet-style references.

Usage:
    python calibrate_grid.py video.mp4
    python calibrate_grid.py video.mp4 --frame 100
    python calibrate_grid.py video.mp4 --cols 26 --rows 20
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

import cv2
import numpy as np


# Grid configuration
DEFAULT_COLS = 78  # A-BZ (finer grid)
DEFAULT_ROWS = 60  # 1-60 (finer grid)


def col_to_letter(col: int) -> str:
    """Convert column number (0-based) to letter (A, B, ... Z, AA, AB...)."""
    result = ""
    col += 1  # 1-based for calculation
    while col > 0:
        col -= 1
        result = chr(col % 26 + ord('A')) + result
        col //= 26
    return result


def letter_to_col(letter: str) -> int:
    """Convert letter (A, B, ... Z, AA, AB...) to column number (0-based)."""
    result = 0
    for char in letter.upper():
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result - 1


def parse_cell_ref(ref: str) -> Tuple[int, int]:
    """Parse cell reference like 'A1' or 'AB12' into (col, row) 0-based."""
    match = re.match(r'^([A-Za-z]+)(\d+)$', ref.strip())
    if not match:
        raise ValueError(f"Invalid cell reference: {ref}")
    col = letter_to_col(match.group(1))
    row = int(match.group(2)) - 1  # Convert to 0-based
    return (col, row)


def parse_cell_range(range_str: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Parse cell range like 'A1:B2' into ((col1, row1), (col2, row2))."""
    range_str = range_str.strip().upper()
    
    # Handle single cell (treat as 1x1 range)
    if ':' not in range_str:
        cell = parse_cell_ref(range_str)
        return (cell, cell)
    
    parts = range_str.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}")
    
    start = parse_cell_ref(parts[0])
    end = parse_cell_ref(parts[1])
    
    # Normalize so start <= end
    col1, col2 = min(start[0], end[0]), max(start[0], end[0])
    row1, row2 = min(start[1], end[1]), max(start[1], end[1])
    
    return ((col1, row1), (col2, row2))


def grid_range_to_roi(range_tuple: Tuple[Tuple[int, int], Tuple[int, int]], 
                       cols: int, rows: int) -> Tuple[float, float, float, float]:
    """Convert grid range to ROI ratios (x1, y1, x2, y2) as 0.0-1.0."""
    (col1, row1), (col2, row2) = range_tuple
    
    cell_width = 1.0 / cols
    cell_height = 1.0 / rows
    
    x1 = col1 * cell_width
    y1 = row1 * cell_height
    x2 = (col2 + 1) * cell_width  # +1 because we want to include the end cell
    y2 = (row2 + 1) * cell_height
    
    return (x1, y1, x2, y2)


def create_grid_image(frame: np.ndarray, cols: int = DEFAULT_COLS, 
                       rows: int = DEFAULT_ROWS) -> np.ndarray:
    """Overlay a labeled grid on the frame with full cell reference in each cell."""
    h, w = frame.shape[:2]
    output = frame.copy()
    
    cell_width = w / cols
    cell_height = h / rows
    
    # Colors
    grid_color = (0, 255, 255)  # Yellow
    text_color = (255, 255, 255)  # White for edge labels
    cell_label_color = (0, 180, 255)  # Orange for cell labels
    bg_color = (0, 0, 0)  # Black background for labels
    
    # Draw vertical lines
    for i in range(cols + 1):
        x = int(i * cell_width)
        cv2.line(output, (x, 0), (x, h), grid_color, 1)
    
    # Draw horizontal lines
    for i in range(rows + 1):
        y = int(i * cell_height)
        cv2.line(output, (0, y), (w, y), grid_color, 1)
    
    # Draw ALL column labels at top (A, B, C, ... on black background strip)
    cv2.rectangle(output, (0, 0), (w, 20), bg_color, -1)
    for i in range(cols):
        label = col_to_letter(i)
        label_x = int((i + 0.3) * cell_width)
        font_scale = 0.3 if cols > 52 else 0.35
        cv2.putText(output, label, (label_x, 14), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
    
    # Draw ALL row labels on left (1, 2, 3, ... on black background strip)
    cv2.rectangle(output, (0, 20), (25, h), bg_color, -1)
    for i in range(rows):
        label = str(i + 1)
        label_y = int((i + 0.65) * cell_height)
        font_scale = 0.3 if rows > 40 else 0.35
        cv2.putText(output, label, (3, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
    
    # Draw full cell reference (e.g., "B15") inside EVERY cell
    # Adjust font size based on cell dimensions
    min_cell_dim = min(cell_width, cell_height)
    if min_cell_dim < 15:
        font_scale = 0.18
    elif min_cell_dim < 20:
        font_scale = 0.22
    else:
        font_scale = 0.25
    
    for row in range(rows):
        for col in range(cols):
            cell_ref = f"{col_to_letter(col)}{row + 1}"
            
            # Position in bottom-left of cell
            text_x = int(col * cell_width + 1)
            text_y = int((row + 1) * cell_height - 2)
            
            # Draw cell reference
            cv2.putText(output, cell_ref, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, cell_label_color, 1)
    
    # Add instruction text at bottom
    instruction = f"Grid: {cols}x{rows} | Full cell references shown (e.g., B15)"
    cv2.rectangle(output, (0, h - 22), (w, h), bg_color, -1)
    cv2.putText(output, instruction, (10, h - 6), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
    
    return output


def extract_frame(video_path: str, frame_number: int = 0) -> Optional[np.ndarray]:
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    return frame if ret else None


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info


# Fields to calibrate with descriptions
CALIBRATION_FIELDS = [
    ('latitude', 'LATITUDE value (e.g., "6.39329")'),
    ('longitude', 'LONGITUDE value (e.g., "2.3018")'),
    ('timestamp', 'FULL TIMESTAMP display (e.g., "Dec 7, 2025 | 16:53:26 UTC")'),
    ('altitude', 'BAROMETRIC ALTITUDE value (e.g., "22,000 ft")'),
    ('ground_speed', 'GROUND SPEED value (e.g., "140 kts")'),
    ('vertical_speed', 'VERTICAL SPEED value (e.g., "+128 fpm")'),
    ('track_heading', 'TRACK heading value (e.g., "104°")'),
]

# Manual input fields (constant throughout flight)
MANUAL_FIELDS = [
    ('track_name', 'Track/aircraft name (e.g., "Unknown Aircraft Nigeria 1") - for identification'),
    ('icao_address', 'ICAO 24-bit hex address (e.g., "654321") - same for entire flight'),
    ('squawk', 'Squawk code (e.g., "1200") - same for entire flight'),
]


def run_calibration(video_path: str, frame_number: int = 0, 
                    cols: int = DEFAULT_COLS, rows: int = DEFAULT_ROWS) -> dict:
    """Run interactive grid calibration."""
    video_path = Path(video_path)
    
    # Get video info
    info = get_video_info(str(video_path))
    if not info:
        print(f"ERROR: Cannot open video: {video_path}")
        return {}
    
    print(f"\n{'='*60}")
    print(f"Flight Tracker Grid Calibration")
    print(f"{'='*60}")
    print(f"Video: {video_path.name}")
    print(f"Resolution: {info['width']}x{info['height']}")
    print(f"Grid: {cols} columns × {rows} rows")
    print(f"{'='*60}\n")
    
    # Extract frame
    frame = extract_frame(str(video_path), frame_number)
    if frame is None:
        print(f"ERROR: Cannot read frame {frame_number}")
        return {}
    
    # Create calibration folder
    calib_folder = video_path.parent / "calibration"
    calib_folder.mkdir(exist_ok=True)
    
    # Create gridded image
    grid_image = create_grid_image(frame, cols, rows)
    
    # Save grid image to calibration folder
    grid_path = calib_folder / f"{video_path.stem}_grid.png"
    cv2.imwrite(str(grid_path), grid_image)
    print(f"✓ Calibration grid saved: {grid_path}")
    print(f"\n>>> OPEN THIS IMAGE and identify each field's location <<<\n")
    
    # Also save original frame for reference
    orig_path = calib_folder / f"{video_path.stem}_frame.png"
    cv2.imwrite(str(orig_path), frame)
    print(f"✓ Original frame saved: {orig_path}\n")
    
    print(f"{'='*60}")
    print("Enter grid cell ranges for each field.")
    print("Format: A1:B2 (top-left:bottom-right of the VALUE text)")
    print("Press Enter to skip a field, or 'q' to quit.")
    print(f"{'='*60}\n")
    
    # Collect calibration data
    calibration = {
        'video': str(video_path),
        'resolution': [info['width'], info['height']],
        'grid': {'cols': cols, 'rows': rows},
        'frame_number': frame_number,
        'rois': {}
    }
    
    for field_name, description in CALIBRATION_FIELDS:
        while True:
            try:
                response = input(f"{description}\n  → Cell range for '{field_name}': ").strip()
                
                if response.lower() == 'q':
                    print("\nCalibration aborted.")
                    return {}
                
                if not response:
                    print(f"  (skipped)\n")
                    break
                
                # Parse and validate
                range_tuple = parse_cell_range(response)
                roi = grid_range_to_roi(range_tuple, cols, rows)
                
                # Validate range is within grid
                (col1, row1), (col2, row2) = range_tuple
                if col2 >= cols or row2 >= rows:
                    print(f"  ERROR: Range exceeds grid bounds. Max is {col_to_letter(cols-1)}{rows}")
                    continue
                
                calibration['rois'][field_name] = {
                    'grid_range': response.upper(),
                    'roi': roi
                }
                print(f"  ✓ {response.upper()} → ROI: ({roi[0]:.3f}, {roi[1]:.3f}, {roi[2]:.3f}, {roi[3]:.3f})\n")
                break
                
            except ValueError as e:
                print(f"  ERROR: {e}. Try again.\n")
    
    # Ask for manual fields (constants for entire flight)
    print(f"\n{'='*60}")
    print("Manual entry for constant values (same throughout flight)")
    print("Press Enter to skip, or 'q' to quit.")
    print(f"{'='*60}\n")
    
    calibration['manual'] = {}
    for field_name, description in MANUAL_FIELDS:
        response = input(f"{description}\n  → Value for '{field_name}': ").strip()
        
        if response.lower() == 'q':
            print("\nCalibration aborted.")
            return {}
        
        if response:
            calibration['manual'][field_name] = response.upper() if field_name == 'icao_address' else response
            print(f"  ✓ Saved: {calibration['manual'][field_name]}\n")
        else:
            print(f"  (skipped)\n")
    
    # Save calibration config (system file for extraction)
    if calibration['rois']:
        calib_folder = video_path.parent / "calibration"
        calib_folder.mkdir(exist_ok=True)
        config_path = calib_folder / f"{video_path.stem}_calibration.json"
        with open(config_path, 'w') as f:
            json.dump(calibration, f, indent=2)
    
    return calibration


def load_calibration(config_path: str) -> dict:
    """Load calibration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def calibration_to_roi_config(calibration: dict):
    """Convert calibration dict to ROIConfig values."""
    rois = {}
    for field_name, data in calibration.get('rois', {}).items():
        rois[field_name] = tuple(data['roi'])
    return rois


def main():
    parser = argparse.ArgumentParser(
        description='Grid-based ROI calibration for flight track extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s video.mp4
  %(prog)s video.mp4 --frame 100
  %(prog)s video.mp4 --cols 30 --rows 25
        '''
    )
    
    parser.add_argument('video', nargs='?', help='Video file path')
    parser.add_argument('--frame', '-f', type=int, default=50,
                        help='Frame number to use for calibration (default: 50)')
    parser.add_argument('--cols', '-c', type=int, default=DEFAULT_COLS,
                        help=f'Number of grid columns (default: {DEFAULT_COLS})')
    parser.add_argument('--rows', '-r', type=int, default=DEFAULT_ROWS,
                        help=f'Number of grid rows (default: {DEFAULT_ROWS})')
    
    args = parser.parse_args()
    
    # If no video provided, ask user
    video_path = args.video
    if not video_path:
        print("=" * 60)
        print("Flight Track Extractor - Calibration")
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
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)
    
    calibration = run_calibration(
        video_path, 
        frame_number=args.frame,
        cols=args.cols,
        rows=args.rows
    )
    
    if calibration:
        print("\n" + "=" * 60)
        print("✅ Calibration complete!")
        print("=" * 60)
        
        # Ask if user wants to proceed with extraction
        print("\nDo you want to proceed with extraction? (Y/N)")
        response = input("→ ").strip().lower()
        
        if response in ['y', 'yes', '']:
            print("\n" + "=" * 60)
            print("Starting extraction...")
            print("=" * 60)
            
            # Import and run extraction
            from flight_extract import FlightTrackExtractor, ROIConfig
            import json
            
            # Load calibration
            calib_folder = Path(video_path).parent / "calibration"
            config_path = calib_folder / f"{Path(video_path).stem}_calibration.json"
            
            with open(config_path, 'r') as f:
                calib_data = json.load(f)
            
            # Setup ROI config
            roi_config = ROIConfig()
            for field_name, data in calib_data.get('rois', {}).items():
                if hasattr(roi_config, field_name):
                    setattr(roi_config, field_name, tuple(data['roi']))
            
            manual_values = calib_data.get('manual', {})
            
            # Setup output directory
            output_dir = Path(video_path).parent / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Run extraction
            extractor = FlightTrackExtractor(
                video_path=video_path,
                output_dir=str(output_dir),
                sample_interval=2.0,
                roi_config=roi_config
            )
            
            extractor.manual_track_name = manual_values.get('track_name')
            extractor.manual_icao = manual_values.get('icao_address')
            extractor.manual_squawk = manual_values.get('squawk')
            
            points = extractor.extract()
            
            # Show report
            valid_points = [p for p in points if p.is_valid_position()]
            valid_count = len(valid_points)
            
            print(f"\n{'='*60}")
            print(f"📊 EXTRACTION REPORT")
            print(f"{'='*60}")
            
            print(f"\n🎬 Video: {Path(video_path).name}")
            
            if manual_values:
                print(f"\n🎯 Calibration:")
                if manual_values.get('track_name'):
                    print(f"   Track name:  {manual_values['track_name']}")
                if manual_values.get('icao_address'):
                    print(f"   ICAO:        {manual_values['icao_address']}")
                if manual_values.get('squawk'):
                    print(f"   Squawk:      {manual_values['squawk']}")
            
            print(f"\n📈 Extraction results:")
            print(f"   Total samples:     {len(points)}")
            print(f"   Valid positions:   {valid_count} ({100*valid_count/len(points):.1f}%)")
            
            if valid_points:
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
                
                timestamps = [p.timestamp_utc for p in valid_points if p.timestamp_utc]
                if timestamps:
                    print(f"\n⏱️  Time range: {timestamps[0]} to {timestamps[-1]}")
            
            # Save outputs
            print(f"\n📁 Output files (in {output_dir}):")
            path = extractor.save_csv()
            print(f"   {Path(path).name}")
            print(f"      └─ Check 'issues' column for flagged data")
            path = extractor.save_kml()
            print(f"   {Path(path).name}")
            alt_f = getattr(extractor, '_kml_alt_filtered', 0)
            if alt_f > 0:
                median = getattr(extractor, '_kml_median_alt', 0)
                print(f"      └─ {alt_f} impossible altitudes corrected (median={median} ft)")
            path = extractor.save_geojson()
            print(f"   {Path(path).name}")
            
            print(f"\n{'='*60}")
            print(f"✅ Done!")
        else:
            print("\nTo run extraction later:")
            print(f'  python flight_extract.py "{video_path}" -f all')


if __name__ == '__main__':
    main()
