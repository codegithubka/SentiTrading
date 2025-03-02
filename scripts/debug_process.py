#!/usr/bin/env python3
"""
Simple debug script to test the basic processing of Reddit data.
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Define paths
input_dir = "data/raw/reddit"
output_dir = "data/processed/reddit"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Find the most recent posts file
posts_files = list(Path(input_dir).glob("reddit_posts_*.csv"))
if not posts_files:
    print(f"No Reddit posts files found in {input_dir}")
    sys.exit(1)

# Sort by modification time (most recent first)
posts_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
most_recent_posts = posts_files[0]

print(f"Loading posts from {most_recent_posts}")
posts_df = pd.read_csv(most_recent_posts)
print(f"Loaded {len(posts_df)} posts")

# Simple processing - just copy the data with an added column
posts_df['processed'] = True

# Save to output
output_file = os.path.join(output_dir, "debug_processed_posts.csv")
posts_df.to_csv(output_file, index=False)
print(f"Saved processed data to {output_file}")

print("Debug processing completed successfully")
