import praw
import yaml
import sys
from pathlib import Path

# Load config
config_path = "config/data_sources.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("Using credentials from config file")

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=config['reddit']['client_id'],
    client_secret=config['reddit']['client_secret'],
    user_agent=config['reddit']['user_agent']
)

print("Successfully authenticated")

# Test fetching subreddit
subreddit_name = "wallstreetbets"
subreddit = reddit.subreddit(subreddit_name)
print(f"Accessing subreddit: r/{subreddit_name}")

# Try to fetch some posts
print("Fetching posts...")
count = 0
for post in subreddit.hot(limit=5):
    count += 1
    print(f"{count}. Post: {post.title[:60]}...")
    print(f"   Author: {post.author}")
    print(f"   Score: {post.score}")
    print(f"   Comments: {post.num_comments}")
    print("")

print("Test completed successfully")
