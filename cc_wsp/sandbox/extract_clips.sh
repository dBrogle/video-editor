#!/bin/bash
# Extract hook segments and copy body clips into per-habit folders
set -e

BASE="cc_wsp/videos"
HOOKS="$BASE/bcai_top_habits/IMG_4039.mov"
HOOKS2="$BASE/bcai_top_habits/IMG_4038.MOV"

# Create folders and extract hooks from IMG_4039 (best takes = last take per habit)
# Adding 0.5s padding on each side for editing room

# Habit 22 - Delegation (39.71 - 52.68)
mkdir -p "$BASE/bcai_top_habits_22"
ffmpeg -y -ss 39.2 -to 53.2 -i "$HOOKS" -c copy "$BASE/bcai_top_habits_22/hook.mov" 2>/dev/null
cp "$BASE/bcai_top_habits/IMG_4040.mov" "$BASE/bcai_top_habits_22/body.mov"
echo "✓ bcai_top_habits_22 (Delegation)"

# Habit 21 - Financial Discipline (69.88 - 81.88)
mkdir -p "$BASE/bcai_top_habits_21"
ffmpeg -y -ss 69.4 -to 82.4 -i "$HOOKS" -c copy "$BASE/bcai_top_habits_21/hook.mov" 2>/dev/null
cp "$BASE/bcai_top_habits/IMG_4041.mov" "$BASE/bcai_top_habits_21/body.mov"
echo "✓ bcai_top_habits_21 (Financial Discipline)"

# Habit 20 - Daily Spiritual Practice (86.04 - 98.04)
mkdir -p "$BASE/bcai_top_habits_20"
ffmpeg -y -ss 85.5 -to 98.5 -i "$HOOKS" -c copy "$BASE/bcai_top_habits_20/hook.mov" 2>/dev/null
cp "$BASE/bcai_top_habits/IMG_4042.mov" "$BASE/bcai_top_habits_20/body.mov"
echo "✓ bcai_top_habits_20 (Daily Spiritual Practice)"

# Habit 19 - Daily Social Connection (110.62 - 124.07)
mkdir -p "$BASE/bcai_top_habits_19"
ffmpeg -y -ss 110.1 -to 124.6 -i "$HOOKS" -c copy "$BASE/bcai_top_habits_19/hook.mov" 2>/dev/null
cp "$BASE/bcai_top_habits/IMG_4043.mov" "$BASE/bcai_top_habits_19/body.mov"
echo "✓ bcai_top_habits_19 (Daily Social Connection)"

# Habit 18 - Leadership (154.24 - 164.61)
mkdir -p "$BASE/bcai_top_habits_18"
ffmpeg -y -ss 153.7 -to 165.1 -i "$HOOKS" -c copy "$BASE/bcai_top_habits_18/hook.mov" 2>/dev/null
cp "$BASE/bcai_top_habits/IMG_4044.mov" "$BASE/bcai_top_habits_18/body.mov"
echo "✓ bcai_top_habits_18 (Leadership)"

# Habits 17-15 - hooks only (no body clips yet)
# Habit 17 - Daily Walking (169.41 - 182.54)
mkdir -p "$BASE/bcai_top_habits_17"
ffmpeg -y -ss 168.9 -to 183.0 -i "$HOOKS" -c copy "$BASE/bcai_top_habits_17/hook.mov" 2>/dev/null
echo "✓ bcai_top_habits_17 (Daily Walking) - hook only"

# Habit 16 - Creativity (199.33 - 210.25)
mkdir -p "$BASE/bcai_top_habits_16"
ffmpeg -y -ss 198.8 -to 210.8 -i "$HOOKS" -c copy "$BASE/bcai_top_habits_16/hook.mov" 2>/dev/null
echo "✓ bcai_top_habits_16 (Creativity) - hook only"

# Habit 15 - Waking Up Early (213.69 - 226.14)
mkdir -p "$BASE/bcai_top_habits_15"
ffmpeg -y -ss 213.2 -to 226.6 -i "$HOOKS" -c copy "$BASE/bcai_top_habits_15/hook.mov" 2>/dev/null
echo "✓ bcai_top_habits_15 (Waking Up Early) - hook only"

echo ""
echo "All done! 8 folders created."
