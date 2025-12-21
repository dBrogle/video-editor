# Two-Stage Feedback Loop Implementation

## Summary

Successfully refactored Step 7 of the video editing pipeline to use a two-stage feedback loop with separate agents for sentence selection and timestamp adjustment.

## Changes Made

### 1. New Files Created

#### `src/services/agents/sentence_selection_agent.py`
- **SentenceSelectionAgent** class for handling sentence keep/remove decisions
- Works with `s4_editing_result.json`
- Tools available:
  - `keep_sentence` - Mark a sentence to be kept
  - `remove_sentence` - Mark a sentence to be removed
  - `approve` - Approve selection and move to next stage

#### `src/services/agents/timestamp_adjustment_agent.py`
- Renamed from `first_cut_agent.py`
- **TimestampAdjustmentAgent** class for fine-tuning timestamps
- Works with `s5_adjusted_sentences.json`
- Tools available:
  - `adjust_timestamp` - Modify sentence start/end times
  - `approve` - Finalize timestamps

### 2. Modified Files

#### `src/services/agents/__init__.py`
- Updated imports to export both new agents
- Removed `FirstCutAgent` reference

#### `src/pipeline.py`
- Completely rewrote `feedback_loop_for_cut()` function
- Implemented two sequential stages:
  - **Stage 1**: Sentence Selection
    - Load editing result
    - Get user feedback on sentence selection
    - Update editing result based on feedback
    - Regenerate adjusted sentences programmatically
    - Regenerate video to show changes
    - Loop until approved
  - **Stage 2**: Timestamp Adjustment
    - Regenerate adjusted sentences from approved editing result
    - Get user feedback on timestamps/pacing
    - Update adjusted sentences based on feedback
    - Regenerate video to show changes
    - Loop until approved

#### `main.py`
- Updated menu text for Step 7 to reflect two-stage approach

#### `README.md`
- Updated documentation with new two-stage workflow
- Added examples for both agents
- Updated code samples

### 3. Deleted Files

#### `src/services/agents/first_cut_agent.py`
- Replaced by `timestamp_adjustment_agent.py`

## Workflow

### Stage 1: Sentence Selection
1. User reviews video with current sentence selection
2. Provides feedback like "Remove sentence 5" or "Keep sentence 12"
3. Agent updates `s4_editing_result.json` (keep: true/false)
4. System regenerates `s5_adjusted_sentences.json` programmatically
5. System regenerates `s6_downsampled_edited.mp4`
6. Loop continues until user approves

### Stage 2: Timestamp Adjustment
1. User reviews video with approved sentences and programmatic timestamps
2. Provides feedback like "Cut 2 seconds from start" or "Reduce pause between 3 and 4"
3. Agent updates `s5_adjusted_sentences.json` (timestamp fields)
4. System regenerates `s6_downsampled_edited.mp4`
5. Loop continues until user approves

## Key Design Decisions

1. **Sequential stages**: Sentence selection must be approved before timestamp adjustment to avoid conflicts
2. **Programmatic regeneration**: After sentence selection changes, s5 is regenerated programmatically (not by agent)
3. **Separate agents**: Each agent has a focused responsibility and works with its own JSON file
4. **Independent files**: Each agent lives in its own file for clarity and maintainability
5. **Video regeneration**: Video is regenerated after each change in both stages so user can see results

## Benefits

- **Clearer separation of concerns**: Sentence selection vs timestamp adjustment
- **Better user experience**: User can focus on one aspect at a time
- **More reliable**: No conflicts between sentence selection and timestamp adjustments
- **More maintainable**: Each agent is simpler and easier to understand
- **Follows workflow**: Matches the natural editing workflow (what to include, then how to pace it)

## Testing Checklist

- [ ] Step 7 runs without errors
- [ ] Stage 1 can keep/remove sentences
- [ ] Stage 1 regenerates video after changes
- [ ] Stage 1 approval moves to Stage 2
- [ ] Stage 2 can adjust timestamps
- [ ] Stage 2 regenerates video after changes
- [ ] Stage 2 approval completes the feedback loop
- [ ] Changes are persisted to correct JSON files
- [ ] No linter errors in any modified files


