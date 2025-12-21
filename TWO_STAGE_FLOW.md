# Two-Stage Feedback Loop - Visual Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STEP 7: TWO-STAGE FEEDBACK LOOP                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1: SENTENCE SELECTION                            │
│                   (Which sentences to keep/remove?)                         │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────┐
   │ Load s4_editing_     │
   │ result.json          │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Generate s5_adjusted │──────┐
   │ _sentences.json      │      │ Programmatic
   │ (programmatically)   │      │ silence removal
   └──────────┬───────────┘      │
              │                   │
              ▼                   │
   ┌──────────────────────┐      │
   │ Generate s6_         │◄─────┘
   │ downsampled_edited   │
   │ .mp4                 │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ User reviews video   │
   │ and provides         │
   │ feedback             │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ SentenceSelection    │
   │ Agent processes      │
   │ feedback             │
   └──────────┬───────────┘
              │
              ▼
        ┌────────────┐
        │ Approved?  │───No──┐
        └─────┬──────┘       │
              │ Yes          │
              │              │
              │              ▼
              │    ┌──────────────────┐
              │    │ Update s4_editing│
              │    │ _result.json     │
              │    └────────┬─────────┘
              │             │
              │             │
              │             └──────────┐
              │                        │
              │                        ▼
              │              ┌──────────────────┐
              │              │ Loop back to top │
              │              └──────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: TIMESTAMP ADJUSTMENT                            │
│                    (How to pace the sentences?)                             │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────┐
   │ Regenerate s5_       │
   │ adjusted_sentences   │
   │ from approved s4     │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Generate s6_         │
   │ downsampled_edited   │
   │ .mp4                 │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ User reviews video   │
   │ and provides         │
   │ feedback on timing   │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ TimestampAdjustment  │
   │ Agent processes      │
   │ feedback             │
   └──────────┬───────────┘
              │
              ▼
        ┌────────────┐
        │ Approved?  │───No──┐
        └─────┬──────┘       │
              │ Yes          │
              │              │
              │              ▼
              │    ┌──────────────────┐
              │    │ Update s5_       │
              │    │ adjusted_        │
              │    │ sentences.json   │
              │    └────────┬─────────┘
              │             │
              │             ▼
              │    ┌──────────────────┐
              │    │ Regenerate s6_   │
              │    │ downsampled_     │
              │    │ edited.mp4       │
              │    └────────┬─────────┘
              │             │
              │             └──────────┐
              │                        │
              │                        ▼
              │              ┌──────────────────┐
              │              │ Loop back        │
              │              └──────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Both stages approved │
   │ Proceed to Step 8    │
   └──────────────────────┘
```

## Agent Tools

### SentenceSelectionAgent Tools
```
┌─────────────────────────────────────────────────┐
│ Tool: keep_sentence                             │
│ • Marks sentence index to keep: true            │
│ • Updates s4_editing_result.json                │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Tool: remove_sentence                           │
│ • Marks sentence index to keep: false           │
│ • Updates s4_editing_result.json                │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Tool: approve                                   │
│ • Finalizes sentence selection                  │
│ • Moves to Stage 2                              │
└─────────────────────────────────────────────────┘
```

### TimestampAdjustmentAgent Tools
```
┌─────────────────────────────────────────────────┐
│ Tool: adjust_timestamp                          │
│ • Modifies sentence timestamp fields            │
│ • Can adjust: original_start, original_end,     │
│   adjusted_start, adjusted_end                  │
│ • Uses word-level timestamps for precision      │
│ • Updates s5_adjusted_sentences.json            │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Tool: approve                                   │
│ • Finalizes timestamp adjustments               │
│ • Completes Step 7                              │
└─────────────────────────────────────────────────┘
```

## Example Feedback Flows

### Stage 1 Example
```
User: "Remove sentences 6, 7, and 8 - they're just filler"
  ↓
SentenceSelectionAgent:
  - Thoughts: "User wants to remove sentences 6, 7, and 8 as filler content"
  - Actions:
    1. remove_sentence(sentence_index="6")
    2. remove_sentence(sentence_index="7")
    3. remove_sentence(sentence_index="8")
  ↓
System:
  - Updates s4_editing_result.json
  - Regenerates s5_adjusted_sentences.json (programmatic)
  - Regenerates s6_downsampled_edited.mp4
  ↓
User reviews updated video...
```

### Stage 2 Example
```
User: "The intro is too slow, cut the first 1.5 seconds"
  ↓
TimestampAdjustmentAgent:
  - Thoughts: "User wants to speed up the intro by cutting first 1.5 seconds"
  - Actions:
    1. adjust_timestamp(sentence_index="1", field="adjusted_start", new_value=1.5)
  ↓
System:
  - Updates s5_adjusted_sentences.json
  - Regenerates s6_downsampled_edited.mp4
  ↓
User reviews updated video...
```


