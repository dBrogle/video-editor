# Stage 8: AI-Generated Image Overlay Implementation

## Overview
Add AI-generated images as overlays to the edited video based on LLM analysis of the transcript.

## Architecture Components

### 1. Image Generator Service (Abstract + Implementations)
- **Location**: `src/services/image_generation/`
- **Files**:
  - `base.py` - Abstract base class `ImageGeneratorService`
  - `openrouter.py` - OpenRouter implementation (Gemini, Flux models)
  - `dalle.py` - DALL-E implementation (legacy, not used)
  - `__init__.py` - Exports

### 2. Image Planning Agent
- **Location**: `src/services/agents/image_planning_agent.py`
- **Responsibilities**:
  - First pass: Analyze transcript, generate image descriptions with sentence mappings
  - Generate detailed prompts for image generator
  - Second pass (future): Process user feedback and adjust images

### 3. Image Metadata Management
- **Location**: Utility functions in `src/util.py` or new `src/services/image_manager.py`
- **Responsibilities**:
  - Save/load image metadata JSON
  - Track image-to-sentence mappings
  - Manage image file paths

### 4. MLT Video Service Enhancement
- **Location**: `src/services/video/mlt_video_service.py`
- **Responsibilities**:
  - Add image overlay support to MLT XML generation
  - Handle image safe zone positioning (contain mode)
  - Manage image timing based on sentence timestamps

## Data Models

### ImageDescription (Pydantic Model)
```python
class ImageDescription(BaseModel):
    description: str  # Human-readable description
    detailed_prompt: str  # Detailed prompt for image generator
    sentence_ids: List[str]  # e.g., ["1", "2", "3"]
```

### ImageMetadata (Pydantic Model)
```python
class ImageMetadata(BaseModel):
    filename: str
    prompt: str  # The prompt used to generate it
    sentence_ids: List[str]
    generated_at: str  # ISO timestamp
    generator_service: str  # e.g., "dalle", "stability"
```

### ImagesMetadataFile (Pydantic Model)
```python
class ImagesMetadataFile(BaseModel):
    images: List[ImageMetadata]
```

## File Structure
```
assets/
  {base_name}/
    images/
      image_001.png
      image_002.png
      ...
      images_metadata.json
    s8_with_images_downsampled.mp4
```

## Implementation Steps - First Pass

### Phase 1: Setup Infrastructure
1. ✅ Create `src/services/image_generation/` directory
2. ✅ Create abstract base class `ImageGeneratorService` in `base.py`
3. ✅ Create concrete implementation (e.g., DALL-E) in `dalle.py`
4. ✅ Add data models to `src/models.py`
5. ✅ Add constants for image safe zone to `src/constants.py`

### Phase 2: Image Planning Agent
6. ✅ Create `src/services/agents/image_planning_agent.py`
7. ✅ Implement first-pass prompt:
   - Input: General instruction + sentences dict
   - Output: List of ImageDescription objects
8. ✅ Test agent with sample transcript

### Phase 3: Image Generation
9. ✅ Implement image generation service
10. ✅ Create utility functions for:
    - Creating images directory
    - Saving images with proper naming (image_001.png, etc.)
    - Creating/updating images_metadata.json
11. ✅ Generate all images in parallel (asyncio)

### Phase 4: MLT Video Integration
12. ✅ Enhance `mlt_video_service.py` to support image overlays
13. ✅ Implement image positioning in safe zone (contain mode)
14. ✅ Calculate image timing from sentence timestamps
15. ✅ Generate MLT XML with image tracks
16. ✅ Export video with images

### Phase 5: Pipeline Integration
17. ✅ Add stage 8 to pipeline
18. ✅ Update `main.py` to include stage 8 option
19. ✅ Test end-to-end flow

## Implementation Steps - Second Pass (Future)

### Phase 6: User Feedback Loop
20. ⏳ Create user prompt for feedback
21. ⏳ Implement feedback agent with tools:
    - `approve_video()` - Mark as complete
    - `regenerate_image(image_id, new_prompt)` - Create new image
    - `remove_image(image_id)` - Delete image
    - `move_image(image_id, new_sentence_ids)` - Change timing
22. ⏳ Loop until user approves

### Phase 7: Final Upsampling
23. ⏳ Apply images to full-resolution video
24. ⏳ Export final high-quality output

## Technical Decisions

### Image Safe Zone
- **Position**: 60th-80th percentile height, 30th-70th percentile width
- **Size**: Configurable via constants (default: 20% height span, 40% width span)
- **Mode**: Contain (scale to fit, maintain aspect ratio)
- **Z-index**: Above video, below any text overlays

### Image Timing
- **Start**: Beginning of first sentence in sentence_ids
- **End**: End of last sentence in sentence_ids
- **Transitions**: None for first pass (instant on/off)

### Parallel Processing
- Generate all images concurrently using asyncio
- Use semaphore to limit concurrent API calls (e.g., max 3 at once)

### Error Handling
- If image generation fails, log error and continue with other images
- Allow partial success (some images generated, others failed)
- Save metadata only for successfully generated images

## Testing Strategy
1. Unit tests for image planning agent
2. Mock image generator for testing without API calls
3. Test MLT generation with sample images
4. End-to-end test with short video clip

## Future Enhancements
- Multiple image safe zones
- Image transitions (fade in/out, slide, etc.)
- Image effects (zoom, pan, etc.)
- Support for video clips as overlays
- Batch processing multiple videos

