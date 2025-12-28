# Animal Tracker Architecture Refactor

## Overview

This document outlines a proposed refactoring to simplify the detection and classification pipeline, separating real-time motion capture from species classification.

## Current Architecture (Problems)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        REAL-TIME PIPELINE                           │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  RTSP    │───▶│   Detector   │───▶│   Tracker    │              │
│  │  Stream  │    │  (BioCLIP)   │    │  (ByteTrack) │              │
│  └──────────┘    └──────────────┘    └──────────────┘              │
│        │                │                    │                      │
│        ▼                ▼                    ▼                      │
│   ┌─────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │ Buffer  │    │   Species    │    │    Event     │              │
│   │ Frames  │    │   Labels     │    │    State     │              │
│   └─────────┘    └──────────────┘    └──────────────┘              │
│        │                │                    │                      │
│        └────────────────┴────────────────────┘                      │
│                         │                                           │
│                         ▼                                           │
│                  ┌──────────────┐                                   │
│                  │  Save Clip   │                                   │
│                  │  with Label  │                                   │
│                  └──────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     POST-PROCESSING (Reanalyze)                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Video   │───▶│   Detector   │───▶│   Tracker    │              │
│  │   File   │    │  (BioCLIP)   │    │  (ByteTrack) │              │
│  └──────────┘    └──────────────┘    └──────────────┘              │
│                         │                    │                      │
│                         ▼                    ▼                      │
│                  ┌──────────────┐    ┌──────────────┐              │
│                  │    Merge     │───▶│   Rename     │              │
│                  │    Tracks    │    │    Clip      │              │
│                  └──────────────┘    └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### Problems with Current Architecture

1. **Duplicate Logic**: Species classification happens twice (real-time + reanalyze) with potentially different results
2. **Inconsistent Tracking**: Real-time tracker creates fragmented tracks; post-process creates different tracks
3. **Wasted GPU Time**: Running full BioCLIP inference during real-time when we just need "is animal present?"
4. **Conflicting Results**: Real-time says "canidae", reanalyze says "animal" or different count
5. **Complex Debugging**: Hard to know which engine produced which results

---

## Proposed Architecture (Simplified)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME PIPELINE (Simplified)                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  RTSP    │───▶│   Motion /   │───▶│   Event      │              │
│  │  Stream  │    │   Animal     │    │   Trigger    │              │
│  └──────────┘    │   Detector   │    │              │              │
│        │         └──────────────┘    └──────────────┘              │
│        │                                    │                       │
│        ▼                                    ▼                       │
│   ┌─────────┐                        ┌──────────────┐              │
│   │ Buffer  │───────────────────────▶│  Save Clip   │              │
│   │ Frames  │                        │  (unclassified)             │
│   └─────────┘                        └──────────────┘              │
│                                             │                       │
│                                             ▼                       │
│                                      ┌──────────────┐              │
│                                      │  Queue for   │              │
│                                      │  Processing  │              │
│                                      └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 UNIFIED CLASSIFICATION ENGINE                       │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Video   │───▶│   Detector   │───▶│   Tracker    │              │
│  │   File   │    │  (BioCLIP)   │    │  (ByteTrack) │              │
│  └──────────┘    └──────────────┘    └──────────────┘              │
│                         │                    │                      │
│                         ▼                    ▼                      │
│                  ┌──────────────┐    ┌──────────────┐              │
│                  │    Merge     │───▶│   Species    │              │
│                  │    Tracks    │    │   Decision   │              │
│                  └──────────────┘    └──────────────┘              │
│                                             │                       │
│                         ┌───────────────────┼───────────────────┐   │
│                         ▼                   ▼                   ▼   │
│                  ┌──────────┐        ┌──────────┐        ┌────────┐│
│                  │  Rename  │        │  Create  │        │  Save  ││
│                  │   Clip   │        │ Thumbnail│        │  Log   ││
│                  └──────────┘        └──────────┘        └────────┘│
└─────────────────────────────────────────────────────────────────────┘
                          ▲
                          │
              ┌───────────┴───────────┐
              │                       │
       ┌──────┴──────┐         ┌──────┴──────┐
       │ Auto-process│         │  Reanalyze  │
       │  (on save)  │         │   Button    │
       └─────────────┘         └─────────────┘
```

---

## Design Decisions

### 1. Real-Time Detection: Motion vs Animal

**Option A: Pure Motion Detection (Simplest)**
- Use frame differencing or background subtraction
- Very fast, no GPU needed
- Risk: More false positives (leaves, shadows)

**Option B: Animal Presence Detection (Recommended)**
- Run BioCLIP with high threshold, only care about "animal" class
- Still uses GPU but simpler prompt: "Is there an animal? Y/N"
- Lower threshold since we just need presence, not species

**Option C: Keep Current Detection, Ignore Species**
- Continue running full BioCLIP
- Only use detection boxes for "animal present" trigger
- Species label from real-time is ignored
- Most compatible with current code

**Recommendation**: Option C initially (minimal code change), migrate to Option B later for performance.

### 2. Clip Naming Convention

**Current**: `{timestamp}_{species}_{camera}.mp4`
**Proposed**: 
- Initial save: `{timestamp}_unprocessed_{camera}.mp4`
- After processing: `{timestamp}_{species}_{camera}.mp4`

This makes it clear which clips have been processed.

### 3. Processing Queue

New clips go into a processing queue:
```python
@dataclass
class ProcessingJob:
    clip_path: Path
    camera_id: str
    priority: int  # Higher = process first
    created_at: datetime
    settings: ProcessingSettings
```

Queue is processed in background thread/process.

### 4. Unified Processing Settings

All species classification uses these settings:
```python
@dataclass
class ProcessingSettings:
    # Detection
    sample_rate: int = 3           # Analyze every Nth frame
    confidence_threshold: float = 0.3
    generic_confidence: float = 0.5
    
    # Tracking
    tracking_enabled: bool = True
    lost_track_buffer: int = 120   # Frames to keep lost track
    
    # Merging
    same_species_gap: int = 120    # Max gap for same-species merge
    hierarchical_gap: int = 120    # Max gap for generic→specific merge
    min_specific_detections: int = 2
    
    # Output
    max_thumbnails: int = 3
    save_processing_log: bool = True
```

Settings can be:
- Set globally in config
- Overridden per-camera
- Adjusted in UI for reanalysis

### 5. Web UI Changes

**Recording Detail Page:**
```
┌─────────────────────────────────────────────────────┐
│  Recording: 2025-12-28_143022_canidae_front.mp4    │
├─────────────────────────────────────────────────────┤
│  [Video Player]                                     │
│                                                     │
│  Species: mammalia_carnivora_canidae (95.8%)       │
│  Tracks: 1 animal detected                          │
│  Duration: 22.3s | Analyzed: 660 frames            │
├─────────────────────────────────────────────────────┤
│  Processing Settings            [▼ Expand]         │
│  ┌───────────────────────────────────────────────┐ │
│  │ Sample Rate: [3] frames                       │ │
│  │ Confidence Threshold: [0.30]                  │ │
│  │ Generic Confidence: [0.50]                    │ │
│  │ Track Merge Gap: [120] frames                 │ │
│  │ □ Enable Hierarchical Merging                 │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  [Reanalyze with Settings]  [Reset to Defaults]    │
├─────────────────────────────────────────────────────┤
│  Processing Log               [▼ Expand]           │
│  Frame 51: mammalia_mammal (72.4%) → Track 1       │
│  Frame 54: canidae (90.4%) → Track 1               │
│  ...                                                │
└─────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Unify Post-Processing (Current Sprint) ✅ COMPLETE
- [x] Add hierarchical track merging
- [x] Add processing settings to log
- [x] Create ProcessingSettings dataclass
- [x] Update ClipPostProcessor to accept unified settings
- [x] Update reprocess API to accept settings overrides
- [x] Add `unified_post_processing` config option
- [x] Add settings UI panel to recording detail page

### Phase 2: Simplify Real-Time Pipeline
- [ ] Remove species tracking from real-time EventState
- [ ] Save clips as "unprocessed" initially
- [ ] Auto-queue clips for processing after save
- [ ] Add processing queue manager

### Phase 3: Web UI Enhancements  
- [ ] Add processing settings panel to recording detail
- [ ] Show processing queue status
- [ ] Allow batch reprocessing with custom settings

### Phase 4: Performance Optimization (Future)
- [ ] Option to use lighter detection model for real-time
- [ ] Parallel processing queue for multiple clips
- [ ] GPU batching for multiple frames

---

## Migration Path

1. **No Breaking Changes**: Existing clips continue to work
2. **Gradual Rollout**: 
   - Phase 1 improves reanalyze without changing real-time
   - Phase 2 can be feature-flagged
3. **Backward Compatible Naming**: Support both old and new filename formats

---

## API Changes

### New/Modified Endpoints

```
POST /api/clips/{path}/reprocess
Body: {
    "sample_rate": 3,
    "confidence_threshold": 0.3,
    "generic_confidence": 0.5,
    "tracking_enabled": true,
    "hierarchical_merge": true,
    "merge_gap": 120
}

GET /api/processing/queue
Response: {
    "pending": 5,
    "processing": "front/2025-12-28_143022.mp4",
    "completed_today": 42
}

GET /api/processing/settings
Response: {
    "defaults": { ... },
    "camera_overrides": { "front": { ... } }
}
```

---

## Questions to Resolve

1. **Auto-process timing**: Process immediately after clip save, or batch every N minutes?
2. **Notification timing**: Notify on clip save (fast, less accurate) or after processing (slower, accurate)?
3. **Storage**: Keep unprocessed originals, or overwrite with processed version?
4. **Real-time display**: Show "animal detected" or wait for species classification?

---

## Multi-Animal Tracking

The system supports tracking **multiple animals simultaneously** in the same frame. This is critical for:
- Multiple birds at a feeder
- A group of deer passing through
- Different species interacting (cat chasing bird)

### How It Works

ByteTrack assigns each detected animal a unique track ID based on spatial continuity:

```
Frame 1:  [Bird A at (100,100)]  [Bird B at (300,100)]
             Track 1                  Track 2

Frame 2:  [Bird A at (110,105)]  [Bird B at (290,95)]
             Track 1                  Track 2

Frame 3:  [Bird A at (120,110)]  [Bird B leaves frame]
             Track 1                  Track 2 ends
```

### Distinguishing Multiple Animals vs Fragmented Tracking

| Scenario | Tracks Overlap in Time? | Merge? |
|----------|------------------------|--------|
| 2 birds at feeder | Yes (both visible frame 1-100) | **No** - keep as 2 animals |
| 1 dog, tracking lost | No (Track 1: frames 1-50, Track 2: frames 80-150) | **Yes** - merge into 1 |
| Bird leaves, different bird arrives | No, but different species | **No** - different species |
| Same bird, detected as "animal" then "sparrow" | No, compatible hierarchy | **Yes** - merge, use "sparrow" |

### Merge Rules (Implemented)

1. **Never merge overlapping tracks** - If both tracks have detections at the same frame, they're different animals
2. **Same species merge** - Non-overlapping tracks with identical species labels merge if gap < 120 frames  
3. **Hierarchical merge** - Generic tracks ("animal") absorbed into specific tracks ("canidae") if:
   - No temporal overlap
   - Gap < 120 frames
   - Species are hierarchically compatible (both mammals, both birds, etc.)
   - Specific track has ≥2 detections (reliable identification)

### Output Format

```json
{
  "tracks": [
    {
      "track_id": 1,
      "species": "mammalia_carnivora_canidae",
      "confidence": 0.958,
      "frames_visible": "51-660",
      "detections": 45
    },
    {
      "track_id": 2, 
      "species": "aves_passeriformes_northern_cardinal",
      "confidence": 0.892,
      "frames_visible": "200-350",
      "detections": 28
    }
  ],
  "animal_count": 2
}
```

---

## Success Metrics

- Single animal in frame → exactly 1 track reported
- Multiple animals in frame → correct count of tracks (one per animal)
- Consistent species classification (reanalyze = auto-process results)
- Consistent species classification (reanalyze = auto-process results)
- Clear audit trail of what settings produced what results
- Reduced code duplication between real-time and post-process
