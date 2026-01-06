# Calendar UI Implementation Plan

Technical specification for implementing a smart calendar interface for recordings in Animal Tracker.

## Overview

Replace the flat chronological list at `/recordings` with a calendar-based navigation system that organizes recordings by year → month → week → day, with filtering and statistics.

---

## Current Architecture

### Storage Structure
```
storage/clips/
├── {camera_id}/
│   └── {YYYY}/
│       └── {MM}/
│           └── {DD}/
│               ├── {timestamp}_{species}.mp4
│               ├── {timestamp}_{species}.log.json
│               └── {timestamp}_{species}_thumb_{species}.jpg
```

### Existing Code
- **`web.py`**: `_scan_recordings()` returns flat list with metadata (path, camera, date, time, size, species, thumbnails)
- **`storage.py`**: `build_clip_path()` creates the `camera/YYYY/MM/DD/` structure
- **Endpoint**: `GET /recordings` renders HTML list

---

## Phase 1: Backend Calendar API

### New Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recordings/calendar` | GET | Aggregated counts by year/month/day |
| `/api/recordings/calendar/{year}` | GET | Month summaries for a year |
| `/api/recordings/calendar/{year}/{month}` | GET | Day summaries for a month |
| `/api/recordings/day/{date}` | GET | Full clip list for YYYY-MM-DD |

### New Methods in `WebServer` class

```python
def _build_calendar_data(self, clips: list) -> dict:
    """
    Group clips into hierarchical calendar structure.
    
    Returns:
        {
            "years": {
                2026: {
                    "total": 45,
                    "months": {
                        1: {
                            "total": 12,
                            "days": {
                                6: {
                                    "count": 5,
                                    "species": ["Bird", "Dog"],
                                    "cameras": ["cam1", "cam2"],
                                    "first_clip_time": "08:23",
                                    "last_clip_time": "18:45"
                                }
                            }
                        }
                    }
                }
            },
            "cameras": ["cam1", "cam2"],  # All unique cameras
            "species": ["Bird", "Dog", "Cat"]  # All unique species
        }
    """

def _get_clips_for_date(self, date_str: str, camera: str = None, species: str = None) -> list:
    """
    Filter clips for a specific date with optional filters.
    
    Args:
        date_str: "YYYY-MM-DD" format
        camera: Optional camera filter
        species: Optional species filter
    
    Returns:
        List of clip dicts sorted by time
    """

async def handle_calendar_api(self, request) -> web.Response:
    """GET /api/recordings/calendar - Returns full calendar structure as JSON"""

async def handle_day_api(self, request) -> web.Response:
    """GET /api/recordings/day/{date} - Returns clips for specific date"""
```

### Route Registration
```python
# Add to __init__
self.app.router.add_get('/api/recordings/calendar', self.handle_calendar_api)
self.app.router.add_get('/api/recordings/day/{date}', self.handle_day_api)
```

### Response Format Examples

**GET /api/recordings/calendar**
```json
{
  "years": {
    "2026": {
      "total": 156,
      "months": {
        "1": {
          "total": 45,
          "days": {
            "6": {"count": 8, "species": ["Bird", "Squirrel"], "cameras": ["front_yard"]},
            "5": {"count": 12, "species": ["Bird", "Dog"], "cameras": ["front_yard", "backyard"]}
          }
        }
      }
    }
  },
  "filters": {
    "cameras": ["front_yard", "backyard"],
    "species": ["Bird", "Dog", "Squirrel", "Cat"]
  }
}
```

**GET /api/recordings/day/2026-01-06**
```json
{
  "date": "2026-01-06",
  "clips": [
    {
      "path": "front_yard/2026/01/06/1736175823_bird.mp4",
      "camera": "front_yard",
      "time": "08:23:43",
      "species": "Bird",
      "size_mb": 2.4,
      "thumbnails": [
        {"url": "/clips/front_yard/2026/01/06/1736175823_bird_thumb_bird.jpg", "species": "Bird"}
      ]
    }
  ],
  "summary": {
    "total": 8,
    "by_species": {"Bird": 5, "Squirrel": 3},
    "by_camera": {"front_yard": 8}
  }
}
```

### Estimated Time: 2 hours

---

## Phase 2: Calendar Month View

Break into sub-phases to avoid response limits.

---

### Phase 2A: Base HTML Structure & CSS (~45 min)

Replace `handle_recordings` HTML with new calendar layout structure.

**Changes:**
- New page layout with calendar container
- Navigation bar (prev/next month)  
- View tabs placeholder (Month/List)
- Calendar grid container
- Keep existing video modal & bulk actions

**CSS to add:**
```css
/* Calendar Container */
.calendar-container { background: #1a1a1a; border-radius: 12px; padding: 16px; }
.calendar-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
.calendar-nav { display: flex; align-items: center; gap: 16px; }
.nav-btn { background: #333; border: none; color: #fff; width: 40px; height: 40px; border-radius: 8px; cursor: pointer; font-size: 1.2em; }
.nav-btn:hover { background: #444; }
.current-month { font-size: 1.3em; font-weight: 600; margin: 0; }

/* View Tabs */
.view-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
.view-tab { background: #333; border: none; color: #aaa; padding: 8px 16px; border-radius: 8px; cursor: pointer; }
.view-tab.active { background: #4CAF50; color: white; }

/* Calendar Grid */
.calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; }
.calendar-weekday { text-align: center; padding: 8px; color: #888; font-size: 0.85em; font-weight: 500; }
.calendar-day { min-height: 70px; background: #2a2a2a; border-radius: 8px; padding: 8px; cursor: pointer; position: relative; }
.calendar-day:hover { background: #333; }
.calendar-day.today { border: 2px solid #4CAF50; }
.calendar-day.has-recordings { background: #1e3a1e; }
.calendar-day.other-month { opacity: 0.4; }
.day-number { font-size: 0.9em; font-weight: 600; }
.day-count { position: absolute; bottom: 6px; right: 6px; background: #4CAF50; color: white; font-size: 0.7em; padding: 2px 8px; border-radius: 10px; }
```

---

### Phase 2B: JavaScript CalendarApp Core (~30 min)

Add JavaScript state management and API loading.

**Functions:**
```javascript
const CalendarApp = {
  state: { view: 'month', year: 2026, month: 1, calendarData: null, dayClips: null },
  
  async init() {
    this.parseUrlParams();
    await this.loadCalendarData();
    this.render();
  },
  
  parseUrlParams() {
    const params = new URLSearchParams(window.location.search);
    // Parse ?view=month&year=2026&month=1
  },
  
  async loadCalendarData() {
    const res = await fetch('/api/recordings/calendar');
    this.state.calendarData = await res.json();
  },
  
  updateUrl() {
    const url = new URL(window.location);
    url.searchParams.set('view', this.state.view);
    url.searchParams.set('year', this.state.year);
    url.searchParams.set('month', this.state.month);
    history.replaceState({}, '', url);
  }
};
```

---

### Phase 2C: Month Grid Rendering (~45 min)

Implement `renderMonthView()` to build calendar grid.

**Functions:**
```javascript
renderMonthView() {
  const grid = document.getElementById('calendarGrid');
  const { year, month } = this.state;
  
  // Get first day of month and total days
  const firstDay = new Date(year, month - 1, 1).getDay();
  const daysInMonth = new Date(year, month, 0).getDate();
  
  // Build grid HTML with weekday headers + day cells
  // Mark days with recordings from calendarData
  // Highlight today
}

getDayData(year, month, day) {
  // Lookup recording count from this.state.calendarData
  const y = this.state.calendarData?.years?.[String(year)];
  const m = y?.months?.[String(month)];
  return m?.days?.[String(day)] || null;
}
```

---

### Phase 2D: Navigation & View Switching (~30 min)

Add month navigation and view tab switching.

**Functions:**
```javascript
prevMonth() {
  if (this.state.month === 1) {
    this.state.month = 12;
    this.state.year--;
  } else {
    this.state.month--;
  }
  this.updateUrl();
  this.render();
}

nextMonth() {
  if (this.state.month === 12) {
    this.state.month = 1;
    this.state.year++;
  } else {
    this.state.month++;
  }
  this.updateUrl();
  this.render();
}

setView(view) {
  this.state.view = view;
  this.updateUrl();
  this.render();
}

goToToday() {
  const now = new Date();
  this.state.year = now.getFullYear();
  this.state.month = now.getMonth() + 1;
  this.updateUrl();
  this.render();
}
```

---

### Phase 2E: List View Fallback (~30 min)

Keep original list view accessible via "List" tab.

**Implementation:**
- When `view === 'list'`, render the original recordings list
- Reuse existing clip card HTML generation
- Keep all existing play/delete functionality

---

### Phase 2 Testing Checklist

- [ ] Calendar grid renders with correct days
- [ ] Today is highlighted
- [ ] Days with recordings show green + count badge
- [ ] Prev/Next month navigation works
- [ ] URL updates when navigating
- [ ] List view shows original recordings list
- [ ] View tabs switch correctly

### Estimated Total: ~3 hours (5 sub-phases)

---

## Phase 3: Day Detail Panel

### UI Structure

```
┌─────────────────────────────────────────┐
│ 📅 January 6, 2026           ✕ Close    │
│ 8 recordings • front_yard               │
├─────────────────────────────────────────┤
│                                          │
│ 🌅 Morning (6 AM - 12 PM)        4 clips│
│ ┌─────────────────────────────────────┐ │
│ │ ☐ 🐦 Bird         8:23 AM   2.4 MB │ │
│ │   [thumb]  front_yard               │ │
│ ├─────────────────────────────────────┤ │
│ │ ☐ 🐿️ Squirrel    9:45 AM   1.8 MB │ │
│ │   [thumb]  front_yard               │ │
│ └─────────────────────────────────────┘ │
│                                          │
│ 🌞 Afternoon (12 PM - 6 PM)      3 clips│
│ ┌─────────────────────────────────────┐ │
│ │ ☐ 🐦 Bird         2:15 PM   3.1 MB │ │
│ └─────────────────────────────────────┘ │
│                                          │
│ 🌙 Evening (6 PM - 12 AM)        1 clip │
│ ┌─────────────────────────────────────┐ │
│ │ ☐ 🦝 Raccoon      8:45 PM   4.2 MB │ │
│ └─────────────────────────────────────┘ │
│                                          │
├─────────────────────────────────────────┤
│ [Select All]              [Delete (0)]  │
└─────────────────────────────────────────┘
```

### Implementation

```javascript
function showDayPanel(dateStr, clips) {
  const grouped = groupByTimeOfDay(clips);
  // Morning: 6-12, Afternoon: 12-18, Evening: 18-24, Night: 0-6
  
  const html = `
    <div class="day-panel">
      <div class="day-header">
        <h3>${formatDate(dateStr)}</h3>
        <span>${clips.length} recordings</span>
        <button onclick="closeDayPanel()">✕</button>
      </div>
      ${renderTimeGroups(grouped)}
      <div class="day-actions">
        <label><input type="checkbox" id="selectAllDay"> Select All</label>
        <button id="deleteDayBtn" disabled>Delete (0)</button>
      </div>
    </div>
  `;
  
  document.getElementById('dayPanelContainer').innerHTML = html;
  document.getElementById('dayPanelContainer').classList.add('visible');
}

function groupByTimeOfDay(clips) {
  return {
    morning: clips.filter(c => c.hour >= 6 && c.hour < 12),
    afternoon: clips.filter(c => c.hour >= 12 && c.hour < 18),
    evening: clips.filter(c => c.hour >= 18 && c.hour < 24),
    night: clips.filter(c => c.hour >= 0 && c.hour < 6)
  };
}
```

### Estimated Time: 2 hours

---

## Phase 4: Filters Sidebar

### UI Structure

```
┌──────────────────────────┐
│ Filters                ✕ │
├──────────────────────────┤
│                          │
│ Quick Select             │
│ [Today] [This Week]      │
│ [This Month] [All Time]  │
│                          │
│ Date Range               │
│ From: [2026-01-01]       │
│ To:   [2026-01-06]       │
│                          │
│ Cameras                  │
│ ☑ front_yard        (45) │
│ ☑ backyard          (23) │
│ ☐ driveway           (8) │
│                          │
│ Species                  │
│ ☑ Bird              (34) │
│ ☑ Squirrel          (18) │
│ ☑ Dog               (12) │
│ ☐ Cat                (5) │
│                          │
│ [Clear All] [Apply]      │
└──────────────────────────┘
```

### Filter Logic

```javascript
function applyFilters() {
  const filters = {
    cameras: getCheckedValues('camera-filter'),
    species: getCheckedValues('species-filter'),
    dateRange: {
      start: document.getElementById('dateFrom').value,
      end: document.getElementById('dateTo').value
    }
  };
  
  CalendarApp.state.filters = filters;
  CalendarApp.render();  // Re-render with filters
}

function filterClips(clips, filters) {
  return clips.filter(clip => {
    if (filters.cameras.length && !filters.cameras.includes(clip.camera)) return false;
    if (filters.species.length && !filters.species.includes(clip.species)) return false;
    if (filters.dateRange.start && clip.date < filters.dateRange.start) return false;
    if (filters.dateRange.end && clip.date > filters.dateRange.end) return false;
    return true;
  });
}
```

### Estimated Time: 1.5 hours

---

## Phase 5: Statistics Header

### UI Structure

```
┌─────────────────────────────────────────────────────────────┐
│  📊 January 2026 Summary                                     │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │    156   │  │  Bird    │  │ front_yd │  │  8-9 AM  │    │
│  │ Clips    │  │ Top Spp  │  │ Top Cam  │  │ Peak Hr  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                              │
│  Activity Heatmap (by hour)                                  │
│  ░░▓▓████████▓▓▓▓░░░░░░░░                                   │
│  0  4  8  12  16  20  24                                     │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

```javascript
function renderStats(clips) {
  const stats = calculateStats(clips);
  
  return `
    <div class="stats-bar">
      <div class="stat-card">
        <div class="stat-value">${stats.total}</div>
        <div class="stat-label">Clips</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.topSpecies}</div>
        <div class="stat-label">Top Species</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.topCamera}</div>
        <div class="stat-label">Top Camera</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.peakHour}</div>
        <div class="stat-label">Peak Hour</div>
      </div>
    </div>
    <div class="activity-heatmap">
      ${renderHourlyHeatmap(stats.byHour)}
    </div>
  `;
}

function calculateStats(clips) {
  const bySpecies = {};
  const byCamera = {};
  const byHour = new Array(24).fill(0);
  
  clips.forEach(clip => {
    bySpecies[clip.species] = (bySpecies[clip.species] || 0) + 1;
    byCamera[clip.camera] = (byCamera[clip.camera] || 0) + 1;
    byHour[clip.hour]++;
  });
  
  return {
    total: clips.length,
    topSpecies: Object.entries(bySpecies).sort((a,b) => b[1]-a[1])[0]?.[0] || '-',
    topCamera: Object.entries(byCamera).sort((a,b) => b[1]-a[1])[0]?.[0] || '-',
    peakHour: byHour.indexOf(Math.max(...byHour)),
    byHour
  };
}
```

### Estimated Time: 1 hour

---

## Testing Checklist

### Phase 1
- [ ] `/api/recordings/calendar` returns valid JSON
- [ ] Calendar data correctly groups by year/month/day
- [ ] `/api/recordings/day/YYYY-MM-DD` returns clips for that date
- [ ] Empty dates return empty arrays, not errors

### Phase 2
- [ ] Month view renders correctly
- [ ] Navigation (prev/next month) works
- [ ] Days with recordings are highlighted
- [ ] Clicking a day triggers day detail load
- [ ] Today is visually distinct

### Phase 3
- [ ] Day panel shows clips grouped by time
- [ ] Thumbnails load correctly
- [ ] Video playback works
- [ ] Bulk selection works
- [ ] Delete functionality works

### Phase 4
- [ ] Camera filter updates calendar view
- [ ] Species filter updates calendar view
- [ ] Date range picker works
- [ ] Quick select buttons work
- [ ] Filters persist during navigation

### Phase 5
- [ ] Stats calculate correctly
- [ ] Heatmap renders
- [ ] Stats update when filters change

---

## Migration Notes

- The existing `/recordings` endpoint will be replaced with the new calendar UI
- The old list view remains accessible via the "List" tab
- All existing functionality (play, delete, bulk delete) is preserved
- URL format changes: `/recordings` → `/recordings?view=month&date=2026-01`

---

## File Changes Summary

| File | Changes |
|------|---------|
| `src/animaltracker/web.py` | ~400 lines added (new methods + HTML/CSS/JS) |
| No new dependencies | Pure HTML/CSS/JS (matches existing architecture) |

---

## Rollback Plan

If issues arise, the original `handle_recordings` can be restored from git:
```bash
git checkout HEAD -- src/animaltracker/web.py
```
