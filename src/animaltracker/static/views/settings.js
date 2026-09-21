/* ============================================================================
   views/settings.js — the configuration editor.

   Route: /app/settings   (optional query: ?section=general.detection | <cameraId>)

   WHAT THIS SCREEN EDITS
   config/cameras.yml, through GET/POST /api/config (configstore.py). The
   server reads and validates the FILE, so what is shown here is what the
   next start will load, with every schema default filled in. The running
   process is annotated on top of that: whether a camera is running, its
   stream state, and a list of reasons the process is out of date with the
   file (a camera added, a stream URI changed) that a restart would clear.

   THE RULES THIS FILE KEEPS
     · Edits are STAGED. Nothing leaves the browser until Save; the draft
       survives switching sections, and leaving the route asks first.
     · One inventory. GENERAL_SECTIONS and CAMERA_GROUPS below are the only
       description of a field: its path, its kind, its range, its hint, and
       whether the pipeline reads it live or at startup. Normalisation,
       rendering, dirty tracking, validation, the payload and the YAML
       preview are all derived from that list, so a field cannot be shown
       but not saved, or saved but not validated.
     · The payload is COMPLETE: every camera, every managed key. The server
       merges it into the file, so a key the UI does not manage (ebird:,
       log_level:) is never touched, and a camera missing from the list is
       a removal — which is why removal is an explicit, confirmed action.
     · The save is optimistic and rolls back field-for-field on failure.
       A 400 carries the server's own field paths; they land on the fields.
     · Live and restart-only fields are told apart at every step: the badge
       on the label, the count in the save bar, the toast after saving and
       the banner that offers the restart.

   Built entirely with the core: h() for DOM (no innerHTML with model data),
   keyedList for the nav and the species chips, dialog/toast for reporting.
   iOS 15 floor: no optional chaining, no ??, no .at().
   ========================================================================= */

import { h, clear, on, delegate, keyedList } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { toast } from '../core/toast.js';
import { dialog, isOverlayOpen } from '../core/overlay.js';
import { router } from '../core/router.js';
import { speciesClass } from '../core/format.js';

/* --------------------------------------------------------------------------
   STATIC CATALOGUES
   ------------------------------------------------------------------------ */

var SPECIES_CATALOG = [
  ['Common Wildlife', ['deer', 'coyote', 'fox', 'raccoon', 'opossum', 'skunk', 'rabbit',
    'squirrel', 'chipmunk', 'groundhog', 'armadillo', 'porcupine', 'beaver']],
  ['Large Mammals', ['bear', 'moose', 'elk', 'mountain lion', 'cougar', 'bobcat', 'lynx',
    'wolf', 'wild boar', 'javelina', 'bison', 'antelope']],
  ['Birds', ['bird', 'turkey', 'hawk', 'owl', 'eagle', 'vulture', 'crow',
    'heron', 'duck', 'goose', 'pheasant', 'quail', 'dove', 'woodpecker']],
  ['Farm Animals', ['horse', 'cow', 'sheep', 'goat', 'pig', 'chicken', 'donkey', 'llama']],
  ['Pets & Domestic', ['dog', 'cat', 'person']],
  ['Other', ['snake', 'turtle', 'frog', 'lizard', 'alligator', 'fish',
    'bat', 'mouse', 'rat', 'mole', 'weasel', 'otter', 'mink', 'badger']]
];

var TRANSPORT_OPTIONS = [
  ['tcp', 'TCP (reliable)'],
  ['udp', 'UDP (lower latency)']
];

var BACKEND_OPTIONS = [
  ['megadetector', 'MegaDetector (animal / person / vehicle)'],
  ['yolo', 'YOLO (fast, generic classes)'],
  ['speciesnet', 'SpeciesNet (species labels)']
];

var SPECIESNET_VERSIONS = [
  ['v4.0.3a', 'v4.0.3a — classifies the detection crop'],
  ['v4.0.3b', 'v4.0.3b — classifies the full frame']
];

var PRIORITY_OPTIONS = [
  ['-2', 'Lowest'],
  ['-1', 'Low'],
  ['0', 'Normal'],
  ['1', 'High']
];

var SOUND_OPTIONS = [
  ['', 'Default'], ['pushover', 'Pushover'], ['bike', 'Bike'], ['bugle', 'Bugle'],
  ['cashregister', 'Cash Register'], ['classical', 'Classical'], ['cosmic', 'Cosmic'],
  ['falling', 'Falling'], ['gamelan', 'Gamelan'], ['incoming', 'Incoming'],
  ['intermission', 'Intermission'], ['magic', 'Magic'], ['mechanical', 'Mechanical'],
  ['pianobar', 'Piano Bar'], ['siren', 'Siren'], ['spacealarm', 'Space Alarm'],
  ['tugboat', 'Tugboat'], ['none', 'None (silent)']
];

var CAMERA_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$/;
var ENV_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
var DEFAULT_SECTION = 'general.detection';

/* --------------------------------------------------------------------------
   THE FIELD INVENTORY
   A spec: { key, kind, label, hint, restart, min, max, step, int, scale, unit,
             options, numeric, nullable, required, pattern, patternHint, mono,
             placeholder, upper, emptyMeans, recent, advanced, when, controls,
             requiredWhen }
   kind: number | pct | slider | switch | select | text | env | list | species
         | destinations (the Pushover recipient list) | recipients (a camera's pick of them)
   restart: the pipeline reads this at startup (badge + banner), else live.
   scale: display = model × scale (min/max/step are in display units).
   when: 'onvif' | 'ptz' — rendered only while that block is switched on.
   controls: the switch that owns a `when` group; toggling re-renders it.
   ------------------------------------------------------------------------ */

var GENERAL_SECTIONS = [
  {
    id: 'general.detection', label: 'Detection', iconName: 'sparkle',
    blurb: 'Which models run, and where in the world the cameras are.',
    groups: [
      { id: 'detectors', legend: 'Detectors', hint: 'Models load once at startup.', fields: [
        { key: 'detector.realtime_backend', kind: 'select', options: BACKEND_OPTIONS, restart: true,
          label: 'Real-time detector',
          hint: 'Runs on every live frame to open events and drive PTZ tracking. MegaDetector is the accurate choice; YOLO is faster on a weak GPU.' },
        { key: 'detector.postprocess_backend', kind: 'select', options: BACKEND_OPTIONS, restart: true,
          label: 'Post-processing detector',
          hint: 'Re-runs on each saved clip to decide the species label. SpeciesNet is the only backend that names species.' },
        { key: 'detector.model_path', kind: 'text', mono: true, restart: true, required: true,
          label: 'YOLO weights', placeholder: 'models/yolo11n.pt',
          hint: 'Path to the .pt file. Used only when a backend above is YOLO.' },
        { key: 'detector.speciesnet_version', kind: 'select', options: SPECIESNET_VERSIONS, restart: true,
          label: 'SpeciesNet version',
          hint: 'Downloaded from Kaggle on first use (about 1.5 GB).' }
      ] },
      { id: 'location', legend: 'Location priors', hint: 'SpeciesNet drops species that do not occur here.', fields: [
        { key: 'detector.country', kind: 'text', nullable: true, restart: true, upper: true,
          pattern: /^[A-Z]{3}$/, patternHint: 'a three-letter ISO code such as USA or CAN',
          label: 'Country', placeholder: 'USA', hint: 'ISO 3166-1 alpha-3 code.' },
        { key: 'detector.admin1_region', kind: 'text', nullable: true, restart: true, upper: true,
          pattern: /^[A-Z0-9]{2,3}$/, patternHint: 'a two-letter state code such as MN',
          label: 'State or province', placeholder: 'MN', hint: 'US state code; leave empty outside the US.' },
        { key: 'detector.latitude', kind: 'number', nullable: true, restart: true, min: -90, max: 90, step: 0.0001,
          label: 'Latitude', hint: 'Optional. Narrows the species range further than the region alone.' },
        { key: 'detector.longitude', kind: 'number', nullable: true, restart: true, min: -180, max: 180, step: 0.0001,
          label: 'Longitude', hint: 'Optional.' }
      ] },
      { id: 'confidence', legend: 'Confidence', fields: [
        /* detector.generic_confidence used to sit here as "the fallback where
           a camera sets none". Every camera's own generic confidence has a
           value (the schema default is 90%) and there is no way to clear it,
           so the fallback could never apply and the slider did nothing. The
           per-camera control below Detection is the real one. */
      ] }
    ]
  },
  {
    id: 'general.recording', label: 'Recording', iconName: 'film',
    blurb: 'Clip length, post-processing and false-positive cleanup.',
    groups: [
      { id: 'buffer', legend: 'Clip buffer', fields: [
        { key: 'clip.pre_seconds', kind: 'number', min: 1, max: 60, step: 1, int: true,
          label: 'Pre-event buffer (seconds)',
          hint: 'Video kept from before the trigger. The in-memory buffer is sized at startup to at least 30 s, so a larger value needs a restart to be fully honoured.' },
        { key: 'clip.post_seconds', kind: 'number', min: 1, max: 60, step: 1, int: true,
          label: 'Post-event buffer (seconds)', hint: 'Recording continues this long after the last detection.' },
        { key: 'clip.max_event_seconds', kind: 'number', min: 30, max: 600, step: 10, int: true,
          label: 'Maximum event length (seconds)', hint: 'An event is closed and saved after this long even if the animal stays.' },
        { key: 'clip.thumbnail_cropped', kind: 'switch',
          label: 'Cropped thumbnails', hint: 'Zoom thumbnails to the detection. Off keeps the full frame with a box.' }
      ] },
      { id: 'postprocess', legend: 'Post-processing', hint: 'Each saved clip is re-analysed for its species label.', fields: [
        { key: 'clip.post_analysis', kind: 'switch',
          label: 'Post-analysis', hint: 'Re-run the post-processing detector on saved clips.' },
        { key: 'clip.unified_post_processing', kind: 'switch',
          label: 'Analyse the saved file',
          hint: 'Recommended. Read frames back from the MP4 rather than from memory, so results match a later reprocess exactly.' },
        { key: 'clip.recover_unfinished_clips', kind: 'switch',
          label: 'Finish interrupted analyses',
          hint: 'A restart mid-analysis leaves a clip unclassified with no key frames. Shortly after startup and every half hour, such clips are analysed in the background, newest first, whenever no live event needs the post-processor.' },
        { key: 'clip.post_analysis_confidence', kind: 'pct',
          label: 'Species confidence', hint: 'Threshold for a specific species label in post-analysis (lower catches more).' },
        { key: 'clip.post_analysis_generic_confidence', kind: 'pct',
          label: 'Generic confidence', hint: 'Threshold for generic labels (animal, bird) in post-analysis.' },
        { key: 'clip.sample_rate', kind: 'number', min: 1, max: 30, step: 1, int: true,
          label: 'Sample every Nth frame', hint: '1 analyses every frame; higher is faster and coarser.' },
        { key: 'clip.post_analysis_frames', kind: 'number', min: 0, max: 1000, step: 10, int: true, advanced: true,
          label: 'Frames to analyse', hint: '0 picks about one frame per second of clip automatically.' },
        { key: 'clip.max_concurrent_postprocess', kind: 'number', min: 1, max: 8, step: 1, int: true, restart: true, advanced: true,
          label: 'Concurrent jobs', hint: 'Clips processed at once. Each job holds a model on the GPU; the pool is sized at startup.' }
      ] },
      { id: 'falsepos', legend: 'False positives', fields: [
        { key: 'clip.delete_if_no_animal', kind: 'switch',
          label: 'Delete clips with no animal', hint: 'When post-analysis finds nothing, delete the clip and skip the alert.' },
        { key: 'clip.min_detection_frames', kind: 'number', min: 1, max: 100, step: 1, int: true,
          label: 'Minimum detection frames', hint: 'Sampled frames that must contain an animal before a clip counts as real.' },
        { key: 'clip.min_reptile_detection_frames', kind: 'number', min: 1, max: 100, step: 1, int: true, advanced: true,
          label: 'Minimum frames for reptiles', hint: 'Stricter floor for class-only reptile and amphibian labels, which pipes and hoses trigger.' }
      ] },
      { id: 'merging', legend: 'Track merging', hint: 'How the post-processor joins detections into one animal.', fields: [
        { key: 'clip.tracking_enabled', kind: 'switch',
          label: 'Object tracking',
          hint: 'Follow the same animal across frames (ByteTrack). Post-processing picks this up immediately; the live tracker is built at startup.' },
        { key: 'clip.track_merge_gap', kind: 'number', min: 10, max: 500, step: 10, int: true,
          label: 'Same-species merge gap (frames)', hint: 'Largest gap between two tracks of one species that still counts as one animal.' },
        { key: 'clip.spatial_merge_enabled', kind: 'switch',
          label: 'Spatial merge', hint: 'Join tracks that overlap in place even when the species label differs.' },
        { key: 'clip.spatial_merge_iou', kind: 'pct', min: 10, max: 90, step: 5, advanced: true,
          label: 'Spatial overlap (IoU)', hint: 'Minimum box overlap for a spatial merge; 30% is the recommended value.' },
        { key: 'clip.spatial_merge_reach', kind: 'number', min: 0, max: 5, step: 0.1, advanced: true,
          label: 'Spatial reach (body lengths)',
          hint: 'Also merge a track that starts within this many body lengths of where the last one ended. 0 = overlap only.' },
        { key: 'clip.hierarchical_merge_enabled', kind: 'switch', advanced: true,
          label: 'Hierarchical merge', hint: 'Fold generic "animal" tracks into the specific species track they overlap.' },
        { key: 'clip.single_animal_mode', kind: 'switch', advanced: true,
          label: 'Single animal mode', hint: 'Force every track in a clip into one animal. Only when there is never more than one.' }
      ] }
    ]
  },
  {
    id: 'general.storage', label: 'Storage', iconName: 'disk',
    blurb: 'Where clips and logs live, and how long clips are kept.',
    groups: [
      { id: 'paths', legend: 'Paths', fields: [
        { key: 'storage_root', kind: 'text', mono: true, required: true, restart: true,
          label: 'Storage root', hint: 'Clips are written under <root>/clips/<camera>/<year>/<month>/<day>.' },
        { key: 'logs_root', kind: 'text', mono: true, required: true, restart: true,
          label: 'Logs root', hint: 'Application and web access logs.' }
      ] },
      { id: 'retention', legend: 'Retention', hint: 'Applied when the cleanup command runs; preview it with --dry-run.', fields: [
        /* Enforced by the cleanup command, which prunes nothing on its own
           schedule: it runs when you run it, or from the ssd-cleaner timer
           where that is installed. A clip goes with its key frames and log,
           and "keep at least" is a floor nothing overrides. */
        { key: 'retention.min_days', kind: 'number', min: 1, max: 365, step: 1, int: true,
          label: 'Keep at least (days)', hint: 'A floor: a clip this young is never removed, even when the disk is full.' },
        { key: 'retention.max_days', kind: 'number', min: 1, max: 3650, step: 1, int: true,
          label: 'Keep at most (days)', hint: 'Clips older than this are removed when cleanup runs.' },
        { key: 'retention.max_utilization_pct', kind: 'slider', min: 50, max: 95, step: 5, unit: '%', restart: true,
          label: 'Disk usage ceiling', hint: 'Above this, cleanup removes the oldest clips first — never past the floor.' }
      ] }
    ]
  },
  {
    id: 'general.notifications', label: 'Notifications', iconName: 'external',
    blurb: 'Pushover alerts, and the species that never alert.',
    groups: [
      { id: 'pushover', legend: 'Pushover', hint: 'Only variable names are stored here. The values live in config/secrets.env and can be set under Secrets below.', fields: [
        { key: 'notification.pushover_app_token_env', kind: 'env', required: true,
          label: 'App token variable', hint: 'Environment variable holding the Pushover application token. A destination can name its own.' },
        { key: 'notification.destinations', kind: 'destinations',
          label: 'Destinations',
          hint: 'Who can receive alerts. Each destination names the variable holding a Pushover user or group key. Every camera alerts every destination unless it picks some in its own Notifications section.',
          emptyMeans: 'No destinations yet — every camera alerts the fallback user key variable below.' },
        { key: 'notification.pushover_user_key_env', kind: 'env', nullable: true,
          label: 'Fallback user key variable', hint: 'Used only while no destinations are defined. Several keys can be comma-separated in secrets.env.' },
        { key: 'notification.web_base_url', kind: 'text', nullable: true, mono: true,
          label: 'Web UI base URL', placeholder: 'http://192.168.1.195:8080',
          hint: 'Makes each alert link straight to its clip.' }
      ] },
      { id: 'exclusions', legend: 'Global exclusions', fields: [
        { key: 'exclusion_list', kind: 'species',
          label: 'Never alert for these species', hint: 'Applies to every camera, on top of its own exclude list.',
          emptyMeans: 'No global exclusions — every species alerts.' }
      ] },
      { id: 'secrets', legend: 'Secrets', custom: 'secrets', fields: [] }
    ]
  },
  {
    id: 'general.system', label: 'System', iconName: 'settings',
    blurb: 'Process-level settings and the running service.',
    groups: [
      { id: 'process', legend: 'Process', fields: [
        { key: 'metrics_port', kind: 'number', min: 1, max: 65535, step: 1, int: true, restart: true,
          label: 'Metrics port', hint: 'Prometheus metrics endpoint.' },
        { key: 'timezone', kind: 'text', nullable: true, restart: true,
          label: 'Timezone', placeholder: 'America/Chicago',
          hint: 'IANA zone for clip times in the UI and alerts. Empty uses the server clock.' }
      ] }
    ]
  }
];

var CAMERA_GROUPS = [
  { id: 'identity', legend: 'Identity', fields: [
    { key: 'name', kind: 'text', required: true,
      label: 'Name', hint: 'Shown on Live, Recordings and in alerts.' },
    { key: 'location', kind: 'text', nullable: true,
      label: 'Location', hint: 'Free-text placement note.' },
    { key: 'detect_enabled', kind: 'switch',
      label: 'Detection', hint: 'Off keeps the stream and the Live view but never opens an event.' }
  ] },
  { id: 'stream', legend: 'Stream', probe: 'rtsp',
    hint: 'The stream is opened at startup; changes here take effect after a restart.', fields: [
    { key: 'rtsp.uri', kind: 'text', mono: true, required: true, restart: true,
      label: 'RTSP URI', placeholder: 'rtsp://user:pass@192.168.1.50:554/stream1',
      hint: 'Passed to FFmpeg as-is. Credentials written here are stored in cameras.yml in plain text.' },
    { key: 'rtsp.transport', kind: 'select', options: TRANSPORT_OPTIONS, restart: true,
      label: 'Transport', hint: 'TCP is reliable; UDP has lower latency. rtsps:// streams need TCP.' },
    { key: 'rtsp.hwaccel', kind: 'switch', restart: true,
      label: 'Hardware decoding (CUDA)', hint: 'Decode on the NVIDIA GPU through FFmpeg. Falls back to software if the stream fails to open.' },
    { key: 'rtsp.latency_ms', kind: 'number', min: 0, max: 5000, step: 100, int: true, restart: true,
      label: 'Latency buffer (ms)', hint: 'Jitter buffer for the stream reader.' },
    { key: 'rtsp.frame_skip', kind: 'number', min: 0, max: 30, step: 1, int: true, restart: true,
      label: 'Frame skip', hint: 'Run detection on every Nth frame. 0 or 1 analyses every frame; 3 analyses one in three.' },
    { key: 'inference_max_width', kind: 'number', min: 0, max: 4096, step: 160, int: true,
      label: 'Inference width cap (px)',
      hint: 'Downscale frames wider than this before detection. Boxes are mapped back, so clips and PTZ are unaffected. 0 = off.' }
  ] },
  { id: 'onvif', legend: 'ONVIF (PTZ control)', probe: 'onvif', fields: [
    { key: 'onvif.enabled', kind: 'switch', restart: true, controls: 'onvif',
      label: 'ONVIF control', hint: 'Needed for PTZ moves, presets and auto-tracking. Off for fixed cameras.' },
    { key: 'onvif.host', kind: 'text', mono: true, restart: true, when: 'onvif', requiredWhen: 'onvif',
      label: 'Host', placeholder: '192.168.1.50', hint: 'Camera IP or hostname.' },
    { key: 'onvif.port', kind: 'number', min: 1, max: 65535, step: 1, int: true, restart: true, when: 'onvif',
      label: 'Port', hint: 'Usually 80, 8000 or 8899.' },
    { key: 'onvif.profile', kind: 'text', nullable: true, mono: true, restart: true, when: 'onvif',
      label: 'Media profile', placeholder: 'Profile_1',
      hint: 'Profile token, or part of one. Test ONVIF lists what the camera offers; empty uses the first profile.' },
    { key: 'onvif.username_env', kind: 'env', restart: true, when: 'onvif', requiredWhen: 'onvif',
      label: 'Username variable', hint: 'Environment variable in config/secrets.env holding the ONVIF user.' },
    { key: 'onvif.password_env', kind: 'env', restart: true, when: 'onvif', requiredWhen: 'onvif',
      label: 'Password variable', hint: 'Variable holding the ONVIF password.' }
  ] },
  { id: 'thresholds', legend: 'Detection thresholds', fields: [
    { key: 'thresholds.confidence', kind: 'pct',
      label: 'Species confidence', hint: 'Minimum score for a specific label to count.' },
    { key: 'thresholds.generic_confidence', kind: 'pct',
      label: 'Generic confidence', hint: 'Higher bar for vague labels (animal, bird, mammal).' },
    { key: 'thresholds.min_frames', kind: 'number', min: 1, max: 30, step: 1, int: true,
      label: 'Minimum frames', hint: 'Consecutive frames with a detection before an event opens.' },
    { key: 'thresholds.min_duration', kind: 'number', min: 0, max: 30, step: 0.5,
      label: 'Minimum duration (seconds)', hint: 'How long detections must persist before an event opens.' },
    { key: 'thresholds.min_detection_area', kind: 'number', scale: 100, unit: '%', min: 0, max: 50, step: 0.05, advanced: true,
      label: 'Minimum detection size (% of frame)',
      hint: 'Boxes smaller than this are ignored. 0.5% filters leaves and noise; lower it for distant animals.' },
    { key: 'thresholds.tracking_min_detection_area', kind: 'number', scale: 100, unit: '%', min: 0, max: 50, step: 0.01, advanced: true,
      label: 'Minimum size while following a subject (% of frame)',
      hint: 'Relaxed floor used while the PTZ tracker is following a subject, and for a box that overlaps the subject of an open event, so an animal that sits down or walks away keeps its clip going.' },
    { key: 'thresholds.blur_threshold', kind: 'number', min: 0, max: 1000, step: 10, advanced: true,
      label: 'Blur threshold', hint: 'Frames with Laplacian variance below this are skipped as blurry. 0 disables; 50–100 suits most cameras.' },
    { key: 'thresholds.ptz_settle_time', kind: 'number', min: 0, max: 5, step: 0.1, advanced: true,
      label: 'PTZ settle time (seconds)', hint: 'Ignore detections this long after a PTZ move while the image steadies.' }
  ] },
  { id: 'ptz', legend: 'PTZ auto-tracking',
    hint: 'The tracker is built at startup; changes take effect after a restart.', fields: [
    { key: 'ptz_tracking.enabled', kind: 'switch', restart: true, controls: 'ptz',
      label: 'Auto-tracking', hint: 'Use this camera’s detections to aim a PTZ head.' },
    { key: 'ptz_tracking.target_camera_id', kind: 'select', options: 'cameras', restart: true, nullable: true,
      label: 'Drives the PTZ of', hint: 'The camera whose head moves. Leave empty when this camera tracks itself.' },
    { key: 'ptz_tracking.self_track', kind: 'switch', restart: true, controls: 'ptz',
      label: 'Self-track', hint: 'This camera centres on its own detections (the zoom camera in a wide + zoom pair). Needs ONVIF control on this camera.' },
    { key: 'ptz_tracking.multi_camera_tracking', kind: 'switch', restart: true, when: 'ptz',
      label: 'Hand over to the target camera', hint: 'Once the animal is in the target camera’s frame, its detections steer for finer control.' },
    { key: 'ptz_tracking.target_fill_pct', kind: 'pct', min: 10, max: 95, step: 5, restart: true, when: 'ptz',
      label: 'Target frame fill', hint: 'How much of the frame the animal should fill; zoom adjusts toward it.' },
    { key: 'ptz_tracking.track_enabled', kind: 'switch', restart: true, when: 'ptz',
      label: 'Follow detections', hint: 'Off keeps patrol only. The Live page can toggle this while running.' },
    { key: 'ptz_tracking.patrol_enabled', kind: 'switch', restart: true, when: 'ptz',
      label: 'Patrol when idle', hint: 'Sweep, or step through presets, while nothing is detected.' },
    { key: 'ptz_tracking.patrol_presets', kind: 'list', restart: true, when: 'ptz',
      label: 'Patrol presets', placeholder: '1, 2, 3', hint: 'Preset tokens to cycle through, comma-separated. Empty means a continuous sweep.' },
    { key: 'ptz_tracking.patrol_dwell_time', kind: 'number', min: 2, max: 120, step: 1, restart: true, when: 'ptz',
      label: 'Dwell per preset (seconds)', hint: 'How long the head rests at each preset.' },
    { key: 'ptz_tracking.patrol_speed', kind: 'number', min: 0.02, max: 1, step: 0.02, restart: true, when: 'ptz',
      label: 'Patrol sweep speed', hint: 'Fraction of full speed; slow sweeps detect better.' },
    { key: 'ptz_tracking.patrol_return_delay', kind: 'number', min: 0.5, max: 30, step: 0.5, restart: true, when: 'ptz',
      label: 'Return to patrol after (seconds)', hint: 'Quiet time with no sighting from any camera before patrol resumes.' },
    { key: 'ptz_tracking.investigate_enabled', kind: 'switch', restart: true, when: 'ptz',
      label: 'Investigate small detections', hint: 'Slew the zoom camera to tiny wide-angle candidates to confirm them.' },
    { key: 'ptz_tracking.investigate_min_area', kind: 'number', scale: 100, unit: '%', min: 0, max: 10, step: 0.01, restart: true, when: 'ptz', advanced: true,
      label: 'Investigate above (% of frame)', hint: 'Smallest wide-angle box worth a look.' },
    { key: 'ptz_tracking.investigate_timeout', kind: 'number', min: 0.5, max: 30, step: 0.5, restart: true, when: 'ptz', advanced: true,
      label: 'Investigate timeout (seconds)', hint: 'Time the zoom camera has to confirm before the spot is rejected.' },
    { key: 'ptz_tracking.investigate_cooldown', kind: 'number', min: 0, max: 600, step: 5, restart: true, when: 'ptz', advanced: true,
      label: 'Investigate cooldown (seconds)', hint: 'Do not revisit a rejected spot for this long.' },
    { key: 'ptz_tracking.investigate_cooldown_radius', kind: 'number', min: 0, max: 0.5, step: 0.01, restart: true, when: 'ptz', advanced: true,
      label: 'Cooldown radius (fraction of frame)', hint: 'How close to a rejected spot counts as the same spot.' },
    { key: 'ptz_tracking.min_detection_area', kind: 'number', scale: 100, unit: '%', min: 0, max: 10, step: 0.05, restart: true, when: 'ptz', advanced: true,
      label: 'Tracker minimum size (% of frame)', hint: 'Detections below this never steer the head.' },
    { key: 'ptz_tracking.pan_scale', kind: 'number', min: 0.1, max: 2, step: 0.05, restart: true, when: 'ptz', advanced: true,
      label: 'Pan scale', hint: 'PTZ pan range as a fraction of the wide-angle field of view (calibration).' },
    { key: 'ptz_tracking.tilt_scale', kind: 'number', min: 0.1, max: 2, step: 0.05, restart: true, when: 'ptz', advanced: true,
      label: 'Tilt scale', hint: 'PTZ tilt range as a fraction of the wide-angle field of view.' },
    { key: 'ptz_tracking.pan_center_x', kind: 'number', min: 0, max: 1, step: 0.01, restart: true, when: 'ptz', advanced: true,
      label: 'Pan centre X', hint: 'Where PTZ (0,0) lands on the wide frame, 0–1 from the left.' },
    { key: 'ptz_tracking.tilt_center_y', kind: 'number', min: 0, max: 1, step: 0.01, restart: true, when: 'ptz', advanced: true,
      label: 'Tilt centre Y', hint: 'Where PTZ (0,0) lands on the wide frame, 0–1 from the top.' },
    { key: 'ptz_tracking.smoothing', kind: 'number', min: 0, max: 0.9, step: 0.05, restart: true, when: 'ptz', advanced: true,
      label: 'Smoothing', hint: '0 reacts instantly; 0.9 is very smooth.' },
    { key: 'ptz_tracking.update_interval', kind: 'number', min: 0.05, max: 2, step: 0.05, restart: true, when: 'ptz', advanced: true,
      label: 'Update interval (seconds)', hint: 'Time between PTZ commands.' },
    { key: 'ptz_tracking.move_min_duration', kind: 'number', min: 0, max: 5, step: 0.1, restart: true, when: 'ptz', advanced: true,
      label: 'Minimum move (seconds)', hint: 'A tracking move runs at least this long before a no-detection tick may stop it.' },
    { key: 'ptz_tracking.tracking_step_duration', kind: 'number', min: 0.05, max: 2, step: 0.05, restart: true, when: 'ptz', advanced: true,
      label: 'Maximum move pulse (seconds)', hint: 'A tracking move is stopped automatically after this long.' },
    { key: 'ptz_tracking.low_fill_threshold', kind: 'number', min: 0.01, max: 1, step: 0.01, restart: true, when: 'ptz', advanced: true,
      label: 'Low-fill threshold', hint: 'Below this frame fill the velocity caps apply.' },
    { key: 'ptz_tracking.low_fill_velocity_cap', kind: 'number', min: 0.01, max: 1, step: 0.01, restart: true, when: 'ptz', advanced: true,
      label: 'Low-fill velocity cap', hint: 'Top pan/tilt speed on small targets.' },
    { key: 'ptz_tracking.low_fill_cap_full_offset', kind: 'number', min: 0.01, max: 1, step: 0.01, restart: true, when: 'ptz', advanced: true,
      label: 'Low-fill cap full offset', hint: 'Offset at which the cap reaches its full value.' },
    { key: 'ptz_tracking.cam1_fallback_delay', kind: 'number', min: 0, max: 30, step: 0.5, restart: true, when: 'ptz', advanced: true,
      label: 'Source fallback delay (seconds)', hint: 'After the target camera drove tracking, suppress source-camera repositioning this long.' },
    { key: 'ptz_tracking.zoom_fov_calibration_path', kind: 'text', mono: true, nullable: true, restart: true, when: 'ptz', advanced: true,
      label: 'Zoom FOV calibration file', placeholder: 'config/zoom_fov_calibration.json',
      hint: 'Created by the zoom-calibrate command; maps the zoom view into the wide frame.' },
    { key: 'ptz_tracking.visibility_recovery_enabled', kind: 'switch', restart: true, when: 'ptz', advanced: true,
      label: 'Visibility recovery', hint: 'Use the wide camera plus the calibration to recentre or zoom out when the zoom camera loses its target.' },
    { key: 'ptz_tracking.visibility_recovery_min_overlap', kind: 'number', min: 0, max: 1, step: 0.05, restart: true, when: 'ptz', advanced: true,
      label: 'Recovery minimum overlap', hint: 'Fraction of the wide detection that must fall inside the predicted zoom view to count as visible.' },
    { key: 'ptz_tracking.visibility_recovery_edge_margin', kind: 'number', min: 0, max: 0.5, step: 0.01, restart: true, when: 'ptz', advanced: true,
      label: 'Recovery edge margin', hint: 'Fraction of the zoom view treated as edge; edge targets trigger recentre and zoom-out.' },
    { key: 'ptz_tracking.visibility_recovery_zoom_out_velocity', kind: 'number', min: -1, max: 0, step: 0.05, restart: true, when: 'ptz', advanced: true,
      label: 'Recovery zoom-out velocity', hint: 'Negative values zoom out.' },
    { key: 'ptz_tracking.visibility_recovery_zoom_in_velocity', kind: 'number', min: 0, max: 1, step: 0.05, restart: true, when: 'ptz', advanced: true,
      label: 'Recovery zoom-in velocity', hint: 'Used only when the target is centred and the zoom camera is still wide.' },
    { key: 'ptz_tracking.visibility_recovery_zoom_in_max_zoom', kind: 'number', min: 0, max: 1, step: 0.05, restart: true, when: 'ptz', advanced: true,
      label: 'Recovery zoom-in ceiling', hint: 'Recovery may zoom in only while the current zoom is below this.' },
    { key: 'ptz_tracking.visibility_recovery_zoom_in_fill_threshold', kind: 'number', min: 0, max: 0.5, step: 0.01, restart: true, when: 'ptz', advanced: true,
      label: 'Recovery zoom-in fill threshold', hint: 'Largest wide-frame fill that still justifies a cautious zoom-in.' },
    { key: 'ptz_tracking.visibility_recovery_velocity_cap', kind: 'number', min: 0.01, max: 1, step: 0.01, restart: true, when: 'ptz', advanced: true,
      label: 'Recovery velocity cap', hint: 'Top pan/tilt speed for recovery pulses.' }
  ] },
  { id: 'species', legend: 'Species filters', fields: [
    { key: 'include_species', kind: 'species',
      label: 'Detect only these species', hint: 'Leave empty to detect everything.',
      emptyMeans: 'Nothing selected — every species is detected.' },
    { key: 'exclude_species', kind: 'species', recent: true,
      label: 'Always ignore these species',
      hint: 'Ignored even when detected. Recent detections on this camera come first, with their clip counts.',
      emptyMeans: 'Nothing excluded on this camera.' }
  ] },
  { id: 'notify', legend: 'Notifications', fields: [
    { key: 'notification.priority', kind: 'select', options: PRIORITY_OPTIONS, numeric: true,
      label: 'Priority', hint: 'Pushover priority for alerts from this camera.' },
    { key: 'notification.sound', kind: 'select', options: SOUND_OPTIONS, nullable: true,
      label: 'Sound', hint: 'Pushover sound.' },
    { key: 'notification.destinations', kind: 'recipients', nullable: true,
      label: 'Send alerts to', hint: 'Every destination defined under General → Notifications, or only the ones ticked here.' }
  ] }
];

/* Flat views of the inventory, built once. */
var GENERAL_SPECS = [];
var GENERAL_SECTION_BY_KEY = {};
var CAMERA_SPECS = [];
(function buildIndexes() {
  var i, j, k;
  for (i = 0; i < GENERAL_SECTIONS.length; i++) {
    var sec = GENERAL_SECTIONS[i];
    for (j = 0; j < sec.groups.length; j++) {
      for (k = 0; k < sec.groups[j].fields.length; k++) {
        var spec = sec.groups[j].fields[k];
        spec.parts = spec.key.split('.');
        GENERAL_SPECS.push(spec);
        GENERAL_SECTION_BY_KEY[spec.key] = sec.id;
      }
    }
  }
  for (i = 0; i < CAMERA_GROUPS.length; i++) {
    for (j = 0; j < CAMERA_GROUPS[i].fields.length; j++) {
      var cspec = CAMERA_GROUPS[i].fields[j];
      cspec.parts = cspec.key.split('.');
      CAMERA_SPECS.push(cspec);
    }
  }
}());

/* --------------------------------------------------------------------------
   TINY UTILITIES
   ------------------------------------------------------------------------ */

var uidN = 0;
function uid(prefix) { uidN += 1; return prefix + '-' + uidN; }

function clone(v) { return JSON.parse(JSON.stringify(v)); }

function isArray(v) { return Object.prototype.toString.call(v) === '[object Array]'; }

function getAt(obj, path) {
  var cur = obj;
  for (var i = 0; i < path.length; i++) {
    if (cur === null || cur === undefined) return undefined;
    cur = cur[path[i]];
  }
  return cur;
}

function setAt(obj, path, value) {
  var cur = obj;
  for (var i = 0; i < path.length - 1; i++) {
    if (cur[path[i]] === null || typeof cur[path[i]] !== 'object') cur[path[i]] = {};
    cur = cur[path[i]];
  }
  cur[path[path.length - 1]] = value;
}

function pathKey(path) { return path.join(''); }

function normList(v) {
  if (!isArray(v)) return '';
  var out = [];
  for (var i = 0; i < v.length; i++) {
    var item = v[i];
    out.push(item && typeof item === 'object'
      ? JSON.stringify(item, Object.keys(item).sort())
      : String(item).toLowerCase());
  }
  out.sort();
  return out.join('\u0001');
}

function eqValue(a, b) {
  if (isArray(a) || isArray(b)) {
    /* A list and "no list" differ: for a camera's destinations null means
       every destination and [] means none. */
    if (!isArray(a) || !isArray(b)) return false;
    return normList(a) === normList(b);
  }
  var aEmpty = a === null || a === undefined || a === '';
  var bEmpty = b === null || b === undefined || b === '';
  if (aEmpty || bEmpty) return aEmpty && bEmpty;
  if (typeof a === 'boolean' || typeof b === 'boolean') return !!a === !!b;
  if (typeof a === 'number' || typeof b === 'number') return Number(a) === Number(b);
  return String(a) === String(b);
}

function titleCase(s) {
  var str = String(s || '');
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function plural(n, one, many) { return n === 1 ? one : (many || one + 's'); }

function toNumber(v) {
  if (v === null || v === undefined || v === '') return NaN;
  var n = Number(v);
  return isFinite(n) ? n : NaN;
}

function roundTo(value, step) {
  if (!step) return value;
  var r = Math.round(value / step) * step;
  return Math.round(r * 1e6) / 1e6;
}

function decimalsOf(step) {
  var s = String(step);
  var e = s.indexOf('e-');
  if (e >= 0) return Number(s.slice(e + 2));
  var i = s.indexOf('.');
  return i < 0 ? 0 : s.length - i - 1;
}

function num(v, fallback) {
  var n = Number(v);
  return v === null || v === undefined || v === '' || !isFinite(n) ? fallback : n;
}

function bool(v, fallback) {
  if (v === true || v === false) return v;
  if (v === null || v === undefined) return fallback;
  return !!v;
}

function strList(v) {
  if (!isArray(v)) return [];
  var out = [];
  for (var i = 0; i < v.length; i++) {
    var s = String(v[i] === null || v[i] === undefined ? '' : v[i]).trim();
    if (s) out.push(s);
  }
  return out;
}

function envName(id, kind) {
  return String(id || 'cam').toUpperCase().replace(/[^A-Z0-9]/g, '_') + '_ONVIF_' + kind;
}

/* A Pushover destination as the draft holds it: four strings, never null. */
function normDestination(d) {
  var src = d && typeof d === 'object' ? d : {};
  function str(v) { return v === null || v === undefined ? '' : String(v).trim(); }
  return { id: str(src.id), name: str(src.name), user_key_env: str(src.user_key_env), app_token_env: str(src.app_token_env) };
}

function destList(v) {
  if (!isArray(v)) return [];
  var out = [];
  for (var i = 0; i < v.length; i++) out.push(normDestination(v[i]));
  return out;
}

function destinationIds(model) {
  var list = getAt(model, ['general', 'notification', 'destinations']);
  var ids = [];
  if (isArray(list)) for (var i = 0; i < list.length; i++) if (list[i] && list[i].id) ids.push(String(list[i].id));
  return ids;
}

/* A camera's pick, reduced to destinations that exist (a hand-edited file
   can name one that was removed; the server refuses those on save). */
function knownRecipients(list, model) {
  var ids = destinationIds(model);
  var out = [];
  for (var i = 0; i < list.length; i++) if (ids.indexOf(list[i]) >= 0 && out.indexOf(list[i]) < 0) out.push(list[i]);
  return out;
}

function hostFromUri(uri) {
  var m = /^[a-z]+:\/\/(?:[^@\/]*@)?([^:\/?#]+)/i.exec(String(uri || ''));
  return m ? m[1] : '';
}

/* --------------------------------------------------------------------------
   NORMALISATION — the API payload becomes the draft model.
   model = { general: {...}, cameras: { id: cam }, order: [id] }
   ------------------------------------------------------------------------ */

function coerce(spec, v) {
  switch (spec.kind) {
    case 'number':
      if (spec.nullable && (v === null || v === undefined || v === '')) return null;
      return num(v, spec.min === undefined ? 0 : Math.max(0, spec.min));
    case 'pct':
    case 'slider':
      return num(v, 0);
    case 'switch':
      return bool(v, false);
    case 'select':
      if (spec.numeric) return num(v, 0);
      return v === null || v === undefined ? '' : String(v);
    case 'text':
    case 'env':
      return v === null || v === undefined ? '' : String(v);
    case 'list':
    case 'species':
      return strList(v);
    case 'destinations':
      return destList(v);
    case 'recipients':
      return isArray(v) ? strList(v) : null;
    default:
      return v;
  }
}

function normalizeGeneral(g) {
  var src = g && typeof g === 'object' ? g : {};
  var out = {};
  for (var i = 0; i < GENERAL_SPECS.length; i++) {
    var spec = GENERAL_SPECS[i];
    setAt(out, spec.parts, coerce(spec, getAt(src, spec.parts)));
  }
  return out;
}

function normalizeCamera(c, id) {
  var src = c && typeof c === 'object' ? c : {};
  var cam = { id: String(id) };
  var onvifSrc = src.onvif && typeof src.onvif === 'object' ? src.onvif : null;
  for (var i = 0; i < CAMERA_SPECS.length; i++) {
    var spec = CAMERA_SPECS[i];
    if (spec.key === 'onvif.enabled') {
      setAt(cam, spec.parts, !!(onvifSrc && onvifSrc.host));
      continue;
    }
    setAt(cam, spec.parts, coerce(spec, getAt(src, spec.parts)));
  }
  if (!cam.onvif.port) cam.onvif.port = 80;
  if (!cam.onvif.username_env) cam.onvif.username_env = envName(id, 'USER');
  if (!cam.onvif.password_env) cam.onvif.password_env = envName(id, 'PASS');
  cam.runtime = src.runtime && typeof src.runtime === 'object' ? src.runtime : { running: false };
  cam.recent_detections = src.recent_detections && typeof src.recent_detections === 'object'
    ? src.recent_detections : {};
  return cam;
}

function normalize(raw) {
  var src = raw && typeof raw === 'object' ? raw : {};
  var model = { general: normalizeGeneral(src.general), cameras: {}, order: [] };
  var cams = isArray(src.cameras) ? src.cameras : [];
  for (var i = 0; i < cams.length; i++) {
    var c = cams[i] || {};
    var id = c.id === null || c.id === undefined ? '' : String(c.id);
    if (!id || model.cameras[id]) continue;
    model.cameras[id] = normalizeCamera(c, id);
    model.order.push(id);
  }
  for (var p = 0; p < model.order.length; p++) {
    var pick = model.cameras[model.order[p]].notification;
    if (pick && isArray(pick.destinations)) pick.destinations = knownRecipients(pick.destinations, model);
  }
  return model;
}

/* Visibility of a `when` field on a camera draft. */
function blockOn(cam, which) {
  if (!cam) return false;
  if (which === 'onvif') return !!(cam.onvif && cam.onvif.enabled);
  if (which === 'ptz') return !!(cam.ptz_tracking && (cam.ptz_tracking.enabled || cam.ptz_tracking.self_track));
  return true;
}

function specVisible(spec, cam) {
  return !spec.when || blockOn(cam, spec.when);
}

/* --------------------------------------------------------------------------
   THE PAYLOAD — derived from the same inventory. Throws rather than emit
   something partial: a half-object here is a half-written cameras.yml.
   ------------------------------------------------------------------------ */

function PayloadError(message) {
  var e = new Error(message);
  e.name = 'PayloadError';
  return e;
}

function serialize(spec, v) {
  var s;
  switch (spec.kind) {
    case 'number': {
      if (v === null || v === undefined || v === '') {
        if (spec.nullable) return null;
        throw PayloadError(spec.label + ' is empty.');
      }
      var n = Number(v);
      if (!isFinite(n)) throw PayloadError(spec.label + ' is not a number.');
      return n;
    }
    case 'pct':
    case 'slider': {
      var m = Number(v);
      if (!isFinite(m)) throw PayloadError(spec.label + ' is not a number.');
      return m;
    }
    case 'switch':
      return !!v;
    case 'select':
      if (spec.numeric) return Number(v);
      s = v === null || v === undefined ? '' : String(v);
      if (spec.nullable && !s) return null;
      return s;
    case 'text':
    case 'env':
      s = String(v === null || v === undefined ? '' : v).trim();
      if (spec.upper) s = s.toUpperCase();
      if (spec.nullable && !s) return null;
      return s;
    case 'list':
    case 'species':
      return strList(v);
    case 'destinations': {
      var dests = destList(v);
      var outList = [];
      for (var d = 0; d < dests.length; d++) {
        outList.push({
          id: dests[d].id, name: dests[d].name || null,
          user_key_env: dests[d].user_key_env, app_token_env: dests[d].app_token_env || null
        });
      }
      return outList;
    }
    case 'recipients':
      return isArray(v) ? strList(v) : null;
    default:
      return v;
  }
}

function buildGeneralPayload(model) {
  var out = {};
  for (var i = 0; i < GENERAL_SPECS.length; i++) {
    var spec = GENERAL_SPECS[i];
    setAt(out, spec.parts, serialize(spec, getAt(model.general, spec.parts)));
  }
  return out;
}

function buildCameraPayload(model, id) {
  var cam = model.cameras[id];
  if (!cam) throw PayloadError('Camera ' + id + ' is missing from the draft.');
  var out = { id: id };
  for (var i = 0; i < CAMERA_SPECS.length; i++) {
    var spec = CAMERA_SPECS[i];
    if (spec.parts[0] === 'onvif') continue;
    /* Hidden `when` fields are still sent: the block is written whole, and
       they keep the last value the operator saw. */
    var value = serialize(spec, getAt(cam, spec.parts));
    if (spec.kind === 'recipients' && isArray(value)) value = knownRecipients(value, model);
    setAt(out, spec.parts, value);
  }
  if (cam.onvif && cam.onvif.enabled) {
    var host = String(cam.onvif.host || '').trim();
    if (!host) throw PayloadError('Camera ' + id + ': ONVIF is on but has no host.');
    var profile = String(cam.onvif.profile || '').trim();
    out.onvif = {
      host: host,
      port: Number(cam.onvif.port) || 80,
      profile: profile || null,
      username_env: String(cam.onvif.username_env || '').trim(),
      password_env: String(cam.onvif.password_env || '').trim()
    };
  } else {
    out.onvif = null;
  }
  return out;
}

function buildPayload(model) {
  if (!model.order.length) throw PayloadError('No cameras are in the draft — refusing to write an empty camera set.');
  var cameras = [];
  for (var i = 0; i < model.order.length; i++) cameras.push(buildCameraPayload(model, model.order[i]));
  var payload = { general: buildGeneralPayload(model), cameras: cameras };
  if (!payload.general.clip || !payload.general.retention || !payload.general.detector ||
      !payload.general.notification || !isArray(payload.general.exclusion_list)) {
    throw PayloadError('The settings payload came out incomplete — nothing was sent.');
  }
  return payload;
}

/* --------------------------------------------------------------------------
   VALIDATION — the same rules the controls enforce, run over the whole
   draft (an edit in a section you are not looking at still counts).
   ------------------------------------------------------------------------ */

function displayRange(spec) {
  var unit = spec.unit || '';
  return spec.min + unit + ' and ' + spec.max + unit;
}

function validateSpec(spec, value, path, prefix, out) {
  var label = (prefix ? prefix + ' — ' : '') + spec.label;
  var s, n;
  switch (spec.kind) {
    case 'number':
      if (value === null || value === undefined || value === '') {
        if (!spec.nullable) out.push({ path: path, message: label + ' must be a number.' });
        return;
      }
      n = toNumber(value);
      if (!isFinite(n)) { out.push({ path: path, message: label + ' must be a number.' }); return; }
      n = n * (spec.scale || 1);
      if (n < spec.min - 1e-9 || n > spec.max + 1e-9) {
        out.push({ path: path, message: label + ' must be between ' + displayRange(spec) + '.' });
      }
      return;
    case 'pct':
      n = toNumber(value);
      if (!isFinite(n)) { out.push({ path: path, message: label + ' must be a number.' }); return; }
      n = Math.round(n * 100);
      if (n < (spec.min === undefined ? 0 : spec.min) || n > (spec.max === undefined ? 100 : spec.max)) {
        out.push({ path: path, message: label + ' must be between ' + (spec.min === undefined ? 0 : spec.min) + '% and ' + (spec.max === undefined ? 100 : spec.max) + '%.' });
      }
      return;
    case 'slider':
      n = toNumber(value);
      if (!isFinite(n)) { out.push({ path: path, message: label + ' must be a number.' }); return; }
      if (n < spec.min || n > spec.max) out.push({ path: path, message: label + ' must be between ' + displayRange(spec) + '.' });
      return;
    case 'text':
    case 'env':
      s = String(value === null || value === undefined ? '' : value).trim();
      if (spec.upper) s = s.toUpperCase();
      if (!s) {
        if (spec.required) out.push({ path: path, message: label + ' is required.' });
        return;
      }
      if (spec.kind === 'env' && !ENV_RE.test(s)) {
        out.push({ path: path, message: label + ' must be an environment variable name (letters, digits and underscores).' });
        return;
      }
      if (spec.pattern && !spec.pattern.test(s)) {
        out.push({ path: path, message: label + ' must be ' + (spec.patternHint || 'in the expected format') + '.' });
      }
      return;
    case 'destinations': {
      var dests = destList(value);
      var seenIds = {};
      for (var i = 0; i < dests.length; i++) {
        var d = dests[i];
        var row = path.concat([String(i)]);
        var who = d.name || d.id || ('destination ' + (i + 1));
        if (!CAMERA_ID_RE.test(d.id)) {
          out.push({ path: row.concat(['id']), message: label + ' — ' + who + ': the id must be letters, digits, - or _ (up to 32 characters).' });
        } else if (seenIds[d.id]) {
          out.push({ path: row.concat(['id']), message: label + ' — the id "' + d.id + '" is used twice.' });
        }
        seenIds[d.id] = 1;
        if (!d.user_key_env) {
          out.push({ path: row.concat(['user_key_env']), message: label + ' — ' + who + ': the user key variable is required.' });
        } else if (!ENV_RE.test(d.user_key_env)) {
          out.push({ path: row.concat(['user_key_env']), message: label + ' — ' + who + ': the user key variable must be an environment variable name (letters, digits and underscores).' });
        }
        if (d.app_token_env && !ENV_RE.test(d.app_token_env)) {
          out.push({ path: row.concat(['app_token_env']), message: label + ' — ' + who + ': the app token variable must be an environment variable name (letters, digits and underscores).' });
        }
      }
      return;
    }
    default:
      return;
  }
}

function validateModel(model) {
  var out = [];
  var i, spec, path;

  for (i = 0; i < GENERAL_SPECS.length; i++) {
    spec = GENERAL_SPECS[i];
    path = ['general'].concat(spec.parts);
    validateSpec(spec, getAt(model, path), path, '', out);
  }
  var minD = Number(getAt(model, ['general', 'retention', 'min_days']));
  var maxD = Number(getAt(model, ['general', 'retention', 'max_days']));
  if (isFinite(minD) && isFinite(maxD) && minD > maxD) {
    out.push({
      path: ['general', 'retention', 'min_days'],
      message: 'Keep at least (' + minD + ' d) cannot exceed keep at most (' + maxD + ' d).'
    });
  }
  var dests = destList(getAt(model, ['general', 'notification', 'destinations']));
  var fallback = String(getAt(model, ['general', 'notification', 'pushover_user_key_env']) || '').trim();
  if (!dests.length && !fallback) {
    out.push({ path: ['general', 'notification', 'pushover_user_key_env'],
      message: 'Add a destination or set the fallback user key variable — otherwise no alert can be sent.' });
  }

  for (var c = 0; c < model.order.length; c++) {
    var id = model.order[c];
    var cam = model.cameras[id];
    if (!cam) continue;
    var prefix = cam.name || id;
    for (i = 0; i < CAMERA_SPECS.length; i++) {
      spec = CAMERA_SPECS[i];
      if (!specVisible(spec, cam)) continue;
      path = ['cameras', id].concat(spec.parts);
      var value = getAt(cam, spec.parts);
      if (spec.requiredWhen && blockOn(cam, spec.requiredWhen)) {
        var sv = String(value === null || value === undefined ? '' : value).trim();
        if (!sv) { out.push({ path: path, message: prefix + ' — ' + spec.label + ' is required while ' + (spec.requiredWhen === 'onvif' ? 'ONVIF control' : 'auto-tracking') + ' is on.' }); continue; }
      }
      validateSpec(spec, value, path, prefix, out);
    }
    var ptz = cam.ptz_tracking || {};
    var target = String(ptz.target_camera_id || '');
    if (target && target === id) {
      out.push({ path: ['cameras', id, 'ptz_tracking', 'target_camera_id'],
        message: prefix + ' — a camera cannot target itself; use Self-track instead.' });
    } else if (target && !model.cameras[target]) {
      out.push({ path: ['cameras', id, 'ptz_tracking', 'target_camera_id'],
        message: prefix + ' — target camera "' + target + '" is not in the configuration.' });
    } else if (target && model.cameras[target] && !blockOn(model.cameras[target], 'onvif')) {
      out.push({ path: ['cameras', id, 'ptz_tracking', 'target_camera_id'],
        message: prefix + ' — target camera "' + (model.cameras[target].name || target) + '" has no ONVIF control, so its head cannot be moved.' });
    }
    if (ptz.enabled && !target && !ptz.self_track) {
      out.push({ path: ['cameras', id, 'ptz_tracking', 'target_camera_id'],
        message: prefix + ' — auto-tracking is on but neither a target camera nor Self-track is set, so it would do nothing.' });
    }
    if (ptz.self_track && !blockOn(cam, 'onvif')) {
      out.push({ path: ['cameras', id, 'ptz_tracking', 'self_track'],
        message: prefix + ' — self-tracking needs ONVIF control on this camera.' });
    }
  }
  return out;
}

/* --------------------------------------------------------------------------
   A VERY SMALL YAML WRITER  (preview only — never parsed back)
   ------------------------------------------------------------------------ */

function yamlScalar(v) {
  if (v === null || v === undefined) return 'null';
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (typeof v === 'number') return String(v);
  var s = String(v);
  if (s === '' || /[:#\-{}\[\],&*?|>'"%@`]/.test(s) || /^\s|\s$/.test(s) || /^[0-9.]+$/.test(s) ||
      /^(true|false|null|yes|no|on|off)$/i.test(s)) {
    return '"' + s.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
  }
  return s;
}

function toYaml(value, indent) {
  var pad = new Array(indent + 1).join('  ');
  var lines = [];
  var k, i;
  if (isArray(value)) {
    if (!value.length) return pad + '[]\n';
    for (i = 0; i < value.length; i++) {
      var item = value[i];
      if (item && typeof item === 'object' && !isArray(item)) {
        /* The object's lines sit one level in; the first takes the dash. */
        var block = toYaml(item, indent + 1).replace(/\n$/, '');
        lines.push(pad + '- ' + block.slice(pad.length + 2));
      } else {
        lines.push(pad + '- ' + yamlScalar(item));
      }
    }
    return lines.join('\n') + '\n';
  }
  if (value && typeof value === 'object') {
    for (k in value) {
      if (!Object.prototype.hasOwnProperty.call(value, k)) continue;
      var v = value[k];
      if (v && typeof v === 'object') {
        if (isArray(v) && !v.length) { lines.push(pad + k + ': []'); continue; }
        lines.push(pad + k + ':');
        lines.push(toYaml(v, indent + 1).replace(/\n$/, ''));
      } else {
        lines.push(pad + k + ': ' + yamlScalar(v));
      }
    }
    return lines.join('\n') + '\n';
  }
  return pad + yamlScalar(value) + '\n';
}

/* ==========================================================================
   SESSION STATE
   ========================================================================= */

var S = null;

function newSession() {
  return {
    root: null,
    baseline: null,      /* what the server last confirmed */
    draft: null,         /* what the operator is editing */
    env: {},             /* env var name -> present? */
    secrets: null,       /* { file, exists, variables: [{ name, used_by, live }] } */
    defaults: {},        /* schema defaults from the server */
    restart: null,       /* { required, reasons, supported, unit } */
    configPath: '',
    backupDir: '',
    section: DEFAULT_SECTION,
    fields: [],          /* controllers for the CURRENTLY RENDERED section */
    fieldByKey: {},
    groupHosts: {},      /* group id -> { el, group, ctx } for targeted re-render */
    openAdvanced: {},    /* group id -> details open? */
    offs: [],
    abort: null,
    refreshAbort: null,
    saving: false,
    restarting: false,
    destroyed: false,
    leaveDialog: null,   /* the "discard unsaved changes?" dialog, while open */
    unapplied: null,     /* live settings in the file the process has not taken */
    navGeneral: null,
    navCameras: null,
    panelEl: null,
    bannerEl: null,
    selbar: null,
    saveBtn: null,
    resetBtn: null,
    countEl: null,
    countDetailEl: null,
    selbarShown: false,
    contentEl: null,
    savedTimers: [],
    envTimer: null,
    envPending: {}
  };
}

function track(off) { if (off) S.offs.push(off); return off; }

function currentCamera() {
  return S.draft && S.draft.cameras[S.section] ? S.draft.cameras[S.section] : null;
}

/* ==========================================================================
   FIELD CONTROLLERS
   Each returns { path, key, el, setDirty, setError, setStatus, focus, sync }.
   ========================================================================= */

function restartBadge() {
  return h('span.badge.sp--unknown', { title: 'Takes effect after a restart', text: 'Restart' });
}

function fieldShell(o) {
  var labelId = uid('lbl');
  var hintId = uid('hint');
  var el = h('div.field');
  var dot = h('span.field__dirty', { hidden: true, 'aria-hidden': 'true' });
  var labelEl = h((o.labelTag || 'label.field__label') + '#' + labelId, { text: o.label });
  labelEl.appendChild(dot);
  if (o.restart) labelEl.appendChild(restartBadge());
  var statusEl = h('span.field__status', { 'aria-live': 'polite' });
  var hintEl = h('p.field__hint#' + hintId, { text: o.hint || '' });
  var errEl = h('p.field__error', { hidden: true, role: 'alert' });
  return {
    el: el, dot: dot, labelEl: labelEl, labelId: labelId,
    hintEl: hintEl, hintId: hintId, errEl: errEl, statusEl: statusEl
  };
}

function baseController(shell, path, extra) {
  var savedTimer = null;
  var ctl = {
    path: path,
    key: pathKey(path),
    el: shell.el,
    label: extra.label,
    focus: extra.focus,
    sync: extra.sync,
    setDirty: function (on_) {
      shell.dot.hidden = !on_;
      if (on_) shell.dot.setAttribute('title', 'Changed — not yet saved');
    },
    setError: function (msg) {
      if (msg) {
        shell.errEl.hidden = false;
        clear(shell.errEl);
        shell.errEl.appendChild(icon('alert', { size: 'sm' }));
        shell.errEl.appendChild(h('span', { text: msg }));
        shell.hintEl.hidden = true;
        if (extra.setInvalid) extra.setInvalid(true);
      } else {
        shell.errEl.hidden = true;
        clear(shell.errEl);
        shell.hintEl.hidden = false;
        if (extra.setInvalid) extra.setInvalid(false);
      }
    },
    setStatus: function (kind, text) {
      clear(shell.statusEl);
      shell.statusEl.className = 'field__status';
      if (!kind) return;
      if (kind === 'saved') {
        shell.statusEl.classList.add('field__saved');
        shell.statusEl.appendChild(icon('check', { size: 'sm' }));
        shell.statusEl.appendChild(h('span', { text: text || 'Saved' }));
        if (savedTimer) clearTimeout(savedTimer);
        savedTimer = setTimeout(function () {
          if (shell.statusEl.isConnected) { clear(shell.statusEl); shell.statusEl.className = 'field__status'; }
        }, 2400);
        S.savedTimers.push(savedTimer);
      } else if (kind === 'error') {
        shell.statusEl.appendChild(icon('alert', { size: 'sm' }));
        shell.statusEl.appendChild(h('span.t-danger', { text: text || 'Not saved' }));
      }
    }
  };
  return ctl;
}

/* --- read-only fact ------------------------------------------------------ */

function readonlyField(o) {
  var shell = fieldShell({ label: o.label, hint: o.hint });
  var input = h('input.input', {
    type: 'text', value: o.value === null || o.value === undefined ? '' : String(o.value),
    disabled: true, readonly: true, 'aria-describedby': shell.hintId
  });
  input.id = uid('ro');
  shell.labelEl.setAttribute('for', input.id);
  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(input);
  shell.el.appendChild(shell.hintEl);
  return { el: shell.el, readonly: true };
}

/* --- number + stepper ---------------------------------------------------- */

function numberField(o) {
  var shell = fieldShell({ label: o.label, hint: o.hint, restart: o.restart });
  var scale = o.scale || 1;
  var decimals = decimalsOf(o.step);
  var displayStep = Math.pow(10, -decimals);
  var inputId = uid('num');
  var input = h('input.input#' + inputId, {
    type: 'number', inputmode: 'decimal',
    min: String(o.min), max: String(o.max), step: String(o.step),
    placeholder: o.nullable ? 'empty' : null,
    'aria-describedby': shell.hintId
  });
  shell.labelEl.setAttribute('for', inputId);

  var dec = h('button.icon-btn.icon-btn--dense', {
    type: 'button', 'aria-label': 'Decrease ' + o.label
  }, icon('minus'));
  var inc = h('button.icon-btn.icon-btn--dense', {
    type: 'button', 'aria-label': 'Increase ' + o.label
  }, icon('plus'));
  var unitEl = o.unit ? h('span.field__hint', { text: o.unit, 'aria-hidden': 'true' }) : null;

  var stepper = h('div.stepper', dec, input, unitEl, inc);
  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(stepper);
  shell.el.appendChild(shell.hintEl);
  shell.el.appendChild(shell.errEl);
  shell.el.appendChild(shell.statusEl);

  function toDisplay(m) { return roundTo(m * scale, displayStep); }
  function toModel(d) { return o.int ? Math.round(d) : roundTo(d / scale, 1e-6); }

  function currentDisplay() {
    var m = toNumber(getAt(S.draft, o.path));
    return isFinite(m) ? toDisplay(m) : null;
  }

  function render() {
    var v = currentDisplay();
    input.value = v === null ? '' : String(v);
    dec.disabled = v !== null && v <= o.min;
    inc.disabled = v !== null && v >= o.max;
  }

  function commit(next, fromTyping) {
    if (next === '' || next === null) {
      if (!o.nullable) { ctl.setError(o.label + ' must be a number.'); return; }
      ctl.setError(null);
      setAt(S.draft, o.path, null);
      if (!fromTyping) render();
      onModelChanged();
      return;
    }
    var n = toNumber(next);
    if (!isFinite(n)) { ctl.setError(o.label + ' must be a number.'); return; }
    if (n < o.min || n > o.max) {
      ctl.setError(o.label + ' must be between ' + displayRange(o) + '.');
      /* Do NOT write an out-of-range value into the draft: the model stays
         sendable at all times, and the error says why. */
      return;
    }
    ctl.setError(null);
    var model = toModel(n);
    if (!eqValue(model, getAt(S.draft, o.path))) {
      setAt(S.draft, o.path, model);
      onModelChanged();
    }
    if (!fromTyping) render();
    else { dec.disabled = n <= o.min; inc.disabled = n >= o.max; }
  }

  function nudge(dir, mult) {
    var step = o.step * (mult || 1);
    var cur = currentDisplay();
    var v = (cur === null ? Math.min(Math.max(0, o.min), o.max) : cur) + dir * step;
    v = Math.min(o.max, Math.max(o.min, roundTo(v, displayStep)));
    commit(v, false);
    input.focus();
  }

  track(on(dec, 'click', function () { nudge(-1, 1); }));
  track(on(inc, 'click', function () { nudge(1, 1); }));
  track(on(input, 'input', function () {
    /* An empty box is mid-edit, not an error: say nothing until blur. */
    if (input.value === '' || input.value === '-') { ctl.setError(null); return; }
    commit(input.value, true);
  }));
  track(on(input, 'blur', function () {
    if (input.value === '') { if (o.nullable) commit('', false); else render(); ctl.setError(null); return; }
    var n = toNumber(input.value);
    if (!isFinite(n)) { render(); ctl.setError(null); return; }
    commit(Math.min(o.max, Math.max(o.min, n)), false);
  }));
  track(on(input, 'keydown', function (ev) {
    if (ev.key !== 'ArrowUp' && ev.key !== 'ArrowDown') return;
    if (!ev.shiftKey) return;
    ev.preventDefault();
    nudge(ev.key === 'ArrowUp' ? 1 : -1, 10);
  }));

  var ctl = baseController(shell, o.path, {
    label: o.label,
    focus: function () { input.focus(); },
    sync: function () { render(); ctl.setError(null); },
    setInvalid: function (bad) {
      if (bad) input.setAttribute('aria-invalid', 'true');
      else input.removeAttribute('aria-invalid');
    }
  });
  render();
  return ctl;
}

/* --- slider -------------------------------------------------------------- */

function sliderField(o) {
  var shell = fieldShell({ label: o.label, hint: o.hint, restart: o.restart, labelTag: 'span.field__label' });
  var toModel = o.toModel || function (v) { return v; };
  var fromModel = o.fromModel || function (v) { return v; };
  var unit = o.unit || '';

  var fill = h('div.slider__fill');
  var knob = h('div.slider__knob');
  var rail = h('div.slider__rail', fill, knob);
  var out = h('output.slider__output');
  var el = h('div.slider', {
    role: 'slider', tabIndex: 0,
    'aria-labelledby': shell.labelId,
    'aria-describedby': shell.hintId,
    'aria-valuemin': String(o.min),
    'aria-valuemax': String(o.max)
  }, rail, out);

  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(el);
  shell.el.appendChild(shell.hintEl);
  shell.el.appendChild(shell.errEl);
  shell.el.appendChild(shell.statusEl);

  function display() {
    var m = toNumber(getAt(S.draft, o.path));
    if (!isFinite(m)) m = toModel(o.min);
    var v = fromModel(m);
    return Math.min(o.max, Math.max(o.min, v));
  }

  var ctl = null;
  function render() {
    var v = display();
    var frac = (v - o.min) / (o.max - o.min || 1);
    el.style.setProperty('--v', String(frac));
    el.setAttribute('aria-valuenow', String(v));
    el.setAttribute('aria-valuetext', v + unit);
    out.textContent = v + unit;
    if (ctl) {
      var raw = toNumber(getAt(S.draft, o.path));
      var shown = fromModel(isFinite(raw) ? raw : toModel(o.min));
      if (isFinite(shown) && (shown < o.min || shown > o.max)) {
        ctl.setError(o.label + ' is ' + shown + unit + ' on disk, outside the allowed ' +
          o.min + unit + '–' + o.max + unit + '. Move the slider to correct it.');
      } else {
        ctl.setError(null);
      }
    }
  }

  function commit(v) {
    var next = Math.min(o.max, Math.max(o.min, roundTo(v, o.step)));
    var model = roundTo(toModel(next), 0.0001);
    if (eqValue(model, getAt(S.draft, o.path))) { render(); return; }
    setAt(S.draft, o.path, model);
    render();
    onModelChanged();
  }

  function fromPointer(clientX) {
    var r = rail.getBoundingClientRect();
    if (!r.width) return;
    var frac = (clientX - r.left) / r.width;
    frac = Math.min(1, Math.max(0, frac));
    commit(o.min + frac * (o.max - o.min));
  }

  var dragging = false;
  track(on(el, 'pointerdown', function (ev) {
    if (ev.button !== undefined && ev.button !== 0) return;
    dragging = true;
    el.classList.add('slider--dragging');
    if (el.setPointerCapture) { try { el.setPointerCapture(ev.pointerId); } catch (e) {} }
    el.focus();
    fromPointer(ev.clientX);
    ev.preventDefault();
  }));
  track(on(el, 'pointermove', function (ev) { if (dragging) fromPointer(ev.clientX); }));
  function endDrag() {
    if (!dragging) return;
    dragging = false;
    el.classList.remove('slider--dragging');
  }
  track(on(el, 'pointerup', endDrag));
  track(on(el, 'pointercancel', endDrag));
  track(on(el, 'blur', endDrag));

  track(on(el, 'keydown', function (ev) {
    var v = display();
    var k = ev.key;
    var handled = true;
    if (k === 'ArrowRight' || k === 'ArrowUp') v += ev.shiftKey ? Math.min(1, o.step) : o.step;
    else if (k === 'ArrowLeft' || k === 'ArrowDown') v -= ev.shiftKey ? Math.min(1, o.step) : o.step;
    else if (k === 'PageUp') v += o.step * 4;
    else if (k === 'PageDown') v -= o.step * 4;
    else if (k === 'Home') v = o.min;
    else if (k === 'End') v = o.max;
    else handled = false;
    if (!handled) return;
    ev.preventDefault();
    commit(v);
  }));

  ctl = baseController(shell, o.path, {
    label: o.label,
    focus: function () { el.focus(); },
    sync: function () { render(); }
  });
  render();
  return ctl;
}

function pctSlider(o) {
  return sliderField({
    path: o.path, label: o.label, hint: o.hint, restart: o.restart,
    min: o.min === undefined ? 0 : o.min,
    max: o.max === undefined ? 100 : o.max,
    step: o.step === undefined ? 5 : o.step,
    unit: '%',
    toModel: function (v) { return roundTo(v / 100, 0.0001); },
    fromModel: function (m) { return Math.round(m * 100); }
  });
}

/* --- switch -------------------------------------------------------------- */

function switchField(o) {
  var titleId = uid('sw');
  var hintId = uid('swh');
  var dot = h('span.field__dirty', { hidden: true, 'aria-hidden': 'true' });
  var titleEl = h('span.switch-row__title#' + titleId, { text: o.label });
  titleEl.appendChild(dot);
  if (o.restart) titleEl.appendChild(restartBadge());
  var hintEl = h('span.switch-row__hint#' + hintId, { text: o.hint || '' });
  var stateEl = h('span.switch-row__state');
  var row = h('div.switch-row', {
    role: 'switch', tabIndex: 0,
    'aria-labelledby': titleId,
    'aria-describedby': hintId
  },
    h('span.switch-row__text', titleEl, hintEl),
    stateEl,
    h('span.switch', h('span.switch__knob')));
  var errEl = h('p.field__error', { hidden: true, role: 'alert' });
  var statusEl = h('span.field__status', { 'aria-live': 'polite' });
  var el = h('div.field', row, errEl, statusEl);

  function value() { return !!getAt(S.draft, o.path); }

  function render() {
    var v = value();
    row.setAttribute('aria-checked', v ? 'true' : 'false');
    stateEl.textContent = v ? 'On' : 'Off';
  }

  function toggle() {
    setAt(S.draft, o.path, !value());
    render();
    onModelChanged();
    if (o.onToggle) o.onToggle();
  }

  track(on(row, 'click', toggle));
  track(on(row, 'keydown', function (ev) {
    if (ev.key !== ' ' && ev.key !== 'Enter' && ev.key !== 'Spacebar') return;
    ev.preventDefault();
    toggle();
  }));

  render();

  var savedTimer = null;
  return {
    path: o.path,
    key: pathKey(o.path),
    el: el,
    label: o.label,
    focus: function () { row.focus(); },
    sync: render,
    setDirty: function (on_) { dot.hidden = !on_; },
    setError: function (msg) {
      clear(errEl);
      if (msg) {
        errEl.hidden = false;
        errEl.appendChild(icon('alert', { size: 'sm' }));
        errEl.appendChild(h('span', { text: msg }));
      } else {
        errEl.hidden = true;
      }
    },
    setStatus: function (kind, text) {
      clear(statusEl);
      statusEl.className = 'field__status';
      if (kind === 'saved') {
        statusEl.classList.add('field__saved');
        statusEl.appendChild(icon('check', { size: 'sm' }));
        statusEl.appendChild(h('span', { text: text || 'Saved' }));
        if (savedTimer) clearTimeout(savedTimer);
        savedTimer = setTimeout(function () {
          if (statusEl.isConnected) { clear(statusEl); statusEl.className = 'field__status'; }
        }, 2400);
        S.savedTimers.push(savedTimer);
      } else if (kind === 'error') {
        statusEl.appendChild(icon('alert', { size: 'sm' }));
        statusEl.appendChild(h('span.t-danger', { text: text || 'Not saved' }));
      }
    }
  };
}

/* --- select -------------------------------------------------------------- */

function selectField(o) {
  var shell = fieldShell({ label: o.label, hint: o.hint, restart: o.restart });
  var selId = uid('sel');
  var sel = h('select.select__el#' + selId, { 'aria-describedby': shell.hintId });
  var options = o.options.slice();
  var current = getAt(S.draft, o.path);
  var curStr = current === null || current === undefined ? '' : String(current);
  var known = false;
  for (var i = 0; i < options.length; i++) if (String(options[i][0]) === curStr) known = true;
  /* A value on disk that the list does not know (a removed target camera, an
     unlisted sound) stays visible rather than silently jumping to the first
     option and being saved as such. */
  if (!known && curStr !== '') options.push([curStr, curStr + ' (not in the list)']);
  for (var j = 0; j < options.length; j++) {
    sel.appendChild(h('option', { value: String(options[j][0]) }, options[j][1]));
  }
  shell.labelEl.setAttribute('for', selId);
  var wrap = h('div.select', sel, h('span.select__chevron', icon('chevron-down', { size: 'sm' })));

  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(wrap);
  shell.el.appendChild(shell.hintEl);
  shell.el.appendChild(shell.errEl);
  shell.el.appendChild(shell.statusEl);

  function render() {
    var v = getAt(S.draft, o.path);
    sel.value = v === null || v === undefined ? '' : String(v);
  }

  track(on(sel, 'change', function () {
    var raw = sel.value;
    setAt(S.draft, o.path, o.numeric ? Number(raw) : raw);
    ctl.setError(null);
    onModelChanged();
  }));

  var ctl = baseController(shell, o.path, {
    label: o.label,
    focus: function () { sel.focus(); },
    sync: render,
    setInvalid: function (bad) {
      if (bad) sel.setAttribute('aria-invalid', 'true');
      else sel.removeAttribute('aria-invalid');
    }
  });
  render();
  return ctl;
}

/* --- text / env / list --------------------------------------------------- */

function textField(o) {
  /* o.kind: 'text' (string) | 'env' (variable name + presence tag) | 'list'
     (comma-separated -> array) */
  var shell = fieldShell({ label: o.label, hint: o.hint, restart: o.restart });
  var inputId = uid('txt');
  var input = h('input.input#' + inputId, {
    type: 'text', autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false',
    placeholder: o.placeholder || null,
    'aria-describedby': shell.hintId
  });
  if (o.mono || o.kind === 'env') input.classList.add('input--mono');
  shell.labelEl.setAttribute('for', inputId);

  var tag = null;
  if (o.kind === 'env') {
    tag = h('span.envtag', { text: '…' });
    shell.labelEl.appendChild(tag);
  }

  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(input);
  shell.el.appendChild(shell.hintEl);
  shell.el.appendChild(shell.errEl);
  shell.el.appendChild(shell.statusEl);

  function fromDraft() {
    var v = getAt(S.draft, o.path);
    if (o.kind === 'list') return strList(v).join(', ');
    return v === null || v === undefined ? '' : String(v);
  }

  function parse(text) {
    if (o.kind === 'list') {
      var parts = String(text).split(',');
      var out = [];
      for (var i = 0; i < parts.length; i++) { var p = parts[i].trim(); if (p) out.push(p); }
      return out;
    }
    return text;
  }

  function refreshTag() {
    if (!tag) return;
    var name = String(getAt(S.draft, o.path) || '').trim();
    tag.className = 'envtag';
    if (!name) { tag.textContent = 'no name'; return; }
    var present = S.env[name];
    if (present === true) { tag.classList.add('envtag--set'); tag.textContent = 'set'; }
    else if (present === false) { tag.classList.add('envtag--unset'); tag.textContent = 'not set'; }
    else { tag.textContent = 'unchecked'; scheduleEnvCheck(name); }
  }

  function render() {
    input.value = fromDraft();
    refreshTag();
  }

  track(on(input, 'input', function () {
    setAt(S.draft, o.path, parse(input.value));
    ctl.setError(null);
    onModelChanged();
    if (tag) refreshTag();
  }));
  track(on(input, 'blur', function () {
    var text = input.value.trim();
    if (o.upper) text = text.toUpperCase();
    var parsed = parse(text);
    if (!eqValue(parsed, getAt(S.draft, o.path)) || text !== input.value) {
      setAt(S.draft, o.path, parsed);
      input.value = o.kind === 'list' ? strList(parsed).join(', ') : text;
      onModelChanged();
    }
    if (tag) refreshTag();
    var problems = [];
    validateSpec(o.spec, getAt(S.draft, o.path), o.path, '', problems);
    ctl.setError(problems.length ? problems[0].message : null);
  }));

  var ctl = baseController(shell, o.path, {
    label: o.label,
    focus: function () { input.focus(); },
    sync: function () { render(); ctl.setError(null); },
    setInvalid: function (bad) {
      if (bad) input.setAttribute('aria-invalid', 'true');
      else input.removeAttribute('aria-invalid');
    }
  });
  ctl.refreshEnv = refreshTag;
  render();
  return ctl;
}

/* Names typed into env fields are checked for presence on the server, in a
   batch, without their values ever travelling. */
function scheduleEnvCheck(name) {
  if (!name || S.envPending[name] || Object.prototype.hasOwnProperty.call(S.env, name)) return;
  S.envPending[name] = 1;
  if (S.envTimer) clearTimeout(S.envTimer);
  S.envTimer = setTimeout(function () {
    if (!S || !S || S.destroyed) return;
    var names = [];
    for (var k in S.envPending) if (Object.prototype.hasOwnProperty.call(S.envPending, k)) names.push(k);
    S.envPending = {};
    S.envTimer = null;
    if (!names.length) return;
    api.probeCamera({ env: names }, { signal: S.abort.signal, timeout: 8000 }).then(function (res) {
      if (!S || S.destroyed || !res || !res.env) return;
      for (var n in res.env) if (Object.prototype.hasOwnProperty.call(res.env, n)) S.env[n] = !!res.env[n];
      refreshEnvTags();
    }, function () { /* the tag simply stays "unchecked" */ });
  }, 700);
}

/* --- species multi-select ------------------------------------------------ */

function speciesField(o) {
  var labelId = uid('spl');
  var el = h('section.field', { 'aria-labelledby': labelId });
  var dot = h('span.field__dirty', { hidden: true, 'aria-hidden': 'true' });
  var head = h('h3.field__label#' + labelId, { text: o.label });
  head.appendChild(dot);
  el.appendChild(head);
  el.appendChild(h('p.field__hint', { text: o.hint || '' }));

  var searchId = uid('spq');
  var searchInput = h('input.search__input#' + searchId, {
    type: 'search', placeholder: 'Search species…',
    'aria-label': 'Filter ' + o.label, autocomplete: 'off',
    autocapitalize: 'off', spellcheck: 'false'
  });
  var searchBox = h('div.search.search--block',
    h('span.search__icon', icon('search', { size: 'sm' })),
    searchInput);

  var clearBtn = h('button.chip.chip--clear', { type: 'button' }, h('span', { text: 'Clear all' }));
  var countEl = h('p.field__hint', { role: 'status', 'aria-live': 'polite' });
  var groupsHost = h('div.stack.stack--tight');
  var emptyNote = h('p.field__hint', { hidden: true, text: 'No species match that filter.' });

  el.appendChild(h('div.row.row--wrap', h('div.row__grow', searchBox), clearBtn));
  el.appendChild(countEl);
  el.appendChild(groupsHost);
  el.appendChild(emptyNote);

  var groups = [];
  function addGroup(id, label, items) {
    var listEl = h('div.chip-row.chip-row--wrap', { role: 'group', 'aria-label': label });
    var wrap = h('div', h('p.overline', { text: label }), listEl);
    groups.push({ id: id, items: items, wrap: wrap, listEl: listEl });
    groupsHost.appendChild(wrap);
  }

  function currentList() {
    var v = getAt(S.draft, o.path);
    return isArray(v) ? v : [];
  }

  var seen = {};
  var recentItems = [];
  var k;
  if (o.recent) {
    var recentKeys = [];
    for (k in o.recent) if (Object.prototype.hasOwnProperty.call(o.recent, k)) recentKeys.push(k);
    recentKeys.sort(function (a, b) { return (o.recent[b] || 0) - (o.recent[a] || 0); });
    for (var r = 0; r < recentKeys.length; r++) {
      recentItems.push({ name: recentKeys[r], count: o.recent[recentKeys[r]] });
      seen[recentKeys[r].toLowerCase()] = 1;
    }
  }
  var catalogSeen = {};
  var cat;
  for (cat = 0; cat < SPECIES_CATALOG.length; cat++) {
    for (var s = 0; s < SPECIES_CATALOG[cat][1].length; s++) catalogSeen[SPECIES_CATALOG[cat][1][s].toLowerCase()] = 1;
  }
  var customItems = [];
  var startList = currentList();
  for (var q = 0; q < startList.length; q++) {
    var low = String(startList[q]).toLowerCase();
    if (seen[low] || catalogSeen[low]) continue;
    seen[low] = 1;
    customItems.push({ name: startList[q], count: null });
  }

  if (recentItems.length) addGroup('recent', 'Recent detections', recentItems);
  if (customItems.length) addGroup('custom', 'In this list', customItems);
  for (cat = 0; cat < SPECIES_CATALOG.length; cat++) {
    var names = SPECIES_CATALOG[cat][1];
    var items = [];
    for (var n2 = 0; n2 < names.length; n2++) items.push({ name: names[n2], count: null });
    addGroup('cat' + cat, SPECIES_CATALOG[cat][0], items);
  }

  var filter = '';

  function selectedSet() {
    var set = {};
    var list = currentList();
    for (var i = 0; i < list.length; i++) set[String(list[i]).toLowerCase()] = 1;
    return set;
  }

  function render() {
    var set = selectedSet();
    var shown = 0;
    for (var g = 0; g < groups.length; g++) {
      var grp = groups[g];
      var visible = [];
      for (var i = 0; i < grp.items.length; i++) {
        var it = grp.items[i];
        if (filter && it.name.toLowerCase().indexOf(filter) < 0) continue;
        visible.push(it);
      }
      shown += visible.length;
      grp.wrap.hidden = visible.length === 0;
      keyedList(grp.listEl, visible, {
        key: function (item) { return item.name.toLowerCase(); },
        create: function (item) {
          return h('button.chip', {
            type: 'button', 'class': speciesClass(item.name), dataset: { species: item.name }
          },
            h('span.chip__dot', { 'aria-hidden': 'true' }),
            h('span.chip__label'),
            item.count === null || item.count === undefined ? null
              : h('span.chip__count', { text: String(item.count) }));
        },
        update: function (node, item) {
          var labelNode = node.querySelector('.chip__label');
          if (labelNode) labelNode.textContent = titleCase(item.name);
          var on_ = !!set[item.name.toLowerCase()];
          node.setAttribute('aria-pressed', on_ ? 'true' : 'false');
          node.setAttribute('aria-label', titleCase(item.name) + (on_ ? ' — in list' : ' — not in list'));
        }
      });
    }
    emptyNote.hidden = shown !== 0;
    var count = currentList().length;
    countEl.textContent = count === 0 ? (o.emptyMeans || 'Nothing selected.') : count + ' ' + plural(count, 'species', 'species') + ' selected';
    clearBtn.disabled = count === 0;
  }

  function toggle(name) {
    var list = currentList().slice();
    var low = String(name).toLowerCase();
    var idx = -1;
    for (var i = 0; i < list.length; i++) if (String(list[i]).toLowerCase() === low) { idx = i; break; }
    if (idx >= 0) list.splice(idx, 1);
    else list.push(name);
    setAt(S.draft, o.path, list);
    render();
    onModelChanged();
  }

  track(delegate(groupsHost, 'click', '.chip[data-species]', function (ev, node) {
    ev.preventDefault();
    toggle(node.dataset.species);
  }));
  track(on(searchInput, 'input', function () {
    filter = searchInput.value.trim().toLowerCase();
    if (filter) searchBox.classList.add('search--filled');
    else searchBox.classList.remove('search--filled');
    render();
  }));
  track(on(clearBtn, 'click', function () {
    if (!currentList().length) return;
    setAt(S.draft, o.path, []);
    render();
    onModelChanged();
  }));

  render();

  return {
    path: o.path,
    key: pathKey(o.path),
    el: el,
    label: o.label,
    focus: function () { searchInput.focus(); },
    sync: render,
    setDirty: function (on_) { dot.hidden = !on_; },
    setError: function () {},
    setStatus: function () {}
  };
}

/* --- Pushover destinations (general) ------------------------------------ */

/* The list lives at general.notification.destinations and cameras refer to
   entries by id, so the id is set once, in the add dialog, and never edited
   in place: a rename would silently detach every camera that picked it. */

function slugId(name) {
  var s = String(name || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return s.slice(0, 32).replace(/-+$/, '');
}

function envFromId(id) {
  return 'PUSHOVER_USER_KEY_' + String(id || '').toUpperCase().replace(/[^A-Z0-9]/g, '_');
}

function destinationList() {
  var v = S.draft ? getAt(S.draft, ['general', 'notification', 'destinations']) : null;
  return isArray(v) ? v : [];
}

function envTag(tag, name) {
  tag.className = 'envtag';
  if (!name) { tag.textContent = 'no name'; return; }
  var present = S.env[name];
  if (present === true) { tag.classList.add('envtag--set'); tag.textContent = 'set'; }
  else if (present === false) { tag.classList.add('envtag--unset'); tag.textContent = 'not set'; }
  else { tag.textContent = 'unchecked'; scheduleEnvCheck(name); }
}

function destinationsField(o) {
  var shell = fieldShell({ label: o.label, hint: o.hint, labelTag: 'h3.field__label' });
  shell.el = h('section.field', { 'aria-labelledby': shell.labelId });
  var listEl = h('div.destlist');
  var countEl = h('p.field__hint', { role: 'status', 'aria-live': 'polite' });
  var addBtn = h('button.btn.btn--secondary.btn--sm', { type: 'button' },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('plus', { size: 'sm' })),
    h('span.btn__label', 'Add destination'));
  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(shell.hintEl);
  shell.el.appendChild(shell.errEl);
  shell.el.appendChild(listEl);
  shell.el.appendChild(countEl);
  shell.el.appendChild(h('div.row.row--wrap', addBtn));
  shell.el.appendChild(shell.statusEl);

  var rows = [];

  function list() {
    var v = getAt(S.draft, o.path);
    if (!isArray(v)) { v = []; setAt(S.draft, o.path, v); }
    return v;
  }

  function titleFor(d) { return d.name || d.id; }

  function refreshTags() {
    var cur = list();
    for (var i = 0; i < rows.length && i < cur.length; i++) {
      var keyName = String(cur[i].user_key_env || '').trim();
      envTag(rows[i].tags.user_key_env, keyName);
      rows[i].setBtns.user_key_env.lastChild.textContent = S.env[keyName] === true ? 'Replace value' : 'Set value';
      var tok = String(cur[i].app_token_env || '').trim();
      rows[i].tags.app_token_env.hidden = !tok;
      envTag(rows[i].tags.app_token_env, tok);
      rows[i].setBtns.app_token_env.lastChild.textContent = S.env[tok] === true ? 'Replace value' : 'Set value';
    }
  }

  function envInput(idx, key, label, hint, placeholder) {
    var id = uid('denv');
    var input = h('input.input.input--mono#' + id, {
      type: 'text', autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false',
      placeholder: placeholder || null
    });
    var tag = h('span.envtag', { text: '…' });
    var lab = h('label.field__label', { 'for': id, text: label });
    lab.appendChild(tag);
    /* The value itself is set here, write-only, without leaving the card. */
    var setBtn = h('button.btn.btn--secondary.btn--sm', { type: 'button' }, h('span.btn__label', 'Set value'));
    track(on(setBtn, 'click', function () {
      var cur = list()[idx];
      var name = String(cur && cur[key] || '').trim();
      if (!name) { toast.info('Name the variable first.'); input.focus(); return; }
      if (!ENV_RE.test(name)) { toast.info('That is not a variable name.', { detail: 'Letters, digits and underscores only.' }); input.focus(); return; }
      setSecretDialog({
        name: name, live: true,
        used_by: [(key === 'user_key_env' ? 'user key' : 'app token') + ' of destination \'' + titleFor(cur) + '\'']
      }, secretsFile());
    }));
    var wrap = h('div.field', lab, h('div.secretset', input, setBtn), h('p.field__hint', { text: hint }));
    var current = list()[idx];
    input.value = current && current[key] ? String(current[key]) : '';
    track(on(input, 'input', function () {
      var cur = list();
      if (!cur[idx]) return;
      cur[idx][key] = input.value;
      ctl.setError(null);
      onModelChanged();
      refreshTags();
    }));
    track(on(input, 'blur', function () {
      var cur = list();
      if (!cur[idx]) return;
      var t = input.value.trim();
      if (t !== input.value) { input.value = t; cur[idx][key] = t; onModelChanged(); }
      refreshTags();
    }));
    return { wrap: wrap, input: input, tag: tag, setBtn: setBtn };
  }

  function removeAt(idx) {
    var cur = list();
    var gone = cur[idx];
    if (!gone) return;
    cur.splice(idx, 1);
    /* Cameras that picked it by id lose the tick; a saved reference to a
       destination that no longer exists is refused by the server. */
    var touched = 0;
    for (var c = 0; c < S.draft.order.length; c++) {
      var cam = S.draft.cameras[S.draft.order[c]];
      var sel = cam && cam.notification ? cam.notification.destinations : null;
      if (!isArray(sel)) continue;
      var at = sel.indexOf(gone.id);
      if (at >= 0) { sel.splice(at, 1); touched += 1; }
    }
    render();
    onModelChanged();
    toast.info('Removed ' + titleFor(gone) + ' from the draft', {
      detail: (touched ? 'Unticked on ' + touched + ' ' + plural(touched, 'camera') + '. ' : '') + 'Save to write the change.'
    });
    addBtn.focus();
  }

  function render() {
    clear(listEl);
    rows = [];
    var cur = list();
    for (var i = 0; i < cur.length; i++) {
      (function (idx) {
        var d = cur[idx];
        var nameId = uid('dname');
        var nameInput = h('input.input#' + nameId, { type: 'text', autocomplete: 'off', placeholder: d.id });
        nameInput.value = d.name || '';
        var title = h('span.destcard__title', { text: titleFor(d) });
        var removeBtn = h('button.icon-btn.icon-btn--danger', { type: 'button', 'aria-label': 'Remove ' + titleFor(d) },
          icon('trash', { size: 'sm' }));
        track(on(removeBtn, 'click', function () { removeAt(idx); }));
        track(on(nameInput, 'input', function () {
          var l = list();
          if (!l[idx]) return;
          l[idx].name = nameInput.value;
          title.textContent = titleFor(l[idx]);
          ctl.setError(null);
          onModelChanged();
        }));
        var key = envInput(idx, 'user_key_env', 'User key variable',
          'Holds this person\'s Pushover user or group key. A comma-separated list sends to each key.');
        var tok = envInput(idx, 'app_token_env', 'App token variable',
          'Optional: a different Pushover application token for this destination.', 'uses the app token above');
        var card = h('div.destcard',
          h('div.destcard__head', title, h('span.destcard__id', { text: d.id }), removeBtn),
          h('div.destcard__grid',
            h('div.field', h('label.field__label', { 'for': nameId, text: 'Name' }), nameInput,
              h('p.field__hint', { text: 'Shown here and in each camera\'s picker.' })),
            key.wrap, tok.wrap));
        rows.push({
          card: card,
          inputs: { name: nameInput, user_key_env: key.input, app_token_env: tok.input },
          tags: { user_key_env: key.tag, app_token_env: tok.tag },
          setBtns: { user_key_env: key.setBtn, app_token_env: tok.setBtn }
        });
        listEl.appendChild(card);
      }(i));
    }
    var n = cur.length;
    countEl.textContent = n
      ? n + ' ' + plural(n, 'destination') + ' · every camera alerts all of them unless it picks some.'
      : (o.emptyMeans || 'No destinations.');
    refreshTags();
  }

  track(on(addBtn, 'click', function () {
    addDestinationDialog(function (dest) {
      list().push(dest);
      /* The section may have been rebuilt while the dialog was open (a
         quiet refresh); draw through whichever controller is live now. */
      var live = S.fieldByKey[ctl.key] || ctl;
      live.sync();
      onModelChanged();
      live.focus([String(list().length - 1), 'name']);
    });
  }));

  var lastSub = null;
  var ctl = baseController(shell, o.path, {
    label: o.label,
    focus: function (sub) {
      var s = sub || lastSub;
      var row = s && rows[Number(s[0])];
      var input = row && s[1] && row.inputs[s[1]] ? row.inputs[s[1]] : null;
      if (input) input.focus(); else addBtn.focus();
    },
    sync: function () { render(); ctl.setError(null); }
  });
  var plainSetError = ctl.setError;
  /* `sub` is what is left of a field path after the list's own key: the row
     index and the property, from validateSpec or a server problem path. */
  ctl.setError = function (msg, sub) {
    plainSetError(msg);
    for (var i = 0; i < rows.length; i++) {
      for (var k in rows[i].inputs) {
        if (Object.prototype.hasOwnProperty.call(rows[i].inputs, k)) rows[i].inputs[k].removeAttribute('aria-invalid');
      }
    }
    lastSub = msg ? (sub || null) : null;
    var row = sub && rows[Number(sub[0])];
    if (msg && row && sub[1] && row.inputs[sub[1]]) row.inputs[sub[1]].setAttribute('aria-invalid', 'true');
  };
  ctl.refreshEnv = refreshTags;
  render();
  return ctl;
}

function secretsFile() {
  return S.secrets && S.secrets.file ? S.secrets.file : 'config/secrets.env';
}

function addDestinationDialog(onAdd) {
  var existing = destinationList();
  var f = { name: '', id: '', user_key_env: '', app_token_env: '', user_key: '', app_token: '' };
  var touched = { id: false, key: false };
  var errs = {};
  var els = {};

  function hasId(id) {
    for (var i = 0; i < existing.length; i++) if (existing[i].id === id) return true;
    return false;
  }
  function uniqueId(base) {
    var root = base || 'dest';
    var id = root;
    var n = 1;
    while (hasId(id)) { n += 1; id = root + n; }
    return id.slice(0, 32);
  }

  function field(key, label, hint, inputEl) {
    var id = uid('addd');
    inputEl.id = id;
    var err = h('p.field__error', { hidden: true, role: 'alert' });
    var wrap = h('div.field', h('label.field__label', { 'for': id, text: label }), inputEl, h('p.field__hint', { text: hint }), err);
    els[key] = { input: inputEl, err: err, hint: wrap.querySelector('.field__hint') };
    return wrap;
  }

  function showErrors() {
    for (var k in els) {
      if (!Object.prototype.hasOwnProperty.call(els, k)) continue;
      var e = els[k];
      clear(e.err);
      if (errs[k]) {
        e.err.hidden = false;
        e.err.appendChild(icon('alert', { size: 'sm' }));
        e.err.appendChild(h('span', { text: errs[k] }));
        e.hint.hidden = true;
        e.input.setAttribute('aria-invalid', 'true');
      } else {
        e.err.hidden = true;
        e.hint.hidden = false;
        e.input.removeAttribute('aria-invalid');
      }
    }
  }

  function validate() {
    errs = {};
    var id = f.id.trim();
    if (!CAMERA_ID_RE.test(id)) errs.id = 'Letters, digits, - or _ only, up to 32 characters.';
    else if (hasId(id)) errs.id = 'A destination with this id already exists.';
    var env = f.user_key_env.trim();
    if (!env) errs.user_key_env = 'Name the variable in config/secrets.env that holds the user key.';
    else if (!ENV_RE.test(env)) errs.user_key_env = 'Letters, digits and underscores only.';
    var tok = f.app_token_env.trim();
    if (tok && !ENV_RE.test(tok)) errs.app_token_env = 'Letters, digits and underscores only.';
    if (f.user_key.trim() && env && !/^PUSHOVER_/.test(env) && S.env[env] === undefined) {
      errs.user_key = 'Only a PUSHOVER_… variable can take its value here before the destination is saved; name it that way, or set the value under Secrets after saving.';
    }
    if (f.app_token.trim() && !tok) errs.app_token = 'Name the app token variable above, or leave this empty.';
    showErrors();
    var any = false;
    for (var k in errs) if (Object.prototype.hasOwnProperty.call(errs, k)) any = true;
    return !any;
  }

  var nameInput = h('input.input', { type: 'text', placeholder: 'Brandon', autocomplete: 'off' });
  var idInput = h('input.input.input--mono', { type: 'text', autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false' });
  var keyInput = h('input.input.input--mono', { type: 'text', autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false' });
  var keyValue = h('input.input.input--mono', { type: 'password', autocomplete: 'new-password', autocapitalize: 'off', spellcheck: 'false', placeholder: 'paste the 30-character key' });
  var showKeys = h('input.check__box', { type: 'checkbox' });
  var tokInput = h('input.input.input--mono', { type: 'text', placeholder: 'uses the app token above', autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false' });
  var tokValue = h('input.input.input--mono', { type: 'password', autocomplete: 'new-password', autocapitalize: 'off', spellcheck: 'false' });
  f.id = uniqueId('');
  idInput.value = f.id;
  f.user_key_env = envFromId(f.id);
  keyInput.value = f.user_key_env;

  var tokValueField = field('app_token', 'App token', 'The token itself, for the variable named above. Written to ' + secretsFile() + ' now; never shown again.', tokValue);
  tokValueField.hidden = true;
  showKeys.addEventListener('change', function () {
    keyValue.type = showKeys.checked ? 'text' : 'password';
    tokValue.type = showKeys.checked ? 'text' : 'password';
  });

  var content = h('div.stack',
    field('name', 'Name', 'Who this is. Shown in each camera\'s picker.', nameInput),
    field('id', 'Id', 'Short and permanent: cameras refer to it, so it cannot be renamed later.', idInput),
    field('user_key_env', 'User key variable', 'The variable in ' + secretsFile() + ' that holds their Pushover user or group key. Reuse PUSHOVER_USER_KEY for the key already in use.', keyInput),
    field('user_key', 'User key', 'Their user key from the Pushover app. Written to ' + secretsFile() + ' and applied the moment you add the destination; never shown again. Leave empty if the variable is already set.', keyValue),
    h('label.check', showKeys, h('span.check__label', 'Show keys while typing')),
    field('app_token_env', 'App token variable', 'Optional: a variable holding a different Pushover application token, for a destination on another Pushover account.', tokInput),
    tokValueField);

  nameInput.addEventListener('input', function () {
    f.name = nameInput.value;
    if (!touched.id) { f.id = uniqueId(slugId(f.name)); idInput.value = f.id; }
    if (!touched.key) { f.user_key_env = envFromId(f.id); keyInput.value = f.user_key_env; }
    if (errs.id) { delete errs.id; showErrors(); }
  });
  idInput.addEventListener('input', function () {
    touched.id = true;
    f.id = idInput.value;
    if (!touched.key) { f.user_key_env = envFromId(f.id.trim()); keyInput.value = f.user_key_env; }
    if (errs.id) { delete errs.id; showErrors(); }
  });
  keyInput.addEventListener('input', function () {
    touched.key = true;
    f.user_key_env = keyInput.value;
    if (errs.user_key_env || errs.user_key) { delete errs.user_key_env; delete errs.user_key; showErrors(); }
  });
  keyValue.addEventListener('input', function () {
    f.user_key = keyValue.value;
    if (errs.user_key) { delete errs.user_key; showErrors(); }
  });
  tokInput.addEventListener('input', function () {
    f.app_token_env = tokInput.value;
    tokValueField.hidden = !tokInput.value.trim();
    if (errs.app_token_env || errs.app_token) { delete errs.app_token_env; delete errs.app_token; showErrors(); }
  });
  tokValue.addEventListener('input', function () {
    f.app_token = tokValue.value;
    if (errs.app_token) { delete errs.app_token; showErrors(); }
  });

  var busy = false;
  var dlg = dialog({
    role: 'dialog',
    title: 'Add a destination',
    body: 'Someone who can receive alerts. The destination is added to the draft and saved with your changes; a key pasted here goes into ' + secretsFile() + ' right away.',
    width: 560,
    content: content,
    initialFocus: nameInput,
    actions: [
      { label: 'Cancel', variant: 'secondary', value: null },
      { label: 'Add destination', variant: 'primary', value: 'add', keepOpen: true, onSelect: function () {
        if (busy || !validate()) return;
        var dest = { id: f.id.trim(), name: f.name.trim(), user_key_env: f.user_key_env.trim(), app_token_env: f.app_token_env.trim() };
        var writes = [];
        if (f.user_key.trim()) writes.push({ name: dest.user_key_env, value: f.user_key, field: 'user_key' });
        if (dest.app_token_env && f.app_token.trim()) writes.push({ name: dest.app_token_env, value: f.app_token, field: 'app_token' });
        var addBtn = dlg.el.querySelector('.btn--primary');
        function done() {
          busy = false;
          if (addBtn) addBtn.disabled = false;
          onAdd(dest);
          dlg.close('added');
          if (writes.length) {
            toast.success((dest.name || dest.id) + ': key saved', {
              detail: 'Applied to the running service. Save changes to add the destination itself.'
            });
          }
        }
        /* The keys are written one after another before the destination
           joins the draft, so a refused write keeps the dialog open with
           the error on the field it belongs to. */
        function write(i) {
          if (i >= writes.length) { done(); return; }
          api.setSecret({ name: writes[i].name, value: writes[i].value }, { signal: S.abort.signal }).then(function () {
            if (!S || !S || S.destroyed) return;
            S.env[writes[i].name] = true;
            refreshEnvTags();
            write(i + 1);
          }, function (e) {
            busy = false;
            if (addBtn) addBtn.disabled = false;
            if (!S || S.destroyed || api.isAbort(e)) return;
            errs[writes[i].field] = api.describe(e);
            showErrors();
          });
        }
        busy = true;
        if (addBtn) addBtn.disabled = true;
        write(0);
      } }
    ]
  });
  window.setTimeout(function () { try { nameInput.focus(); } catch (e) {} }, 0);
}

/* --- Recipients (a camera's pick of the destinations) ------------------- */

function recipientsField(o) {
  var shell = fieldShell({ label: o.label, hint: o.hint, labelTag: 'h3.field__label' });
  shell.el = h('section.field', { 'aria-labelledby': shell.labelId });
  var body = h('div.recipients');
  var countEl = h('p.field__hint', { role: 'status', 'aria-live': 'polite' });
  shell.el.appendChild(shell.labelEl);
  shell.el.appendChild(body);
  shell.el.appendChild(countEl);
  shell.el.appendChild(shell.hintEl);
  shell.el.appendChild(shell.errEl);
  shell.el.appendChild(shell.statusEl);

  var radioName = uid('rcp');
  var allRadio = null;
  var someRadio = null;
  var goBtn = null;
  var boxes = [];

  function value() {
    var v = getAt(S.draft, o.path);
    return isArray(v) ? v : null;
  }

  function refresh() {
    var dests = destinationList();
    var v = value();
    var all = v === null;
    if (allRadio) { allRadio.checked = all; someRadio.checked = !all; }
    var picked = 0;
    for (var i = 0; i < boxes.length; i++) {
      var on_ = !all && v.indexOf(boxes[i].id) >= 0;
      boxes[i].input.checked = all ? true : on_;
      boxes[i].input.disabled = all;
      if (on_) picked += 1;
    }
    countEl.className = 'field__hint';
    if (!dests.length) countEl.textContent = '';
    else if (all) countEl.textContent = 'All ' + dests.length + ' ' + plural(dests.length, 'destination') + ', including any added later.';
    else if (picked) countEl.textContent = picked + ' of ' + dests.length + ' ' + plural(dests.length, 'destination') + '.';
    else { countEl.textContent = 'None ticked — this camera sends no alerts.'; countEl.classList.add('field__hint--warn'); }
  }

  function commit(next) {
    setAt(S.draft, o.path, next);
    refresh();
    ctl.setError(null);
    onModelChanged();
  }

  function render() {
    clear(body);
    boxes = [];
    allRadio = null;
    someRadio = null;
    goBtn = null;
    var dests = destinationList();
    if (!dests.length) {
      goBtn = h('button.btn.btn--secondary.btn--sm', { type: 'button' }, h('span.btn__label', 'Define destinations'));
      track(on(goBtn, 'click', function () { selectSection('general.notifications'); }));
      body.appendChild(h('p.field__hint', { text: 'No destinations are defined yet, so alerts from this camera go to the fallback user key variable.' }));
      body.appendChild(h('div.row.row--wrap', goBtn));
      refresh();
      return;
    }
    allRadio = h('input.check__box', { type: 'radio', name: radioName, value: 'all' });
    someRadio = h('input.check__box', { type: 'radio', name: radioName, value: 'some' });
    body.appendChild(h('label.check', allRadio, h('span.check__label', 'Every destination')));
    body.appendChild(h('label.check', someRadio, h('span.check__label', 'Only these:')));
    var sub = h('div.recipients__sub');
    for (var i = 0; i < dests.length; i++) {
      (function (d) {
        var box = h('input.check__box', { type: 'checkbox', value: d.id });
        var lab = h('label.check', box,
          h('span.check__label', h('span', { text: d.name || d.id }), h('span.destcard__id', { text: d.user_key_env })));
        track(on(box, 'change', function () {
          var cur = value();
          var next = isArray(cur) ? cur.slice() : [];
          var at = next.indexOf(d.id);
          if (box.checked && at < 0) next.push(d.id);
          if (!box.checked && at >= 0) next.splice(at, 1);
          commit(next);
        }));
        boxes.push({ id: d.id, input: box });
        sub.appendChild(lab);
      }(dests[i]));
    }
    body.appendChild(sub);
    track(on(allRadio, 'change', function () { if (allRadio.checked) commit(null); }));
    track(on(someRadio, 'change', function () {
      if (!someRadio.checked) return;
      /* Start from everyone ticked; untick to narrow. */
      var ids = [];
      for (var k = 0; k < dests.length; k++) ids.push(dests[k].id);
      commit(ids);
    }));
    refresh();
  }

  var ctl = baseController(shell, o.path, {
    label: o.label,
    focus: function () { if (allRadio) allRadio.focus(); else if (goBtn) goBtn.focus(); },
    sync: function () { render(); ctl.setError(null); }
  });
  render();
  return ctl;
}

/* --- Secrets: write-only values for the variables the config names ------ */

/* Unlike everything else on this page these are NOT staged: a value is
   written to config/secrets.env and into the running process the moment
   the dialog is confirmed, and nothing ever reads it back — the row only
   ever shows "set" or "not set". The list comes from the SAVED file
   (S.secrets, from GET /api/config), never from the draft: the server
   accepts only names the saved configuration references. */

function refreshEnvTags() {
  for (var i = 0; i < S.fields.length; i++) if (S.fields[i].refreshEnv) S.fields[i].refreshEnv();
}

function renderSecretsCard() {
  var meta = S.secrets || {};
  var file = meta.file || 'config/secrets.env';
  var vars = isArray(meta.variables) ? meta.variables : [];
  var wrap = h('div.stack.stack--tight');
  wrap.appendChild(h('p.field__hint', { text: 'Values are written to ' + file +
    ' and applied to the running service at once; they are never shown again. Pushover variables can also be set ' +
    'from a destination\'s card; anything else appears here once the saved configuration names it.' }));
  if (!vars.length) {
    wrap.appendChild(h('p.field__hint', { text: 'The saved configuration names no variables yet.' }));
    return wrap;
  }
  var list = h('div.secretlist');
  for (var i = 0; i < vars.length; i++) {
    (function (v) {
      var tag = h('span.envtag', { text: '…' });
      var usedText = (isArray(v.used_by) ? v.used_by : []).join(', ');
      var setBtn = h('button.btn.btn--secondary.btn--sm', { type: 'button', 'aria-label': 'Set the value of ' + v.name },
        h('span.btn__label', 'Set value'));
      var removeBtn = h('button.btn.btn--ghost.btn--sm', { type: 'button', 'aria-label': 'Remove ' + v.name + ' from ' + file },
        h('span.btn__label', 'Remove'));
      function refresh() {
        envTag(tag, v.name);
        var present = S.env[v.name] === true;
        removeBtn.hidden = !present;
        setBtn.lastChild.textContent = present ? 'Replace value' : 'Set value';
      }
      track(on(setBtn, 'click', function () { setSecretDialog(v, file); }));
      track(on(removeBtn, 'click', function () { removeSecretDialog(v, file); }));
      var row = h('div.secretrow',
        h('div.secretrow__text',
          h('div.secretrow__head', h('span.secretrow__name', { text: v.name }), tag),
          h('p.field__hint', { text: usedText + (v.live ? '' : ' · read when the service starts, so restart after changing it') })),
        h('div.secretrow__actions', setBtn, removeBtn));
      /* Registered like a field so the batched env probe and a saved value
         refresh this tag too; it is never dirty and never validated. */
      S.fields.push({
        key: 'secret:' + v.name, path: ['secrets', v.name], el: row, label: v.name,
        refreshEnv: refresh, setDirty: function () {}, setError: function () {}, setStatus: function () {}
      });
      refresh();
      list.appendChild(row);
    }(vars[i]));
  }
  wrap.appendChild(list);
  return wrap;
}

function setSecretDialog(v, file) {
  var input = h('input.input.input--mono', {
    type: 'password', autocomplete: 'new-password', autocapitalize: 'off', spellcheck: 'false',
    'aria-label': 'Value for ' + v.name
  });
  var show = h('input.check__box', { type: 'checkbox' });
  show.addEventListener('change', function () { input.type = show.checked ? 'text' : 'password'; });
  var err = h('p.field__error', { hidden: true, role: 'alert' });
  var hint = h('p.field__hint', { text: 'Pasted straight into ' + file + '. It is never shown again; set it again to replace it.' });
  function showErr(msg) {
    clear(err);
    if (msg) {
      err.hidden = false;
      err.appendChild(icon('alert', { size: 'sm' }));
      err.appendChild(h('span', { text: msg }));
      hint.hidden = true;
      input.setAttribute('aria-invalid', 'true');
    } else {
      err.hidden = true;
      hint.hidden = false;
      input.removeAttribute('aria-invalid');
    }
  }
  input.addEventListener('input', function () { showErr(null); });
  var content = h('div.stack',
    h('div.field', h('label.field__label', { text: 'Value' }), input, hint, err),
    h('label.check', show, h('span.check__label', 'Show while typing')));
  var busy = false;
  var dlg = dialog({
    role: 'dialog',
    title: 'Set ' + v.name,
    body: (isArray(v.used_by) ? v.used_by : []).join(', ') +
      (v.live ? '. Takes effect the moment it is saved.' : '. Read when the service starts, so restart afterwards.'),
    width: 520,
    content: content,
    initialFocus: input,
    actions: [
      { label: 'Cancel', variant: 'secondary', value: null },
      { label: 'Save value', variant: 'primary', value: 'save', keepOpen: true, onSelect: function () {
        if (busy) return;
        var value = input.value;
        if (!value.trim()) { showErr('Paste the value first.'); input.focus(); return; }
        busy = true;
        api.setSecret({ name: v.name, value: value }, { signal: S.abort.signal }).then(function (res) {
          busy = false;
          if (!S || !S || S.destroyed) return;
          S.env[v.name] = true;
          input.value = '';
          dlg.close('saved');
          refreshEnvTags();
          toast.success(v.name + ' set', {
            detail: (res && res.live ? 'Applied to the running service.' : 'Saved; read at the next restart.') +
              (res && res.backup ? ' The previous secrets.env is in backups.' : '')
          });
        }, function (e) {
          busy = false;
          if (!S || S.destroyed || api.isAbort(e)) return;
          showErr(api.describe(e));
        });
      } }
    ]
  });
  window.setTimeout(function () { try { input.focus(); } catch (e) {} }, 0);
}

function removeSecretDialog(v, file) {
  var dlg = dialog({
    role: 'alertdialog',
    tone: 'danger',
    title: 'Remove ' + v.name + '?',
    body: 'Its line is deleted from ' + file + ' and the running service forgets the value at once. ' +
      'Anything using it (' + (isArray(v.used_by) ? v.used_by : []).join(', ') + ') stops working until it is set again.',
    actions: [
      { label: 'Keep it', variant: 'secondary', value: false, focus: true },
      { label: 'Remove', variant: 'danger', value: true }
    ]
  });
  dlg.result.then(function (yes) {
    if (yes !== true || !S || S.destroyed) return;
    api.setSecret({ name: v.name, value: '' }, { signal: S.abort.signal }).then(function () {
      if (!S || !S || S.destroyed) return;
      S.env[v.name] = false;
      refreshEnvTags();
      toast.info(v.name + ' removed from ' + file);
    }, function (e) {
      if (!S || S.destroyed || api.isAbort(e)) return;
      toast.error(v.name + ' was not removed.', { detail: api.describe(e) });
    });
  });
}

/* ==========================================================================
   SPEC -> CONTROLLER
   ========================================================================= */

function cameraOptions(selfId) {
  var out = [['', 'None']];
  for (var i = 0; i < S.draft.order.length; i++) {
    var id = S.draft.order[i];
    if (id === selfId) continue;
    var cam = S.draft.cameras[id];
    out.push([id, (cam && cam.name ? cam.name : id) + ' (' + id + ')']);
  }
  return out;
}

function renderField(spec, ctx) {
  var path = ctx.base.concat(spec.parts);
  var o = { path: path, label: spec.label, hint: spec.hint, restart: !!spec.restart, spec: spec };
  var ctl;
  switch (spec.kind) {
    case 'number':
      ctl = numberField(Object.assign(o, {
        min: spec.min, max: spec.max, step: spec.step, int: !!spec.int,
        nullable: !!spec.nullable, scale: spec.scale, unit: spec.unit
      }));
      break;
    case 'pct':
      ctl = pctSlider(Object.assign(o, { min: spec.min, max: spec.max, step: spec.step }));
      break;
    case 'slider':
      ctl = sliderField(Object.assign(o, { min: spec.min, max: spec.max, step: spec.step, unit: spec.unit }));
      break;
    case 'switch':
      ctl = switchField(Object.assign(o, {
        onToggle: spec.controls ? function () { rerenderGroup(ctx.groupId, spec); } : null
      }));
      break;
    case 'select':
      ctl = selectField(Object.assign(o, {
        options: spec.options === 'cameras' ? cameraOptions(ctx.cameraId) : spec.options,
        numeric: !!spec.numeric
      }));
      break;
    case 'text':
    case 'env':
    case 'list':
      ctl = textField(Object.assign(o, {
        kind: spec.kind, mono: !!spec.mono, placeholder: spec.placeholder, upper: !!spec.upper
      }));
      break;
    case 'species':
      ctl = speciesField(Object.assign(o, {
        emptyMeans: spec.emptyMeans,
        recent: spec.recent && ctx.camera ? ctx.camera.recent_detections : null
      }));
      break;
    case 'destinations':
      ctl = destinationsField(Object.assign(o, { emptyMeans: spec.emptyMeans }));
      break;
    case 'recipients':
      ctl = recipientsField(o);
      break;
    default:
      return null;
  }
  ctl.group = ctx.groupId;
  ctl.spec = spec;
  S.fields.push(ctl);
  S.fieldByKey[ctl.key] = ctl;
  return ctl;
}

/* ==========================================================================
   GROUPS AND SECTIONS
   ========================================================================= */

function groupKey(ctx, group) { return (ctx.cameraId || 'general') + ':' + group.id; }

function renderGroup(group, ctx) {
  var gid = groupKey(ctx, group);
  ctx = Object.assign({}, ctx, { groupId: gid });
  var fs = h('fieldset.fieldset', h('legend.fieldset__legend', { text: group.legend }));
  if (group.hint) fs.appendChild(h('p.field__hint', { text: group.hint }));

  var main = [];
  var adv = [];
  for (var i = 0; i < group.fields.length; i++) {
    var spec = group.fields[i];
    if (ctx.camera && !specVisible(spec, ctx.camera)) continue;
    (spec.advanced ? adv : main).push(spec);
  }
  for (var m = 0; m < main.length; m++) {
    var ctl = renderField(main[m], ctx);
    if (ctl) fs.appendChild(ctl.el);
  }
  if (group.custom === 'secrets') fs.appendChild(renderSecretsCard());
  if (group.probe === 'rtsp' && ctx.camera) fs.appendChild(rtspProbeRow(ctx.cameraId));
  if (group.probe === 'onvif' && ctx.camera && blockOn(ctx.camera, 'onvif')) fs.appendChild(onvifProbeRow(ctx.cameraId));

  if (adv.length) {
    var details = h('details.disclosure', { open: !!S.openAdvanced[gid] });
    var summary = h('summary.disclosure__summary',
      h('span.disclosure__chev', { 'aria-hidden': 'true' }, icon('chevron-down', { size: 'sm' })),
      h('span', { text: 'Advanced · ' + adv.length + ' ' + plural(adv.length, 'setting') }));
    var body = h('div.disclosure__body');
    for (var a = 0; a < adv.length; a++) {
      var actl = renderField(adv[a], ctx);
      if (actl) body.appendChild(actl.el);
    }
    details.appendChild(summary);
    details.appendChild(body);
    track(on(details, 'toggle', function () { S.openAdvanced[gid] = details.open; }));
    fs.appendChild(details);
  }

  S.groupHosts[gid] = { el: fs, group: group, ctx: ctx };
  return fs;
}

/* Re-render one group in place (a controlling switch changed which fields
   apply). Controllers of the old group are dropped; focus returns to the
   switch that was toggled. */
function rerenderGroup(gid, focusSpec) {
  var host = S.groupHosts[gid];
  if (!host || !host.el.isConnected) return;
  var keep = [];
  var byKey = {};
  for (var i = 0; i < S.fields.length; i++) {
    if (S.fields[i].group !== gid) { keep.push(S.fields[i]); byKey[S.fields[i].key] = S.fields[i]; }
  }
  S.fields = keep;
  S.fieldByKey = byKey;
  var ctx = host.ctx;
  ctx.camera = ctx.cameraId ? S.draft.cameras[ctx.cameraId] : null;
  var fresh = renderGroup(host.group, { base: ctx.base, cameraId: ctx.cameraId, camera: ctx.camera });
  host.el.parentNode.replaceChild(fresh, host.el);
  /* ONVIF on/off changes what PTZ validation says; the PTZ group's fields
     do not change, so only dirty state needs a refresh. */
  refreshDirty();
  if (focusSpec) {
    var ctl = S.fieldByKey[pathKey(ctx.base.concat(focusSpec.parts))];
    if (ctl && ctl.focus) ctl.focus();
  }
}

/* --- connection probes --------------------------------------------------- */

function probeRow(label, iconName, run) {
  var btn = h('button.btn.btn--secondary.btn--sm', { type: 'button' },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon(iconName, { size: 'sm' })),
    h('span.btn__spinner', { 'aria-hidden': 'true' }, h('span.spinner')),
    h('span.btn__label', label));
  var out = h('div.probe', { role: 'status', 'aria-live': 'polite' });
  var el = h('div.field', h('div.row.row--wrap', btn, h('div.row__grow', out)));

  function show(kind, text, detailNode) {
    clear(out);
    out.className = 'probe' + (kind === 'ok' ? ' probe--ok' : kind === 'bad' ? ' probe--bad' : '');
    if (kind === 'busy') out.appendChild(h('span.spinner'));
    else out.appendChild(icon(kind === 'ok' ? 'check' : 'alert', { size: 'sm' }));
    var body = h('div.stack.stack--tight', h('span', { text: text }));
    if (detailNode) body.appendChild(detailNode);
    out.appendChild(body);
  }

  track(on(btn, 'click', function () {
    btn.setAttribute('aria-busy', 'true');
    show('busy', 'Testing…');
    run().then(function (res) {
      btn.removeAttribute('aria-busy');
      if (!S || !S || S.destroyed) return;
      show(res.ok ? 'ok' : 'bad', res.text, res.detail || null);
    }, function (err) {
      btn.removeAttribute('aria-busy');
      if (!S || S.destroyed || api.isAbort(err)) return;
      show('bad', api.describe(err));
    });
  }));
  return el;
}

function rtspProbeRow(cameraId) {
  return probeRow('Test stream', 'live', function () {
    var cam = S.draft.cameras[cameraId];
    var body = { rtsp: { uri: String(cam.rtsp.uri || '').trim(), transport: cam.rtsp.transport || 'tcp' } };
    return api.probeCamera(body, { signal: S.abort.signal }).then(function (res) {
      var r = res && res.rtsp ? res.rtsp : { ok: false, error: 'No result.' };
      if (r.ok) {
        return {
          ok: true,
          text: 'Stream opened: ' + r.width + '×' + r.height + (r.fps ? ' at ' + r.fps + ' fps' : ''),
          detail: h('span.probe__detail', { text: 'software decode · ' + r.elapsed_ms + ' ms' })
        };
      }
      return { ok: false, text: r.error || 'The stream did not open.',
        detail: r.elapsed_ms ? h('span.probe__detail', { text: r.elapsed_ms + ' ms' }) : null };
    });
  });
}

function onvifProbeRow(cameraId) {
  return probeRow('Test ONVIF', 'refresh', function () {
    var cam = S.draft.cameras[cameraId];
    var body = { onvif: {
      host: String(cam.onvif.host || '').trim(), port: Number(cam.onvif.port) || 80,
      username_env: String(cam.onvif.username_env || '').trim(),
      password_env: String(cam.onvif.password_env || '').trim()
    } };
    return api.probeCamera(body, { signal: S.abort.signal }).then(function (res) {
      var r = res && res.onvif ? res.onvif : { ok: false, error: 'No result.' };
      if (r.env) {
        for (var n in r.env) if (Object.prototype.hasOwnProperty.call(r.env, n)) S.env[n] = !!r.env[n];
        for (var i = 0; i < S.fields.length; i++) if (S.fields[i].refreshEnv) S.fields[i].refreshEnv();
      }
      if (!r.ok) return { ok: false, text: r.error || 'ONVIF did not answer.' };
      var dev = r.device || {};
      var who = [dev.manufacturer, dev.model].filter(function (x) { return x && x !== 'unknown'; }).join(' ');
      var list = h('div.probe__profiles');
      var profiles = isArray(r.profiles) ? r.profiles : [];
      for (var p = 0; p < profiles.length; p++) {
        (function (prof) {
          var useProfile = h('button.chip', { type: 'button', 'aria-pressed': String(cam.onvif.profile || '') === String(prof.token || '') ? 'true' : 'false' },
            h('span.chip__label', { text: String(prof.token || '?') }));
          track(on(useProfile, 'click', function () {
            setAt(S.draft, ['cameras', cameraId, 'onvif', 'profile'], String(prof.token || ''));
            var ctl = S.fieldByKey[pathKey(['cameras', cameraId, 'onvif', 'profile'])];
            if (ctl && ctl.sync) ctl.sync();
            var chips = list.querySelectorAll('.chip');
            for (var c = 0; c < chips.length; c++) chips[c].setAttribute('aria-pressed', chips[c] === useProfile ? 'true' : 'false');
            onModelChanged();
          }));
          var row = h('div.probe__profile', useProfile);
          if (prof.uri) {
            row.appendChild(h('span.probe__detail', { text: prof.uri }));
            var useUri = h('button.btn.btn--ghost.btn--sm', { type: 'button' }, h('span.btn__label', 'Use as RTSP URI'));
            track(on(useUri, 'click', function () {
              setAt(S.draft, ['cameras', cameraId, 'rtsp', 'uri'], String(prof.uri));
              var uctl = S.fieldByKey[pathKey(['cameras', cameraId, 'rtsp', 'uri'])];
              if (uctl && uctl.sync) uctl.sync();
              onModelChanged();
              toast.info('RTSP URI replaced with the profile’s stream address', {
                detail: 'Add user:password@ before the host if the camera needs credentials.'
              });
            }));
            row.appendChild(useUri);
          }
          list.appendChild(row);
        }(profiles[p]));
      }
      return {
        ok: true,
        text: (who ? who + ' answered' : 'ONVIF answered') + ' with ' + profiles.length + ' ' + plural(profiles.length, 'profile') +
          (profiles.length ? '. Pick one to use it as the media profile.' : '.'),
        detail: profiles.length ? list : null
      };
    });
  });
}

/* --- headers ------------------------------------------------------------- */

function yamlButton() {
  var btn = h('button.btn.btn--secondary.btn--sm', { type: 'button' },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('layers', { size: 'sm' })),
    h('span.btn__label', 'Reveal in YAML'));
  track(on(btn, 'click', revealYaml));
  return btn;
}

function statusWord(cam) {
  if (!cam) return { word: '', cls: '' };
  if (cam.isNew) return { word: 'new', cls: 'tab__meta--stale' };
  var rt = cam.runtime || {};
  if (!rt.running) return { word: 'not running', cls: 'tab__meta--off' };
  if (rt.state === 'live') return { word: 'live', cls: 'tab__meta--live' };
  if (rt.state === 'stale') return { word: 'stale', cls: 'tab__meta--stale' };
  return { word: 'offline', cls: 'tab__meta--off' };
}

function cameraStatusLine(cam) {
  var line = h('p.statusline');
  var rt = cam.runtime || {};
  if (cam.isNew) {
    line.appendChild(h('span.camrow__dot.camrow__dot--stale', { 'aria-hidden': 'true' }));
    line.appendChild(h('span', { text: 'New camera — not saved yet. Save, then restart to start it.' }));
    return line;
  }
  if (!rt.running) {
    line.appendChild(h('span.camrow__dot.camrow__dot--offline', { 'aria-hidden': 'true' }));
    line.appendChild(h('span', { text: 'Not running. The process has not been restarted since this camera was configured.' }));
    return line;
  }
  var dotCls = rt.state === 'live' ? 'camrow__dot--live' : rt.state === 'stale' ? 'camrow__dot--stale' : 'camrow__dot--offline';
  line.appendChild(h('span.camrow__dot.' + dotCls, { 'aria-hidden': 'true' }));
  var words = rt.state === 'live' ? 'Live' : rt.state === 'stale' ? 'Stale' : 'Offline';
  if (rt.frame_age !== null && rt.frame_age !== undefined) words += ' · last frame ' + rt.frame_age + ' s ago';
  line.appendChild(h('span', { text: words }));
  line.appendChild(h('span.statusline__sep', { 'aria-hidden': 'true', text: '·' }));
  if (rt.onvif_connected) line.appendChild(h('span', { text: 'ONVIF ' + (rt.profile_token || 'connected') }));
  else line.appendChild(h('span', { text: 'no ONVIF session' }));
  if (rt.has_tracker) {
    line.appendChild(h('span.statusline__sep', { 'aria-hidden': 'true', text: '·' }));
    line.appendChild(h('span', { text: 'PTZ tracker active' }));
  }
  return line;
}

function renderGeneralSection(host, sec) {
  for (var i = 0; i < sec.groups.length; i++) host.appendChild(renderGroup(sec.groups[i], { base: ['general'], cameraId: null, camera: null }));
  if (sec.id === 'general.system') host.appendChild(renderServiceCard());
}

function renderServiceCard() {
  var fs = h('fieldset.fieldset', h('legend.fieldset__legend', { text: 'Service' }));
  var facts = h('dl.facts',
    h('div', h('dt', 'Configuration file'), h('dd', { text: S.configPath || 'config/cameras.yml' })),
    h('div', h('dt', 'Backups'), h('dd', { text: (S.backupDir || 'config/backups') + ' (last 20 saves)' })),
    h('div', h('dt', 'Service unit'), h('dd', { text: S.restart && S.restart.unit ? S.restart.unit : 'not running under systemd' })));
  fs.appendChild(facts);
  var restartBtn = h('button.btn.btn--secondary', { type: 'button', disabled: !(S.restart && S.restart.supported) },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('refresh', { size: 'sm' })),
    h('span.btn__label', 'Restart service'));
  track(on(restartBtn, 'click', restartService));
  fs.appendChild(h('div.field',
    h('div.row.row--wrap', restartBtn),
    h('p.field__hint', { text: S.restart && S.restart.supported
      ? 'Reloads the models and reopens every stream; detection pauses for roughly a minute.'
      : 'Restart is only available when the service runs under systemd. Restart it by hand: sudo systemctl restart animaltracker' })));
  return fs;
}

function renderCameraSection(host, id) {
  var cam = S.draft.cameras[id];
  if (!cam) {
    host.appendChild(h('p.field__hint', { text: 'That camera is no longer in the configuration.' }));
    return;
  }
  var ctx = { base: ['cameras', id], cameraId: id, camera: cam };
  for (var i = 0; i < CAMERA_GROUPS.length; i++) host.appendChild(renderGroup(CAMERA_GROUPS[i], ctx));

  var removeBtn = h('button.btn.btn--danger', { type: 'button' },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('trash', { size: 'sm' })),
    h('span.btn__label', 'Remove camera'));
  track(on(removeBtn, 'click', function () { removeCamera(id); }));
  host.appendChild(h('div.dangerzone',
    h('div.dangerzone__text',
      h('p.dangerzone__title', { text: 'Remove this camera' }),
      h('p.dangerzone__hint', { text: 'It leaves cameras.yml when you save and stops at the next restart. Its recordings stay on disk.' })),
    removeBtn));
}

function sectionTitle() {
  if (S.section.indexOf('general.') === 0) {
    var sec = findGeneralSection(S.section);
    return sec ? sec.label : 'Settings';
  }
  var cam = currentCamera();
  return cam ? (cam.name || cam.id) : S.section;
}

function findGeneralSection(id) {
  for (var i = 0; i < GENERAL_SECTIONS.length; i++) if (GENERAL_SECTIONS[i].id === id) return GENERAL_SECTIONS[i];
  return null;
}

function renderSection() {
  S.fields = [];
  S.fieldByKey = {};
  S.groupHosts = {};
  clear(S.panelEl);

  var titleId = uid('sec');
  var titleText = sectionTitle();
  var headStack = h('div.stack.stack--tight', h('h2.t-h3#' + titleId, { tabIndex: -1, text: titleText }));
  var isGeneral = S.section.indexOf('general.') === 0;
  var sec = isGeneral ? findGeneralSection(S.section) : null;
  var cam = isGeneral ? null : currentCamera();
  if (sec && sec.blurb) headStack.appendChild(h('p.settings__blurb', { text: sec.blurb }));
  if (cam) {
    headStack.appendChild(h('p.settings__blurb', { text: 'Camera id ' + cam.id + (cam.location ? ' · ' + cam.location : '') }));
    headStack.appendChild(cameraStatusLine(cam));
  }
  S.panelEl.setAttribute('aria-labelledby', titleId);
  S.panelEl.appendChild(h('div.settings__head', headStack, yamlButton()));

  var body = h('div.stack.stack--loose.field-form');
  S.panelEl.appendChild(body);

  if (sec) renderGeneralSection(body, sec);
  else if (cam) renderCameraSection(body, S.section);
  else {
    body.appendChild(h('div.empty',
      h('div.empty__art', icon('camera', { size: 'lg' })),
      h('h3.empty__title', 'Nothing to show here'),
      h('p.empty__body', 'Pick a section on the left, or add a camera.')));
  }

  renderNav();
  refreshDirty();
}

function selectSection(id, opts) {
  var o = opts || {};
  if (id === S.section && !o.force) return;
  S.section = id;
  renderSection();
  if (!o.silent) router.setQuery({ section: id === DEFAULT_SECTION ? null : id }, { replace: true });
  var heading = S.panelEl.querySelector('h2');
  if (heading && heading.focus) { try { heading.focus({ preventScroll: false }); } catch (e) { heading.focus(); } }
  if (S.panelEl.scrollIntoView && window.matchMedia && !window.matchMedia('(min-width: 1024px)').matches) {
    S.panelEl.scrollIntoView({ block: 'start' });
  }
}

/* ==========================================================================
   DIRTY STATE
   change = { path, key, section, restart, kind: 'field' | 'added' | 'removed', label }
   ========================================================================= */

function computeChanges() {
  var list = [];
  var map = {};
  if (!S.baseline || !S.draft) return { list: list, map: map };
  var i, spec, path, k;

  for (i = 0; i < GENERAL_SPECS.length; i++) {
    spec = GENERAL_SPECS[i];
    path = ['general'].concat(spec.parts);
    if (!eqValue(getAt(S.draft, path), getAt(S.baseline, path))) {
      k = pathKey(path);
      map[k] = 1;
      list.push({ path: path, key: k, section: GENERAL_SECTION_BY_KEY[spec.key], restart: !!spec.restart, kind: 'field', label: spec.label });
    }
  }

  for (var c = 0; c < S.draft.order.length; c++) {
    var id = S.draft.order[c];
    var dcam = S.draft.cameras[id];
    var bcam = S.baseline.cameras[id];
    if (!bcam) {
      list.push({ path: ['cameras', id], key: pathKey(['cameras', id]), section: id, restart: true, kind: 'added', label: (dcam.name || id) + ' added' });
      continue;
    }
    for (i = 0; i < CAMERA_SPECS.length; i++) {
      spec = CAMERA_SPECS[i];
      if (spec.when && !blockOn(dcam, spec.when) && !blockOn(bcam, spec.when)) continue;
      path = ['cameras', id].concat(spec.parts);
      if (!eqValue(getAt(S.draft, path), getAt(S.baseline, path))) {
        k = pathKey(path);
        map[k] = 1;
        list.push({ path: path, key: k, section: id, restart: !!spec.restart, kind: 'field', label: spec.label });
      }
    }
  }
  for (var b = 0; b < S.baseline.order.length; b++) {
    var bid = S.baseline.order[b];
    if (!S.draft.cameras[bid]) {
      list.push({ path: ['cameras', bid], key: pathKey(['cameras', bid]), section: bid, restart: true, kind: 'removed',
        label: (S.baseline.cameras[bid].name || bid) + ' removed' });
    }
  }
  return { list: list, map: map };
}

function onModelChanged() { refreshDirty(); }

function refreshDirty() {
  var d = computeChanges();
  var n = d.list.length;
  var restarts = 0;
  var bySection = {};
  for (var j = 0; j < d.list.length; j++) {
    if (d.list[j].restart) restarts += 1;
    var sec = d.list[j].section;
    bySection[sec] = (bySection[sec] || 0) + 1;
  }

  for (var i = 0; i < S.fields.length; i++) S.fields[i].setDirty(!!d.map[S.fields[i].key]);

  var lists = [S.navGeneral, S.navCameras];
  for (var l = 0; l < lists.length; l++) {
    if (!lists[l]) continue;
    var buttons = lists[l].querySelectorAll('[data-section]');
    for (var b = 0; b < buttons.length; b++) {
      var el = buttons[b];
      var count = bySection[el.dataset.section] || 0;
      var dot = el.querySelector('.field__dirty');
      if (dot) {
        dot.hidden = count === 0;
        if (count) dot.setAttribute('title', count + ' unsaved ' + plural(count, 'change'));
      }
    }
  }

  if (S.countEl) S.countEl.firstChild.nodeValue = n + ' unsaved ' + plural(n, 'change');
  if (S.countDetailEl) {
    S.countDetailEl.textContent = restarts
      ? restarts + ' of them ' + plural(restarts, 'takes', 'take') + ' effect after a restart'
      : 'Applied to the running process on save';
  }
  if (S.saveBtn) S.saveBtn.disabled = n === 0 || S.saving;
  if (S.resetBtn) S.resetBtn.disabled = n === 0 || S.saving;

  var show = n > 0;
  if (show !== S.selbarShown) {
    S.selbarShown = show;
    store.setChrome({ selbar: show ? S.selbar : null });
  }
  return d;
}

/* ==========================================================================
   SAVE / RESET
   ========================================================================= */

function setSaveBusy(busy) {
  if (!S.saveBtn) return;
  if (busy) S.saveBtn.setAttribute('aria-busy', 'true');
  else S.saveBtn.removeAttribute('aria-busy');
  S.saveBtn.disabled = busy;
  if (S.resetBtn) S.resetBtn.disabled = busy;
  if (S.selbar) {
    if (busy) S.selbar.classList.add('selbar--acting');
    else S.selbar.classList.remove('selbar--acting');
  }
}

function sectionOfPath(path) {
  if (path[0] === 'general') {
    /* general.notification.destinations.1.user_key_env belongs to the
       section that owns the longest known prefix. */
    for (var n = path.length; n > 1; n--) {
      var sec = GENERAL_SECTION_BY_KEY[path.slice(1, n).join('.')];
      if (sec) return sec;
    }
    return DEFAULT_SECTION;
  }
  return path[1];
}

/* The controller for a path: an exact match, else the one owning the
   longest prefix, with the rest of the path (a list index and property)
   handed back so the controller can point at the row. */
function controllerFor(path) {
  for (var n = path.length; n > 0; n--) {
    var ctl = S.fieldByKey[pathKey(path.slice(0, n))];
    if (ctl) return { ctl: ctl, rest: path.slice(n) };
  }
  return null;
}

function focusPath(path) {
  var sec = sectionOfPath(path);
  if (sec !== S.section) selectSection(sec, { silent: false });
  var hit = controllerFor(path);
  if (!hit) {
    /* The field may sit inside a closed Advanced fold. Open every fold in
       its group and try again. */
    for (var g in S.groupHosts) {
      if (!Object.prototype.hasOwnProperty.call(S.groupHosts, g)) continue;
      var det = S.groupHosts[g].el.querySelector('details.disclosure');
      if (det && !det.open) { det.open = true; S.openAdvanced[g] = true; }
    }
    hit = controllerFor(path);
  }
  if (!hit) return;
  var ctl = hit.ctl;
  var det2 = ctl.el.closest ? ctl.el.closest('details.disclosure') : null;
  if (det2 && !det2.open) det2.open = true;
  if (ctl.el && ctl.el.scrollIntoView) ctl.el.scrollIntoView({ block: 'center' });
  if (ctl.focus) ctl.focus(hit.rest);
}

function serverPath(p) {
  /* 'cameras.cam1.rtsp.uri' -> ['cameras','cam1','rtsp','uri'];
     'general.clip.pre_seconds' -> ['general','clip','pre_seconds'] */
  return String(p || '').split('.');
}

/** Write the draft. Resolves true when config/cameras.yml was written (or
    there was nothing to write), false when nothing was saved — so a caller
    that meant "save and then leave" knows whether it may leave. */
function save(opts) {
  /* `apply` posts even when the form is not dirty. That is the only way to
     reconcile a value edited in the file behind this page: the payload then
     equals the file, so there is nothing to "change", and without this the
     Save button looked like it did nothing at all. */
  var force = !!(opts && opts.apply);
  if (S.saving || !S.draft || !S.baseline) return Promise.resolve(false);

  var problems = validateModel(S.draft);
  if (problems.length) {
    var first = problems[0];
    focusPath(first.path);
    var phit = controllerFor(first.path);
    if (phit && phit.ctl.setError) phit.ctl.setError(first.message, phit.rest);
    toast.danger('Nothing was saved — ' + problems.length + ' ' + plural(problems.length, 'field') + ' failed validation.', {
      detail: first.message
    });
    return Promise.resolve(false);
  }

  var payload;
  try {
    payload = buildPayload(S.draft);
  } catch (err) {
    toast.error('Nothing was sent — the settings payload could not be built.', {
      detail: String(err && err.message ? err.message : err)
    });
    return Promise.resolve(false);
  }

  var d = computeChanges();
  var changed = d.list.slice();
  var n = changed.length;
  if (!n && !force) return Promise.resolve(true);   /* nothing to lose */
  var restarts = 0;
  for (var r = 0; r < changed.length; r++) if (changed[r].restart) restarts += 1;

  var prevBaseline = clone(S.baseline);
  S.saving = true;
  setSaveBusy(true);

  /* Optimistic: the baseline advances now, so the form reads as saved while
     the write is in flight. `prevBaseline` is the rollback. */
  S.baseline = clone(S.draft);
  refreshDirty();

  var progress = toast.progress(n ? 'Writing config/cameras.yml…' : 'Applying the file to the running process…', {
    detail: n
      ? n + ' ' + plural(n, 'change') + ' · ' + S.draft.order.length + ' ' + plural(S.draft.order.length, 'camera')
      : 'Nothing to write; taking the settings already in the file.'
  });

  return api.saveConfig(payload, { timeout: 30000, signal: S.abort.signal }).then(function (res) {
    if (!S || S.destroyed) return false;
    progress.close();
    S.saving = false;
    setSaveBusy(false);

    for (var i = 0; i < changed.length; i++) {
      var ctl = S.fieldByKey[changed[i].key];
      if (ctl && ctl.setStatus) ctl.setStatus('saved', 'Saved');
    }
    for (var id in S.draft.cameras) if (Object.prototype.hasOwnProperty.call(S.draft.cameras, id)) delete S.draft.cameras[id].isNew;

    S.restart = res && res.restart ? res.restart : S.restart;
    /* The save applied every live setting the file held, this one included. */
    S.unapplied = { count: 0, keys: [] };
    renderBanner();
    var live = res && isArray(res.applied_live) ? res.applied_live.length : 0;
    var detail;
    if (S.restart && S.restart.required) {
      detail = (restarts ? restarts + ' ' + plural(restarts, 'change') + ' ' + plural(restarts, 'waits', 'wait') + ' for a restart' : 'A restart is pending') +
        (live ? '; ' + live + ' applied to the running process' : '') + '. See the banner above.';
    } else {
      detail = live ? 'Applied to the running process immediately' : 'Written to disk';
    }
    var applied = isArray(res && res.applied_live) ? res.applied_live.length : 0;
    if (n) {
      toast.success(n + ' ' + plural(n, 'change') + ' saved to config/cameras.yml', {
        detail: detail + (res && res.backup ? ' · backup kept' : '')
      });
    } else {
      toast.success(applied
        ? applied + ' ' + plural(applied, 'setting') + ' from the file applied to the running process'
        : 'The running process already matches the file', { detail: null });
    }
    refreshDirty();
    /* Pick up the server's normalised copy (schema defaults for a new camera,
       runtime annotations) without disturbing anything the operator typed
       since — the quiet reload declines when the draft is dirty. */
    load({ quiet: true, force: true });
    return true;
  }, function (err) {
    if (!S || S.destroyed) return false;
    progress.close();
    S.saving = false;
    setSaveBusy(false);
    if (api.isAbort(err)) return false;

    /* Roll the baseline back: every edit becomes dirty again, still in the
       controls, still editable. A failed write must never eat work. */
    S.baseline = prevBaseline;
    refreshDirty();
    for (var j = 0; j < changed.length; j++) {
      var ctl2 = S.fieldByKey[changed[j].key];
      if (ctl2 && ctl2.setStatus) ctl2.setStatus('error', 'Not saved');
    }

    var body = err && err.body ? err.body : null;
    var serverProblems = body && isArray(body.problems) ? body.problems : [];
    if (serverProblems.length) {
      var firstPath = serverPath(serverProblems[0].path);
      focusPath(firstPath);
      for (var p = 0; p < serverProblems.length; p++) {
        var shit = controllerFor(serverPath(serverProblems[p].path));
        if (shit && shit.ctl.setError) shit.ctl.setError(serverProblems[p].message, shit.rest);
      }
      toast.error('config/cameras.yml was NOT written — the server rejected ' + serverProblems.length + ' ' + plural(serverProblems.length, 'field') + '.', {
        detail: serverProblems[0].path + ': ' + serverProblems[0].message,
        retry: save
      });
      return false;
    }
    toast.error('config/cameras.yml was NOT written — your edits are still here.', {
      detail: api.describe(err),
      retry: save
    });
    return false;
  });
}

function resetDraft() {
  var d = computeChanges();
  if (!d.list.length) return;
  var n = d.list.length;
  var dlg = dialog({
    role: 'alertdialog',
    tone: 'danger',
    title: 'Discard ' + n + ' unsaved ' + plural(n, 'change') + '?',
    body: 'The form returns to the values the server last confirmed. Nothing on disk changes.',
    stakes: describeChanges(d.list),
    actions: [
      { label: 'Keep editing', variant: 'secondary', value: false, focus: true },
      { label: 'Discard changes', variant: 'danger', value: true }
    ]
  });
  dlg.result.then(function (v) {
    if (v !== true || !S || S.destroyed) return;
    S.draft = clone(S.baseline);
    if (S.section.indexOf('general.') !== 0 && !S.draft.cameras[S.section]) S.section = DEFAULT_SECTION;
    renderSection();
    refreshDirty();
    toast.info(n + ' ' + plural(n, 'change') + ' discarded');
  });
}

function describeChanges(list) {
  var seen = {};
  var names = [];
  for (var i = 0; i < list.length; i++) {
    var sec = list[i].section;
    if (seen[sec]) continue;
    seen[sec] = 1;
    var gs = findGeneralSection(sec);
    if (gs) names.push(gs.label);
    else {
      var cam = S.draft.cameras[sec] || (S.baseline && S.baseline.cameras[sec]);
      names.push(cam ? (cam.name || sec) : sec);
    }
  }
  return list.length + ' ' + plural(list.length, 'field') + ' across ' + names.join(', ');
}

/* ==========================================================================
   CAMERAS: ADD AND REMOVE
   ========================================================================= */

function suggestId() {
  var n = 1;
  while (S.draft.cameras['cam' + n]) n += 1;
  return 'cam' + n;
}

function newCameraFromDefaults(id, fields) {
  var base = S.defaults && S.defaults.camera ? clone(S.defaults.camera) : {};
  base.id = id;
  base.name = fields.name;
  base.location = fields.location || null;
  base.rtsp = Object.assign(base.rtsp || {}, {
    uri: fields.uri, transport: fields.transport || 'tcp', hwaccel: !!fields.hwaccel
  });
  if (fields.onvif) {
    base.onvif = {
      host: fields.host, port: Number(fields.port) || 80, profile: null,
      username_env: envName(id, 'USER'), password_env: envName(id, 'PASS')
    };
  } else {
    base.onvif = null;
  }
  if (fields.copyFrom && S.draft.cameras[fields.copyFrom]) {
    var src = S.draft.cameras[fields.copyFrom];
    base.thresholds = clone(src.thresholds);
    base.include_species = clone(src.include_species);
    base.exclude_species = clone(src.exclude_species);
    base.notification = clone(src.notification);
    base.inference_max_width = src.inference_max_width;
    base.detect_enabled = src.detect_enabled;
    base.rtsp.latency_ms = src.rtsp.latency_ms;
    base.rtsp.frame_skip = src.rtsp.frame_skip;
  }
  var cam = normalizeCamera(base, id);
  cam.isNew = true;
  cam.runtime = { running: false };
  return cam;
}

function addCameraDialog() {
  var f = { id: suggestId(), name: '', location: '', uri: '', transport: 'tcp', hwaccel: false, copyFrom: '', onvif: false, host: '', port: 80 };
  var errs = {};
  var els = {};

  function field(key, label, hint, inputEl) {
    var id = uid('add');
    inputEl.id = id;
    var err = h('p.field__error', { hidden: true, role: 'alert' });
    var wrap = h('div.field', h('label.field__label', { 'for': id, text: label }), inputEl, h('p.field__hint', { text: hint }), err);
    els[key] = { input: inputEl, err: err, hint: wrap.querySelector('.field__hint') };
    return wrap;
  }

  function showErrors() {
    for (var k in els) {
      if (!Object.prototype.hasOwnProperty.call(els, k)) continue;
      var e = els[k];
      clear(e.err);
      if (errs[k]) {
        e.err.hidden = false;
        e.err.appendChild(icon('alert', { size: 'sm' }));
        e.err.appendChild(h('span', { text: errs[k] }));
        e.hint.hidden = true;
        e.input.setAttribute('aria-invalid', 'true');
      } else {
        e.err.hidden = true;
        e.hint.hidden = false;
        e.input.removeAttribute('aria-invalid');
      }
    }
  }

  function validate() {
    errs = {};
    var id = f.id.trim();
    if (!CAMERA_ID_RE.test(id)) errs.id = 'Letters, digits, - or _ only, up to 32 characters. The id names the clip folder and the URLs.';
    else if (S.draft.cameras[id]) errs.id = 'A camera with this id already exists.';
    if (!f.name.trim()) errs.name = 'Give the camera a name.';
    if (!f.uri.trim()) errs.uri = 'The stream address is required.';
    else if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(f.uri.trim())) errs.uri = 'Expected a URL such as rtsp://host:554/stream.';
    if (f.onvif && !f.host.trim()) errs.host = 'The ONVIF host is required when ONVIF control is on.';
    showErrors();
    var any = false;
    for (var k in errs) if (Object.prototype.hasOwnProperty.call(errs, k)) any = true;
    return !any;
  }

  var idInput = h('input.input.input--mono', { type: 'text', value: f.id, autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false' });
  var nameInput = h('input.input', { type: 'text', placeholder: 'Bird feeder', autocomplete: 'off' });
  var locInput = h('input.input', { type: 'text', placeholder: 'Back yard', autocomplete: 'off' });
  var uriInput = h('input.input.input--mono', { type: 'text', placeholder: 'rtsp://user:pass@192.168.1.50:554/stream1', autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false' });
  var transportSel = h('select.select__el');
  for (var t = 0; t < TRANSPORT_OPTIONS.length; t++) transportSel.appendChild(h('option', { value: TRANSPORT_OPTIONS[t][0] }, TRANSPORT_OPTIONS[t][1]));
  var hwCheck = h('input.check__box', { type: 'checkbox' });
  var copySel = h('select.select__el');
  copySel.appendChild(h('option', { value: '' }, 'Schema defaults'));
  for (var c = 0; c < S.draft.order.length; c++) {
    var cid = S.draft.order[c];
    copySel.appendChild(h('option', { value: cid }, (S.draft.cameras[cid].name || cid) + ' (' + cid + ')'));
  }
  var onvifCheck = h('input.check__box', { type: 'checkbox' });
  var hostInput = h('input.input.input--mono', { type: 'text', placeholder: '192.168.1.50', autocomplete: 'off', spellcheck: 'false' });
  var portInput = h('input.input.input--mono', { type: 'number', min: '1', max: '65535', step: '1', value: '80', inputmode: 'numeric' });
  var onvifBlock = h('div.stack', { hidden: true },
    field('host', 'ONVIF host', 'Defaults to the host in the stream address.', hostInput),
    field('port', 'ONVIF port', 'Usually 80, 8000 or 8899.', portInput),
    h('p.field__hint', { text: 'Credentials are read from ' + envName(f.id, 'USER') + ' and ' + envName(f.id, 'PASS') + ' in config/secrets.env; you can rename them after adding.' }));

  var probe = probeRow('Test stream', 'live', function () {
    return api.probeCamera({ rtsp: { uri: f.uri.trim(), transport: f.transport } }, { signal: S.abort.signal }).then(function (res) {
      var r = res && res.rtsp ? res.rtsp : { ok: false, error: 'No result.' };
      if (r.ok) return { ok: true, text: 'Stream opened: ' + r.width + '×' + r.height + (r.fps ? ' at ' + r.fps + ' fps' : '') };
      return { ok: false, text: r.error || 'The stream did not open.' };
    });
  });

  var content = h('div.stack.stack--loose',
    h('div.stack',
      field('id', 'Camera id', 'Short and permanent: it names the clip folder and the URLs.', idInput),
      field('name', 'Name', 'Shown on Live, Recordings and in alerts.', nameInput),
      field('location', 'Location', 'Optional placement note.', locInput)),
    h('fieldset.fieldset', h('legend.fieldset__legend', { text: 'Stream' }),
      field('uri', 'RTSP URI', 'Passed to FFmpeg as-is; include user:password@ if the camera needs it.', uriInput),
      field('transport', 'Transport', 'TCP unless the camera needs UDP.', h('div.select', transportSel, h('span.select__chevron', icon('chevron-down', { size: 'sm' })))),
      h('label.check', hwCheck, h('span.check__label', 'Decode on the GPU (CUDA)')),
      probe),
    h('fieldset.fieldset', h('legend.fieldset__legend', { text: 'Detection & alerts' }),
      field('copyFrom', 'Start from', 'Thresholds, species filters and notification settings are copied from this camera.',
        h('div.select', copySel, h('span.select__chevron', icon('chevron-down', { size: 'sm' }))))),
    h('fieldset.fieldset', h('legend.fieldset__legend', { text: 'PTZ' }),
      h('label.check', onvifCheck, h('span.check__label', 'This camera has ONVIF control (pan/tilt/zoom)')),
      onvifBlock));

  var dlg = null;
  function bind(input, key, evName, read) {
    input.addEventListener(evName || 'input', function () {
      f[key] = read ? read() : input.value;
      if (errs[key]) { delete errs[key]; showErrors(); }
    });
  }
  bind(idInput, 'id');
  bind(nameInput, 'name');
  bind(locInput, 'location');
  bind(uriInput, 'uri');
  uriInput.addEventListener('blur', function () { if (!f.host && f.onvif) { f.host = hostFromUri(f.uri); hostInput.value = f.host; } });
  bind(transportSel, 'transport', 'change');
  bind(hwCheck, 'hwaccel', 'change', function () { return hwCheck.checked; });
  bind(copySel, 'copyFrom', 'change');
  onvifCheck.addEventListener('change', function () {
    f.onvif = onvifCheck.checked;
    onvifBlock.hidden = !f.onvif;
    if (f.onvif && !f.host) { f.host = hostFromUri(f.uri); hostInput.value = f.host; }
  });
  bind(hostInput, 'host');
  bind(portInput, 'port');

  dlg = dialog({
    role: 'dialog',
    title: 'Add a camera',
    body: 'The camera is added to the draft. Save writes it to cameras.yml; a restart starts it.',
    width: 640,
    content: content,
    initialFocus: nameInput,
    actions: [
      { label: 'Cancel', variant: 'secondary', value: null },
      { label: 'Add camera', variant: 'primary', value: 'add', keepOpen: true, onSelect: function () {
        if (!validate()) return;
        var id = f.id.trim();
        var cam = newCameraFromDefaults(id, {
          name: f.name.trim(), location: f.location.trim(), uri: f.uri.trim(), transport: f.transport,
          hwaccel: f.hwaccel, copyFrom: f.copyFrom, onvif: f.onvif, host: f.host.trim(), port: f.port
        });
        S.draft.cameras[id] = cam;
        S.draft.order.push(id);
        dlg.close('added');
        selectSection(id);
        toast.info((cam.name || id) + ' added to the draft', {
          detail: 'Review its settings, then Save. It starts after the next restart.'
        });
      } }
    ]
  });
  /* dialog() only focuses actions; put the caret in the first field. */
  window.setTimeout(function () { try { nameInput.focus(); } catch (e) {} }, 0);
}

function removeCamera(id) {
  var cam = S.draft.cameras[id];
  if (!cam) return;
  if (S.draft.order.length <= 1) {
    toast.info('Add the replacement camera first', { detail: 'The configuration must keep at least one camera.' });
    return;
  }
  var wasNew = !S.baseline.cameras[id];
  var dlg = dialog({
    role: 'alertdialog',
    tone: 'danger',
    title: 'Remove ' + (cam.name || id) + '?',
    body: wasNew
      ? 'It was only added to this draft; nothing on disk changes.'
      : 'It leaves cameras.yml when you save and stops at the next restart. Recordings under clips/' + id + ' stay on disk.',
    stakes: wasNew ? null : 'Camera ' + id + ' · ' + (cam.runtime && cam.runtime.running ? 'currently running' : 'not running'),
    actions: [
      { label: 'Keep it', variant: 'secondary', value: false, focus: true },
      { label: 'Remove camera', variant: 'danger', value: true }
    ]
  });
  dlg.result.then(function (v) {
    if (v !== true || !S || S.destroyed) return;
    delete S.draft.cameras[id];
    var idx = S.draft.order.indexOf(id);
    if (idx >= 0) S.draft.order.splice(idx, 1);
    for (var g in S.openAdvanced) if (g.indexOf(id + ':') === 0) delete S.openAdvanced[g];
    selectSection(S.draft.order.length ? S.draft.order[Math.max(0, idx - 1)] : DEFAULT_SECTION, { force: true });
    toast.info((cam.name || id) + ' removed from the draft', {
      detail: wasNew ? 'Nothing to save.' : 'Save to write the change; the camera stops after a restart.'
    });
  });
}

/* ==========================================================================
   RESTART
   ========================================================================= */

function renderBanner() {
  if (!S.bannerEl) return;
  clear(S.bannerEl);
  var r = S.restart;
  var drift = S.unapplied && S.unapplied.count ? S.unapplied : null;
  if (!r || !r.required) {
    /* No restart needed, but the file may still hold live settings the
       running process has not taken — edited by hand, or by another writer.
       They used to be invisible: this page reads the file, so it showed them
       as though they were in force. */
    if (drift) {
      S.bannerEl.hidden = false;
      S.bannerEl.appendChild(driftNotice(drift));
      return;
    }
    S.bannerEl.hidden = true;
    return;
  }
  S.bannerEl.hidden = false;
  if (drift) S.bannerEl.appendChild(driftNotice(drift));
  var reasons = isArray(r.reasons) ? r.reasons : [];
  var shown = reasons.slice(0, 4);
  var list = h('ul.notice__list');
  for (var i = 0; i < shown.length; i++) list.appendChild(h('li', { text: shown[i] }));
  if (reasons.length > shown.length) list.appendChild(h('li', { text: 'and ' + (reasons.length - shown.length) + ' more' }));
  var body = h('div.notice__body',
    h('p.notice__title', { text: 'The running process is out of date with cameras.yml' }),
    list,
    h('p.notice__hint', { text: r.supported
      ? 'Restarting reloads the models and reopens every stream; detection pauses for roughly a minute.'
      : 'Restart the service by hand to apply these: sudo systemctl restart animaltracker' }));
  var actions = h('div.notice__actions');
  if (r.supported) {
    var btn = h('button.btn.btn--primary.btn--sm', { type: 'button', disabled: S.restarting },
      h('span.btn__icon', { 'aria-hidden': 'true' }, icon('refresh', { size: 'sm' })),
      h('span.btn__label', 'Restart now'));
    track(on(btn, 'click', restartService));
    actions.appendChild(btn);
  }
  S.bannerEl.appendChild(h('div.notice.notice--warn', { role: 'status' }, icon('alert'), body, actions));
}

/** Live settings cameras.yml has changed that the running process has not
    taken. A restart is not what they need — Save applies them. */
function driftNotice(drift) {
  var keys = isArray(drift.keys) ? drift.keys : [];
  var shown = keys.slice(0, 4);
  var list = h('ul.notice__list');
  for (var i = 0; i < shown.length; i++) list.appendChild(h('li', { text: shown[i] }));
  if (keys.length > shown.length) {
    list.appendChild(h('li', { text: 'and ' + (keys.length - shown.length) + ' more' }));
  }
  var body = h('div.notice__body',
    h('p.notice__title', {
      text: drift.count + ' ' + plural(drift.count, 'setting') + ' in cameras.yml ' +
        (drift.count === 1 ? 'is' : 'are') + ' not in effect'
    }),
    list,
    h('p.notice__hint', {
      text: 'The file was changed outside this page. These apply without a restart — '
        + 'press Save to take them, or restart the service.'
    }));
  var actions = h('div.notice__actions');
  var btn = h('button.btn.btn--primary.btn--sm', { type: 'button' },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('check', { size: 'sm' })),
    h('span.btn__label', 'Apply them'));
  track(on(btn, 'click', function () { save({ apply: true }); }));
  actions.appendChild(btn);
  return h('div.notice.notice--warn', { role: 'status' }, icon('alert'), body, actions);
}

function restartService() {
  if (S.restarting) return;
  var dirty = computeChanges().list.length;
  var dlg = dialog({
    role: 'alertdialog',
    tone: 'danger',
    icon: 'refresh',
    title: 'Restart Animal Tracker?',
    body: 'Detection and recording stop while the process reloads its models and reopens every stream, typically for 15–60 seconds. A clip being recorded right now is cut short.',
    stakes: dirty ? dirty + ' unsaved ' + plural(dirty, 'change') + ' in this form will NOT be part of the restart — save first.' : null,
    actions: [
      { label: 'Cancel', variant: 'secondary', value: false, focus: true },
      { label: 'Restart service', variant: 'danger', value: true }
    ]
  });
  dlg.result.then(function (v) {
    if (v !== true || !S || S.destroyed) return;
    S.restarting = true;
    renderBanner();
    var progress = toast.progress('Restarting Animal Tracker…', { detail: 'Asking systemd to restart the service' });
    api.restart({ signal: S.abort.signal }).then(function () {
      if (!S || !S || S.destroyed) return;
      waitForServer(progress);
    }, function (err) {
      if (!S || !S || S.destroyed) return;
      progress.close();
      S.restarting = false;
      renderBanner();
      if (api.isAbort(err)) return;
      toast.error('The service was not restarted.', { detail: api.describe(err) });
    });
  });
}

function waitForServer(progress) {
  var started = Date.now();
  var sawDown = false;
  function tick() {
    if (!S || !S || S.destroyed) return;
    var elapsed = Math.round((Date.now() - started) / 1000);
    if (elapsed > 300) {
      progress.close();
      S.restarting = false;
      renderBanner();
      toast.error('The server has not come back after 5 minutes.', {
        detail: 'Check it with: journalctl -u animaltracker -n 100'
      });
      return;
    }
    api.cameras({ timeout: 3000, signal: S.abort.signal }).then(function () {
      if (!S || !S || S.destroyed) return;
      if (sawDown || elapsed > 20) {
        progress.close();
        S.restarting = false;
        toast.success('Animal Tracker is back', { detail: 'Restarted in about ' + elapsed + ' s' });
        load({});
        return;
      }
      progress.update({ detail: 'Waiting for the old process to stop… ' + elapsed + ' s' });
      window.setTimeout(tick, 2000);
    }, function (err) {
      if (!S || S.destroyed || api.isAbort(err)) return;
      sawDown = true;
      progress.update({ detail: 'Waiting for the service to come back… ' + elapsed + ' s' });
      window.setTimeout(tick, 2000);
    });
  }
  window.setTimeout(tick, 2500);
}

/* ==========================================================================
   YAML PREVIEW
   ========================================================================= */

function revealYaml() {
  var text;
  try {
    if (S.section.indexOf('general.') === 0) {
      text = 'general:\n' + toYaml(buildGeneralPayload(S.draft), 1);
    } else {
      var cam = buildCameraPayload(S.draft, S.section);
      var body = toYaml(cam, 2);
      /* "  id: cam1" -> "- id: cam1" so the block reads as a list item */
      text = 'cameras:\n' + body.replace(/^ {4}/, '  - ');
    }
  } catch (err) {
    text = '# This section cannot be serialised yet:\n# ' + String(err && err.message ? err.message : err);
  }
  var pre = h('pre.code.mono', { tabIndex: 0, style: { overflowX: 'auto', whiteSpace: 'pre', margin: '0' } });
  pre.textContent = text;
  dialog({
    role: 'dialog',
    title: 'What this section writes',
    body: 'The managed keys as they will be merged into config/cameras.yml. Keys the editor does not manage are left as they are; comments are not preserved.',
    width: 640,
    content: pre,
    actions: [{ label: 'Close', variant: 'secondary', value: null, focus: true }]
  });
}

/* ==========================================================================
   NAV + LAYOUT
   ========================================================================= */

function navButton(item) {
  return h('button.tab', { type: 'button', dataset: { section: item.id } },
    h('span', { 'aria-hidden': 'true' }, icon(item.iconName, { size: 'sm' })),
    h('span.truncate', { dataset: { role: 'label' } }),
    h('span.field__dirty', { hidden: true, 'aria-hidden': 'true' }),
    h('span.tab__meta', { dataset: { role: 'meta' } }));
}

function navUpdate(node, item) {
  var lab = node.querySelector('[data-role="label"]');
  if (lab) lab.textContent = item.label;
  var meta = node.querySelector('[data-role="meta"]');
  if (meta) {
    meta.textContent = item.meta || '';
    meta.className = 'tab__meta' + (item.metaCls ? ' ' + item.metaCls : '');
    meta.hidden = !item.meta;
  }
  if (item.id === S.section) node.setAttribute('aria-current', 'page');
  else node.removeAttribute('aria-current');
  node.setAttribute('aria-label', item.label + ' settings' + (item.meta ? ' — ' + item.meta : ''));
}

function renderNav() {
  var general = [];
  for (var i = 0; i < GENERAL_SECTIONS.length; i++) {
    general.push({ id: GENERAL_SECTIONS[i].id, label: GENERAL_SECTIONS[i].label, iconName: GENERAL_SECTIONS[i].iconName });
  }
  keyedList(S.navGeneral, general, { key: function (it) { return it.id; }, create: navButton, update: navUpdate });

  var cams = [];
  for (var c = 0; c < S.draft.order.length; c++) {
    var id = S.draft.order[c];
    var cam = S.draft.cameras[id];
    var sw = statusWord(cam);
    cams.push({ id: id, label: cam.name || id, iconName: 'camera', meta: sw.word, metaCls: sw.cls });
  }
  keyedList(S.navCameras, cams, { key: function (it) { return it.id; }, create: navButton, update: navUpdate });
}

function buildNav() {
  S.navGeneral = h('div.settings__navlist', { role: 'list' });
  S.navCameras = h('div.settings__navlist', { role: 'list' });
  var addBtn = h('button.btn.btn--secondary.btn--sm', { type: 'button' },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('plus', { size: 'sm' })),
    h('span.btn__label', 'Add camera'));
  track(on(addBtn, 'click', addCameraDialog));
  var nav = h('nav.settings__nav', { 'aria-label': 'Settings sections' },
    h('div.settings__navgroup', h('p.overline', { text: 'General' }), S.navGeneral),
    h('div.settings__navgroup', h('p.overline', { text: 'Cameras' }), S.navCameras, h('div', addBtn)));
  track(delegate(nav, 'click', '[data-section]', function (ev, node) {
    ev.preventDefault();
    selectSection(node.dataset.section);
  }));
  return nav;
}

function buildBody() {
  clear(S.contentEl);

  var intro = h('p.settings__intro',
    icon('info', { size: 'sm' }),
    h('span', { text: 'Saving rewrites config/cameras.yml; the previous version is kept in config/backups (last 20). Comments in the file are not preserved. Fields marked Restart are read at startup.' }));

  S.bannerEl = h('div', { hidden: true });
  var nav = buildNav();
  S.panelEl = h('section.settings__panel', { role: 'region', tabIndex: -1 });
  var layout = h('div.settings', nav, S.panelEl);

  S.contentEl.appendChild(h('div.stack.stack--loose', intro, S.bannerEl, layout));

  renderBanner();
  renderSection();
}

/* ==========================================================================
   SAVE BAR
   ========================================================================= */

function buildSelbar() {
  var count = h('div.selbar__count', { role: 'status', 'aria-live': 'polite' });
  count.appendChild(document.createTextNode('0 unsaved changes'));
  var detail = h('span');
  count.appendChild(detail);
  S.countEl = count;
  S.countDetailEl = detail;

  var reset = h('button.btn.btn--secondary', { type: 'button', disabled: true },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('undo', { size: 'sm' })),
    h('span.btn__label', 'Reset'));
  var saveBtn = h('button.btn.btn--primary', { type: 'button', disabled: true },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('check', { size: 'sm' })),
    h('span.btn__spinner', { 'aria-hidden': 'true' }, h('span.spinner')),
    h('span.btn__label', 'Save changes'));
  S.resetBtn = reset;
  S.saveBtn = saveBtn;

  track(on(reset, 'click', resetDraft));
  track(on(saveBtn, 'click', save));

  return h('div.selbar', count, h('div.spacer'),
    h('div.selbar__actions', reset, h('span.selbar__sep'), saveBtn));
}

/* ==========================================================================
   LOADING
   ========================================================================= */

function skeleton() {
  var host = h('div.stack.stack--loose');
  for (var i = 0; i < 6; i++) {
    host.appendChild(h('div.stack.stack--tight',
      h('span.skel.skel--text', { style: { width: '30%' } }),
      h('span.skel.skel--row')));
  }
  host.setAttribute('aria-hidden', 'true');
  return host;
}

function errorState(err, retry) {
  var body = err && err.body ? err.body : null;
  var problems = body && isArray(body.problems) ? body.problems : [];
  var box = h('div.empty.empty--error',
    h('div.empty__art', icon('alert', { size: 'lg' })),
    h('h2.empty__title', problems.length ? 'cameras.yml does not validate' : 'Settings could not be loaded'),
    h('p.empty__body', { text: api.describe(err) }));
  if (problems.length) {
    var list = h('ul.notice__list', { style: { textAlign: 'left' } });
    for (var i = 0; i < Math.min(problems.length, 8); i++) list.appendChild(h('li', { text: problems[i].path + ': ' + problems[i].message }));
    box.appendChild(h('div.empty__cause', list));
    box.appendChild(h('p.empty__body', 'Fix the file by hand (a backup of the last good version is in config/backups if the editor wrote it), then try again.'));
  }
  box.appendChild(h('p.empty__endpoint', { text: 'GET /api/config' }));
  var actions = h('div.empty__actions');
  var again = h('button.btn.btn--primary', { type: 'button' },
    h('span.btn__icon', { 'aria-hidden': 'true' }, icon('refresh', { size: 'sm' })),
    h('span.btn__label', 'Try again'));
  track(on(again, 'click', retry));
  actions.appendChild(again);
  box.appendChild(actions);
  return box;
}

function applyServerMeta(raw) {
  S.env = raw && raw.env && typeof raw.env === 'object' ? raw.env : {};
  S.secrets = raw && raw.secrets && typeof raw.secrets === 'object' ? raw.secrets : null;
  S.defaults = raw && raw.defaults && typeof raw.defaults === 'object' ? raw.defaults : {};
  S.restart = raw && raw.restart ? raw.restart : { required: false, reasons: [], supported: false, unit: null };
  S.unapplied = raw && raw.unapplied ? raw.unapplied : { count: 0, keys: [] };
  S.configPath = raw && raw.config_path ? String(raw.config_path) : '';
  S.backupDir = raw && raw.backup_dir ? String(raw.backup_dir) : '';
}

function load(opts) {
  var o = opts || {};
  if (!o.quiet) {
    clear(S.contentEl);
    S.contentEl.appendChild(skeleton());
  }
  if (S.refreshAbort) { try { S.refreshAbort.abort(); } catch (e) {} S.refreshAbort = null; }
  var ctrl = null;
  if (typeof AbortController === 'function') {
    ctrl = new AbortController();
    S.refreshAbort = ctrl;
  }
  return api.config({ signal: ctrl ? ctrl.signal : undefined, timeout: 20000 }).then(function (raw) {
    if (!S || !S || S.destroyed) return;
    S.refreshAbort = null;
    var model = normalize(raw);
    if (o.quiet) {
      /* A background refresh must never overwrite work in progress: a
         dirty draft, a focused control, or a dialog whose callback still
         points at the controls it was opened from. */
      if (computeChanges().list.length) return;
      if (!S.panelEl) return;
      if (!o.force && (S.panelEl.contains(document.activeElement) || isOverlayOpen())) return;
      applyServerMeta(raw);
      S.baseline = model;
      S.draft = clone(model);
      if (S.section.indexOf('general.') !== 0 && !S.draft.cameras[S.section]) S.section = DEFAULT_SECTION;
      renderBanner();
      renderSection();
      return;
    }
    applyServerMeta(raw);
    S.baseline = model;
    S.draft = clone(model);
    if (!findGeneralSection(S.section) && !S.draft.cameras[S.section]) S.section = DEFAULT_SECTION;
    buildBody();
  }, function (err) {
    if (!S || S.destroyed || api.isAbort(err)) return;
    S.refreshAbort = null;
    if (o.quiet) {
      toast.danger('Could not refresh settings', { detail: api.describe(err) });
      return;
    }
    clear(S.contentEl);
    S.contentEl.appendChild(errorState(err, function () { load({}); }));
    toast.error('Settings could not be loaded.', { detail: api.describe(err), retry: function () { load({}); } });
  });
}

/* ==========================================================================
   NAVIGATION GUARD
   ========================================================================= */

/**
 * Ask before leaving with unsaved edits, then do what the operator chose.
 * `proceed()` is how this particular exit leaves once it may.
 * Returns false when there is something to lose (the caller must not leave).
 */
function confirmLeave(proceed) {
  var d = computeChanges();
  if (!d.list.length) return true;
  if (S.leaveDialog) return false;      /* one at a time: a double Back press */

  var n = d.list.length;
  S.leaveDialog = dialog({
    role: 'alertdialog',
    tone: 'danger',
    title: 'Discard ' + n + ' unsaved ' + plural(n, 'change') + '?',
    body: 'Leaving this screen throws away edits that were never written to config/cameras.yml.',
    stakes: describeChanges(d.list),
    actions: [
      { label: 'Stay here', variant: 'secondary', value: 'stay', focus: true },
      { label: 'Save and leave', variant: 'primary', value: 'save' },
      { label: 'Discard and leave', variant: 'danger', value: 'go' }
    ]
  });
  S.leaveDialog.result.then(function (v) {
    S.leaveDialog = null;
    if (!S || !S || S.destroyed) return;
    if (v === 'go') {
      S.baseline = clone(S.draft);   /* silence the guard, then navigate */
      refreshDirty();
      proceed();
    } else if (v === 'save') {
      /* "Save and leave" used to save and stay, so the click looked ignored.
         Leaving waits for the write: a rejected save keeps the edits here. */
      save().then(function (saved) {
        if (saved && S && !S.destroyed) proceed();
      });
    }
  });
  return false;
}

/* Anything that can run after unmount checks the session first: `unmount`
   sets S to null, so `S.destroyed` on its own throws. The settings page
   starts long requests (a 20 s config load, a 30 s save) and, since
   "Save and leave", deliberately navigates away while one is in flight. */
function installGuards() {
  track(on(window, 'beforeunload', function (ev) {
    if (!S || S.destroyed || !computeChanges().list.length) return;
    ev.preventDefault();
    ev.returnValue = '';
    return '';
  }));

  /* In-app links (tab bar, app bar, rail) are plain anchors that app.js
     intercepts. We run first, in the capture phase, and only when there is
     something to lose. */
  track(on(document, 'click', function (ev) {
    if (!S || S.destroyed || S.saving) return;
    if (ev.defaultPrevented || ev.button !== 0) return;
    if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
    var node = ev.target;
    while (node && node !== document && node.tagName !== 'A') node = node.parentElement;
    if (!node || node === document || !node.getAttribute) return;
    var href = node.getAttribute('href');
    if (!href || href.charAt(0) === '#') return;
    if (node.target && node.target !== '_self') return;
    if (node.host && node.host !== window.location.host) return;
    var here = window.location.pathname + window.location.search;
    if (href === here) return;
    if (!computeChanges().list.length) return;

    ev.preventDefault();
    ev.stopPropagation();
    confirmLeave(function () { router.navigate(href); });
  }, true));

  /* Back and Forward reach no anchor: without this the whole draft went away
     in silence on a Back press or a trackpad swipe. The router puts the URL
     back for us and leaves the asking to this. */
  track(router.guard(function (to) {
    if (!S || S.destroyed || S.saving) return true;
    return confirmLeave(function () { router.navigate(to); });
  }));

  track(on(document, 'visibilitychange', function () {
    if (!S || !S || S.destroyed) return;
    if (document.hidden) {
      if (S.refreshAbort) { S.refreshAbort.abort(); S.refreshAbort = null; }
      return;
    }
    if (!S.baseline || S.restarting) return;
    if (computeChanges().list.length) return;   /* never clobber pending edits */
    load({ quiet: true });
  }));
}

/* ==========================================================================
   THE VIEW
   ========================================================================= */

function sectionFromQuery(q) {
  if (!q) return null;
  var s = q.section ? String(q.section) : (q.camera ? String(q.camera) : '');
  if (!s) return null;
  if (s === 'global') return DEFAULT_SECTION;
  return s;
}

export var view = {
  mount: function (root, ctx) {
    S = newSession();
    S.root = root;
    S.destroyed = false;
    if (typeof AbortController === 'function') S.abort = new AbortController();
    else S.abort = { signal: undefined, abort: function () {} };

    var wanted = sectionFromQuery(ctx && ctx.query);
    if (wanted) S.section = wanted;

    S.selbar = buildSelbar();

    store.setChrome({
      title: 'Settings',
      subtitle: 'Detection, recording, storage, notifications and cameras',
      actions: [],
      toolbar: null,
      rail: null,
      norail: true,
      selbar: null,
      mods: []
    });

    root.appendChild(h('h1.visually-hidden', { tabIndex: -1, text: 'Settings' }));
    S.contentEl = h('div');
    root.appendChild(S.contentEl);

    installGuards();
    load({});
  },

  /* Same route, new query (our own setQuery, or Back/Forward): switch the
     section without reloading, so the draft survives. */
  update: function (ctx) {
    if (!S || !S.draft) return;
    var wanted = sectionFromQuery(ctx && ctx.query) || DEFAULT_SECTION;
    if (wanted === S.section) return;
    if (!findGeneralSection(wanted) && !S.draft.cameras[wanted]) return;
    selectSection(wanted, { silent: true });
  },

  unmount: function () {
    if (!S) return;
    S.destroyed = true;
    /* The dialog lives on <body>: leaving the route does not remove it. */
    if (S.leaveDialog) { try { S.leaveDialog.close(null); } catch (e0) {} S.leaveDialog = null; }
    if (S.abort && S.abort.abort) { try { S.abort.abort(); } catch (e) {} }
    if (S.refreshAbort) { try { S.refreshAbort.abort(); } catch (e2) {} }
    if (S.envTimer) clearTimeout(S.envTimer);
    for (var i = 0; i < S.offs.length; i++) {
      try { S.offs[i](); } catch (e3) {}
    }
    for (var t = 0; t < S.savedTimers.length; t++) clearTimeout(S.savedTimers[t]);
    S.offs = [];
    S.fields = [];
    S.fieldByKey = {};
    store.setChrome({ selbar: null });
    S = null;
  }
};
