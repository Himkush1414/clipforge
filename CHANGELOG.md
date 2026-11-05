# Changelog

## Unreleased

### Added

- Works with no channel: a "No channel" option, a warning with three choices when nothing is connected, and three clip folders (Unassigned, Queued, Saved)
- Per-clip and multi-select actions: Move to Saved, Move to Queued, Upload to channel, Download, Delete
- A "clips are waiting to upload" banner once a channel is connected
- Right-to-left captions for Arabic, Persian, Urdu and Hebrew
- A Settings page for the Saved path and disk limits, plus disk warnings and a refusal to start jobs when space is low
- A repeatable real-browser test, `e2e/run.sh`, which covers the clip player and drag and drop
- Engine core: FFmpeg wrapper, media probe, hardware detection by real test encodes, SQLite models, restart-safe job queue, 24-hour clip cleanup
- Word-level transcription with GPU and CPU fallback
- Frame-exact scene cut and motion analysis, audio energy and silence analysis
- Moment scoring by genre, with cut boundaries that never touch a spoken word
- Word-by-word caption engine with seven presets, safe margins and a real libass preview
- Export pipeline: exact-frame cuts, reframing, joins, two-pass loudness normalization, NVENC with CPU fallback, quality checks
- Local titles and descriptions from the transcript
- Local API with required source rights and origin checks
- Web UI: new job, jobs, clips with countdown, preset editor, channels
- Auto mode that explains each decision
- Official API uploads for YouTube, Facebook and Instagram, with a hard stop on copyright claims

### Changed

- The automatic cleanup can now delete only inside the Unassigned folder, using real-path checks, and logs every deletion and refusal
- Renders write to the temp folder first, so clip folders never hold partial files

### Fixed

- Clips with many cuts no longer run out of memory on long videos: they now render one segment at a time
- Restarted jobs resume without duplicate clips
- The worker thread survives errors, so queued jobs cannot stall
- API errors keep their CORS headers, so the UI shows the real message
- Silent static videos now produce clips
- Sentences no longer run together in descriptions when Whisper drops punctuation
