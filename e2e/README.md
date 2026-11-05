# End-to-end browser run

`e2e/run.sh` starts the real API and web app in a throwaway folder, runs real jobs, and drives the UI with Chromium. It blocks all network access, so no platform is contacted, and it cannot touch your clips.

## Setup

Install Playwright outside the repo:

```bash
mkdir -p ~/clipforge-work/qa && cd ~/clipforge-work/qa
npm init -y && npm i playwright
PLAYWRIGHT_BROWSERS_PATH=$PWD/browsers npx playwright install chromium-headless-shell
```

If Chromium is missing system libraries and you have no sudo, unpack them from `.deb` files into a folder and write that folder's path to `~/clipforge-work/qa/libpath.txt`.

## Run

```bash
e2e/run.sh
```

It exits non-zero if any check fails. The sample video comes from `scripts/make_sample.py`.

## Limits

Drag and drop sends a real file through the page's real drop handler using a synthetic event, because a headless browser has no operating system drag.
