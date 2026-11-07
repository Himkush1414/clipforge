// Real browser run against a real API and real render jobs. See e2e/run.sh.
// Nothing leaves the machine: non-local requests are recorded and blocked.
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const WEB = 'http://localhost:3000';
const API = 'http://127.0.0.1:8765';
const SAMPLE = process.env.E2E_SAMPLE;
const SMALL = process.env.E2E_SMALL_VIDEO;
const SHOTS = process.env.E2E_SHOTS;
const SEED = process.env.E2E_SEED_SCRIPT;
const PYTHON = process.env.E2E_PYTHON;

let passed = 0;
let failed = 0;
function check(name, ok, detail = '') {
  if (ok) passed++; else failed++;
  console.log(ok ? 'PASS' : 'FAIL', name, detail);
}
const getJson = async (p) => (await fetch(API + p)).json();
const shot = (page, name) => page.screenshot({ path: path.join(SHOTS, `${name}.png`), fullPage: true });

async function pickRights(page) {
  await page.getByLabel('What rights do you have to this footage?').selectOption('own');
}

async function waitForJobs(count) {
  for (let i = 0; i < 300; i++) {
    const jobs = await getJson('/api/jobs');
    if (jobs.length >= count && jobs.every((j) => ['done', 'failed', 'cancelled'].includes(j.status))) return jobs;
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error('jobs did not finish');
}

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1360, height: 950 } });
  const external = [];
  await ctx.route('**/*', (route) => {
    const url = route.request().url();
    if (/^https?:\/\/(127\.0\.0\.1|localhost)/.test(url) || /^(blob|data):/.test(url)) return route.continue();
    external.push(url);
    return route.abort();
  });
  const page = await ctx.newPage();
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 200)); });
  page.on('pageerror', (e) => errors.push('pageerror ' + String(e).slice(0, 200)));

  // ---------- drag and drop, with a real File through the real drop handler ----------
  await page.goto(WEB + '/');
  await page.getByRole('heading', { name: 'New job' }).waitFor();
  const bytes = fs.readFileSync(SMALL).toString('base64');
  const transfer = await page.evaluateHandle((b64) => {
    const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const dt = new DataTransfer();
    dt.items.add(new File([raw], 'dropped small clip.mp4', { type: 'video/mp4' }));
    return dt;
  }, bytes);
  const zone = page.getByRole('button', { name: /Drop a video here/ });
  await zone.dispatchEvent('dragover', { dataTransfer: transfer });
  await page.waitForTimeout(100);
  check('drag over highlights the drop zone', /border-accent/.test(await zone.getAttribute('class')));
  await zone.dispatchEvent('drop', { dataTransfer: transfer });
  await page.getByText(/640×360/).waitFor({ timeout: 30000 });
  check('dropping a file uploads it and selects it', (await getJson('/api/sources')).some((s) => s.filename === 'dropped small clip.mp4'));
  await zone.dispatchEvent('dragover', { dataTransfer: transfer });
  await zone.dispatchEvent('dragleave', { dataTransfer: transfer });
  await page.waitForTimeout(100);
  check('leaving removes the highlight', !/border-accent/.test(await zone.getAttribute('class')));
  const bad = await page.evaluateHandle(() => {
    const dt = new DataTransfer();
    dt.items.add(new File(['not a video'], 'notes.txt', { type: 'text/plain' }));
    return dt;
  });
  await zone.dispatchEvent('drop', { dataTransfer: bad });
  await page.locator('main [role=alert]').first().waitFor({ timeout: 8000 });
  check('dropping a non-video shows a clear error', /Unsupported file type/.test(await page.locator('main [role=alert]').first().innerText()));

  // ---------- rights gate and the file picker ----------
  const start = page.getByRole('button', { name: 'Start job' });
  await page.locator('input[type=file]').setInputFiles(SAMPLE);
  await page.getByText(/1920×1080/).waitFor({ timeout: 120000 });
  check('file picker upload works', true);
  check('start blocked without rights', await start.isDisabled());
  const rights = page.getByLabel('What rights do you have to this footage?');
  await rights.selectOption('creative_commons');
  check('Creative Commons without credit blocked', await start.isDisabled());
  await page.getByLabel('Credit text').fill('Sample, CC BY 4.0');
  check('Creative Commons with credit allowed', !(await start.isDisabled()));
  await pickRights(page);

  // ---------- no channel ----------
  const channelSelect = page.getByLabel(/^Channel/);
  check('channel step offers "No channel: just make the clips"', await channelSelect.locator('option', { hasText: 'No channel: just make the clips' }).count() === 1);
  await page.getByRole('radio', { name: 'Manual: I choose' }).click();
  await page.getByLabel('Genre').selectOption('podcast');
  await page.getByLabel('Clip length (seconds)').fill('15');
  await page.getByLabel('Number of clips').fill('3');
  await start.click();
  const dialog = page.getByRole('alertdialog');
  await dialog.waitFor();
  check('warning says no channel is connected', await dialog.getByRole('heading', { name: 'No channel is connected.' }).isVisible());
  for (const name of ['Just make the clips', 'Make now, upload later']) check(`warning offers "${name}"`, await dialog.getByRole('button', { name }).isVisible());
  check('warning offers "Go to Channels"', await dialog.getByRole('link', { name: 'Go to Channels' }).isVisible());
  await shot(page, 'no-channel-warning');
  await page.keyboard.press('Escape');
  check('Escape closes it without starting a job', await dialog.count() === 0 && (await getJson('/api/jobs')).length === 0);
  await start.click();
  await dialog.getByRole('button', { name: 'Just make the clips' }).click();
  await page.waitForURL('**/jobs');
  await page.getByText('sample_workshop.mp4').first().waitFor();
  await page.waitForTimeout(2000);
  await shot(page, 'job-running');
  let jobs = await waitForJobs(1);
  check('job finished', jobs[0].status === 'done', jobs[0].error || '');
  check('rights saved with the job', jobs[0].rights_type === 'own');

  await page.goto(WEB + '/');
  const sampleOption = await page.getByLabel("Your videos").locator("option", { hasText: "sample_workshop" }).first().getAttribute("value");
  await page.getByLabel("Your videos").selectOption(sampleOption);
  await pickRights(page);
  await page.getByRole('radio', { name: 'Manual: I choose' }).click();
  await page.getByLabel('Genre').selectOption('podcast');
  await page.getByLabel('Clip length (seconds)').fill('10');
  await page.getByLabel('Number of clips').fill('2');
  await page.getByRole('button', { name: 'Start job' }).click();
  await page.getByRole('alertdialog').getByRole('button', { name: 'Make now, upload later' }).click();
  await page.waitForURL('**/jobs');
  jobs = await waitForJobs(2);
  check('second job finished', jobs.every((j) => j.status === 'done'), jobs.map((j) => j.error).join(' '));

  await page.goto(WEB + '/');
  await pickRights(page);
  await page.getByLabel('Your videos').selectOption({ index: 1 });
  await page.getByRole('button', { name: 'Start job' }).click();
  await page.getByRole('alertdialog').getByRole('link', { name: 'Go to Channels' }).click();
  await page.waitForURL('**/channels');
  check('Go to Channels goes there without starting a job', (await getJson('/api/jobs')).length === 2);

  // ---------- folders ----------
  await page.goto(WEB + '/clips');
  await page.waitForSelector('article');
  const tab = (name) => page.getByRole('tab', { name: new RegExp(`^${name}`) });
  check('Unassigned tab holds the "just make" clips', /\(3\)/.test(await tab('Unassigned').innerText()), await tab('Unassigned').innerText());
  check('Queued tab holds the "upload later" clips', /\(2\)/.test(await tab('Queued').innerText()), await tab('Queued').innerText());
  const first = await page.getByText(/Deleted in/).first().innerText();
  await page.waitForTimeout(2200);
  check('Unassigned countdown ticks', first !== await page.getByText(/Deleted in/).first().innerText(), first);
  await page.locator('article').first().getByRole('button', { name: /Quality:/ }).click();
  await shot(page, 'clips-unassigned');
  check('no upload control while no channel exists', await page.getByText('Upload to channel', { exact: true }).count() === 0);

  // ---------- why this clip ----------
  const reasonCount = await page.getByText('Why this clip:').count();
  const cardCount = await page.locator('article').count();
  check('every clip says why it was chosen', cardCount > 0 && reasonCount === cardCount, `${reasonCount} of ${cardCount}`);
  check('the reason is plain words', (await page.getByText('Why this clip:').first().locator('..').innerText()).length > 25);

  // ---------- the clip player ----------
  const player = page.locator('article video').first();
  await player.scrollIntoViewIfNeeded();
  const played = await player.evaluate(async (v) => {
    v.muted = true;
    v.preload = 'auto';
    v.load();
    await new Promise((res, rej) => {
      v.addEventListener('canplay', res, { once: true });
      v.addEventListener('error', () => rej(new Error('media error ' + (v.error && v.error.code))), { once: true });
      setTimeout(() => rej(new Error('timed out loading, readyState ' + v.readyState)), 20000);
    });
    await v.play();
    await new Promise((r) => setTimeout(r, 1500));
    const during = { time: v.currentTime, paused: v.paused, width: v.videoWidth, height: v.videoHeight, duration: v.duration };
    v.pause();
    v.currentTime = Math.min(5, v.duration / 2);
    await new Promise((res, rej) => {
      v.addEventListener('seeked', res, { once: true });
      setTimeout(() => rej(new Error('seek timed out')), 10000);
    });
    return { ...during, seekedTo: v.currentTime, error: v.error ? v.error.code : null };
  }).catch((e) => ({ failure: String(e) }));
  check('clip player loads and plays', !played.failure && played.time > 0.5 && played.paused === false, JSON.stringify(played));
  check('clip player shows the right size', played.width === 1080 && played.height === 1920, `${played.width}x${played.height}`);
  check('clip player seeks', !played.failure && Math.abs(played.seekedTo - Math.min(5, played.duration / 2)) < 0.2, String(played.seekedTo));

  // ---------- single move, multi-select, bulk delete ----------
  await page.locator('article').first().getByRole('button', { name: 'Move to Saved' }).click();
  await page.waitForFunction(() => /\(1\)/.test(document.querySelector('[role=tab]:nth-child(3)').innerText), null, { timeout: 8000 });
  check('Move to Saved moves one clip', /\(2\)/.test(await tab('Unassigned').innerText()) && /\(1\)/.test(await tab('Saved').innerText()));
  await tab('Queued').click();
  check('Queued clips show the timer as paused', await page.getByText('Timer paused').first().isVisible());
  await tab('Unassigned').click();
  await page.getByLabel('Select all clips in this folder').check();
  check('select all shows the count', await page.getByText('2 selected').isVisible());
  const bar = page.locator('div', { hasText: '2 selected' }).filter({ has: page.getByRole('button', { name: 'Delete', exact: true }) }).last();
  await bar.getByRole('button', { name: 'Move to Saved' }).click();
  await page.waitForFunction(() => /\(3\)/.test(document.querySelector('[role=tab]:nth-child(3)').innerText), null, { timeout: 8000 });
  check('multi-select moves many clips', /\(0\)/.test(await tab('Unassigned').innerText()) && /\(3\)/.test(await tab('Saved').innerText()));
  await tab('Saved').click();
  check('Saved clips say they are kept', await page.getByText('Kept until you delete it').first().isVisible());
  const savedBefore = (await getJson('/api/storage')).folders.saved;
  check('files really moved to the Saved folder on disk', savedBefore > 0, `${savedBefore} bytes`);
  await page.locator('article').first().getByRole('checkbox').check();
  await page.getByRole('button', { name: 'Delete', exact: true }).first().click();
  await page.getByRole('button', { name: 'Delete 1 for good' }).click();
  await page.waitForFunction(() => /\(2\)/.test(document.querySelector('[role=tab]:nth-child(3)').innerText), null, { timeout: 8000 });
  check('bulk delete needs a confirmation, then deletes', (await getJson('/api/storage')).folders.saved < savedBefore);
  await shot(page, 'clips-saved');

  // ---------- rights travel with moved clips ----------
  const all = await getJson('/api/clips');
  check('every kept clip still has its rights record', all.length === 4 && all.every((c) => c.rights_type === 'own'), `${all.length} clips`);

  // ---------- connected channel: banner ----------
  execFileSync(PYTHON, [SEED], { stdio: 'pipe' });
  await page.goto(WEB + '/clips');
  await page.getByText('2 clips are waiting to upload').waitFor({ timeout: 8000 });
  check('banner counts the waiting clips', true);
  await shot(page, 'queued-banner');
  await page.getByRole('button', { name: 'Upload now' }).click();
  await page.getByText(/sent to the upload queue/).waitFor({ timeout: 8000 });
  check('Upload now sends them to the queue', (await getJson('/api/uploads')).length === 2);
  for (let i = 0; i < 30; i++) {
    const uploads = await getJson('/api/uploads');
    if (uploads.every((u) => u.status === 'failed')) break;
    await page.waitForTimeout(1000);
  }
  const queuedAfter = (await getJson('/api/clips')).filter((c) => c.folder === 'queued');
  check('a failed upload (network blocked) keeps clips Queued with no timer', queuedAfter.length === 2 && queuedAfter.every((c) => c.seconds_left === null));
  await page.goto(WEB + '/clips');
  await tab('Queued').click();
  check('per-clip upload control appears once a channel exists', await page.getByText(/^Upload to channel/).first().isVisible());

  // ---------- settings and disk guard ----------
  await page.goto(WEB + '/settings');
  await page.getByLabel('Folder path').waitFor();
  await page.getByLabel('Folder path').fill('/etc/nope');
  await page.getByRole('button', { name: 'Save settings' }).click();
  await page.getByText('That is a system folder.').waitFor({ timeout: 5000 });
  check('an unsafe Saved path is refused with a clear message', true);
  const custom = path.join(process.env.CLIPFORGE_WORK_DIR, 'custom-saved');
  await page.getByLabel('Folder path').fill(custom);
  await page.getByLabel('Warn when Saved passes (GB)').fill('0.000001');
  await page.getByRole('button', { name: 'Save settings' }).click();
  await page.getByText(`Now using: ${custom}`).waitFor({ timeout: 5000 });
  check('Saved path can be changed', true);
  await page.reload();
  await page.getByText(/past your .* GB limit/).waitFor({ timeout: 8000 });
  check('disk banner warns when a limit is passed', true);
  await shot(page, 'settings-warning');

  // ---------- presets and channels still work ----------
  await page.goto(WEB + '/presets');
  await page.waitForSelector('[aria-label="Live caption preview"]');
  await page.getByRole('button', { name: 'Real render: frame' }).click();
  await page.waitForSelector('img[alt="Exact libass render of this preset"]', { timeout: 30000 });
  check('real caption render works', await page.locator('img[alt="Exact libass render of this preset"]').evaluate((i) => i.complete && i.naturalWidth > 0));
  await page.goto(WEB + '/channels');
  await page.getByText('Google (YouTube)').waitFor({ timeout: 8000 });
  check('channels page shows the platform setups', await page.getByText('Meta (Facebook and Instagram)').isVisible());

  const phone = await browser.newContext({ viewport: { width: 390, height: 800 } });
  await phone.route('**/*', (r) => (/^https?:\/\/(127\.0\.0\.1|localhost)|^(blob|data):/.test(r.request().url()) ? r.continue() : r.abort()));
  const m = await phone.newPage();
  for (const route of ['/', '/jobs', '/clips', '/presets', '/channels', '/settings']) {
    await m.goto(WEB + route);
    await m.waitForTimeout(600);
    const over = await m.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    check(`phone width has no sideways scroll on ${route}`, over <= 0, `${over}px`);
  }

  check('nothing tried to leave this machine', external.length === 0, external.join(' ').slice(0, 200));
  // Expected: the two refused actions in this run answer with HTTP errors, which browsers log.
  const unexpected = [...new Set(errors)].filter((e) => !/status of (4\d\d|507)/.test(e));
  check('no unexpected console or page errors', unexpected.length === 0, unexpected.join(' | ').slice(0, 500));
  await browser.close();
  console.log(`\n${passed}/${passed + failed} passed`);
  process.exit(failed ? 1 : 0);
})().catch((e) => {
  console.log('E2E CRASHED', String(e).slice(0, 900));
  console.log(`\n${passed}/${passed + failed} passed before the crash`);
  process.exit(2);
});
