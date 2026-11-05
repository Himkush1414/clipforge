# Platform uploads

clipforge uploads only through official APIs. These rules were read from each platform's documentation on 2026-11-05. They change often, so check the linked pages before relying on them.

> **Status: no live upload has been made yet.** The code follows the documentation and passes tests against simulated platforms. Your first real upload is the real test, so start with a private YouTube upload.

## What you must set up yourself

clipforge cannot do these for you. They need your accounts and, in some cases, a review by the platform.

### YouTube

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project and enable **YouTube Data API v3**.
2. Configure the OAuth consent screen and add your Google account as a test user.
3. Create an OAuth client of type **Web application**, with this redirect address: `http://127.0.0.1:8765/api/oauth/callback`
4. On the **Channels** page, enter the client ID and secret, add a YouTube channel, and press **Connect**.

Scopes requested: `youtube.upload` and `youtube.readonly`.

### Facebook Pages

1. Create an app at [Meta for Developers](https://developers.facebook.com/) using Facebook Login for Business.
2. Add the same redirect address as a valid OAuth redirect. Meta may restrict plain `http` redirects outside development mode, which has not been tested.
3. Enter the app ID and secret on the **Channels** page, add a Facebook channel, and press **Connect**.

Permissions requested: `pages_show_list`, `pages_read_engagement`, `pages_manage_posts`.

### Instagram

Instagram needs a **professional account linked to a Facebook Page**, and uses the same Meta app.

Permissions requested: `instagram_basic`, `instagram_content_publish`, `pages_show_list`, `pages_read_engagement`.

Meta's docs list Standard or Advanced Access. If people without a role on your app will use it, expect App Review. For your own accounts, a role on the app should be enough, but this has not been tested.

## Limits

| Platform | Limit | Source |
|---|---|---|
| YouTube | 100 `videos.insert` calls a day by default, in their own quota bucket. More needs a compliance audit. | [Quota costs](https://developers.google.com/youtube/v3/determine_quota_cost), [Audits](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits) |
| YouTube | 256 GB file limit | [videos.insert](https://developers.google.com/youtube/v3/docs/videos/insert) |
| Facebook | 30 API-published reels per 24 hours, per Page | [Reels Publishing](https://developers.facebook.com/documentation/video-api/guides/reels-publishing) |
| Facebook | 3 to 90 seconds, at least 540x960, 24 to 60 fps, 9:16. Checked before sending. | Same page |
| Instagram | 100 API-published posts per 24 hours. A carousel section of the same page says 50, and the page does not reconcile them. | [Content Publishing](https://developers.facebook.com/documentation/instagram-platform/content-publishing) |
| Instagram | A container must be published within 24 hours | Same page |

Older guides say uploads from unverified YouTube projects are forced to private. Google's current `videos.insert` page says they are not restricted. clipforge defaults to **Private** either way, and you choose Unlisted or Public for each upload.

## How claims are detected

| Platform | What the API reports | What clipforge does |
|---|---|---|
| Facebook | `copyright_check_information`, with matches, owner, action and segments | Holds the reel unpublished until the check is clean. On a match it stops and shows the details. |
| YouTube | `status.rejectionReason` of `copyright`, `claim` or `trademark` | Stops and points you to YouTube Studio for details |
| YouTube | A Content ID claim that leaves the video live is not documented as readable by non-partners | **Not detected.** Check YouTube Studio and use Report a claim. |
| Instagram | No copyright signal is documented | **Not detected.** Check the app and use Report a claim. |

## What happens on a claim

1. Details are logged and shown on the clip.
2. Waiting uploads from the same source video are held.
3. New uploads from that video are refused until you choose an option.
4. You choose one:
   - **Trim:** recut by exact frames to remove the claimed part.
   - **Replace the audio:** upload a track you hold a licence for. This makes a new clip, and the licence note goes in its description.
   - **Dispute:** only if you hold a licence. You file it with the platform. See [YouTube's guide](https://support.google.com/youtube/answer/2797454), or follow the appeal option in Meta's notice.
5. The claimed clip itself can never be uploaded again. Nothing retries, re-edits or re-uploads by itself.

## Privacy of what you send

Titles, descriptions and video files go to a platform only when you press **Upload**. Instagram and Facebook reels are public once published. YouTube uploads stay at the privacy you choose.

Your 24-hour clip cleanup removes the local file only. It does not delete anything already on a platform.
