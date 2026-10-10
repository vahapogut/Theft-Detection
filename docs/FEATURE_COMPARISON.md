# Related Projects and Feature Comparison

This comparison was prepared on October 10, 2026, by reviewing the official GitHub sources listed below. Features were implemented independently; no source code was copied from these projects. JPEG snapshots, a single region of interest (ROI), and CSV downloads already existed. The table describes the functionality added to those foundations.

| Area | Source reviewed | Previous behavior | Added functionality |
| --- | --- | --- | --- |
| Event review | [Frigate review](https://github.com/blakeblackshear/frigate/blob/dev/docs/docs/configuration/review.md), [ZoneMinder tags](https://github.com/ZoneMinder/zoneminder/blob/master/docs/userguide/tags.rst) | No review status or notes | New, reviewed, and false alarm statuses; reviewer identity, notes, and archive protection |
| Search and export | [ZoneMinder filters](https://github.com/ZoneMinder/zoneminder/blob/master/docs/userguide/filterevents.rst) | Browser filtering covered only the latest 100 records; date and pagination controls were inactive | Server filtering by text, date, camera, type, status, and archive state; working pagination; safe CSV export of all matching records |
| Camera recovery | [Viseron camera management](https://github.com/roflcoopter/viseron/blob/dev/viseron/components/ffmpeg/camera.py), [ZoneMinder network settings](https://github.com/ZoneMinder/zoneminder/blob/master/docs/userguide/options/options_network.rst) | No retry after an initial connection failure | Automatic reconnection with increasing, capped retry delays; manual reconnection; last frame time, FPS, and connection status |
| Zones and false alarm controls | [Frigate zones](https://github.com/blakeblackshear/frigate/blob/dev/docs/docs/configuration/zones.md) | One ROI and a fixed dwell time; saving an ROI could deadlock | Multiple named zones; separate dwell times; confidence thresholds, consecutive frame requirements, and alert cooldowns; camera-specific rule switches; deadlock fix |
| Privacy | [motionEye interface](https://github.com/motioneye-project/motioneye/blob/dev/motioneye/templates/main.html), [Frigate masks](https://github.com/blakeblackshear/frigate/blob/dev/docs/docs/configuration/masks.md) | No masks or image attachment preferences for streams and notifications | Black privacy masks applied before analysis, streaming, and recording; optional image attachments for email and Telegram notifications |
| Video evidence | [Frigate recording settings](https://github.com/blakeblackshear/frigate/blob/dev/docs/docs/configuration/record.md), [ZoneMinder event video](https://github.com/ZoneMinder/zoneminder/blob/master/docs/userguide/viewevents.rst) | JPEG snapshots only | H.264 MP4 clips covering footage before and after an alert, with a bounded memory buffer; background encoding and recording capability checks |
| Retention management | [Viseron storage settings](https://github.com/roflcoopter/viseron/blob/dev/viseron/components/storage/config.py) | Files and records accumulated indefinitely | Preview cleanup by age; expiring, single-use confirmation tied to the user; protection for archived evidence and clips still being prepared |
| User permissions | [ZoneMinder users](https://github.com/ZoneMinder/zoneminder/blob/master/docs/userguide/options/options_users.rst) | API, live streams, and evidence required no login | Administrator, operator, and viewer roles; password hashing; expiring sessions; authorization for the API, WebSocket streams, and evidence |
| Actual statistics | Complements the event review workflow | The hourly chart was random; false alarms were estimated as 15% of the total | Actual hourly and weekly database counts, review statuses, and measured CPU/RAM usage |

## Design Differences and Limitations

- Privacy masks completely black out the selected pixels. They serve a different purpose from Frigate's object filter masks.
- Retention cleanup requires an administrator to preview and explicitly confirm deletion. Continuous recording, storage tiers, and size quotas are outside the scope of this release.
- Roles apply across the application; individual camera permissions per user are not yet supported.
- Clip recording requires FFmpeg's H.264 encoder. If it is unavailable, the interface shows this status and JPEG evidence remains available.
- Mask changes apply to new images; previously saved evidence is not modified retroactively.
- Detections record suspicious activity for human review, rather than establishing that a crime occurred. Configure rules and masks for each site.
- A local video source goes offline at the end of the file. Use reconnection to play it again.
- If a camera driver ignores its read timeout, the system stops displaying stale frames. The capture thread cannot be forcibly stopped while its native driver call remains blocked.

## Sources and Licenses

Frigate and Viseron use the MIT license; ZoneMinder uses GPL v2; motionEye uses GPL v3. These links document the feature research and do not identify contributors to this project. Shinobi was excluded from the current feature comparison because its official GitHub README states that the project moved to GitLab.
