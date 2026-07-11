# Browse Page Design Conclusion

## Reference Behavior

Flight Review browse behavior, database structure, and GPS overview image
storage are documented in `docs/flight_review_plot_generation.md`.

Use that behavior as reference material, not as an implementation template.

## Browse Row Layout

Use these columns:

- Upload Date
- Log Date
- Vehicle/Airframe
- Hardware
- Software
- Duration
- Error Count
- Flight Modes
- Tags

Remove only the `Overview` column.

Combine Flight Review's `Type` and `Airframe` columns into one `Vehicle/Airframe` column.

## Vehicle/Airframe Display

The visible browse-page value should be the airframe image only.

The index should still store:

- Vehicle type
- Airframe name
- Airframe group
- Airframe id, such as `SYS_AUTOSTART`

These stored values are for search, filtering, image resolution, and detail views.

## Airframe Images

Use the attributed QGroundControl airframe SVG assets bundled under
`web/airframes`.

Resolve the image from generated PX4 airframe metadata, not from an application
lookup table. Match the log's `SYS_AUTOSTART` value to an `<airframe>` ID, then
use the containing `<airframe_group image="...">` value as the image key. For
example, airframe `13000` is in the `Standard VTOL` group with image key
`VTOLPlane`.

Metadata sources are merged in this order:

1. Bundled QGroundControl `AirframeFactMetaData.xml` as the standalone fallback.
2. `<flight_review_storage_path>/cache/airframes.xml`, when configured, as the
   preferred source.

Fallback behavior:

- Exact airframe image when the metadata key and corresponding SVG both exist
- Generic/unknown airframe image if no exact image is available

## Image Storage

Use Flight Review's file-backed image storage pattern for our airframe images,
but with shared assets instead of per-log generated images.

Our airframe image behavior should be:

- Store airframe images as files, not database blobs.
- Store only an `airframe_image_key` or filename in the browse DB.
- Serve airframe images through a static route such as `/airframe_img/<image>.svg`.
- Use a configured asset/cache directory, for example `<browse_asset_root>/airframes`.
- Fall back to an unknown/generic airframe image when no exact image key resolves.

The key difference is that Flight Review's GPS overview image is per-log, while
our airframe image is shared across many logs.

## Error Count

Add error count to the browse/index data.

Warning count is not part of this design conclusion.

## Flight Modes

Keep a Flight Modes column similar to Flight Review.

## Tags

Tags are user-managed.

The UI should support:

- Creating tags
- Adding existing tags to a log
- Removing tags from a log
- Filtering by selected tags

Tag filtering is must-include only. If multiple tags are selected, a log must contain all selected tags.

The exact tag display/editing area is left for UI design later.

## Search

Do not copy Flight Review's implementation directly.

Use Flight Review as the behavior reference:

- Browse page has a search bar
- Search query can be preserved in the URL
- Search combines with pagination/sorting/filtering
- Search applies to browse-index metadata

Implement it in our own app style:

- Use our local browse index/API
- Search stored metadata such as hardware, software, vehicle type, airframe fields, flight modes, log id/path, and other browse-index text
- Combine the global search with tag filters and date filters
- A log must satisfy all active filter groups

## Date Filters

Upload date range and log date range are separate filters.

If both are set, they are combined:

- Upload date must be within the selected upload-date range
- Log date must be within the selected log-date range

Either range can be unset.

## Data Model Direction

Add a local browse DB/index for browse support.

The browse DB should be our own reconstructed database, not the Flight Review
database. If an existing Flight Review-format database is available, treat it as
an optional read-only import source. Do not mutate the Flight Review database.

The browse page should query only our browse DB/API during normal operation.
Import/rebuild code can read the Flight Review database and log directory, then
populate our browse DB with normalized rows for our UI, search model, and
user-managed tags.

The index should store:

- Upload date
- Log date
- Log path/id
- Vehicle type
- Airframe name/group/id
- Airframe image key
- Hardware
- Software
- Duration
- Error count
- Flight modes
- User-added tags

## Flight Review Import Configuration

Support importing existing Flight Review-format log data through configurable
paths:

- `flight_review_storage_path`
- `flight_review_db_path`
- `flight_review_log_dir`
- `browse_db_path`

`flight_review_storage_path` should be the primary convenience setting when the
external data follows Flight Review's default layout, because Flight Review
derives both `logs.sqlite` and `log_files/` from the storage path.

Path resolution should work as follows:

- If `flight_review_db_path` is set, use it.
- Else if `flight_review_storage_path` is set, use `<flight_review_storage_path>/logs.sqlite`.
- If `flight_review_log_dir` is set, use it.
- Else if `flight_review_storage_path` is set, use `<flight_review_storage_path>/log_files`.
- Store our reconstructed browse database at `browse_db_path`.

The importer should preserve the existing Flight Review database and log files
intact. Rebuild/resync should only write to our browse DB.
