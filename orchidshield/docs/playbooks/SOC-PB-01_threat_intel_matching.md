# SOC-PB-01 Threat-intel indicator matching
Match alert indicators (sha256 hash, ip, domain, url) against the curated intel feed.
Only indicators with confidence of at least 0.6 that have not expired count as known bad.
Expired or low-confidence indicators are context only and must not trigger containment.
Record the feed name and confidence as provenance for every match.
