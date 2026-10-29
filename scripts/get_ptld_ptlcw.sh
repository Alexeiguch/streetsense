#!/usr/bin/env bash
# Fetch PTLD (PUCRS) and PTL-Crosswalk into /workspace/datasets/{ptld,ptlcw}. Never commit the images.
# Licence for both: UNCONFIRMED (see eval/SOURCES.md).
# Status 29 Oct 2026: both hosts require sign-in for anonymous clients, so this exits non-zero.
set -u
D=${DATASETS:-/workspace/datasets}; mkdir -p "$D/ptld" "$D/ptlcw"; rc=0
# PTLD: https://tinyurl.com/y9z7xodw -> legacy Drive folder (no resource key) -> Google sign-in.
python -m gdown "https://drive.google.com/drive/folders/0B2MY7T7S8OmJVVlCTW1jYWxqUVE" -O "$D/ptld" \
  || { echo "PTLD: Drive folder needs sign-in (401)"; rc=1; }
# PTL-Crosswalk: OneDrive share from the GitHub README; shares API returns 401, page 403.
U='https://1drv.ms/f/c/ac5426c77d3b6a62/ElwZ_7ElwzxDv8V8zd7daCQBfp91XE85Rd86yMl1ctgo-w?e=1sfrqy'
B=$(printf %s "$U" | base64 -w0 | tr '/+' '_-' | tr -d =)
code=$(curl -s -o "$D/ptlcw/listing.json" -w '%{http_code}' "https://api.onedrive.com/v1.0/shares/u!$B/root?expand=children")
[ "$code" = 200 ] && echo "PTL-Crosswalk listing saved; download children from listing.json" \
  || { echo "PTL-Crosswalk: OneDrive returned $code (auth required)"; rm -f "$D/ptlcw/listing.json"; rc=1; }
exit $rc
