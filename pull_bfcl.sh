REPO_URL="https://github.com/ShishirPatil/gorilla.git"
SUBDIR="berkeley-function-call-leaderboard"
DEST="bfcl"

TMP=".tmp_bfcl"
rm -rf "$TMP" && mkdir -p "$TMP"
git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" "$TMP"
cd "$TMP"
git sparse-checkout set "$SUBDIR"

# Record the exact commit;
BFCL_COMMIT="$(git rev-parse HEAD)"
echo "BFCL source commit: $BFCL_COMMIT"

# Copy only the Python lib;
cd ..
mkdir -p "$DEST"
rsync -a --delete "$TMP/$SUBDIR/" "$DEST/"
# Stamp the commit you vendored from;
printf "%s\n" "$BFCL_COMMIT" > "$DEST/VENDORED_FROM_BFCL_COMMIT.txt"

# Clean up;
rm -rf "$TMP"
