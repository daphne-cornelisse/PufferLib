OSRS_ASSET_GROUP="${ENV#osrs_}"
if [ -z "$OSRS_ASSET_GROUP" ] || [ "$OSRS_ASSET_GROUP" = "$ENV" ]; then
    echo "Error: OSRS web hook requires an osrs_* environment" >&2
    exit 1
fi

OSRS_PRELOAD_LINES=$(python3 ocean/osrs/scripts/osrs_asset_manifest.py \
    emcc-preload-args ocean/osrs/asset_manifest.json \
    --group core \
    --group "$OSRS_ASSET_GROUP" \
    --group combat_visuals \
    --group gui \
    --group items)

WEB_SRC=("$SRC_FILE")
WEB_SHELL=tools/web/osrs_shell.html
WEB_USE_DEFAULT_PRELOAD=0
WEB_PRELOAD=()
WEB_DEFINES=()
WEB_EXTRA=(-sLZ4=1)
WEB_PUBLISH=0
WEB_REQUIRE_OBS=0

while IFS= read -r line; do
    [ -z "$line" ] && continue
    if [[ "$line" != "--preload-file "* ]]; then
        echo "Error: invalid OSRS preload argument: $line" >&2
        exit 1
    fi
    WEB_PRELOAD+=(--preload-file "${line#--preload-file }")
done <<< "$OSRS_PRELOAD_LINES"

if [ "${#WEB_PRELOAD[@]}" -eq 0 ]; then
    echo "Error: OSRS asset manifest produced no web preload files" >&2
    exit 1
fi
