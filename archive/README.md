# Archived references

`legacy_decoder.py` preserves the previous Thai decoder for explicitly selected
`thai-legacy` image fixtures. The active decoder loads it only for that language;
camera Thai reading uses `thai_decoder.py`.

These are historical manual checks and the previous README, retained for reference.
They use the old synthetic Thai mapping, may play audio or open windows, and are
not part of automated test discovery. Use `tests/` and the current root README.
