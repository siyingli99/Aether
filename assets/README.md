# Asset staging

`assets/` records expected asset layouts; it is not a public download bundle.

## Local-only archive

The development checkout may contain:

```text
assets/local/pandaset095_processed.zip   # approximately 643 MB
```

This archive is an internal local copy of the full processed PandaSet scene 095. It is required only for reconstruction, Phase-B LiDAR training, and related corpus generation. The minimum RGB/scene-editing/Mode-A runtime does not require it.

Extract it safely with:

```bash
python scripts/extract_pandaset095.py
```

The destination defaults to `data/pandaset/processed/`. The script rejects absolute paths, parent traversal, symbolic links, and other unsupported archive entries. It does not download the archive.

## Target layout

`manifest.yaml` lists the expected path, size, and checksum of each optional
asset.
