# Slicer Script Repository

This directory contains the bundled Slicer script repository used by SlicerKimiAgent
to provide context-aware code examples to the AI assistant.

## Contents

### script_repository/

15 markdown files containing official Slicer Python code examples:

| File | Topic |
|------|-------|
| `volumes.md` | Volume loading, saving, rendering, numpy arrays |
| `segmentations.md` | Segmentation, Segment Editor, effects |
| `markups.md` | Fiducials, curves, lines, planes, ROIs |
| `models.md` | Surface models, meshes, STL/OBJ |
| `transforms.md` | Linear/grid transforms, coordinate systems |
| `registration.md` | Image registration (BRAINSFit, etc.) |
| `dicom.md` | DICOM import/export, PACS integration |
| `gui.md` | UI, layouts, views, interaction |
| `plots.md` | Charts, plotting, matplotlib |
| `screencapture.md` | Screenshots, videos, animation |
| `sequences.md` | 4D volume sequences, time series |
| `subjecthierarchy.md` | Data organization, folders |
| `tractography.md` | DTI fiber tracking |
| `batch.md` | Batch processing, machine learning |
| `webserver.md` | Web server, REST API |

## Source

These files are originally from the 3D Slicer documentation:
`https://github.com/Slicer/Slicer/tree/main/Docs/developer_guide/script_repository/`

They are bundled with SlicerKimiAgent to ensure offline availability and version consistency.

## Usage

The `SkillContextManager` class automatically reads from these files based on the user's query.
For example, if the user asks about "volume rendering", it will extract relevant examples from
`volumes.md` and include them in the AI context.

## Updates

To update these files from the upstream Slicer repository:
1. Copy new versions from slicer-source/Docs/developer_guide/script_repository/
2. Replace the files in this directory
3. Restart Slicer
