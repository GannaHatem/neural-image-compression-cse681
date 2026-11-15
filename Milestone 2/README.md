# Milestone 2 package

Start with M2_Progress_Report.pdf. This is a team-review draft, not a claim that the report has been approved or submitted.

report.tex follows the supplied course section structure and can be uploaded with the figures to Overleaf. Its PDF layout will differ from the supplied readable PDF, which was generated separately because this environment's TeX installation lacks LaTeX formats. Merge citations into your existing M1 references.bib and confirm individual contributions before submission. No individual completion credits have been invented.

## Evidence
- 24 Kodak original-resolution images; reference pixels agree across methods.
- 216 neural records, 216 JPEG records, 144 HEVC records.
- RGB PSNR and actual file bytes/bpp validated against saved reconstructions and streams for all records.
- One neural stream additionally spot-decoded in NumPy; maximum difference from supplied PNG was 1 intensity level.
- Mean rates are not exact per-image matched rates. Qualitative grids show nearest sampled rates with actual rates labeled.
- Keep both original uploaded result archives; this compact package contains consolidated CSVs and figures, not all original streams/checkpoints.

## Reproduce locally in VS Code
Dependencies: torch, numpy, Pillow, matplotlib; FFmpeg with libx265. Use your recorded environment for close reproducibility. See config.json in the original result archive for exact training versions. FFmpeg version output is supplied here.

    python m2_codec.py --dataset "G:\Postgrad\Advanced Digital Image and Video Processing\Data Set" --widths 32 64 128 --bits 6 8 10 --steps 5000 --out results_m2_full
    python hevc_m2.py --dataset "G:\Postgrad\Advanced Digital Image and Video Processing\Data Set" --out results_hevc_m2

Independent decoding without the original image/checkpoint:

    python m2_codec.py --decode results_m2_full/kodim01/width64_8bit.inr --png standalone_decoded.png

To regenerate audit CSVs and figures, put results_m2_full and results_hevc_m2 beside these scripts and run:

    python audit_image_results.py
    python audit_hevc_results.py

These two scripts regenerate into review_images and regenerated_review respectively, so existing evidence is preserved. They require the complete original result folders. Neural fitting intentionally uses the evaluated image: this is an overfit codec, not supervised generalization.

## Before submission
Review and understand the technical draft. Confirm the individual contribution table with the team. Merge the existing M1 bibliography, retaining citation style. Include an accurate AI-assistance disclosure as required by the course. Upload code/results to your repository when ready; this package has not been pushed or submitted.
