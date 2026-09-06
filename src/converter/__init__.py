"""Nz-Videomni weight-conversion toolbox.

Started life as a Sulphur-2 -> GGUF (Q4_K_M) converter and now also hosts the
PrunaVAED decoder conversion (``convert-vae``). See README.md.
"""

#: Recorded in converted files' provenance metadata, so any artefact can be
#: traced back to the exact tool that produced it. Bump on behaviour changes.
__version__ = "1.2.0"
