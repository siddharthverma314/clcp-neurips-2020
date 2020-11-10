{ cudaSupport ? true
, overlays ? []
, python ? "python38"
}:
import (import ./sources.nix).nixpkgs {
  config.allowUnfree = true;
  config.cudaSupport = cudaSupport;
  overlays = (import ./overlays.nix { inherit cudaSupport python; }) ++ overlays;
}
