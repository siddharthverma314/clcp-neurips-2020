let
  pkgs = import ./pyrl/nix/nixpkgs.nix {python = "python37";};
  pyrl = import ./pyrl/default.nix {inherit pkgs;};
  adversarial = import ./default.nix {inherit pkgs;};
in
pkgs.mkShell {
  buildInputs = adversarial.pkg_gpu.propagatedBuildInputs ++ [
    pyrl.dev
    pkgs.google-cloud-sdk
  ];
}
