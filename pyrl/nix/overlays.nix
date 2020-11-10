{ cudaSupport ? true
, python ? "python38"
}:
[
  # add sources
  (self: super: {
    sources = import ./sources.nix;
  })

  # top-level pkgs overlays
  (self: super: {
    # batteries included :)
    ffmpeg = super.ffmpeg.override {
      nonfreeLicensing = true;
      nvenc = cudaSupport; # nvidia support
    };

    gitignore = (super.callPackage super.sources.gitignore {}).gitignoreSource;
  })

  # python pkgs overlays
  (self: super: rec {
    pythonOverrides = python-self: python-super: rec {
      blas = super.blas.override { blasProvider = super.mkl; };
      lapack = super.lapack.override { lapackProvider = super.mkl; };

      pytorch = python-super.pytorch.override {
        inherit cudaSupport;
        tensorflow-tensorboard = python-super.tensorflow-tensorboard_2;
      };

      opencv3 = python-super.opencv3.override {
        enableCuda = cudaSupport;
        enableFfmpeg = true;
      };

      opencv4 = python-super.opencv4.override {
        enableCuda = cudaSupport;
        enableFfmpeg = true;
      };

      mujoco-py = python-super.callPackage ./mujoco_py.nix {
        inherit cudaSupport;
        mesa = super.mesa;
        mjKeyPath = ~/secrets/mjkey.txt;
      };

      cpprb = python-super.callPackage ./cpprb.nix {};

      cloudpickle = python-super.callPackage ./cloudpickle.nix {};

      gym = python-super.gym.overrideAttrs (old: {
        postPatch = ''
          substituteInPlace setup.py \
            --replace "pyglet>=1.2.0,<=1.3.2" "pyglet" \
        '';
      });

      # self-made packages
      glfw = python-super.callPackage ./glfw.nix {};
      flatten-dict = python-super.callPackage ./flatten-dict.nix {};
      scikit-video = python-super.callPackage ./scikit-video.nix {};
    };

    "${python}" = super."${python}".override { packageOverrides = pythonOverrides; };
    python3 = self."${python}";
  })
]
