{ mjKeyPath
, cudaSupport ? false
, fetchFromGitHub
, mesa
, python3
, libGL
, gcc
, stdenv
, callPackage
, autoPatchelfHook
, xorg
, lib
, libglvnd
, imageio
, numpy
, cython
, buildPythonPackage
, sources
}:
let
  mujoco = callPackage ./mujoco.nix {};
  src = sources.mujoco-py;
in
buildPythonPackage {
  inherit src;
  pname = "mujoco-py";
  version = "1.50.1.1";
  requirements = builtins.readFile "${src}/requirements.txt";

  python = python3;
  MUJOCO_BUILD_GPU = cudaSupport;
  nativeBuildInputs = [
    autoPatchelfHook
  ];
  propagatedBuildInputs = [
    imageio
    numpy
    cython
    (callPackage ./glfw.nix {})
  ];
  buildInputs = [
    mesa
    mesa.osmesa
    mujoco
    python3
    libGL
    gcc
    stdenv.cc.cc.lib
  ] ++ lib.optionals cudaSupport [ xorg.libX11 libglvnd ];

  # hacks to make the package work
  postInstall = ''
    cat ${mjKeyPath} > $out/lib/${python3.libPrefix}/site-packages/mujoco_py/mjkey.txt
  '' + lib.optionalString cudaSupport ''
    patchelf --add-needed libEGL.so $out/lib/${python3.libPrefix}/site-packages/mujoco_py/cymj.cpython*.so
    patchelf --add-needed libOpenGL.so $out/lib/${python3.libPrefix}/site-packages/mujoco_py/cymj.cpython*.so
  '';

  doCheck = false;
}
