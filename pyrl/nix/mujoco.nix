{ stdenv
, fetchurl
, autoPatchelfHook
, unzip
, libGL
, xorg
, sources
}:
stdenv.mkDerivation {
  pname = "mujoco";
  version = "1.5";
  src = sources.mujoco;

  unpackCmd = "unzip $curSrc";
  nativeBuildInputs = [ unzip ];

  buildInputs = [
    autoPatchelfHook
    stdenv.cc.cc.lib
    libGL
    xorg.libX11
    xorg.libXinerama
    xorg.libXxf86vm
    xorg.libXcursor
    xorg.libXrandr
  ];
  installPhase = ''
    mkdir $out

    # copy required folders
    for folder in bin include model; do
      cp -r $folder $out/$folder
    done

    # make lib folder
    mkdir $out/lib
    ln -s $out/bin/*.so $out/lib/
  '';
  testPhase = ''
    cd sample
    make
  '';
}
