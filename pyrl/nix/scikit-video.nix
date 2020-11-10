{ buildPythonPackage
, fetchFromGitHub
, numpy
, scipy
, scikitlearn
, pillow
, ffmpeg
, nose
, sources
}:
buildPythonPackage rec {
  pname = "scikit-video";
  version = "1.1.11";
  src = sources.scikit-video;
  postPatch = ''
    substituteInPlace skvideo/__init__.py \
      --replace '_FFMPEG_PATH = which("ffprobe")' '_FFMPEG_PATH = "${ffmpeg}/bin"'
  '';
  propagatedBuildInputs = [
    # python inputs
    numpy
    scipy
    scikitlearn
    pillow

    # non-python inputs
    ffmpeg
  ];
  checkInputs = [
    nose
  ];
}
