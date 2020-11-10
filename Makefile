JOBS=4

.PHONY: all default cachix docker

default:
	nix build -f default.nix pkg_cpu pkg_gpu --max-jobs ${JOBS}

cachix:
	make default
	nix path-info -f default.nix pkg_cpu pkg_gpu | cachix push pyrl -j ${JOBS}

docker:
	nix build -f default.nix docker_cpu --max-jobs ${JOBS}
	docker load < $$(nix path-info -f default.nix docker_cpu)
	docker push siddharthverma/adversarial

all:
	make cachix
	make docker
