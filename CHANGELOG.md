# Changelog

## [1.6.4](https://github.com/chodeus/lama-sidecar/compare/v1.6.3...v1.6.4) (2026-10-27)


### Bug Fixes

* **ci:** label main builds from the event, not a detached checkout ([#80](https://github.com/chodeus/lama-sidecar/issues/80)) ([4a4457b](https://github.com/chodeus/lama-sidecar/commit/4a4457bf1a28e6dbdf730cd07fdbf08cefc33511))
* report the release version from the API; rebuild the image on rapidocr bumps ([#79](https://github.com/chodeus/lama-sidecar/issues/79)) ([3e5b862](https://github.com/chodeus/lama-sidecar/commit/3e5b86207c5133de4e1b37db1d13dfe1d97a2e0c))


### Documentation

* say why the Real-ESRGAN model pin isn't tracked ([#82](https://github.com/chodeus/lama-sidecar/issues/82)) ([6564249](https://github.com/chodeus/lama-sidecar/commit/656424993e7b2f9982c516a0f6dea6925b0ee8e9))

## [1.6.3](https://github.com/chodeus/lama-sidecar/compare/v1.6.2...v1.6.3) (2026-10-26)


### Bug Fixes

* **deps:** pin filelock 4.2.4 over torch's index copy (CVE-2026-112940) ([#74](https://github.com/chodeus/lama-sidecar/issues/74)) ([0f87757](https://github.com/chodeus/lama-sidecar/commit/0f877573753d35c9adc81c4e84a36e922ad06984))

## [1.6.2](https://github.com/chodeus/lama-sidecar/compare/v1.6.1...v1.6.2) (2026-10-21)


### Bug Fixes

* **deps:** upgrade pip so its bundled certifi drops the eMudhra roots ([7e10f6f](https://github.com/chodeus/lama-sidecar/commit/7e10f6fe9b090e96587f7edc1d7496f8327186c8))

## [1.6.1](https://github.com/chodeus/lama-sidecar/compare/v1.6.0...v1.6.1) (2026-09-04)


### Bug Fixes

* apply Debian security updates in the image build ([3490f29](https://github.com/chodeus/lama-sidecar/commit/3490f29db1f7421fb5f69199a294cd8b7e82e590))

## [1.6.0](https://github.com/chodeus/lama-sidecar/compare/v1.5.1...v1.6.0) (2026-07-22)


### Features

* mask snap, native boundary refine, and run-count bail fix ([169f0c6](https://github.com/chodeus/lama-sidecar/commit/169f0c65c51c465119a7190f856949b89ba0a865))

## [1.5.1](https://github.com/chodeus/lama-sidecar/compare/v1.5.0...v1.5.1) (2026-07-07)


### Bug Fixes

* keep passing healthcheck probes out of the access log ([a61b61f](https://github.com/chodeus/lama-sidecar/commit/a61b61f496cc9d4fe0fa14406a75b12bd707e497))

## [1.5.0](https://github.com/chodeus/lama-sidecar/compare/v1.4.1...v1.5.0) (2026-07-04)


### Features

* hole-based removal quality, text-detection + logo-upscale endpoints, hardening ([aab5f5e](https://github.com/chodeus/lama-sidecar/commit/aab5f5e6b796464a42b6ead541aa81503fe7ebba))

## [1.4.1](https://github.com/chodeus/lama-sidecar/compare/v1.4.0...v1.4.1) (2026-07-03)


### Chores

* release 1.4.1 ([e6e1eff](https://github.com/chodeus/lama-sidecar/commit/e6e1eff00a306bd292ad8f115580fd76b418b3d1))

## [1.4.0](https://github.com/chodeus/lama-sidecar/compare/v1.3.0...v1.4.0) (2026-06-30)


### Features

* dilate+feather mask to remove logo fringe ghosting ([6292c6a](https://github.com/chodeus/lama-sidecar/commit/6292c6a4ec9e698b6b8f81c93b36262306178590))
