# sentspace


<!-- ABOUT THE PROJECT -->
## About The Project

`sentspace`
aims to characterize language stimuli using a collection of metrics and comparisons.
Imagine you generated a set of sentences based on some optimization algorithm or extracted 
them from an ANN language model. How do these 'artificially' generated sentences compare to 
naturally occurring sentences?

In the present form of `sentspace`, 
the goal is for a user to feed the toolbox sentences and for the toolbox to return sentence feature values
along the many metrics it implements (and will implementat in the future).

## Documentation 
[![CircleCI](https://circleci.com/gh/aalok-sathe/sentspace/tree/main.svg?style=svg)](https://circleci.com/gh/aalok-sathe/sentspace/tree/main)

For more information, [visit the docs!](https://aalok-sathe.github.io/sentspace/index.html)

<!-- request read access to the [project doc](https://docs.google.com/document/d/1O1M7T5Ji6KKRvDfI7KQXe_LJ7l9O6_OZA7TEaVP4f8E/edit#). -->



## Usage

The recommended way to run this project with all its dependencies is using a prebuilt Docker image, `aloxatel/ubuntu:sent-space`.
To use the image as a container using `singularity`, do:

#### **first, some important housekeeping stuff**
- `which singularity` (make sure you have singularity, or load/install it otherwise)
- make sure you have set the ennvironment variables that specify where `singularity` will cache its images. if you don't do this, `singularity` will make assumptions and you may end up with a full disk and an unresponsive server. you need about 6gb of free space at the target location.

#### **next, running the container** (automatically built and deployed to Docker hub)
[![CircleCI](https://circleci.com/gh/aalok-sathe/sentspace/tree/circle-ci.svg?style=svg)](https://circleci.com/gh/aalok-sathe/sentspace/tree/circle-ci)

- `singularity shell docker://aloxatel/ubuntu:sent-space` (or alternatively, from the root of the repo, `bash singularity-shell.sh`). this step can take a while when you run it for the first time as it needs to download the image from docker hub and convert it to singularity image format (`.sif`). however, each subsequent run will execute rapidly.
- [now you are within the container] `source .singularitybashrc`, again from the root of the repo, to activate the environment variables and so on.
- now you are ready to run the module!

For a complete list of options and default values, see the `help` page like so:
```bash
python3 -m sentspace -h
```

### Submodules

In general, each submodule exists as a standalone implementation. You can run each module on its own by specifying it like so:
`python -m sentspace.syntax -h`, which will print out the usage for that submodule.
Below, we provide more information and the capabilities/usage of each submodule in some greater depth.

#### `syntax`

#### `lexical`
Not yet implemented

#### `embedding`
Not yet implemented

#### `semantic`
Not yet implemented

<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**, and no contribution is *too small* to contribute.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

MIT License.



<!-- CONTACT -->
## Contact

- About the project: 
  - Greta Tuckute, EvLab, MIT BCS
- For help:
  - coming soon
