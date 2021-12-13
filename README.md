# sentspace


<!-- ABOUT THE PROJECT -->
## About The Project

`sentspace`
aims to characterize language stimuli using a collection of metrics and comparisons.
Imagine you generated a set of sentences based on some optimization algorithm or extracted 
them from an ANN language model. How do these 'artificially' generated sentences compare to 
naturally occurring sentences?

In the present form of `sentspace`, 
a user can feed `sentspace` sentences and obtain sentence feature values
along many metrics in a single interface.

## Documentation 
[![CircleCI](https://circleci.com/gh/aalok-sathe/sentspace/tree/main.svg?style=svg)](https://circleci.com/gh/aalok-sathe/sentspace/tree/main)

For more information, [visit the docs!](https://aalok-sathe.github.io/sentspace/index.html)

<!-- request read access to the [project doc](https://docs.google.com/document/d/1O1M7T5Ji6KKRvDfI7KQXe_LJ7l9O6_OZA7TEaVP4f8E/edit#). -->



## Usage

Example: get only lexical and syntax features for stimuli from a csv containing columns for 'sentence' and 'index'.
```bash
python3 -m sentspace -lex 1 -syn 1 -emb 0 -sem 0 in/wsj_stimuli.csv
```

Example: get embedding features in a custom script
```python
import sentspace

s = sentspace.Sentence.Sentence('The person purchased two mugs at the price of one.')
emb_feat = sentspace.embedding.get_features(s)
```

Example: parallelize getting features for multiple sentences using multithreading
```python
import sentspace

sentences = [
    'Hello, how may I help you today?',
    'The person purchased three mugs at the price of five!',
    "She's leaving home today.",
    'This is an example sentence we want features of.'
             ]
             
sentences = [*map(sentspace.Sentence.Sentence, sentences)]
lex_feat = sentspace.utils.parallelize(sentspace.lexical.get_features, sentences,
                                       wrap_tqdm=True, desc='Lexical features pipeline')
```


## Installing

The recommended way to run this project with all its dependencies is using a prebuilt Docker image, `aloxatel/sentspace:latest`.
However, you are welcome to manually install all the dependencies locally too, which would be useful to be able to import the 
package into your custom script and extract sentence features there.




### Container-based usage

To use the image as a container using `singularity/docker`:

#### **first, some important housekeeping stuff**
- make sure you have `singularity`/`docker`, or load/install it otherwise
  - `which singularity`   or  `which docker` 
- make sure you have set the ennvironment variables that specify where `singularity/docker` will cache its images. if you don't do this, `singularity` will make assumptions and you may end up with a full disk and an unresponsive server, if running on a server with filesystem restrictions. you should have about 5GB free space at the target location.

#### **next, running the container** (automatically built and deployed to Docker hub)
[![CircleCI](https://circleci.com/gh/aalok-sathe/sentspace/tree/circle-ci.svg?style=svg)](https://circleci.com/gh/aalok-sathe/sentspace/tree/circle-ci)

- `singularity shell docker://aloxatel/sentspace:latest` (or alternatively, from the root of the repo, `bash singularity-shell.sh`). this step can take a while when you run it for the first time as it needs to download the image from docker hub and convert it to singularity image format (`.sif`). however, each subsequent run will execute rapidly. alternatively, use [corresponding commands for Docker](https://docs.docker.com/engine/reference/commandline/exec/).
- now you are inside the container and ready to run Sentspace!

For a complete list of options and default values, see the `help` page like so:
```bash
python3 -m sentspace -h
```

### Manual dependency install (not officially supported; needs elevated privileges)
```bash
# optional (but recommended): 
# create a virtual environment using your favorite method (venv, conda, ...) 
# before any of the following

# install basic packages using apt (you likely already have these)
sudo apt update
sudo apt install python3.8 python3.8-dev python3-pip
sudo apt install python2.7 python2.7-dev 
sudo apt install build-essential git

# install ICU
DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" sudo apt install python3-icu

# install ZS package separately (pypi install fails)
python3.8 -m pip install -U pip cython
git clone https://github.com/njsmith/zs
cd zs && git checkout v0.10.0 && pip install .

# install rest of the requirements using pip
cd .. # make sure you're in the sentspace/ directory
pip install -r ./requirements.txt
polyglot download morph2.en
```



## Submodules

In general, each submodule implements a major class of features. You can run each module on its own by specifying its flag with the module call:
```bash
python -m sentspace -lex 1 <input_file_path>
```

#### `lexical`
Obtain lexical (word-level) features that are not dependendent on the sentence context. 
These features are returned on a word-by-word level and also averaged at the sentence level to provide each sentence a corresponding value.
- typical age of acquisition
- n-gram surprisal `n={1,2,3,4}`
- etc. (comprehensive list will be updated)

#### `syntax`
Description pending

#### `embedding`
Obtain high dimensional representations of sentences using word-embedding and contextualized encoder models.
- glove
- Huggingface model hub (`gpt2-xl`, `bert-base-uncased`)

#### `semantic`
Multi-word features computed using partial or full sentence context.
- PMI (pointwise mutual information)
- Language model-based perplexity/surprisal
*Not Implemented yet*




## Contributing

Any contributions you make are **greatly appreciated**, and no contribution is *too small* to contribute.

1. Fork the project using Github [(how to fork)](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
2. Create your feature/patch branch (`git checkout -b feature/AmazingFeature`)
3. Make some changes! Implement your patch/feature. Test to make sure it works!
4. Commit your bhanges (`git commit -m 'Add some AmazingFeature'`)
5. Push the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request (PR) and we will take a look!

## License & Contact
- `gretatu % mit ^ edu`
- `asathe % mit ^ edu`

(C) 2020-2021 EvLab, MIT BCS
