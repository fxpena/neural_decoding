# Neural decoding
Here are a few tools for decoding neural activity of a pseudo-simultaneous population, which is the collection of all neurons recorded across sessions and/or animals. The main tool is the function decode() which takes in numpy arrays. Check out the demo file testCases.ipynb to see a couple examples of how to use it.

## For Matlab data
I collected and preprocessed neural data in Matlab so I wrote a function prepDecoder() which will read the .mat files in a given folder. It returns a list of numpy arrays which can be given as input to decode()

## How to use
Download or clone the repository. You can call any of the functions in the pseudosimul file from a Jupyter notebook or script. At the top of the file write the following code.

```
from pseudosimul * import
```

### Dependencies
- imblearn
- numpy
- sklearn
- scipy
