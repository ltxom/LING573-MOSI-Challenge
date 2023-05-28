# Sentimetalchemy (STM): A Multi-modal Approach to Sentiment Analysis
## Deliverable 4 Note for Gina & Haotian
There are still some problems of the condor on Patas. For some reasons, it cannot activate the virtual environment, so it probably needs to run without the condor_submit command. Here's how we run our project (D4) on Patas.

```
chmod u+x run_d4_without_condor.sh
./run_d4_without_condor.sh
```
This script may take a while to create the virtual environment (especially when installing the tensorflow).

The code itself should be fast to run since we serialized all time-consuming stuffs into files. Results of two model will be saved into <code>results/D4</code> once it is completed, and outputs will be saved under the <code>outputs/D4/</code> directory.
