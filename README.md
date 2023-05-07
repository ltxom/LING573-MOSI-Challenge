# ATM: Alchemist Transformers-based Multi-modal Sentiment Analysis
## Deliverable 3 Note for Gina & Haotian
There are still some problems of the condor on Patas. For some reasons, it cannot activate the virtual environment, so it probably needs to run without the condor_submit command. Here's how we run our project (D3) on Patas.

```
chmod u+x run_d3_without_condor.sh
./run_d3_without_condor.sh
```
This script may take a while to create the virtual environment (especially when installing the tensorflow).

The code itself should be fast to run since we serialized all time-consuming stuffs into files. Results of two model will be saved into <code>results/d3.txt</code> once it is completed, and visualizations will be saved under the <code>outputs/d3/</code> directory.