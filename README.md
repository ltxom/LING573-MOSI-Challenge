# ATM: Alchemist Transformers-based Multi-modal Sentiment Analysis
## Deliverable 2 Note for Gina & Haotian
There are some problems of the condor on Patas. It probably needs to run without the condor_submit command. Here's how we run our project (D2) on Patas.

```
chmod u+x run_d2_without_condor.sh
./run_d2_without_condor.sh
```
This script may take a while to create the virtual environment (especially when installing the tensorflow).

The code itself should be fast to run since we serialized all time-consuming stuffs into files. Results of two model will be saved into <code>results/d2.txt</code> once it is completed, and four visualizations will be saved under the <code>outputs/</code> directory.