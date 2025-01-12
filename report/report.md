# Fine-tuning CARTE for lung cancer treatment classification

Ayoub Kachkach

## Goal

`/lung_cancer_data.csv` is a ~12k-row dataset of various health and lifestyle indicators for patients at different stages of lung cancer. The learning task is to predict the right treatment (chemo, surgery, ...) based on the patient's indicators.

The goal of this report is to come up with improvements to **CARTE 0.0.24** that would:

- +10% in classification metrics (measured by AUC).
- -20% in memory usage.
- -20% in runtime.

on this particular task. `./task.pdf` contains the full task description.

## Setting a baseline

> **⚠️ Note**: training on GPU was broken initially, but then fixed in https://github.com/soda-inria/carte/commit/adfef5881ad0f454376b5e8d93ca52308a22e2bd.

A first run of CARTE with default hyperparameters on the dataset gives a AUROC of ~0.5 -- equivalent to a random classifier. The model is clearly not learning anything useful. Attempts at getting baselines with simpler models like XGBoost, sweeping over 100 different configurations, again always gives an AUROC of ~0.5 as well. The same stays true even after doing some basic preprocessing like removing the 'Patient_ID' column, or fixing rows in which the "Smoking_Pack_Years" column has non-zero values where "Smoking History" is set to "Never Smoked".

### Taking a closer look at the data

(see `/data_analysis.ipynb` for the all the plots)

Plotting the distribution of numerical values, we can see that they are almost perfectly uniformly distributed:
![alt text](artifacts/dist1.png)
![alt text](artifacts/dist2.png)

The pair-wise correlation matrix shows that there is absolutely no correlation between any of the features (even for things where there should be, like 'Age' and 'Survival_Months')
![alt text](artifacts/correlation.png)

My hunch is that this dataset is composed of completely independently random values (probably synthetic), and that there are no patterns to learn from it. The +10% AUC goal is therefore impossible to reach on this dataset, and I will redirect my efforts on the other goals.

## Reducing runtime+memory usage

### Inference

#### Current state

Running inference using CARTE the first time locally put my computer (16GB RAM) to its knees. I ran a profiler to get a sense of how much memory was being used:
![alt text](artifacts/ram_plot_before.png)

Running [memray](https://github.com/bloomberg/memray) to get a more detailed view shows that most of the memory was used by the 7.5GB FastText model that is fully loaded in memory:
![alt text](artifacts/flamegraph.png).
We can see the interactive flamegraph in `./flamegraph.html`.

The FastText model is loaded when the rows->graph preprocessor is fit as a class attribute ([here](https://github.com/soda-inria/carte/blob/adfef5881ad0f454376b5e8d93ca52308a22e2bd/carte_ai/src/carte_table_to_graph.py#L223)), and is kept in memory for the whole duration of the inference process.

#### Solution

##### **Alternative 1: Lazy loading**

One approach would be to free the FastText model from memory after the preprocessor is fit, and reload it on-demand when we need to do inference. This would reduce the overall memory footprint but would still (1) keep the same peak memory usage (requiring us to still serve this on a high-RAM machine), and (2) tradeoff memory with speed as loading the whole model from disk takes ~40s.

**Conclusion:** this doesn't solve the problem.

##### **Alternative 2: Preprocess table once and cache embeddings**

We could fit our table->graph preprocessor once with the in-memory FastText model but free it immediately and instead cache all embeddings we calculated (for all col names + categorical values in our dataset). We could then use these cached embeddings at inference time instead of loading the whole model.

This however limits CARTE to only processing in-vocab categorical values (those seen and cached while training the preprocessor). Example: we won't be able to handle a high-variance column like a free-text "Doctor's Notes" since it is impossible to preprocess and cache. The ability of handling out-of-vocab values is one of the core stated missions of CARTE and we shouldn't compromise on it.

**Conclusion:** this doesn't solve the problem.

##### **Solution: export FastText to disk in fast-read format**

CARTE only does lookups of very specific words in the FastText model. It doesn't do any expensive operations like knn search which would warrant loading the whole vocab in memory. We can therefore afford to re-export the FastText model to disk in a <k,v>-friendly format, and only load the embeddings we need on-demand.

We went with [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) as it seemed to offer the fastest reads + lowest overhead in storage. We also wrote a new `FastText` class that can read from this store on-demand instead of loading everything in memory.

Conclusion: With this, we reduce the memory footprint of FastText at inference time from 7.5GB to ~0GB and achieve the following:

- 63% reduction in peak memory usage => we can use smaller/cheaper machines for serving.
- 90% reduction in total memory usage => less costs if we pay per resources used.
- 3x speedup preprocessing training data (see figure) and comparable latency processing random rows at inference (see profiler.py).

![alt text](artifacts/inference_ram_disk_vs_mem.png)

![alt text](artifacts/peak_memory.png)

![alt text](artifacts/total_memory.png)
