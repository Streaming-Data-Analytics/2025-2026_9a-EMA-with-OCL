# 2025/2026: Reproducing "Improving Online Continual Learning Performance and Stability with Temporal Ensembles"

Optional project of the [Streaming Data Analytics](https://emanueledellavalle.org/teaching/streaming-data-analytics-2025-26/) course provided by Politecnico di Milano.

Student: **[To be assigned]**

_____
# Brief Description
This project focuses on reproducing the core experiments and results from the [2023 paper](https://arxiv.org/pdf/2306.16817) *"Improving Online Continual Learning Performance and Stability with Temporal Ensembles"* by Soutif-Cormerais et al.

Online Continual Learning (OCL) models suffer from catastrophic forgetting and a phenomenon known as the **stability gap**, which is a temporary but severe performance drop on previously learned tasks when the model starts learning a new task. To combat this, the paper investigates **temporal ensembling**, specifically using the Exponential Moving Average (EMA) of model weights. 

The original research demonstrates significant performance and stability gains using convolutional architectures on standard image datasets. Your objective is to replicate these findings, implementing the EMA weight update mechanism and benchmarking it against standard OCL baselines on **Split-CIFAR100** and **Split-MiniImageNet**.

______

# Background
In OCL, data arrives in small mini-batches and is processed in a single pass. When a model transitions to a new task, the gradient updates aggressively push the weights toward the new task's optimum. This not only causes long-term forgetting but also triggers the stability gap: the representations learned for previous tasks are immediately disrupted before the model can consolidate them.

To mitigate this, the authors propose a **Weight Exponential Moving Average (EMA)** approach. Instead of relying solely on the active "online" weights $\theta_{online}$ updated by backpropagation, the system maintains a "target" model with weights $\theta_{EMA}$:
$$\theta_{EMA}^{(t)} = \lambda \cdot \theta_{EMA}^{(t-1)} + (1 - \lambda) \cdot \theta_{online}^{(t)}$$
where $\lambda \in [0, 1)$ is the momentum parameter. 

Using the EMA weights for evaluation provides a smoother, more stable representation over time. It effectively acts as an ensemble of models from different training steps, reducing the task-recency bias and bridging the stability gap without requiring extra memory for larger replay buffers.

# Project Goals
1. **Implement Weight EMA:** Integrate the target model and the Exponential Moving Average logic into a standard OCL training loop.
2. **Setup the Baselines:** Implement standard baselines like Naive fine-tuning and Experience Replay (ER) using convolutional architectures (e.g., ResNet-18).
3. **Reproduce the Experiments:** Run the experiments on the **Split-CIFAR100** and **Split-MiniImageNet** benchmarks.
4. **Result Verification:** Compare your results against the metrics reported in the original paper. Specifically, analyze if the EMA approach successfully mitigates the stability gap.

# Evaluation Metrics
You must evaluate the models using standard CL metrics:
* **Average Accuracy ($A_n$)**: To measure the overall predictive performance.
* **Forgetting**: To quantify the mitigation of catastrophic forgetting.
* **Stability Gap Analysis**: Plot the accuracy on the first task continuously while training on subsequent tasks to visually inspect the stability gap.

# Deliverables
1. **Codebase:** A reproducible PyTorch/Avalanche repository containing the EMA implementation and the training pipelines.
2. **Jupyter Notebook / Report:** A presentation of your findings. Plot the accuracy curves, report the final metrics in a table, and critically analyze whether the EMA approach improved performance and stability as claimed by the authors.

## Note for Students
* Clone the created repository offline;
* Add your name and surname into the Readme file;
* Add a `requirements.txt` file for code reproducibility;
* Commit your changes to your local repository and push them online.
