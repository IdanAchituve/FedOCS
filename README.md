# Communication Efficient Distributed Learning over Wireless Channels
Vertically distributed learning exploits the local features collected by multiple learning workers to form a better global model. However, data exchange between the workers and the model aggregator for parameter training incurs a heavy communication burden, especially when the learning system is built upon capacity-constrained wireless networks. In this paper, we propose a novel hierarchical distributed learning framework, where each worker separately learns a low-dimensional embedding of their local observed data. Then, they perform communication-efficient distributed max-pooling to efficiently transmit the synthesized input to the aggregator. For data exchange over a shared wireless channel, we propose an opportunistic carrier sensing-based protocol to implement the max-pooling of the output of all the workers. Our simulation experiments show that the proposed learning framework is able to achieve almost the same model accuracy as the learning model using the concatenation of all the raw outputs from the learning workers while significantly reducing the communication load.

### Instructions
Install repo:
```bash
pip install -e .
```

To run the code first enter the required directory:
```bash
cd experiments/[dataset_name]
```
Where [dataset_name] is either cifar, cub, or mnist. Then run trainer.py with the default hyper-parameters.
