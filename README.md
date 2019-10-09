# PredicrionInQueueNets
Do deep prediction in Queue Networks

The queue nets dataset is composed by peices of event logs such as: 
[agentId queueId arivalTime beginTime departTime],
which cannot be simply utilized by the existed ML techs for the input of ML is vector-like data and each feature of the vector should be computable.

Therefor, translating the event logs into some ML applicable dataset is the key issue.
In this project, we apply the methods developed in queue theory, graph neural network and time series prediction methods (LSTM, GRU, Transformers) to automatically translate the event logs to vector-like features and do prediction based on the features.
