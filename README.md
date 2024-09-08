# Ray_VS_PyTorch
Performance comparison between Ray and PyTorch, across different Machine Learning and Big Data related operations. It is developed as a project for the NTUA course Analysis and Design of Information Systems.

During our research we conducted 3 different experiments, across multiple cluster sizes, in a distributed environment: 
* **K-means clustering:** We tested the frameworks on popular machine learning tasks, like clustering, using social network datasets of varying sizes to assess their time performance and clustering quality.
* **PageRank Algorithm:** We implemented the PageRank algorithm on different graph sizes to measure execution time and retrieve the 10 most significant nodes in each graph.
* **X-Ray Image Classification:** We performed more complex deep learning tasks, such as classifing real chest X-ray images to detect pneumonia. We compared the frameworks in terms of training accuracy, loss convergence, and handling custom datasets.



## Our Team Members:
Full Name | NTUA id number (ΑΜ) | Github
| :---: | :---: | :---:
Λιάγκα Αικατερίνη  | el17208 | [Katerina Liagka](https://github.com/LiagkaAikaterini)
Λιάγκας Νικόλαος  | el19221 | [Nikos Liagkas](https://github.com/NikosLiagkas)
Τζέλλας Απόστολος-Στέφανος | el19878 | [Apostolos Stefanos Tzellas](https://github.com/tzellas)

## Navigation to Scripts
| **Experiment** | **PyTorch Script** | **Ray Script**  | **Results Directory** |
|:---: | :---: | :---: | :---: 
| **K-Means Clustering**          | [k-means Script](PyTorch/kmeans/kmeans.py)                 | [Ray K-Means Script](Ray/kmeans/kmeans.py)                     | `/PyTorch/kmeans/res/`, `/Ray/kmeans/res/`     |
| **PageRank Algorithm**          | [PageRank Script](PyTorch/pagerank/pagerank.py)            | [Ray PageRank Script](Ray/pagerank/pagerank.py)                | `/PyTorch/pagerank/res/`, `/Ray/pagerank/res/` |
| **X-Ray Image Classification**  | [Classification Script](PyTorch/pneumonia_classification/pneumonia_classification.py) | [Classification Script](Ray/pneumonia_classification/pneumonia_classification.py) | `/PyTorch/pneumonia_classification/res/`, `/Ray/pneumonia_classification/res/` |
