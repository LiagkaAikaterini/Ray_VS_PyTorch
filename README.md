# Ray_VS_PyTorch
Performance comparison between Ray and PyTorch, across different Machine Learning and Big Data related operations. It is developed as a project for the NTUA course Analysis and Design of Information Systems.

## Our Team Members:
Full Name | NTUA id number (ΑΜ) | Github
| :---: | :---: | :---:
Λιάγκα Αικατερίνη  | el17208 | [Katerina Liagka](https://github.com/LiagkaAikaterini)
Λιάγκας Νικόλαος  | el19221 | [Nikos Liagkas](https://github.com/NikosLiagkas)
Τζέλλας Απόστολος-Στέφανος | el19878 | [Apostolos Stefanos Tzellas](https://github.com/tzellas)

## Project Overview
Our research delved into an in-depth comparative analysis of Ray and PyTorch, highlighting each frameworks’ strengths and weaknesses, providing insight into the optimal use cases for each framework in real-world scenarios.

We conducted 3 different experiments, which were implemented for both Ray and PyTorch, in a distributed environment:
* **K-means clustering:** We tested the frameworks on popular machine learning tasks, like clustering, using social network datasets of varying sizes to assess their time performance and clustering quality.
* **PageRank Algorithm:** We implemented the PageRank algorithm on different graph sizes to measure execution time and retrieve the 10 most significant nodes in each graph.
* **X-Ray Image Classification:** We performed more complex deep learning tasks, such as classifing real chest X-ray images to detect pneumonia. We compared the frameworks in terms of training accuracy, loss convergence, and handling custom datasets.

Our team executed the scripts with various number of nodes and utilized different data types and sizes, to test the frameworks' scalability. 

### Report
Our [report](documents/report.pdf), which is located in the repository's documents folder, provides detailed instructions for installation and setup, listing all the necessary dependencies. It also contains details on the infrastracture, datasets and algorithms we utilized, as well as a thourough analysis of the experiments' results.

### Source Code
Each framework has a dedicated folder in our repository, named Ray and PyTorch, which contain subfolders named after the respective experiments (kmeans, pagerank, pneumonia_classification). Each of these subfolders include the corresponding python script, as well as a res folder with the txt result files from our executions of the experiments.

#### Direct Navigation to the Scripts
| **PyTorch Scripts** | **Ray Scripts**  |
| :---: | :---: |
[k-means Clustering](PyTorch/kmeans/kmeans.py)                 | [k-means Clustering](Ray/kmeans/kmeans.py)                     | 
[PageRank](PyTorch/pagerank/pagerank.py)            | [PageRank](Ray/pagerank/pagerank.py)                | 
[X-Ray Image Classification](PyTorch/pneumonia_classification/pneumonia_classification.py) | [X-Ray Image Classification](Ray/pneumonia_classification/pneumonia_classification.py) |

### Datasets
The datasets are not included in our repository due to their large size. However, they can be downloaded from the official websites, which are also referenced in the "Datasets" section of our [report](documents/report.pdf) :
* [Friendster social network](https://snap.stanford.edu/data/com-Friendster.html) from SNAP
* [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) from Kaggle Datasets
  
In our repository's data folder the resize.py python script can be located, which was leveraged to create smaller data files from the Friendster Social Network Dataset.
