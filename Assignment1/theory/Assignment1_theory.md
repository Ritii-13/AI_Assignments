# AI Assignment - Search Algorithms
## Ritika Thakur | 2022408

####  Question 1:
![alt text](<WhatsApp Image 2024-09-17 at 23.22.13_9f4fa843.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.14_9c6171de.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.14_7c576f29.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.14_6766d65b.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.15_6137d2ce.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.15_11983d86.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.15_657454e7.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.16_8a631896.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.16_206a6f41.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.17_837eb994.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.17_341d8a5a.jpg>)

#### Question 2:
![alt text](<WhatsApp Image 2024-09-17 at 23.22.17_e614715d.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.17_69dc03a8.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.18_eea16e8e.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.18_4f529176.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.18_5128c6f5.jpg>)
![alt text](<WhatsApp Image 2024-09-17 at 23.22.19_8a71320f.jpg>)

#### Question 3:
(a) Implemented in code_2022408.py

(b) We notice that the path obtained to travel from u to v for Iterative Deepening Search and Bidirectional Search is the same for all public test cases. However, it will not always be identical for all possible test cases. This is because during Iterative Deepening Search, the path is obtained by performing a depth-first search with a depth limit. The path obtained is the first path that is found by the algorithm. On the other hand, Bidirectional Search is a graph search algorithm that starts the search from both the source and the destination nodes. The algorithm stops when the two searches meet. The path obtained is the shortest path between the source and the destination nodes. The path obtained by the two algorithms will be the same when the first path found by Iterative Deepening Search is the shortest path between the source and the destination nodes. This will not always be the case, as the path obtained by Iterative Deepening Search may not be the shortest path between the source and the destination nodes. Hence, the path obtained by the two algorithms will not always be identical for all possible test cases.

(c) ![alt text](<WhatsApp Image 2024-09-17 at 04.14.34_4eb6d46c.jpg>)
The above figure shows the total time and space taken by Iterative Deepening Search and Bidirectional Search for the public test cases. We notice that the time and space taken by Iterative Deepening Search is greater than the time and space taken by Bidirectional Search for all public test cases. This is because Iterative Deepening Search is a depth-first search algorithm that performs a depth-first search with a depth limit. The algorithm continues to increase the depth limit until the goal node is found. This results in the algorithm exploring a large number of nodes, which increases the time and space complexity of the algorithm. On the other hand, Bidirectional Search is a graph search algorithm that starts the search from both the source and the destination nodes. The algorithm stops when the two searches meet. This results in the algorithm exploring a smaller number of nodes, which decreases the time and space complexity of the algorithm. Hence, the time and space taken by Iterative Deepening Search is greater than the time and space taken by Bidirectional Search for all public test cases.

Below is a scatter plot showing the same:
![alt text](<WhatsApp Image 2024-09-17 at 04.14.20_11f28c1e.jpg>)

(d) Implemented in code_2022408.py

(e) We notice that the path obtained to travel from u to v is not identical for all public test cases for A star Search and Bidirectional Heuristic Search. This is because A star Search is a graph search algorithm that uses a heuristic function to estimate the cost of reaching the goal node from the current node. The algorithm uses this heuristic function to guide the search towards the goal node. On the other hand, Bidirectional Heuristic Search is a graph search algorithm that starts the search from both the source and the destination nodes. The algorithm stops when the two searches meet. The path obtained is the shortest path between the source and the destination nodes. The path obtained by the two algorithms will be the same when the heuristic function used by A star Search is admissible and consistent. This will not always be the case, as the heuristic function used by A star Search may not be admissible and consistent. Hence, the path obtained by the two algorithms will not always be identical for all possible test cases.

![alt text](<WhatsApp Image 2024-09-17 at 03.47.35_0bb907db.jpg>)
The above figure shows the total time and space taken by A star Search and Bidirectional Heuristic Search for the public test cases. We notice that the time and space taken by A star Search is greater than the time and space taken by Bidirectional Heuristic Search for all public test cases. This is because A star Search is a graph search algorithm that uses a heuristic function to estimate the cost of reaching the goal node from the current node. The algorithm uses this heuristic function to guide the search towards the goal node. The time and space complexity of the algorithm depends on the heuristic function used. On the other hand, Bidirectional Heuristic Search is a graph search algorithm that starts the search from both the source and the destination nodes. The algorithm stops when the two searches meet. This results in the algorithm exploring a smaller number of nodes, which decreases the time and space complexity of the algorithm. Hence, the time and space taken by A star Search is greater than the time and space taken by Bidirectional Heuristic Search for all public test cases.

Below is a scatter plot showing the same:
![alt text](<WhatsApp Image 2024-09-17 at 03.47.39_a7370dfb.jpg>)

(f) Below is a scatter plot comparing all the abpve discussed uninformed and informed search algorithms:
![alt text](<WhatsApp Image 2024-09-17 at 03.40.54_ab4758ab.jpg>)

The total time and space taken by the above discussed uninformed and informed search algorithms are as follows:
![alt text](<WhatsApp Image 2024-09-17 at 03.40.50_43ff0a99.jpg>)

The uninformed search algorithms are Iterative Deepening Search and Bidirectional Search. The informed search algorithms are A star Search and Bidirectional Heuristic Search. \
IDS takes the largest amount of time making it least optimal out of all four algorithms.
Bidirectional Heuristic and Bidirectional Search take approximately the same amount of time, followed by A star Search which is not greater by a large margin but only a small one.\
A star Search takes the most amount of space with IDS not far behind making these te least efficient. Bidirectional Search takes the least amount of space making it the most efficient and Bidierectional Heuristic Search lies comfortably between IDS and Bidirectional Search.\
The metric to obtain the scatter plot is the total amount of time and space taken by all four algirthms for each pair of nodes. The x-axis represents the total amount of space taken by the algorithms and the y-axis represents the total amount of time taken by the algorithms. The scatter plot shows the total time and space taken by the algorithms for each pair of nodes.

#### Benefits of using informed search algorithms over uninformed search algorithms:
1. Informed search algorithms use a heuristic function to estimate the cost of reaching the goal node from the current node. This heuristic function guides the search towards the goal node, which helps in reducing the time and space complexity of the algorithm.
2. Informed search algorithms are more optimal than uninformed search algorithms as they use additional information about the problem domain to guide the search towards the goal node.

#### Limitations of using informed search algorithms over uninformed search algorithms:
1. Informed search algorithms may not always find the optimal path between the source and the destination nodes. This is because the heuristic function used by the algorithm may not be admissible and consistent.
2. Informed search algorithms may take more space than uninformed search algorithms. This is because the algorithm needs to store additional information about the problem domain to guide the search towards the goal node.

(g) Implemented in code_2022408.py