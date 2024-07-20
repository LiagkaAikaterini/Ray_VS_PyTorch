# We used the friendster undirected graph from the Standford Unniversity website SNAP
# The url where you can download the full graph (30 GB) com-friendster.ungraph.txt is 
# https://snap.stanford.edu/data/com-Friendster.html

def reduce_file_size(input, data, target_size_gb):
    input_file = open(input)
    data_file = open(data, 'w')
    size = 0
    target_size = target_size_gb * (1024**3)

    for line in input_file:
        if line.startswith('#'):
            continue  # skip comments
        
        csv_line = ','.join(line.strip().split()) + '\n'
        data_file.write(csv_line)
        
        size += len(line.encode('utf-8'))
        if size >= target_size:
            break
        
    input_file.close()
    data_file.close()
    
# Execute with desired files
input_file = 'data/com-friendster.ungraph.txt'

data_file = 'data/data.csv'
data_file_1 = 'data/data_1.csv'
data_file_2 = 'data/data_2.csv'
data_test = 'data/test_data.csv'

reduce_file_size(input_file, data_file, 10)
reduce_file_size(input_file, data_file_1, 1)
reduce_file_size(input_file, data_file_2, 2.5)
reduce_file_size(input_file, data_test, 0.01)