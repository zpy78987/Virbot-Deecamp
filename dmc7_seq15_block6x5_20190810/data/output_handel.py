import numpy as np

def non_zero_mean(np_arr):
    exist = (np_arr != 0)
    num = np_arr.sum(axis=0)
    den = exist.sum(axis=0)
    return num/den

def output_handle(inputs, stype='mean'):
    """
    input: [dim1, dim2, dim3]   # [3571, 30, 30]
    output: [dim1+dim2-1, dim3]
    """
    shape = np.array(inputs).shape
    dim1, dim2, dim3 = shape[0], shape[1], shape[2]
    seq_len = dim1 + dim2 - 1
    matrix = np.zeros([dim1, seq_len, dim3])
    for i in range(dim1):
        matrix[i, i:i+dim2, :] = inputs[i]
        
    output = []
    for i in range(seq_len):
        temp = matrix[:, i, :]
        # print("temp:")
        # print(temp)
        # print(temp)
        if stype == 'mean':
            # print(np.mean(temp, axis=0))
            temp = non_zero_mean(temp)
            output.append(temp)
        if stype == 'max':
            # print(np.max(temp, axis=0))
            output.append(np.max(temp, axis=0))
    return np.array(output)

if __name__ == "__main__":
    data = [
        [
            [1, 1, 1],
            [1, 1, 1]
        ], [
            [2, 2, 2],
            [3, 3, 3]
        ]
    ]
    # print(np.array(data).shape)
    output = output_handle(data, 'mean')
    print(np.array(output).shape)
    print(output)