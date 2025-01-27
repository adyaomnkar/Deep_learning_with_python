# Deep learning with Python
Here’s a **handy list of essential and useful tensor methods** you’ll frequently use with **PyTorch** (`torch`) while working with tensors. I'll categorize them for better clarity. 🚀

![image](https://github.com/user-attachments/assets/f01ac9ce-fcfa-4555-8e5a-ae3b32806b2d)

## **1. Tensor Creation Methods**
These are used to create new tensors:  
- `torch.tensor(data)`: Creates a tensor from a Python list or NumPy array.  
- `torch.zeros(size)`: Creates a tensor filled with zeros.  
- `torch.ones(size)`: Creates a tensor filled with ones.  
- `torch.full(size, value)`: Creates a tensor filled with a specific value.  
- `torch.arange(start, end, step)`: Creates a tensor with values in a range.  
- `torch.linspace(start, end, steps)`: Creates evenly spaced values.  
- `torch.eye(n)`: Creates an identity matrix (diagonal is 1).  
- `torch.empty(size)`: Creates an uninitialized tensor.  
- `torch.randn(size)`: Creates a tensor with random values from a normal distribution.  
- `torch.randint(low, high, size)`: Creates a tensor with random integers.  
- `torch.rand(size)`: Creates a tensor with random values in `[0, 1)`.  
- `torch.from_numpy(ndarray)`: Converts a NumPy array to a tensor.  

---

## **2. Tensor Shape and Resizing**
Useful for manipulating tensor dimensions:  
- `tensor.shape`: Returns the shape of the tensor.  
- `tensor.view(size)`: Reshapes the tensor (without copying data).  
- `tensor.reshape(size)`: Reshapes, but may copy data if necessary.  
- `tensor.unsqueeze(dim)`: Adds a new dimension at the specified position.  
- `tensor.squeeze(dim)`: Removes dimensions of size 1.  
- `tensor.permute(dims)`: Rearranges dimensions of a tensor.  
- `tensor.transpose(dim0, dim1)`: Swaps two specified dimensions.  
- `torch.cat(tensors, dim)`: Concatenates tensors along a dimension.  
- `torch.stack(tensors, dim)`: Stacks tensors along a new dimension.  

---

## **3. Indexing, Slicing, and Joining**
For accessing or modifying specific parts:  
- `tensor[index]`: Basic indexing.  
- `tensor[: ,1:4]`: Slicing.  
- `tensor.index_select(dim, indices)`: Selects along a specific dimension.  
- `tensor.gather(dim, index)`: Gathers values along an axis based on indices.  
- `tensor.scatter_(dim, index, value)`: Writes values at specific indices.  
- `tensor.split(size_or_sections, dim)`: Splits tensor into chunks.  

---

## **4. Mathematical Operations**
For arithmetic and element-wise operations:  
- `torch.add(tensor1, tensor2)` or `+`: Addition.  
- `torch.sub(tensor1, tensor2)` or `-`: Subtraction.  
- `torch.mul(tensor1, tensor2)` or `*`: Element-wise multiplication.  
- `torch.div(tensor1, tensor2)` or `/`: Division.  
- `torch.matmul(tensor1, tensor2)`: Matrix multiplication.  
- `tensor.pow(exponent)`: Exponentiation.  
- `torch.sum(tensor, dim)`: Sums elements along a dimension.  
- `torch.mean(tensor, dim)`: Computes mean.  
- `torch.max(tensor, dim)`: Returns max value and its index.  
- `torch.min(tensor, dim)`: Returns min value and its index.  
- `torch.exp(tensor)`: Exponential.  
- `torch.log(tensor)`: Natural logarithm.  
- `torch.sqrt(tensor)`: Square root.  

---

## **5. Comparison Operations**
To compare values in tensors:  
- `torch.eq(tensor1, tensor2)`: Element-wise equality check.  
- `torch.ne(tensor1, tensor2)`: Not equal.  
- `torch.gt(tensor1, tensor2)`: Greater than.  
- `torch.lt(tensor1, tensor2)`: Less than.  
- `torch.ge(tensor1, tensor2)`: Greater than or equal.  
- `torch.le(tensor1, tensor2)`: Less than or equal.  

---

## **6. Reduction Operations**
To reduce tensors to a smaller shape:  
- `torch.argmax(tensor, dim)`: Returns index of max value along a dimension.  
- `torch.argmin(tensor, dim)`: Returns index of min value.  
- `torch.cumsum(tensor, dim)`: Cumulative sum.  
- `torch.cumprod(tensor, dim)`: Cumulative product.  
- `torch.prod(tensor, dim)`: Product of elements.  

---

## **7. Tensor Cloning and Memory Management**
To handle copying and saving:  
- `tensor.clone()`: Creates a copy of a tensor.  
- `tensor.detach()`: Detaches a tensor from the computation graph (useful in training).  
- `tensor.requires_grad_()`: Sets `requires_grad` property (for autograd).  
- `tensor.numpy()`: Converts a tensor to a NumPy array.  

---

## **8. GPU and Device Management**
For moving tensors between CPU and GPU:  
- `tensor.to(device)`: Moves tensor to a specified device.  
- `tensor.cuda()`: Moves tensor to GPU (if available).  
- `tensor.cpu()`: Moves tensor to CPU.  
- `torch.device("cuda")`: Specifies GPU.  
- `torch.device("cpu")`: Specifies CPU.  

---

## **9. Other Handy Methods**
- `torch.flatten(tensor, start_dim, end_dim)`: Flattens a tensor.  
- `tensor.item()`: Gets a single value as a Python scalar.  
- `torch.norm(tensor)`: Computes the norm (magnitude).  
- `torch.save(tensor, filepath)`: Saves tensor to a file.  
- `torch.load(filepath)`: Loads tensor from a file.  



This list should cover **most of what you need** for everyday PyTorch tensor operations. If you're curious about any specific method, let me know—I’ll explain with examples! 😊
