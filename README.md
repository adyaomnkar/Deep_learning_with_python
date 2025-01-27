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

Alright, let’s dive deep into the mathematics behind neural networks (NNs) and convolutional neural networks (CNNs) as your AI professor with a knack for relatable examples! 😊 I’ll break it into digestible chunks and connect it with real-life scenarios where applicable.

---

### **Mathematics in Neural Networks (NNs)**

Neural networks involve several layers of mathematics, and they build upon the core concepts you already know from linear regression. Here are the key components:

---

#### 1. **Linear Transformation**  
At the heart of a neural network, each neuron performs the same fundamental task as linear regression:
\[
z = W \cdot X + b
\]
- **\( X \)**: Input data (features).
- **\( W \)**: Weights (how much importance is given to each input feature).
- **\( b \)**: Bias (a constant term to shift the output).

Think of this as a student studying for exams:
- **Input \( X \)**: Time spent studying different subjects.  
- **Weight \( W \)**: Importance of each subject in the exam.  
- **Bias \( b \)**: A bonus mark from the teacher.

This is a linear transformation for one neuron. But neural networks combine this across multiple neurons to form layers!

---

#### 2. **Activation Functions**  
After the linear transformation, the output (\( z \)) is passed through a nonlinear function called an **activation function**. Why? To allow the model to learn complex patterns.

Common activation functions:
- **Sigmoid**: Squashes output to a range between 0 and 1.
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
  Example: Predicting the probability of a binary outcome.

- **ReLU (Rectified Linear Unit)**: Replaces negative values with 0.
  \[
  \text{ReLU}(z) = \max(0, z)
  \]
  Example: Imagine stepping on a treadmill; negative speed doesn’t make sense—it’s either 0 or positive.

- **Softmax**: Converts raw scores into probabilities for multi-class classification.
  \[
  P(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
  \]

---

#### 3. **Loss Function**
The loss function measures how far the model's predictions (\( y_{\text{pred}} \)) are from the actual values (\( y_{\text{true}} \)).

Common loss functions in neural networks:
- **Mean Squared Error (MSE)** (for regression):
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_{\text{true}} - y_{\text{pred}})^2
  \]
  Example: Predicting house prices and comparing them to actual prices.

- **Cross-Entropy Loss** (for classification):
  \[
  \text{Cross-Entropy} = -\frac{1}{n} \sum_{i=1}^n y_i \log(p_i)
  \]
  Example: Measuring how well probabilities match the true labels in image classification.

---

#### 4. **Gradient Descent**
Gradient Descent optimizes the weights (\( W \)) and bias (\( b \)) to minimize the loss. It calculates the gradient (slope of the loss function) and updates parameters iteratively:
\[
W = W - \eta \cdot \frac{\partial L}{\partial W}
\]
- **\(\eta\)**: Learning rate (how big the steps are in each update).
- **\(\frac{\partial L}{\partial W}\)**: Gradient (how much the loss changes when weights are adjusted).

Example: Think of climbing down a hill to reach the lowest point (minimum loss). The steepness of the hill tells you how big your steps should be.

---

#### 5. **Backpropagation**
Backpropagation is how neural networks calculate gradients for **all layers** (not just one layer like in linear regression). It uses the **chain rule** from calculus:
\[
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial W_1}
\]
- The gradients are passed backward through the layers to update weights.

Real-life analogy: Imagine you're passing corrections in a team game. The feedback moves backward, starting from the person closest to the goal.

---

---

### **Mathematics in Convolutional Neural Networks (CNNs)**

Now, let’s get into CNNs. They’re a specialized type of NN designed to work with **images**, and their math revolves around spatial data.

---

#### 1. **Convolution Operation**
Instead of using fully connected layers, CNNs use a **convolution** to extract features like edges, textures, or patterns:
\[
z[i, j] = \sum_{m=0}^{k} \sum_{n=0}^{k} X[i+m, j+n] \cdot W[m, n]
\]
- **\( X \)**: Input image (pixel values).
- **\( W \)**: Filter/kernel (a small matrix of weights).
- **Output \( z \)**: Feature map (the result of the convolution).

Example: Think of a photo filter app. A filter (like blur) slides over an image and adjusts pixels based on the surrounding values.

---

#### 2. **Pooling**
After convolution, CNNs use **pooling** to reduce the spatial dimensions of the image while retaining important features.  
- Common pooling methods:
  - **Max Pooling**: Takes the maximum value from each patch.
  - **Average Pooling**: Takes the average value from each patch.

Example: Max pooling is like summarizing your marks by keeping only the highest score from each subject group.

---

#### 3. **Flattening**
After several convolution and pooling layers, the output is **flattened** into a vector to feed into fully connected layers.

---

#### 4. **Backpropagation in CNNs**
The weights of the filters (\( W \)) are updated during backpropagation, just like in regular NNs. Gradients flow backward through the convolution and pooling layers.

---

---

### **Putting It All Together with Examples**
Let’s say we’re building a CNN to classify images of cats and dogs:
1. **Convolution**: Extract edges (e.g., ears, tails, etc.) from the image.
2. **Pooling**: Downsample the features to make computations efficient.
3. **Activation**: Apply ReLU to add non-linearity and focus on important features.
4. **Fully Connected Layers**: Combine the features to predict probabilities for "cat" or "dog."
5. **Loss Calculation**: Measure the error in predictions using cross-entropy loss.
6. **Gradient Descent**: Update the filters and weights to improve accuracy.

---

### Final Tip: Understand the Flow
Think of neural networks and CNNs as building blocks:
- Linear transformation ➡ Activation function ➡ Loss function ➡ Gradient descent ➡ Backpropagation.
CNNs add an extra layer of power with convolutions and pooling to handle images effectively.

Does this explain the math in a way that feels relatable?



This list should cover **most of what you need** for everyday PyTorch tensor operations. If you're curious about any specific method, let me know—I’ll explain with examples! 😊
