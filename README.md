## In Deep Learning:
Weights are the learnable parameters of the model.

Gradients are the partial derivatives of the loss with respect to each weight.

We use gradients to update weights via backpropagation.

![image](https://github.com/user-attachments/assets/71071673-8d69-4cec-910b-cbe1a44ba94e)

#### 1. Forward pass → Get model predictions
outputs = model(inputs)
#### 2. Calculate loss
loss = loss_fn(outputs, targets)
#### 3. Backward pass → Calculate gradients
loss.backward()
#### 4. Update weights using gradients
optimizer.step()
#### 5. Clear old gradients
optimizer.zero_grad()
