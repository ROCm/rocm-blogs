---
blogpost: true
date: 5 Feb 2024
author: Sean Song
tags: LLM, AI/ML, Generative AI, Tuning, PyTorch
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Using LoRA for efficient fine-tuning: Fundamental principles"
    "keywords": "LoRA, low-rank adaptation, fine-tuning, ROCm, MNIST, model
  training, generative AI"
    "property=og:locale": "en_US"
---

# Using LoRA for efficient fine-tuning: Fundamental principles

[Low-Rank Adaptation of Large Language Models (LoRA)](https://arxiv.org/abs/2106.09685) is used to
address the challenges of fine-tuning large language models (LLMs). Models like GPT and Llama, which
boast billions of parameters, are typically cost-prohibitive to fine-tune for specific tasks or domains.
LoRA preserves pre-trained model weights and incorporates trainable layers within each model block.
This results in a significant reduction in the number of parameters that need to be fine-tuned and
considerably reduces GPU memory requirements. The key benefit of LoRA is that it substantially
decreases the number of trainable parameters--sometimes by a factor of up to 10,000--leading to a
considerable decrease in GPU resource demands.

## Why LoRA works

Pre-trained LLMs have a low “intrinsic dimension” when they are adapted to a new task, which means
that data can be effectively represented or approximated by a lower-dimensional space while retaining
most of its essential information or structure. We can decompose the new weight matrix for the
adapted task into lower-dimensional (smaller) matrices without losing a lot of important information.
We achieve this by low-rank approximation.

The rank of a matrix is a value that gives you an idea of the matrix’s complexity. A low-rank
approximation of a matrix aims to approximate the original matrix as closely as possible, but with a
lower rank. A lower-rank matrix reduces computational complexity, and thus increases the efficiency of
matrix multiplications. Low-rank decomposition refers to the process of effectively approximating
matrix A by deriving low-rank approximations of A. Singular value decomposition (SVD) is a common
method for low-rank decomposition.

Suppose `W` represents the weight matrix in a given neural network layer and suppose `ΔW` is the
weight update for `W` after a full fine-tuning. We can then decompose the weight update matrix `ΔW`
into two smaller matrices: `ΔW = WA*WB`, where `WA` is an `A × r`-dimensional matrix, and `WB` is an
`r × B`-dimensional matrix. Here, we keep the original weight `W` frozen and only train the new
matrices `WA` and `WB`. This summarizes the LoRA method, which is also illustrated in the following
figure.

![LoRA structure](./images/structure.jpg)

## The benefits of LoRA

* **Reduced resource consumption.**
    Fine-tuning deep learning models typically requires substantial computational resources, which can
    be expensive and time-consuming. LoRA reduces the demand for resources while maintaining high
    performance.

* **Faster iterations.**
    LoRA enables rapid iterations, making it easier to experiment with different fine-tuning tasks and
    adapt models quickly.

* **Improved transfer learning.**
    LoRA enhances the effectiveness of transfer learning, as models with LoRA adapters can be
    fine-tuned with fewer data. This is particularly valuable in situations where labeled data are
    scarce.

* **Broad applicability.**
    LoRA is versatile and can be applied across diverse domains, including natural language processing,
    computer vision, and speech recognition.

* **Lower carbon footprint.**
    By reducing computational requirements, LoRA contributes to a greener and more sustainable
    approach to deep learning.

## Train a neural network using the LoRA technique

Our goal is to train a neural network for the classification of the MNIST database of handwritten digits.
We then fine-tune this network to improve its performance for a category in which it doesn't initially
perform well.

* Hardware: AMD Instinct GPU
* Software:
  * [ROCm 5.7+](https://github.com/RadeonOpenCompute/ROCm)
  * [PyTorch 1.7.0+](https://pytorch.org/)
  * tqdm

The code utilized in this blog post includes contributions sourced from [LoRA implementation](https://github.com/hkproj/pytorch-lora/blob/main/lora.ipynb), with due credit attributed to Umar Jamil.

### Getting started

1. Import the packages.

    ```python
    import torch
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torch.nn as nn
    from tqdm import tqdm
    ```

2. Sets the seed for generating random numbers to make the model deterministic.

    ```python
    # Make torch deterministic
    _ = torch.manual_seed(0)
    ```

3. Load the data set.

    ```python
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Load the MNIST data set
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Create a dataloader for the training
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

    # Load the MNIST test set
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ```

4. Create the neural network to classify the digits (we used code that makes it more complicated in
    order to better showcase LoRA).

    ```python
    # Create an overly expensive neural network to classify MNIST digits
    # Daddy got money, so I don't care about efficiency
    class RichBoyNet(nn.Module):
        def __init__(self, hidden_size_1=1000, hidden_size_2=2000):
            super(RichBoyNet,self).__init__()
            self.linear1 = nn.Linear(28*28, hidden_size_1)
            self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
            self.linear3 = nn.Linear(hidden_size_2, 10)
            self.relu = nn.ReLU()

        def forward(self, img):
            x = img.view(-1, 28*28)
            x = self.relu(self.linear1(x))
            x = self.relu(self.linear2(x))
            x = self.linear3(x)
            return x

    net = RichBoyNet().to(device)
    ```

5. Train the network for one epoch to simulate a complete general pre-training on the data. This
    process takes seconds on an AMD Instinct GPU.

    ```python
    def train(train_loader, net, epochs=5, total_iterations_limit=None):
        cross_el = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        total_iterations = 0

        for epoch in range(epochs):
            net.train()

            loss_sum = 0
            num_iterations = 0

            data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            if total_iterations_limit is not None:
                data_iterator.total = total_iterations_limit
            for data in data_iterator:
                num_iterations += 1
                total_iterations += 1
                x, y = data
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                output = net(x.view(-1, 28*28))
                loss = cross_el(output, y)
                loss_sum += loss.item()
                avg_loss = loss_sum / num_iterations
                data_iterator.set_postfix(loss=avg_loss)
                loss.backward()
                optimizer.step()

                if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                    return

    train(train_loader, net, epochs=1)
    ```

    Epoch 1: 100%|██████████| 6000/6000 [00:20<00:00, 299.74it/s, loss=0.237]

    > [!TIP]
    > Keep a copy of the original weights (clone them) in order to see that the original weights weren't
    > altered after fine-tuning.

    ```python
    original_weights = {}
    for name, param in net.named_parameters():
        original_weights[name] = param.clone().detach()
    ```

### Fine-tuning

1. Choose a digit to fine-tune. The pre-trained network performs poorly on digit 9, so we'll fine-tune
   this.

    ```python
    def test():
        correct = 0
        total = 0

        wrong_counts = [0 for i in range(10)]

        with torch.no_grad():
            for data in tqdm(test_loader, desc='Testing'):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                output = net(x.view(-1, 784))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct +=1
                    else:
                        wrong_counts[y[idx]] +=1
                    total +=1
        print(f'Accuracy: {round(correct/total, 3)}')
        for i in range(len(wrong_counts)):
            print(f'wrong counts for the digit {i}: {wrong_counts[i]}')

    test()
    ```

    ```bash
        Testing: 100%|██████████| 1000/1000 [00:02<00:00, 497.86it/s]

        Accuracy: 0.951
        wrong counts for the digit 0: 35
        wrong counts for the digit 1: 31
        wrong counts for the digit 2: 26
        wrong counts for the digit 3: 81
        wrong counts for the digit 4: 34
        wrong counts for the digit 5: 15
        wrong counts for the digit 6: 74
        wrong counts for the digit 7: 67
        wrong counts for the digit 8: 11
        wrong counts for the digit 9: 116
    ```

2. Visualize how many parameters are in the original network before introducing the LoRA matrices.

    ```python
    # Print the size of the weights matrices of the network
    # Save the count of the total number of parameters
    total_parameters_original = 0
    for index, layer in enumerate([net.linear1, net.linear2, net.linear3]):
        total_parameters_original += layer.weight.nelement() + layer.bias.nelement()
        print(f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape}')
    print(f'Total number of parameters: {total_parameters_original:,}')
    ```

    ```python
        Layer 1: W: torch.Size([1000, 784]) + B: torch.Size([1000])
        Layer 2: W: torch.Size([2000, 1000]) + B: torch.Size([2000])
        Layer 3: W: torch.Size([10, 2000]) + B: torch.Size([10])
        Total number of parameters: 2,807,010
    ```

3. Define the LoRA parameterization.

    ```python
    class LoRAParametrization(nn.Module):
        def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
            super().__init__()
            # Section 4.1 of the paper:
            # We use a random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the
            # beginning of training
            self.lora_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
            self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
            nn.init.normal_(self.lora_A, mean=0, std=1)

            # Section 4.1 of the paper:
            # We then scale ∆Wx by α/r , where α is a constant in r.
            # When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we
            # scale the initialization appropriately.
            # As a result, we simply set α to the first r we try and do not tune it.
            # This scaling helps to reduce the need to retune hyperparameters when we vary r.
            self.scale = alpha / rank
            self.enabled = True

        def forward(self, original_weights):
            if self.enabled:
                # Return W + (B*A)*scale
                return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
            else:
                return original_weights
    ```

4. Add the parameterization to our network. You can learn more about
   [PyTorch parametrizations](https://pytorch.org/tutorials/intermediate/parametrizations.html) on
   PyTorch.org.

    ```python
    import torch.nn.utils.parametrize as parametrize

    def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
        # Only add the parameterization to the weight matrix, ignore the Bias

        # From section 4.2 of the paper:
        # We limit our study to only adapting the attention weights for downstream tasks and freeze the
        # MLP modules (so they are not trained in downstream tasks) both for simplicity and
        # parameter-efficiency.
        # [...]
        # We leave the empirical investigation of [...], and biases to a future work.

        features_in, features_out = layer.weight.shape
        return LoRAParametrization(
            features_in, features_out, rank=rank, alpha=lora_alpha, device=device
        )

    parametrize.register_parametrization(
        net.linear1, "weight", linear_layer_parameterization(net.linear1, device)
    )
    parametrize.register_parametrization(
        net.linear2, "weight", linear_layer_parameterization(net.linear2, device)
    )
    parametrize.register_parametrization(
        net.linear3, "weight", linear_layer_parameterization(net.linear3, device)
    )


    def enable_disable_lora(enabled=True):
        for layer in [net.linear1, net.linear2, net.linear3]:
            layer.parametrizations["weight"][0].enabled = enabled
    ```

5. Display the number of parameters added by LoRA.

    ```python
    total_parameters_lora = 0
    total_parameters_non_lora = 0
    for index, layer in enumerate([net.linear1, net.linear2, net.linear3]):
        total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
        total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
        print(
            f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations["weight"][0].lora_A.shape} + Lora_B: {layer.parametrizations["weight"][0].lora_B.shape}'
        )
    # The non-LoRA parameters count must match the original network
    assert total_parameters_non_lora == total_parameters_original
    print(f'Total number of parameters (original): {total_parameters_non_lora:,}')
    print(f'Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}')
    print(f'Parameters introduced by LoRA: {total_parameters_lora:,}')
    parameters_increment = (total_parameters_lora / total_parameters_non_lora) * 100
    print(f'Parameters increment: {parameters_increment:.3f}%')
    ```

    ```python
        Layer 1: W: torch.Size([1000, 784]) + B: torch.Size([1000]) + Lora_A: torch.Size([1, 784]) + Lora_B: torch.Size([1000, 1])
        Layer 2: W: torch.Size([2000, 1000]) + B: torch.Size([2000]) + Lora_A: torch.Size([1, 1000]) + Lora_B: torch.Size([2000, 1])
        Layer 3: W: torch.Size([10, 2000]) + B: torch.Size([10]) + Lora_A: torch.Size([1, 2000]) + Lora_B: torch.Size([10, 1])
        Total number of parameters (original): 2,807,010
        Total number of parameters (original + LoRA): 2,813,804
        Parameters introduced by LoRA: 6,794
        Parameters increment: 0.242%
    ```

6. Freeze all the parameters of the original network and only fine-tune the ones introduced by LoRA.
   Then, fine-tune the model for digit 9 for 100 batches.

    ```python
    # Freeze the non-Lora parameters
    for name, param in net.named_parameters():
        if 'lora' not in name:
            print(f'Freezing non-LoRA parameter {name}')
            param.requires_grad = False

    # Load the MNIST data set again, by keeping only the digit 9
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    exclude_indices = mnist_trainset.targets == 9
    mnist_trainset.data = mnist_trainset.data[exclude_indices]
    mnist_trainset.targets = mnist_trainset.targets[exclude_indices]
    # Create a dataloader for the training
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

    # Train the network with LoRA only on the digit 9 and only for 100 batches (hoping that it would
    # improve the performance on the digit 9)
    train(train_loader, net, epochs=1, total_iterations_limit=100)
    ```

    ```bash
        Freezing non-LoRA parameter linear1.bias
        Freezing non-LoRA parameter linear1.parametrizations.weight.original
        Freezing non-LoRA parameter linear2.bias
        Freezing non-LoRA parameter linear2.parametrizations.weight.original
        Freezing non-LoRA parameter linear3.bias
        Freezing non-LoRA parameter linear3.parametrizations.weight.original

        Epoch 1:  99%|█████████▉| 99/100 [00:00<00:00, 200.52it/s, loss=0.132]
    ```

7. Verify that the fine-tuning didn't alter the original weights (using only those introduced by LoRA).

    ```python
    # Check that the frozen parameters are still unchanged by the finetuning
    assert torch.all(net.linear1.parametrizations.weight.original == original_weights['linear1.weight'])
    assert torch.all(net.linear2.parametrizations.weight.original == original_weights['linear2.weight'])
    assert torch.all(net.linear3.parametrizations.weight.original == original_weights['linear3.weight'])

    enable_disable_lora(enabled=True)
    # Now let's use layer of net.linear1 as an example to check if the Lora is applied to the model
    # correctly as defined in the LoRAParametrization.forward()
    # The new linear1.weight is obtained by the "forward" function of our LoRA parametrization
    # The original weights have been moved to net.linear1.parametrizations.weight.original
    # More info here: https://pytorch.org/tutorials/intermediate/parametrizations.html#inspecting-a-parametrized-module
    assert torch.equal(net.linear1.weight, net.linear1.parametrizations.weight.original + (net.linear1.parametrizations.weight[0].lora_B @ net.linear1.parametrizations.weight[0].lora_A) * net.linear1.parametrizations.weight[0].scale)

    enable_disable_lora(enabled=False)
    # If we disable LoRA, the linear1.weight is the original one
    assert torch.equal(net.linear1.weight, original_weights['linear1.weight'])
    ```

8. Test the network with LoRA enabled (the digit 9 should be classified better).

    ```python
    # Test with LoRA enabled
    enable_disable_lora(enabled=True)
    test()
    ```

    ```bash
    Testing: 100%|██████████| 1000/1000 [00:02<00:00, 471.08it/s]

    Accuracy: 0.905
    wrong counts for the digit 0: 144
    wrong counts for the digit 1: 34
    wrong counts for the digit 2: 30
    wrong counts for the digit 3: 216
    wrong counts for the digit 4: 161
    wrong counts for the digit 5: 73
    wrong counts for the digit 6: 93
    wrong counts for the digit 7: 100
    wrong counts for the digit 8: 95
    wrong counts for the digit 9: 6
    ```

    Test the network with LoRA disabled (the accuracy and errors counts must be the same as the
    original network).

    ```python
    # Test with LoRA disabled
    enable_disable_lora(enabled=False)
    test()
    ```

    ```bash
    Testing: 100%|██████████| 1000/1000 [00:01<00:00, 517.04it/s]

    Accuracy: 0.951
    wrong counts for the digit 0: 35
    wrong counts for the digit 1: 31
    wrong counts for the digit 2: 26
    wrong counts for the digit 3: 81
    wrong counts for the digit 4: 34
    wrong counts for the digit 5: 15
    wrong counts for the digit 6: 74
    wrong counts for the digit 7: 67
    wrong counts for the digit 8: 11
    wrong counts for the digit 9: 116
    ```

> [!NOTE]
> You might observe that fine-tuning has impacted the accuracies of other labels. This is expected, as
> our fine-tuning was exclusively focused on digit 9.
