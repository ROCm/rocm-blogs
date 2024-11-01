import torch
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import torch.distributed as dist

import argparse
import time
import os


def evaluate_on_test_data(model, device, test_loader):
    model.eval()
    correct_num = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += len(labels)
            correct_num += (predicted == labels).sum().item()

    return correct_num / total


def train_epoch(model, dataloader, criterion, optimizer, device, bs):  # bs: bach size
    model.train()  
    total_loss = 0  
    total_correct = 0  

    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        
        optimizer.zero_grad() # reset gradients 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)

        correct_num = (predictions == labels).sum().item()
        total_correct += correct_num

    return total_loss / len(dataloader), total_correct / (len(dataloader)*bs)

def main():

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=100)
    parser.add_argument("--batch_size", type=int, help="Training batch size.", default=1024)
    parser.add_argument("--batch_size_scaled", action="store_true", help="Training batch size for one process.")
    argv = parser.parse_args()


    # we need GPUs
    assert torch.cuda.is_available(), "DDP requires at least one GPU."
    # Initializes the default distributed process group, and this will also initialize the distributed package.
    dist.init_process_group(backend="nccl")


    # training configuration 
    num_epochs =  argv.num_epochs
    batch_size = argv.batch_size
    if argv.batch_size_scaled:
        print("argv.batch_size_scaled", argv.batch_size_scaled, num_epochs, batch_size)
        batch_size //= dist.get_world_size()
    else:
        print("argv.batch_size_scaled", argv.batch_size_scaled, num_epochs, batch_size)
    print(batch_size)
    learning_rate = 0.002
    model_dir = "saved_ddp_models"
    model_filename = "resnet_ddp.pth"
    log_every = 5

    model_filepath = os.path.join(model_dir, model_filename)

    
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Hello from local_rank {local_rank}, global_rank {global_rank}")
    torch.cuda.set_device(local_rank)

    # Wrap the model on the GPU assigned to the current process
    model = torchvision.models.resnet101(weights='IMAGENET1K_V1')
    #change the output since we are using CIFAR100
    model.fc = torch.nn.Linear(model.fc.in_features, 100)

    model = model.to(local_rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=False, transform=transform) 
    test_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False, transform=transform)

    # Sampler that restricts data loading to a subset of the dataset exclusive to the current process
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=DistributedSampler(dataset=train_dataset), shuffle=False, num_workers=16)
    
    # we will only test on rank 0
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9)

    log_epoch = 0
    # Training Loop
    training_start = time.perf_counter()
    for epoch in range(num_epochs):
        log_epoch += 1
        epoch_start_time = time.perf_counter()

        ddp_model.train()
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, local_rank, batch_size)

        torch.cuda.synchronize()
        epoch_end_time = time.perf_counter()

        if  global_rank  == 0:
            print(f"Epoch - {epoch}/{num_epochs}: time - {(epoch_end_time - epoch_start_time):.4f}s || loss_train - {train_loss:.4f} || accuracy_train - {train_acc:.4f}")
            if epoch % log_every == 0:
                test_accuracy = evaluate_on_test_data(model=ddp_model, device=local_rank, test_loader=test_loader)
                print(f"Accuracy on test dataset - {test_accuracy:.4f}")


    torch.cuda.synchronize()
    training_end = time.perf_counter()
    if global_rank == 0:
        print(f"Training took {(training_end - training_start):.4f} s")
        torch.save(ddp_model.state_dict(), model_filepath)

    # Destroy the process group, and deinitialize the distributed package
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()