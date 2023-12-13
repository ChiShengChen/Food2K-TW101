import torch
import tqdm

def val(model, criterion, val_loader, device):
    model.eval()

    # Initialize the running loss and accuracy
    val_loss = 0.0
    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects5 = 0
    
    # Iterate over the batches of the validation loader
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(val_loader):
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, top3_pos = torch.topk(outputs.data, 5)

            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            batch_corrects4 = torch.sum((top3_pos[:, 3] == labels)).data.item()
            batch_corrects5 = torch.sum((top3_pos[:, 4] == labels)).data.item()
            val_corrects5 += (batch_corrects5 + batch_corrects4 + batch_corrects3 + batch_corrects2 + batch_corrects1)

            loss = criterion(outputs, labels)

            # Update the running loss and accuracy
            val_loss += loss.item() * inputs.size(0)

    # Calculate the validation loss and accuracy
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = float(val_corrects1) / len(val_loader.dataset)
    val5_acc = float(val_corrects5) / len(val_loader.dataset)
    return val_loss, val_acc, val5_acc

def val_prenet(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects5 = 0

    val_en_corrects1 = 0
    val_en_corrects2 = 0
    val_en_corrects5 = 0
    with torch.no_grad():
        for (inputs, targets) in tqdm.tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, _, _, output_concat, output1, output2, output3 = model(inputs, True)
            outputs_com = output1 + output2 + output3 + output_concat

            loss = criterion(output_concat, targets)
            val_loss += loss.item() * inputs.size(0)
            _, top3_pos = torch.topk(output_concat.data, 5)
            _, top3_pos_en = torch.topk(outputs_com.data, 5)


            batch_corrects1 = torch.sum((top3_pos[:, 0] == targets)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == targets)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == targets)).data.item()
            batch_corrects4 = torch.sum((top3_pos[:, 3] == targets)).data.item()
            batch_corrects5 = torch.sum((top3_pos[:, 4] == targets)).data.item()
            val_corrects5 += (batch_corrects5 + batch_corrects4 + batch_corrects3 + batch_corrects2 + batch_corrects1)

            batch_corrects1 = torch.sum((top3_pos_en[:, 0] == targets)).data.item()
            val_en_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos_en[:, 1] == targets)).data.item()
            val_en_corrects2+= (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos_en[:, 2] == targets)).data.item()
            batch_corrects4 = torch.sum((top3_pos_en[:, 3] == targets)).data.item()
            batch_corrects5 = torch.sum((top3_pos_en[:, 4] == targets)).data.item()
            val_en_corrects5 += (batch_corrects5 + batch_corrects4 + batch_corrects3 + batch_corrects2 + batch_corrects1)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_corrects1 / len(val_loader.dataset)
    val5_acc = val_corrects5 / len(val_loader.dataset)
    val_acc_en = val_en_corrects1 / len(val_loader.dataset)
    val5_acc_en = val_en_corrects5 / len(val_loader.dataset)
    return val_loss, val_acc, val5_acc, val_acc_en, val5_acc_en 