import os
import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from val_func import val, val_prenet

def train_general(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, num_epochs, store_name, device):
    exp_dir = 'outputs/' + store_name
    
    # Train the model for the specified number of epochs
    max_val_acc = 0
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0


        # Iterate over the batches of the train loader
        for inputs, labels in tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
          
            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Calculate the train loss and accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        

        lr_scheduler.step()

        #evaluation the model
        val_loss, val_acc, val5_acc = val(model, criterion, val_loader, device)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
        
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Epoch %d | train_acc = %.5f | train_loss = %.5f |\n' % (
                epoch, 100. * float(train_acc), train_loss,))

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), './outputs/' + store_name + '/model.pth')
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write(
                'Epoch %d, top1 = %.5f, top5 = %.5f, val_loss = %.6f\n' % (
                    epoch, val_acc, val5_acc, val_loss))


def train_prenet(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, num_epochs, store_name, device):
    exp_dir = 'outputs/' + store_name
    KLLoss = nn.KLDivLoss(reduction="batchmean")

    # Train the model for the specified number of epochs
    max_val_acc = 0
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        running_loss = 0.0
        running_corrects = 0
        u1 = 1
        u2 = 0.5


        # Iterate over the batches of the train loader
        for inputs, labels in tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            # labels = labels.to(device)
            # labels should not be string
            label_to_int = {label: idx for idx, label in enumerate(set(labels))}
            numeric_labels = [label_to_int[label] for label in labels]
            labels = torch.tensor(numeric_labels, dtype=torch.long).to(device)
          
            # Step 1
            optimizer.zero_grad()
            #inputs1 = jigsaw_generator(inputs, 8)
            _, _, _, _, output_1, _, _ = model(inputs, False)
            #print(output_1.shape)
            loss1 = criterion(output_1, labels) * 1
            loss1.backward()
            # if (idx+1) % GRADIENT_ACCUMULATION == 0:
            optimizer.step()

            # Step 2
            optimizer.zero_grad()
            #inputs2 = jigsaw_generator(inputs, 4)

            _, _, _, _, _, output_2, _, = model(inputs, False)
            #print(output_2.shape)
            loss2 = criterion(output_2, labels) * 1
            loss2.backward()
            # if (idx+1) % GRADIENT_ACCUMULATION == 0:
            optimizer.step()

            # Step 3
            optimizer.zero_grad()
            #inputs3 = jigsaw_generator(inputs, 2)
            _, _, _, _, _, _, output_3 = model(inputs, False)
                #print(output_3.shape)
            loss3 = criterion(output_3, labels) * 1
            loss3.backward()
            # if (idx+1) % GRADIENT_ACCUMULATION == 0:
            optimizer.step()


            optimizer.zero_grad()
            x1, x2, x3, output_concat, _, _, _ = model(inputs,True)
            concat_loss = criterion(output_concat, labels) * 2


            #loss4 = -KLLoss(F.softmax(x1, dim=1), F.softmax(x2, dim=1)) / batch_size
            #loss5 = -KLLoss(F.softmax(x1, dim=1), F.softmax(x3, dim=1)) / batch_size
            loss6 = -KLLoss(F.softmax(x2, dim=1), F.softmax(x1, dim=1))
            #loss7 = -KLLoss(F.softmax(x2, dim=1), F.softmax(x3, dim=1)) / batch_size
            loss8 = -KLLoss(F.softmax(x3, dim=1), F.softmax(x1, dim=1))
            loss9 = -KLLoss(F.softmax(x3, dim=1), F.softmax(x2, dim=1))

            Klloss = loss6 + loss8 + loss9

            totalloss = u1 * concat_loss + u2 * Klloss
            totalloss.backward()
            # if (idx+1) % GRADIENT_ACCUMULATION == 0:
            optimizer.step()

            #  training log
            _, predicted = torch.max(output_concat.data, 1)
            running_corrects += predicted.eq(labels.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

        # Calculate the train loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        

        lr_scheduler.step()

        #evaluation the model
        val_loss, val_acc, val5_acc, val_acc_com, val5_acc_com = val_prenet(model, criterion, val_loader, device)

        # Print the epoch results
        print(
            'Epoch: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%%' % (
            epoch, train_loss1 / len(train_loader.dataset), train_loss2 / len(train_loader.dataset),
            train_loss3 / len(train_loader.dataset), train_loss4 / len(train_loader.dataset), train_loss,
                100. * float(train_acc)))
        
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Epoch %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                epoch, 100. * float(train_acc), train_loss, train_loss1 / len(train_loader.dataset), train_loss2 / len(train_loader.dataset), train_loss3 / len(train_loader.dataset),
                train_loss4 / len(train_loader.dataset)))

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), './outputs/' + store_name + '/model.pth')
        
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write(
                'Epoch %d, top1 = %.5f, top5 = %.5f, top1_combined = %.5f, top5_combined = %.5f, test_loss = %.6f\n' % (
                    epoch, val_acc, val5_acc, val_acc_com, val5_acc_com, val_loss))