import torch
from SmoothValue import SmoothedValue, MetricLogger, ConfusionMatrix
from criterion import criterion

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, mask in metric_logger.log_every(data_loader, print_freq, header):
        image, mask = image.to(device), mask.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, mask, join=False)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        for image, mask in metric_logger.log_every(data_loader, 100, header):
            image, mask = image.to(device), mask.to(device)
            output = model(image)['out']

            confmat.update(mask.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat