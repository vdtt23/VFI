
from openpyxl import Workbook, load_workbook
from pathlib import Path

FILE = Path("experiment_results.xlsx")

def init_excel():
    if not FILE.exists():
        wb = Workbook()

        ws1 = wb.active
        ws1.title = "Experiment Summary"
        ws1.append([
            "Config", "Triplets", "Epochs", "Batch Size",
            "Learning Rate", "Best Train L1", "Best Val L1",
            "Test PSNR", "Test SSIM", "Training Time (s)",
            "Checkpoint"
        ])

        ws2 = wb.create_sheet("Epoch History")
        ws2.append([
            "Config", "Epoch", "Train L1",
            "Val L1", "Time (s)"
        ])

        wb.save(FILE)

def log_epoch(config, epoch, train_l1, val_l1, time):
    init_excel()
    wb = load_workbook(FILE)
    ws = wb["Epoch History"]

    ws.append([
        config, epoch, train_l1, val_l1, time
    ])

    wb.save(FILE)

def log_experiment(
    config, triplets, epochs, batch_size, lr,
    train_l1, val_l1, psnr, ssim, total_time,
    checkpoint
):
    init_excel()
    wb = load_workbook(FILE)
    ws = wb["Experiment Summary"]

    ws.append([
        config, triplets, epochs, batch_size,
        lr, train_l1, val_l1, psnr, ssim,
        total_time, checkpoint
    ])

    wb.save(FILE)
