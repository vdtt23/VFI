
from openpyxl import Workbook, load_workbook
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FILE = PROJECT_ROOT / "experiment_results.xlsx"

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

    for row in range(2, ws.max_row + 1):
        if ws.cell(row=row, column=1).value == config and ws.cell(row=row, column=2).value == epoch:
            ws.cell(row=row, column=3).value = train_l1
            ws.cell(row=row, column=4).value = val_l1
            ws.cell(row=row, column=5).value = time
            break
    else:
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

    values = [
        config, triplets, epochs, batch_size,
        lr, train_l1, val_l1, psnr, ssim,
        total_time, checkpoint
    ]
    for row in range(2, ws.max_row + 1):
        if ws.cell(row=row, column=1).value == config:
            for column, value in enumerate(values, start=1):
                ws.cell(row=row, column=column).value = value
            break
    else:
        ws.append(values)

    wb.save(FILE)
