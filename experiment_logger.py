
from openpyxl import Workbook, load_workbook
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FILE = PROJECT_ROOT / "experiment_results.xlsx"

SUMMARY_HEADERS = [
    "Config", "Triplets", "Epochs", "Batch Size",
    "Learning Rate", "Best Train L1", "Best Val L1",
    "Linear Blending PSNR", "Linear Blending SSIM",
    "U-Net PSNR", "U-Net SSIM", "Training Time (s)",
    "Checkpoint",
]


def init_excel():
    if not FILE.exists():
        wb = Workbook()

        ws1 = wb.active
        ws1.title = "Experiment Summary"
        ws1.append(SUMMARY_HEADERS)

        ws2 = wb.create_sheet("Epoch History")
        ws2.append([
            "Config", "Epoch", "Train L1",
            "Val L1", "Time (s)"
        ])

        wb.save(FILE)


def _ensure_summary_columns(ws):
    headers = {
        ws.cell(row=1, column=column).value: column
        for column in range(1, ws.max_column + 1)
    }

    # Preserve metrics written by older versions as U-Net metrics.
    for old_name, new_name in (
        ("Test PSNR", "U-Net PSNR"),
        ("Test SSIM", "U-Net SSIM"),
    ):
        if old_name in headers and new_name not in headers:
            ws.cell(row=1, column=headers[old_name]).value = new_name
            headers[new_name] = headers.pop(old_name)

    for header in SUMMARY_HEADERS:
        if header not in headers:
            headers[header] = ws.max_column + 1
            ws.cell(row=1, column=headers[header]).value = header
    return headers


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
    headers = _ensure_summary_columns(ws)
    values = {
        "Config": config,
        "Triplets": triplets,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "Learning Rate": lr,
        "Best Train L1": train_l1,
        "Best Val L1": val_l1,
        "Training Time (s)": total_time,
        "Checkpoint": checkpoint,
    }
    if psnr is not None:
        values["U-Net PSNR"] = psnr
    if ssim is not None:
        values["U-Net SSIM"] = ssim

    for row in range(2, ws.max_row + 1):
        if ws.cell(row=row, column=headers["Config"]).value == config:
            for header, value in values.items():
                ws.cell(row=row, column=headers[header]).value = value
            break
    else:
        row = ws.max_row + 1
        for header, value in values.items():
            ws.cell(row=row, column=headers[header]).value = value

    wb.save(FILE)


def log_evaluation(config, baseline_psnr, baseline_ssim, unet_psnr, unet_ssim):
    init_excel()
    wb = load_workbook(FILE)
    ws = wb["Experiment Summary"]
    headers = _ensure_summary_columns(ws)

    row = None
    for candidate in range(2, ws.max_row + 1):
        if ws.cell(row=candidate, column=headers["Config"]).value == config:
            row = candidate
            break
    if row is None:
        row = ws.max_row + 1
        ws.cell(row=row, column=headers["Config"]).value = config

    metrics = {
        "Linear Blending PSNR": baseline_psnr,
        "Linear Blending SSIM": baseline_ssim,
        "U-Net PSNR": unet_psnr,
        "U-Net SSIM": unet_ssim,
    }
    for header, value in metrics.items():
        ws.cell(row=row, column=headers[header]).value = value

    wb.save(FILE)
