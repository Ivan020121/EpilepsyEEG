import csv

windows = [1, 2, 5, 10, 20]
chunks = [1, 2, 5, 10, 20]
experts = ["A", "B", "C"]

result_datas = []

for expert in experts:
    for window in windows:
        for chunk in chunks:
            for i in range(5):
                file_path = f"./expert_{expert}_{window}sec_{chunk}chunk_64Hz_resnet50_{i}fold.txt"
                try:
                    with open(file_path, "r") as file:
                        last_line = file.readlines()[-1].strip()
                    data = last_line.split("\t")[3:]
                    data = [float(x) for x in data]
                    data_rounded = [round(x, 5) for x in data]
                    result_data = [window, chunk, i] + data_rounded
                    result_data.append(expert)
                    result_datas.append(result_data)
                except (FileNotFoundError, IndexError):
                    result_datas.append([window, chunk, i, 0, 0, 0, 0, expert])
                    continue

with open("./result.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(result_datas)
