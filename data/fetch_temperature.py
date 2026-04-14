import pandas as pd
from datetime import datetime

def print_metadata(file_path):
    with open(file_path, 'r') as f:
        data_section = False
        count = 0
        for line in f:
            if line.strip().startswith("@data"):
                data_section = True
                continue
            if data_section and line.strip():
                # 값 부분 제외, 메타데이터 부분만 출력
                parts = line.strip().split(":")
                print(f"Line {count}: 메타 부분 = {parts[:-1]}")
                count += 1
                if count >= 5:
                    break

def parse_tsf_temperature(file_path):
    series_dict = {}  # station_id: values
    
    with open(file_path, 'r') as f:
        data_section = False
        for line in f:
            line = line.strip()
            if line.startswith("@data"):
                data_section = True
                continue
            if data_section and line:
                parts = line.split(":")
                series_id = parts[0]
                station_id = parts[1]
                attribute = parts[2]
                values_str = parts[-1].strip()
                
                if attribute == "T_MEAN":
                    values = [float(v) for v in values_str.split(",") if v.strip()]
                    series_dict[station_id] = values
    
    return series_dict

if __name__ == "__main__":
    temp_data = parse_tsf_temperature("temperature_rain_dataset_without_missing_values.tsf")
    print(f"관측소 수: {len(temp_data)}")
    print(f"첫 번째 시계열 길이: {len(list(temp_data.values())[0])}")


