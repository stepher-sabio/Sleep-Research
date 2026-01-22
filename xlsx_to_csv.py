import pandas as pd
import os
from pathlib import Path

input_folder = '/Users/stepher/Desktop/Actigraphy2/data_xlsx'
output_folder = '/Users/stepher/Desktop/Actigraphy2/data_csv'

for file in Path(input_folder).glob('*.xlsx'):
    try:
        df = pd.read_excel(file)
        output_file = os.path.join(output_folder, file.stem + '.csv')
        
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"✗ Error converting {file.name}: {e}")
print("Conversion complete!")

