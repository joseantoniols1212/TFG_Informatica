import pandas as pd
import os

file_name = './comparison_results.csv'
precision = 8
traspose = False

df = pd.read_csv(file_name)

if traspose:
    df = df.transpose()

n_colums = df.shape[1]

table = '\\begin{tabular}{' + 'c' * n_colums + '}\n'
for i, header in enumerate(list(df.columns.values)):
    table += str(header)
    if i != df.shape[1]-1:
        table += ' & '
table += ' \\\\'
for _, row in df.iterrows():
    table += '\n'
    for i, (_, elem) in enumerate(row.to_dict().items()):
        if type(elem) is float:
            table += f'{elem:.{precision}f}'
        else:
            table += str(elem)
        if i != df.shape[1]-1:
            table += ' & '
    table += ' \\\\'
table += '\n\\end{tabular}'

txt_file_name = os.path.splitext(file_name)[0] + ".txt"
with open(txt_file_name, "w") as text_file:
    text_file.write(table)
