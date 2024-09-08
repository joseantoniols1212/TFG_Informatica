import pandas as pd

df = pd.read_csv('comparison_results.csv')

table = '\\begin{tabular}{ccccccc}'
for _, row in df.iterrows():
    table += '\n'
    for i, elem in row.to_dict().items():
        table += ' & '
        table += str(elem)
    table += ' \\\\'
table += '\\end{tabular}'
print(table)
