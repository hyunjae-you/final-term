import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


df = pd.read_csv('test_results.csv', header=None)
df.columns = ['Material ID', 'True', 'Predicted']

plt.figure(figsize=(6, 6))
plt.scatter(df['True'], df['Predicted'], alpha=0.7)
plt.plot([df['True'].min(), df['True'].max()],
         [df['True'].min(), df['True'].max()],
         'r--')

plt.xlabel('True Band Gap')
plt.ylabel('Predicted Band Gap')
plt.title('CGCNN Prediction vs. Ground Truth')
plt.grid(True)
plt.tight_layout()

plt.savefig('prediction_vs_true.jpg', dpi=300)

