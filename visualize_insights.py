import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Generate simulated visualization matrix
np.random.seed(42)
n_points = 300
api_errors = np.random.uniform(0.0, 0.06, n_points)
base_churn = 1 / (1 + np.exp(-10 * (api_errors - 0.03))) 
tickets = np.random.poisson(lam=1.5, size=n_points) + (api_errors * 20).astype(int)
churn_prob = np.clip(base_churn + (tickets * 0.1) + np.random.normal(0, 0.1, n_points), 0, 1)

df = pd.DataFrame({'API_Error_Rate': api_errors, 'Churn_Probability': churn_prob, 'Support_Tickets': tickets})

# Plot Executive Diagnostic Chart
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
scatter = plt.scatter(df['API_Error_Rate']*100, df['Churn_Probability'], 
                      c=df['Support_Tickets'], cmap='coolwarm', alpha=0.8, edgecolors='none', s=60)

plt.axvline(x=3.0, color='red', linestyle='--', linewidth=2, label='Critical 3% Operational Alert Boundary')
plt.title('Boardroom Diagnostic: API Error Threshold vs. Escalating Customer Risk', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Customer API Error Frequency Rate (%)', fontsize=12)
plt.ylabel('Algorithmic Churn Probability Score', fontsize=12)
cb = plt.colorbar(scatter)
cb.set_label('Accumulated Customer Support Tickets', fontsize=11)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()

# Save figure out to project portfolio directories
os.makedirs("output_charts", exist_ok=True)
plt.savefig('output_charts/executive_api_threshold_plot.png', dpi=300)
print("[SUCCESS] Production chart generated in 'output_charts/executive_api_threshold_plot.png'")
